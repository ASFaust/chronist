"""HistStore: per-step histogram records (separate from MetricStore).

Idempotent on (run, key, step): the first write wins, like scalars. Reads
return every record in [start, end), or an evenly strided subset (always
with the first and last step) when there are more than ``max_steps``.
"""
from __future__ import annotations

import os
import sqlite3
import threading
from typing import Dict, List, Optional

import numpy as np

try:
    from typing import Protocol
except ImportError:  # pragma: no cover
    Protocol = object  # type: ignore

from ..codec import compress_maybe, decompress
from ..hist import HistRecord, subsample_indices
from .base import StoreStats
from .sqlite_rows import connect


class HistStore(Protocol):
    def write(self, run_id: int, key_records: Dict[str, List[HistRecord]]) -> None: ...
    def read(self, run_id: int, key: str, start: Optional[int] = None, end: Optional[int] = None,
             max_steps: Optional[int] = None) -> List[HistRecord]: ...
    def at(self, run_id: int, key: str, step: int) -> Optional[HistRecord]: ...
    def keys(self, run_id: int) -> List[str]: ...
    def last(self, run_id: int, key: str) -> Optional[HistRecord]: ...
    def delete_run(self, run_id: int) -> None: ...
    def delete_range(self, run_id: int, start: int) -> None: ...
    def stats(self) -> StoreStats: ...
    def close(self) -> None: ...


SCHEMA = """
CREATE TABLE IF NOT EXISTS keys (key_id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE);
CREATE TABLE IF NOT EXISTS hist (
    run_id INTEGER NOT NULL, key_id INTEGER NOT NULL, step INTEGER NOT NULL,
    wall_time REAL, count INTEGER NOT NULL, nan_count INTEGER NOT NULL,
    posinf_count INTEGER NOT NULL, neginf_count INTEGER NOT NULL,
    min REAL, max REAL, mean REAL, std REAL, lo REAL, hi REAL,
    underflow INTEGER NOT NULL DEFAULT 0, overflow INTEGER NOT NULL DEFAULT 0,
    fixed INTEGER NOT NULL DEFAULT 0, wide INTEGER NOT NULL DEFAULT 0,
    counts BLOB NOT NULL, raw BLOB,
    PRIMARY KEY (run_id, key_id, step)
) WITHOUT ROWID;
"""

COLS = ("step, wall_time, count, nan_count, posinf_count, neginf_count, min, max, mean, std, lo, hi, "
        "underflow, overflow, fixed, wide, counts, raw")


def _nn(x):
    return None if x is None or x != x or x in (float("inf"), float("-inf")) else float(x)


def _row_to_record(row) -> HistRecord:
    (step, wall, count, nan, pinf, ninf, mn, mx, mean, std, lo, hi, under, over, fixed, wide, counts, raw) = row
    f = lambda v: float("nan") if v is None else v  # noqa: E731
    c = np.frombuffer(decompress(counts), dtype="<u8" if wide else "<u4").astype(np.int64)
    return HistRecord(step=step, count=count, nan_count=nan, posinf_count=pinf, neginf_count=ninf,
                      min=f(mn), max=f(mx), mean=f(mean), std=f(std), lo=f(lo), hi=f(hi), counts=c,
                      underflow=under, overflow=over, fixed=bool(fixed),
                      raw=None if raw is None else np.frombuffer(decompress(raw), dtype="<f4").copy(),
                      wall_time=wall)


class SqliteHistStore:
    name = "sqlite_hist"

    def __init__(self, path: str, readonly: bool = False):
        self.path = path
        self.readonly = readonly
        if not readonly:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            con = connect(path)
            con.execute("PRAGMA auto_vacuum=INCREMENTAL")
            con.executescript(SCHEMA)
            con.close()
        self._local = threading.local()
        self._wcon: Optional[sqlite3.Connection] = None
        self._wlock = threading.Lock()
        self._key_ids: Dict[str, int] = {}

    def _rcon(self):
        con = getattr(self._local, "con", None)
        if con is None:
            con = connect(self.path, readonly=True)
            self._local.con = con
        return con

    def _writer(self):
        if self.readonly:
            raise RuntimeError("store opened read-only")
        if self._wcon is None:
            self._wcon = connect(self.path)
        return self._wcon

    def _key_id(self, con, key: str, create: bool) -> Optional[int]:
        kid = self._key_ids.get(key)
        if kid is not None:
            return kid
        row = con.execute("SELECT key_id FROM keys WHERE name=?", (key,)).fetchone()
        if row is None:
            if not create:
                return None
            con.execute("INSERT INTO keys(name) VALUES (?)", (key,))
            row = con.execute("SELECT key_id FROM keys WHERE name=?", (key,)).fetchone()
        self._key_ids[key] = row[0]
        return row[0]

    def _tx(self, fn):
        with self._wlock:
            con = self._writer()
            con.execute("BEGIN")
            try:
                fn(con)
                con.execute("COMMIT")
            except BaseException:
                self._key_ids.clear()
                con.execute("ROLLBACK")
                raise

    # ---- writes ----
    def write(self, run_id: int, key_records: Dict[str, List[HistRecord]]) -> None:
        def _w(con):
            for key, recs in key_records.items():
                kid = self._key_id(con, key, create=True)
                rows = []
                for r in recs:
                    wide = bool(r.counts.size and int(r.counts.max()) >= 2 ** 32)
                    counts = compress_maybe(r.counts.astype("<u8" if wide else "<u4").tobytes())
                    raw = None if r.raw is None else compress_maybe(r.raw.astype("<f4").tobytes())
                    rows.append((run_id, kid, int(r.step), r.wall_time, int(r.count), int(r.nan_count),
                                 int(r.posinf_count), int(r.neginf_count), _nn(r.min), _nn(r.max), _nn(r.mean),
                                 _nn(r.std), _nn(r.lo), _nn(r.hi), int(r.underflow), int(r.overflow),
                                 int(r.fixed), int(wide), counts, raw))
                con.executemany(f"INSERT OR IGNORE INTO hist(run_id, key_id, {COLS}) "
                                f"VALUES ({','.join('?' * 20)})", rows)
        self._tx(_w)

    def delete_run(self, run_id: int) -> None:
        self._tx(lambda con: con.execute("DELETE FROM hist WHERE run_id=?", (run_id,)))

    def delete_range(self, run_id: int, start: int) -> None:
        self._tx(lambda con: con.execute("DELETE FROM hist WHERE run_id=? AND step>=?", (run_id, int(start))))

    # ---- reads ----
    def keys(self, run_id: int) -> List[str]:
        # DISTINCT over the PK prefix: SQLite skips through it via the index
        rows = self._rcon().execute(
            "SELECT k.name FROM keys k WHERE EXISTS (SELECT 1 FROM hist h WHERE h.run_id=? AND h.key_id=k.key_id) "
            "ORDER BY k.name", (run_id,)).fetchall()
        return [r[0] for r in rows]

    def read(self, run_id: int, key: str, start: Optional[int] = None, end: Optional[int] = None,
             max_steps: Optional[int] = None) -> List[HistRecord]:
        con = self._rcon()
        kid = self._key_id(con, key, create=False)
        if kid is None:
            return []
        lo = -(2 ** 63) if start is None else int(start)
        hi = 2 ** 63 - 1 if end is None else int(end)
        con.execute("BEGIN")
        try:
            steps = [r[0] for r in con.execute(
                "SELECT step FROM hist WHERE run_id=? AND key_id=? AND step>=? AND step<? ORDER BY step",
                (run_id, kid, lo, hi))]
            pick = [steps[i] for i in subsample_indices(len(steps), max_steps)]
            out = []
            for i in range(0, len(pick), 500):
                chunk = pick[i:i + 500]
                q = ",".join("?" * len(chunk))
                out += [_row_to_record(r) for r in con.execute(
                    f"SELECT {COLS} FROM hist WHERE run_id=? AND key_id=? AND step IN ({q}) ORDER BY step",
                    (run_id, kid, *chunk))]
            return out
        finally:
            con.execute("COMMIT")

    def at(self, run_id: int, key: str, step: int) -> Optional[HistRecord]:
        """The record at ``step``, or the nearest one before it."""
        con = self._rcon()
        kid = self._key_id(con, key, create=False)
        if kid is None:
            return None
        row = con.execute(f"SELECT {COLS} FROM hist WHERE run_id=? AND key_id=? AND step<=? "
                          "ORDER BY step DESC LIMIT 1", (run_id, kid, int(step))).fetchone()
        return None if row is None else _row_to_record(row)

    def last(self, run_id: int, key: str) -> Optional[HistRecord]:
        return self.at(run_id, key, 2 ** 63 - 1)

    def stats(self) -> StoreStats:
        size = sum(os.path.getsize(self.path + s) for s in ("", "-wal") if os.path.exists(self.path + s))
        n = self._rcon().execute("SELECT count(*) FROM hist").fetchone()[0]
        return StoreStats(bytes_on_disk=size, points=n)

    def vacuum(self, full: bool = False) -> None:
        with self._wlock:
            con = self._writer()
            con.execute("VACUUM" if full else "PRAGMA incremental_vacuum")

    def close(self) -> None:
        if self._wcon is not None:
            self._wcon.close()
            self._wcon = None
        con = getattr(self._local, "con", None)
        if con is not None:
            con.close()
            self._local.con = None


HIST_BACKENDS = {"sqlite_hist": SqliteHistStore}
