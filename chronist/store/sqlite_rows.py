"""SqliteRowStore: one row per point. The v1 default MetricStore.

Concurrency model: exactly one instance per data dir is used for writing (the
server's writer thread). Readers may open the same file with
``readonly=True``; SQLite WAL mode lets them read while the writer commits.
"""
from __future__ import annotations

import math
import os
import sqlite3
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import Buckets, Points, StoreStats, bucket_width

SCHEMA = """
CREATE TABLE IF NOT EXISTS keys (
    key_id INTEGER PRIMARY KEY,
    name   TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS points (
    run_id INTEGER NOT NULL,
    key_id INTEGER NOT NULL,
    step   INTEGER NOT NULL,
    value  REAL,
    wall   REAL,
    PRIMARY KEY (run_id, key_id, step)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS run_keys (
    run_id     INTEGER NOT NULL,
    key_id     INTEGER NOT NULL,
    first_step INTEGER NOT NULL,
    last_step  INTEGER NOT NULL,
    last_value REAL,
    PRIMARY KEY (run_id, key_id)
) WITHOUT ROWID;
"""


def connect(path: str, readonly: bool = False) -> sqlite3.Connection:
    if readonly:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True,
                              check_same_thread=False, isolation_level=None)
    else:
        con = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA busy_timeout=10000")
    con.execute("PRAGMA cache_size=-65536")  # 64 MB page cache
    return con


class SqliteRowStore:
    name = "sqlite_rows"

    def __init__(self, path: str, readonly: bool = False):
        self.path = path
        self.readonly = readonly
        if not readonly:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            con = connect(path)
            con.execute("PRAGMA auto_vacuum=INCREMENTAL")  # only effective on a new file
            con.executescript(SCHEMA)
            con.close()
        self._local = threading.local()
        self._key_ids: Dict[str, int] = {}
        self._wcon: Optional[sqlite3.Connection] = None
        self._wlock = threading.Lock()

    # -- connections ---------------------------------------------------
    def _rcon(self) -> sqlite3.Connection:
        """Per-thread read connection."""
        con = getattr(self._local, "con", None)
        if con is None:
            con = connect(self.path, readonly=True)
            self._local.con = con
        return con

    def _writer(self) -> sqlite3.Connection:
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

    # -- writes --------------------------------------------------------
    def write(self, run_id: int, points: Points) -> None:
        with self._wlock:
            con = self._writer()
            own_tx = not con.in_transaction
            if own_tx:
                con.execute("BEGIN")
            try:
                self._write(con, run_id, points)
                if own_tx:
                    con.execute("COMMIT")
            except BaseException:
                self._key_ids.clear()  # may hold ids from the rolled-back tx
                if own_tx:
                    con.execute("ROLLBACK")
                raise

    def _write(self, con, run_id: int, points: Points) -> None:
        for key, s in points.items():
            if len(s) == 0:
                continue
            kid = self._key_id(con, key, create=True)
            steps = s.step.tolist()
            # SQLite cannot store NaN (it becomes NULL); NULL reads back as NaN.
            vals = [None if v != v else v for v in s.value.tolist()]
            walls = s.wall_time.tolist() if s.wall_time is not None else [None] * len(steps)
            con.executemany(
                "INSERT OR IGNORE INTO points(run_id,key_id,step,value,wall) VALUES (?,?,?,?,?)",
                zip([run_id] * len(steps), [kid] * len(steps), steps, vals, walls),
            )
            i_max = int(np.argmax(s.step))
            i_min = int(np.argmin(s.step))
            con.execute(
                """INSERT INTO run_keys(run_id,key_id,first_step,last_step,last_value)
                   VALUES (?,?,?,?,?)
                   ON CONFLICT(run_id,key_id) DO UPDATE SET
                     first_step = min(first_step, excluded.first_step),
                     last_value = CASE WHEN excluded.last_step > last_step
                                       THEN excluded.last_value ELSE last_value END,
                     last_step  = max(last_step, excluded.last_step)""",
                (run_id, kid, steps[i_min], steps[i_max], vals[i_max]),
            )

    def delete_run(self, run_id: int) -> None:
        with self._wlock:
            con = self._writer()
            own_tx = not con.in_transaction
            if own_tx:
                con.execute("BEGIN")
            con.execute("DELETE FROM points WHERE run_id=?", (run_id,))
            con.execute("DELETE FROM run_keys WHERE run_id=?", (run_id,))
            if own_tx:
                con.execute("COMMIT")

    def delete_range(self, run_id: int, start: int) -> None:
        with self._wlock:
            con = self._writer()
            own_tx = not con.in_transaction
            if own_tx:
                con.execute("BEGIN")
            try:
                con.execute("DELETE FROM points WHERE run_id=? AND step>=?", (run_id, int(start)))
                for (kid,) in con.execute("SELECT key_id FROM run_keys WHERE run_id=?", (run_id,)).fetchall():
                    lo, hi = con.execute("SELECT min(step), max(step) FROM points WHERE run_id=? AND key_id=?",
                                         (run_id, kid)).fetchone()
                    if lo is None:
                        con.execute("DELETE FROM run_keys WHERE run_id=? AND key_id=?", (run_id, kid))
                        continue
                    v = con.execute("SELECT value FROM points WHERE run_id=? AND key_id=? AND step=?",
                                    (run_id, kid, hi)).fetchone()[0]
                    con.execute("UPDATE run_keys SET first_step=?, last_step=?, last_value=? "
                                "WHERE run_id=? AND key_id=?", (lo, hi, v, run_id, kid))
                if own_tx:
                    con.execute("COMMIT")
            except BaseException:
                if own_tx:
                    con.execute("ROLLBACK")
                raise

    def vacuum(self, full: bool = False) -> None:
        """Return free pages to the OS: incremental (cheap) or a full VACUUM."""
        with self._wlock:
            con = self._writer()
            con.execute("VACUUM" if full else "PRAGMA incremental_vacuum")
            con.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def flush(self) -> None:
        pass

    def maintain(self) -> None:
        if self._wcon is not None:
            with self._wlock:
                self._wcon.execute("PRAGMA wal_checkpoint(PASSIVE)")
                self._wcon.execute("PRAGMA optimize")

    # -- reads ---------------------------------------------------------
    def keys(self, run_id: int) -> List[str]:
        rows = self._rcon().execute(
            "SELECT k.name FROM run_keys r JOIN keys k USING(key_id) WHERE r.run_id=? ORDER BY k.name",
            (run_id,)).fetchall()
        return [r[0] for r in rows]

    def last(self, run_id: int, keys: List[str]) -> Dict[str, Tuple[int, float]]:
        out = {}
        con = self._rcon()
        for key in keys:
            row = con.execute(
                "SELECT r.last_step, r.last_value FROM run_keys r JOIN keys k USING(key_id) "
                "WHERE r.run_id=? AND k.name=?", (run_id, key)).fetchone()
            if row is not None:
                out[key] = (row[0], math.nan if row[1] is None else row[1])
        return out

    def step_range(self, run_id: int, key: str) -> Optional[Tuple[int, int]]:
        row = self._rcon().execute(
            "SELECT r.first_step, r.last_step FROM run_keys r JOIN keys k USING(key_id) "
            "WHERE r.run_id=? AND k.name=?", (run_id, key)).fetchone()
        return None if row is None else (row[0], row[1])

    def read(self, run_id: int, key: str, start: Optional[int] = None,
             end: Optional[int] = None, max_points: Optional[int] = None,
             limit: Optional[int] = None) -> Buckets:
        con = self._rcon()
        kid = self._key_id(con, key, create=False)
        if kid is None:
            return Buckets.empty()
        lo = -(2 ** 63) if start is None else int(start)
        hi = 2 ** 63 - 1 if end is None else int(end)
        args = (run_id, kid, lo, hi)
        where = "run_id=? AND key_id=? AND step>=? AND step<?"

        width = 1
        if max_points is not None:
            rng = self.step_range(run_id, key)
            if rng is None:
                return Buckets.empty()
            if max(lo, rng[0]) > min(hi - 1, rng[1]):
                return Buckets.empty()
            # An explicit bound defines the grid; an open bound uses the data.
            eff_lo = lo if start is not None else rng[0]
            eff_hi = hi if end is not None else rng[1] + 1
            width = bucket_width(eff_lo, eff_hi, max_points)

        if width == 1:
            sql = f"SELECT step, value, wall FROM points WHERE {where} ORDER BY step"
            if limit is not None and max_points is None:
                sql += f" LIMIT {int(limit)}"
            cur = con.execute(sql, args)
            parts = []
            while True:  # convert in batches: a list of 10M tuples would cost GBs
                rows = cur.fetchmany(100_000)
                if not rows:
                    break
                parts.append(np.array(rows, dtype=np.float64))  # None -> nan
            if not parts:
                return Buckets.empty()
            a = parts[0] if len(parts) == 1 else np.concatenate(parts)
            wall = a[:, 2] if not np.isnan(a[:, 2]).all() else None
            return Buckets.from_raw(a[:, 0].astype(np.int64), a[:, 1], wall)

        # Floor division that is correct for negative steps too.
        rows = con.execute(
            f"""SELECT (step - ((step % ?) + ?) % ?) / ? AS b,
                       max(step), min(value), max(value), avg(value), count(*)
                FROM points WHERE {where} GROUP BY b ORDER BY b""",
            (width, width, width, width) + args).fetchall()
        if not rows:
            return Buckets.empty()
        a = np.array(rows, dtype=np.float64)
        last_steps = a[:, 1].astype(np.int64)
        # Value and wall time of the last point per bucket: point lookups by PK.
        lv = {}
        sl = last_steps.tolist()
        for i in range(0, len(sl), 500):
            chunk = sl[i:i + 500]
            q = ",".join("?" * len(chunk))
            for st, v, w in con.execute(
                    f"SELECT step, value, wall FROM points WHERE run_id=? AND key_id=? AND step IN ({q})",
                    (run_id, kid, *chunk)):
                lv[st] = (math.nan if v is None else v, math.nan if w is None else w)
        last = np.array([lv[s][0] for s in sl], dtype=np.float64)
        wall = np.array([lv[s][1] for s in sl], dtype=np.float64)
        return Buckets(
            step=last_steps, min=a[:, 2], max=a[:, 3], mean=a[:, 4], last=last,
            count=a[:, 5].astype(np.int64),
            wall_time=None if np.isnan(wall).all() else wall, width=width)

    def stats(self) -> StoreStats:
        size = 0
        for suffix in ("", "-wal"):
            p = self.path + suffix
            if os.path.exists(p):
                size += os.path.getsize(p)
        n = self._rcon().execute("SELECT count(*) FROM points").fetchone()[0]
        return StoreStats(bytes_on_disk=size, points=n)

    def close(self) -> None:
        if self._wcon is not None:
            self._wcon.close()
            self._wcon = None
        con = getattr(self._local, "con", None)
        if con is not None:
            con.close()
            self._local.con = None
