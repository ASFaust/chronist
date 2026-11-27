"""ChunkedStore: compressed chunks + a downsampling pyramid (spec section 5).

Layout (one SQLite file):

    hot(run, key, step, value, wall)        recent tail and late points, row per point
    chunks(run, key, level, start, ...)     level 0: sealed raw points, ~CHUNK points per blob
                                            level L>=1: aggregate buckets of width 16**L steps,
                                            blobs hold the buckets with idx // CHUNK == start
    series(run, key, sealed_upto, ...)      per-series watermark + summary

Points are sealed from ``hot`` into level-0 chunks once CHUNK of them have
accumulated; sealing extends every pyramid level over the buckets that became
complete. Steps are delta-encoded, every array is byte-shuffled, then
zstd-compressed (zlib if ``zstandard`` is not installed).

Idempotency: a point at or below the sealed watermark is checked against the
sealed chunks and dropped if its step exists; otherwise it is a genuine late
point, kept in ``hot`` and merged into the chunks (and the pyramid rebuilt over
the affected range) at the next compaction. Reads account for unmerged late
points separately, so results are exact at all times.

Reads: the bucket width ``w`` from ``bucket_width`` is rounded up to a multiple
of the coarsest level width ``W_L <= w``. The range is then served
recursively: complete level-L buckets fully inside the range, finer levels for
the edges, raw points at the bottom. Every source bucket nests inside exactly
one output bucket, so min/max/mean/last/count are exact.
"""
from __future__ import annotations

import math
import os
import sqlite3
import struct
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import Buckets, Points, StoreStats, bucket_width
from .sqlite_rows import connect

from ..codec import compress as _compress, decompress as _decompress

CHUNK = 4096
FANOUT = 16
MAX_LEVEL = 6  # bucket width 16**6 = 16.7M steps

SCHEMA = """
CREATE TABLE IF NOT EXISTS keys (key_id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE);
CREATE TABLE IF NOT EXISTS hot (
    run_id INTEGER NOT NULL, key_id INTEGER NOT NULL, step INTEGER NOT NULL,
    value REAL, wall REAL,
    PRIMARY KEY (run_id, key_id, step)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS chunks (
    run_id INTEGER NOT NULL, key_id INTEGER NOT NULL, level INTEGER NOT NULL,
    start INTEGER NOT NULL,   -- level 0: first step; level>0: chunk number (idx // CHUNK)
    last INTEGER NOT NULL,    -- level 0: last step;  level>0: last bucket idx
    n INTEGER NOT NULL,
    data BLOB NOT NULL,
    PRIMARY KEY (run_id, key_id, level, start)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS series (
    run_id INTEGER NOT NULL, key_id INTEGER NOT NULL,
    sealed_upto INTEGER,      -- max sealed step, NULL if nothing sealed
    first_step INTEGER NOT NULL, last_step INTEGER NOT NULL, last_value REAL,
    PRIMARY KEY (run_id, key_id)
) WITHOUT ROWID;
"""

I64 = np.int64
F64 = np.float64


# ---------------------------------------------------------------- codec
def _shuffle(a: np.ndarray) -> bytes:
    return np.ascontiguousarray(a).view(np.uint8).reshape(-1, 8).T.tobytes()


def _unshuffle(b: bytes, n: int, dtype) -> np.ndarray:
    return np.frombuffer(b, dtype=np.uint8).reshape(8, n).T.copy().view(dtype).reshape(n)


def encode(arrays: List[np.ndarray], delta: List[bool]) -> bytes:
    n = len(arrays[0])
    parts = [struct.pack("<II", n, len(arrays))]
    for a, d in zip(arrays, delta):
        if d and n:
            a = np.diff(a, prepend=a.dtype.type(0))
        parts.append(_shuffle(a))
    return _compress(b"".join(parts))


def decode(blob: bytes, dtypes: List, delta: List[bool]) -> List[np.ndarray]:
    raw = _decompress(blob)
    n, k = struct.unpack_from("<II", raw)
    out, off = [], 8
    for dt, d in zip(dtypes, delta):
        a = _unshuffle(raw[off:off + 8 * n], n, dt)
        off += 8 * n
        if d:
            a = np.cumsum(a, dtype=dt)
        out.append(a)
    return out


RAW_DT = [I64, F64, F64]
RAW_DELTA = [True, False, False]
AGG_FIELDS = ["idx", "last_step", "min", "max", "sum", "last", "count", "nn", "last_wall"]
AGG_DT = [I64, I64, F64, F64, F64, F64, I64, I64, F64]
AGG_DELTA = [True, True, False, False, False, False, False, False, False]


# ---------------------------------------------------------------- aggregates
class Agg:
    """Bucket aggregates keyed by idx (sorted, unique)."""

    __slots__ = AGG_FIELDS

    def __init__(self, *arrays):
        for f, a in zip(AGG_FIELDS, arrays):
            setattr(self, f, a)

    def arrays(self):
        return [getattr(self, f) for f in AGG_FIELDS]

    def __len__(self):
        return len(self.idx)

    @staticmethod
    def empty() -> "Agg":
        return Agg(*[np.zeros(0, dt) for dt in AGG_DT])

    def select(self, lo: int, hi: int) -> "Agg":
        """Buckets with lo <= idx < hi."""
        a, b = np.searchsorted(self.idx, [lo, hi])
        return Agg(*[x[a:b] for x in self.arrays()])

    def without(self, lo: int, hi: int) -> "Agg":
        m = (self.idx < lo) | (self.idx >= hi)
        return Agg(*[x[m] for x in self.arrays()])


def agg_points(step: np.ndarray, value: np.ndarray, wall: np.ndarray, width: int) -> Agg:
    """Aggregate raw points (sorted by step) into buckets idx = step // width."""
    if len(step) == 0:
        return Agg.empty()
    idx = step // width
    starts = np.flatnonzero(np.r_[True, idx[1:] != idx[:-1]])
    ends = np.r_[starts[1:], len(step)] - 1
    finite = ~np.isnan(value)
    v0 = np.where(finite, value, 0.0)
    with np.errstate(invalid="ignore"):
        mn = np.fmin.reduceat(value, starts)
        mx = np.fmax.reduceat(value, starts)
    return Agg(idx[starts], step[ends], mn, mx, np.add.reduceat(v0, starts), value[ends],
               (ends - starts + 1).astype(I64), np.add.reduceat(finite.astype(I64), starts), wall[ends])


def reduce_aggs(parts: List[Agg], factor: int = 1) -> Agg:
    """Combine partial aggregates (any order, may share idx) after idx //= factor."""
    parts = [p for p in parts if len(p)]
    if not parts:
        return Agg.empty()
    cat = [np.concatenate([getattr(p, f) for p in parts]) for f in AGG_FIELDS]
    idx = cat[0] // factor if factor != 1 else cat[0]
    order = np.lexsort((cat[1], idx))  # by idx, then last_step
    idx = idx[order]
    c = [x[order] for x in cat]
    starts = np.flatnonzero(np.r_[True, idx[1:] != idx[:-1]])
    ends = np.r_[starts[1:], len(idx)] - 1
    with np.errstate(invalid="ignore"):
        mn = np.fmin.reduceat(c[2], starts)
        mx = np.fmax.reduceat(c[3], starts)
    return Agg(idx[starts], c[1][ends], mn, mx, np.add.reduceat(c[4], starts), c[5][ends],
               np.add.reduceat(c[6], starts), np.add.reduceat(c[7], starts), c[8][ends])


def merge_points(a, b):
    """Union of two sorted point sets (step, value, wall); on equal steps a wins."""
    s = np.concatenate([a[0], b[0]])
    v = np.concatenate([a[1], b[1]])
    w = np.concatenate([a[2], b[2]])
    src = np.concatenate([np.zeros(len(a[0]), np.int8), np.ones(len(b[0]), np.int8)])
    order = np.lexsort((src, s))
    s, v, w = s[order], v[order], w[order]
    if len(s) == 0:
        return s, v, w
    keep = np.r_[True, s[1:] != s[:-1]]
    return s[keep], v[keep], w[keep]


# ---------------------------------------------------------------- store
class ChunkedStore:
    name = "chunked"

    def __init__(self, path: str, readonly: bool = False, chunk: int = CHUNK):
        self.chunk = chunk
        self.path = path
        self.readonly = readonly
        if not readonly:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            con = connect(path)
            con.executescript(SCHEMA)
            con.close()
        self._local = threading.local()
        self._key_ids: Dict[str, int] = {}
        self._wcon: Optional[sqlite3.Connection] = None
        self._wlock = threading.RLock()
        self._hot_n: Dict[Tuple[int, int], int] = {}

    # -- connections ---------------------------------------------------
    def _rcon(self) -> sqlite3.Connection:
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

    def _tx(self, fn, *args):
        with self._wlock:
            con = self._writer()
            own = not con.in_transaction
            if own:
                con.execute("BEGIN IMMEDIATE")
            try:
                r = fn(con, *args)
                if own:
                    con.execute("COMMIT")
                return r
            except BaseException:
                self._key_ids.clear()
                self._hot_n.clear()
                if own:
                    con.execute("ROLLBACK")
                raise

    # -- low-level reads (any connection) ------------------------------
    @staticmethod
    def _series_row(con, run_id, kid):
        return con.execute("SELECT sealed_upto, first_step, last_step, last_value FROM series "
                           "WHERE run_id=? AND key_id=?", (run_id, kid)).fetchone()

    @staticmethod
    def _sealed_points(con, run_id, kid, lo, hi):
        """Points from level-0 chunks with lo <= step < hi."""
        rows = con.execute(
            "SELECT data FROM chunks WHERE run_id=? AND key_id=? AND level=0 AND last>=? AND start<? "
            "ORDER BY start", (run_id, kid, lo, hi)).fetchall()
        if not rows:
            return np.zeros(0, I64), np.zeros(0, F64), np.zeros(0, F64)
        ss, vs, ws = [], [], []
        for (blob,) in rows:
            s, v, w = decode(blob, RAW_DT, RAW_DELTA)
            a, b = np.searchsorted(s, [lo, hi])
            ss.append(s[a:b]); vs.append(v[a:b]); ws.append(w[a:b])
        return np.concatenate(ss), np.concatenate(vs), np.concatenate(ws)

    @staticmethod
    def _hot_points(con, run_id, kid, lo, hi):
        rows = con.execute("SELECT step, value, wall FROM hot WHERE run_id=? AND key_id=? AND step>=? AND step<? "
                           "ORDER BY step", (run_id, kid, lo, hi)).fetchall()
        if not rows:
            return np.zeros(0, I64), np.zeros(0, F64), np.zeros(0, F64)
        a = np.array(rows, dtype=F64)
        return a[:, 0].astype(I64), a[:, 1], a[:, 2]

    def _level_buckets(self, con, run_id, kid, level, idx_lo, idx_hi) -> Agg:
        """Stored level buckets with idx_lo <= idx < idx_hi."""
        if idx_hi <= idx_lo:
            return Agg.empty()
        rows = con.execute(
            "SELECT data FROM chunks WHERE run_id=? AND key_id=? AND level=? AND start>=? AND start<=? ORDER BY start",
            (run_id, kid, level, idx_lo // self.chunk, (idx_hi - 1) // self.chunk)).fetchall()
        if not rows:
            return Agg.empty()
        parts = [Agg(*decode(blob, AGG_DT, AGG_DELTA)) for (blob,) in rows]
        allb = Agg(*[np.concatenate([getattr(p, f) for p in parts]) for f in AGG_FIELDS])
        return allb.select(idx_lo, idx_hi)

    # -- writes --------------------------------------------------------
    def write(self, run_id: int, points: Points) -> None:
        self._tx(self._write, run_id, points)

    def _write(self, con, run_id, points):
        for key, s in points.items():
            if len(s) == 0:
                continue
            kid = self._key_id(con, key, create=True)
            order = np.argsort(s.step, kind="stable")
            step = s.step[order]
            value = s.value[order]
            wall = s.wall_time[order] if s.wall_time is not None else np.full(len(step), np.nan)
            keep = np.r_[True, step[1:] != step[:-1]]  # dups inside the batch: first wins
            step, value, wall = step[keep], value[keep], wall[keep]
            row = self._series_row(con, run_id, kid)
            sealed = row[0] if row else None
            if sealed is not None and step[0] <= sealed:
                late = step <= sealed
                ls, lv, lw = step[late], value[late], wall[late]
                ss, _, _ = self._sealed_points(con, run_id, kid, int(ls[0]), int(ls[-1]) + 1)
                fresh_late = ~np.isin(ls, ss)
                step = np.concatenate([ls[fresh_late], step[~late]])
                value = np.concatenate([lv[fresh_late], value[~late]])
                wall = np.concatenate([lw[fresh_late], wall[~late]])
                if len(step) == 0:
                    continue
            i_max = len(step) - 1
            con.execute(
                """INSERT INTO series(run_id,key_id,sealed_upto,first_step,last_step,last_value)
                   VALUES (?,?,NULL,?,?,?)
                   ON CONFLICT(run_id,key_id) DO UPDATE SET
                     first_step = min(first_step, excluded.first_step),
                     last_value = CASE WHEN excluded.last_step > last_step THEN excluded.last_value ELSE last_value END,
                     last_step = max(last_step, excluded.last_step)""",
                (run_id, kid, int(step[0]), int(step[i_max]), None if value[i_max] != value[i_max] else float(value[i_max])))
            hk = (run_id, kid)
            n = self._hot_n.get(hk)
            if n is None:
                n = con.execute("SELECT count(*) FROM hot WHERE run_id=? AND key_id=?", hk).fetchone()[0]
            if n + len(step) >= 2 * self.chunk:
                # big enough to seal: go straight from memory, skipping the hot round-trip
                self._compact(con, run_id, kid, extra=(step, value, wall))
                continue
            before = con.total_changes
            self._insert_hot(con, run_id, kid, step, value, wall)
            self._hot_n[hk] = n + con.total_changes - before

    @staticmethod
    def _insert_hot(con, run_id, kid, step, value, wall):
        con.executemany(
            "INSERT OR IGNORE INTO hot(run_id,key_id,step,value,wall) VALUES (?,?,?,?,?)",
            zip([run_id] * len(step), [kid] * len(step), step.tolist(),
                [None if v != v else v for v in value.tolist()],
                [None if v != v else v for v in wall.tolist()]))

    def _compact(self, con, run_id, kid, force_late: bool = True, extra=None):
        """Seal full chunks from hot (+ ``extra`` in-memory points, which lose
        to existing hot rows on equal steps), merge late points, extend the pyramid."""
        row = self._series_row(con, run_id, kid)
        sealed = row[0]
        hs, hv, hw = self._hot_points(con, run_id, kid, -(2 ** 63), 2 ** 63 - 1)
        if extra is not None:
            hs, hv, hw = merge_points((hs, hv, hw), extra)
        if len(hs) == 0:
            return
        rebuild_lo = None
        # 1. late points (<= sealed watermark): merge into existing level-0 chunks
        if sealed is not None:
            nl = int(np.searchsorted(hs, sealed, side="right"))
            if nl and force_late:
                self._merge_late(con, run_id, kid, hs[:nl], hv[:nl], hw[:nl])
                con.execute("DELETE FROM hot WHERE run_id=? AND key_id=? AND step<=?", (run_id, kid, sealed))
                rebuild_lo = int(hs[0])
            hs, hv, hw = hs[nl:], hv[nl:], hw[nl:]
        # 2. seal full chunks of fresh points
        nseal = (len(hs) // self.chunk) * self.chunk
        new_sealed = sealed
        if nseal:
            for i in range(0, nseal, self.chunk):
                s, v, w = hs[i:i + self.chunk], hv[i:i + self.chunk], hw[i:i + self.chunk]
                con.execute("INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?,?,?)",
                            (run_id, kid, 0, int(s[0]), int(s[-1]), len(s), encode([s, v, w], RAW_DELTA)))
            new_sealed = int(hs[nseal - 1])
            con.execute("DELETE FROM hot WHERE run_id=? AND key_id=? AND step<=?", (run_id, kid, new_sealed))
            con.execute("UPDATE series SET sealed_upto=? WHERE run_id=? AND key_id=?", (new_sealed, run_id, kid))
            # Buckets straddling the old watermark only become complete now,
            # so the rebuild starts there, not at the first newly sealed step.
            seal_lo = int(hs[0]) if sealed is None else min(int(hs[0]), sealed + 1)
            rebuild_lo = seal_lo if rebuild_lo is None else min(rebuild_lo, seal_lo)
        if extra is not None:  # hot now holds exactly the unsealed remainder
            con.execute("DELETE FROM hot WHERE run_id=? AND key_id=?", (run_id, kid))
            self._insert_hot(con, run_id, kid, hs[nseal:], hv[nseal:], hw[nseal:])
        self._hot_n[(run_id, kid)] = len(hs) - nseal
        if rebuild_lo is not None and new_sealed is not None:
            self._build_pyramid(con, run_id, kid, rebuild_lo, new_sealed)

    def _merge_late(self, con, run_id, kid, ls, lv, lw):
        chunks = con.execute("SELECT start, last, data FROM chunks WHERE run_id=? AND key_id=? AND level=0 ORDER BY start",
                             (run_id, kid)).fetchall()
        starts = np.array([c[0] for c in chunks], dtype=I64)
        # each late point goes to the last chunk starting at or before it (or the first chunk)
        target = np.maximum(np.searchsorted(starts, ls, side="right") - 1, 0)
        for ci in np.unique(target):
            m = target == ci
            start, _, blob = chunks[int(ci)]
            s, v, w = merge_points(decode(blob, RAW_DT, RAW_DELTA), (ls[m], lv[m], lw[m]))
            con.execute("DELETE FROM chunks WHERE run_id=? AND key_id=? AND level=0 AND start=?", (run_id, kid, start))
            con.execute("INSERT INTO chunks VALUES (?,?,?,?,?,?,?)",
                        (run_id, kid, 0, int(s[0]), int(s[-1]), len(s), encode([s, v, w], RAW_DELTA)))

    def _build_pyramid(self, con, run_id, kid, lo_step: int, sealed: int):
        """(Re)build all complete level buckets covering steps >= lo_step."""
        for level in range(1, MAX_LEVEL + 1):
            W = FANOUT ** level
            ka = lo_step // W
            kb = (sealed + 1) // W  # exclusive: buckets < kb are complete
            if kb <= ka:
                break
            if level == 1:
                s, v, w = self._sealed_points(con, run_id, kid, ka * W, kb * W)
                new = agg_points(s, v, w, W)
            else:
                lower = self._level_buckets(con, run_id, kid, level - 1, ka * FANOUT, kb * FANOUT)
                new = reduce_aggs([lower], FANOUT)
            self._store_level(con, run_id, kid, level, new, ka, kb)

    def _store_level(self, con, run_id, kid, level, new: Agg, ka: int, kb: int):
        for c in range(ka // self.chunk, (kb - 1) // self.chunk + 1):
            lo, hi = max(ka, c * self.chunk), min(kb, (c + 1) * self.chunk)
            row = con.execute("SELECT data FROM chunks WHERE run_id=? AND key_id=? AND level=? AND start=?",
                              (run_id, kid, level, c)).fetchone()
            parts = [new.select(lo, hi)]
            if row is not None:
                parts.append(Agg(*decode(row[0], AGG_DT, AGG_DELTA)).without(lo, hi))
            merged = reduce_aggs(parts)
            if len(merged) == 0:
                continue
            con.execute("INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?,?,?)",
                        (run_id, kid, level, c, int(merged.idx[-1]), len(merged),
                         encode(merged.arrays(), AGG_DELTA)))

    def delete_run(self, run_id: int) -> None:
        def _del(con):
            for t in ("hot", "chunks", "series"):
                con.execute(f"DELETE FROM {t} WHERE run_id=?", (run_id,))
        self._tx(_del)
        for k in [k for k in self._hot_n if k[0] == run_id]:
            del self._hot_n[k]

    def delete_range(self, run_id: int, start: int) -> None:
        start = int(start)

        def _del(con):
            con.execute("DELETE FROM hot WHERE run_id=? AND step>=?", (run_id, start))
            for (kid,) in con.execute("SELECT key_id FROM series WHERE run_id=?", (run_id,)).fetchall():
                # level 0: drop chunks entirely past the cut, truncate the one straddling it
                con.execute("DELETE FROM chunks WHERE run_id=? AND key_id=? AND level=0 AND start>=?",
                            (run_id, kid, start))
                row = con.execute("SELECT start, data FROM chunks WHERE run_id=? AND key_id=? AND level=0 "
                                  "AND last>=? ORDER BY start", (run_id, kid, start)).fetchone()
                if row is not None:
                    s, v, w = decode(row[1], RAW_DT, RAW_DELTA)
                    k = int(np.searchsorted(s, start))
                    con.execute("DELETE FROM chunks WHERE run_id=? AND key_id=? AND level=0 AND start=?",
                                (run_id, kid, row[0]))
                    if k:
                        con.execute("INSERT INTO chunks VALUES (?,?,?,?,?,?,?)",
                                    (run_id, kid, 0, int(s[0]), int(s[k - 1]), k, encode([s[:k], v[:k], w[:k]], RAW_DELTA)))
                # pyramid: drop every bucket at or after the cut (the straddling one is now incomplete)
                for level in range(1, MAX_LEVEL + 1):
                    kcut = start // FANOUT ** level
                    con.execute("DELETE FROM chunks WHERE run_id=? AND key_id=? AND level=? AND start>?",
                                (run_id, kid, level, kcut // self.chunk))
                    row = con.execute("SELECT data FROM chunks WHERE run_id=? AND key_id=? AND level=? AND start=?",
                                      (run_id, kid, level, kcut // self.chunk)).fetchone()
                    if row is not None:
                        kept = Agg(*decode(row[0], AGG_DT, AGG_DELTA)).select(-(2 ** 62), kcut)
                        con.execute("DELETE FROM chunks WHERE run_id=? AND key_id=? AND level=? AND start=?",
                                    (run_id, kid, level, kcut // self.chunk))
                        if len(kept):
                            con.execute("INSERT INTO chunks VALUES (?,?,?,?,?,?,?)",
                                        (run_id, kid, level, kcut // self.chunk, int(kept.idx[-1]), len(kept),
                                         encode(kept.arrays(), AGG_DELTA)))
                # watermark + summary from what remains
                last0 = con.execute("SELECT max(last), min(start) FROM chunks WHERE run_id=? AND key_id=? AND level=0",
                                    (run_id, kid)).fetchone()
                hot = con.execute("SELECT min(step), max(step) FROM hot WHERE run_id=? AND key_id=?",
                                  (run_id, kid)).fetchone()
                firsts = [x for x in (last0[1], hot[0]) if x is not None]
                lasts = [x for x in (last0[0], hot[1]) if x is not None]
                if not lasts:
                    con.execute("DELETE FROM series WHERE run_id=? AND key_id=?", (run_id, kid))
                    continue
                last_step = max(lasts)
                hv = con.execute("SELECT value FROM hot WHERE run_id=? AND key_id=? AND step=?",
                                 (run_id, kid, last_step)).fetchone()
                if hv is not None:
                    last_value = hv[0]
                else:
                    blob = con.execute("SELECT data FROM chunks WHERE run_id=? AND key_id=? AND level=0 AND last=?",
                                       (run_id, kid, last_step)).fetchone()[0]
                    lv = float(decode(blob, RAW_DT, RAW_DELTA)[1][-1])
                    last_value = None if lv != lv else lv
                con.execute("UPDATE series SET sealed_upto=?, first_step=?, last_step=?, last_value=? "
                            "WHERE run_id=? AND key_id=?", (last0[0], min(firsts), last_step, last_value, run_id, kid))
                self._hot_n.pop((run_id, kid), None)
        self._tx(_del)

    def flush(self) -> None:
        pass

    def maintain(self) -> None:
        """Seal everything sealable and merge all late points."""
        if self.readonly:
            return

        def _m(con):
            rows = con.execute(
                "SELECT h.run_id, h.key_id, count(*), sum(h.step <= coalesce(s.sealed_upto, -9223372036854775808)) "
                "FROM hot h JOIN series s USING(run_id, key_id) GROUP BY h.run_id, h.key_id").fetchall()
            for run_id, kid, n, nlate in rows:
                if n >= self.chunk or nlate:
                    self._compact(con, run_id, kid)
        self._tx(_m)
        with self._wlock:
            self._writer().execute("PRAGMA wal_checkpoint(PASSIVE)")

    # -- reads ---------------------------------------------------------
    def _read_tx(self):
        con = self._rcon()
        con.execute("BEGIN")
        return con

    def keys(self, run_id: int) -> List[str]:
        rows = self._rcon().execute(
            "SELECT k.name FROM series s JOIN keys k USING(key_id) WHERE s.run_id=? ORDER BY k.name",
            (run_id,)).fetchall()
        return [r[0] for r in rows]

    def last(self, run_id: int, keys: List[str]) -> Dict[str, Tuple[int, float]]:
        out = {}
        con = self._rcon()
        for key in keys:
            row = con.execute("SELECT s.last_step, s.last_value FROM series s JOIN keys k USING(key_id) "
                              "WHERE s.run_id=? AND k.name=?", (run_id, key)).fetchone()
            if row is not None:
                out[key] = (row[0], math.nan if row[1] is None else row[1])
        return out

    def step_range(self, run_id: int, key: str) -> Optional[Tuple[int, int]]:
        row = self._rcon().execute("SELECT s.first_step, s.last_step FROM series s JOIN keys k USING(key_id) "
                                   "WHERE s.run_id=? AND k.name=?", (run_id, key)).fetchone()
        return None if row is None else (row[0], row[1])

    def read(self, run_id: int, key: str, start: Optional[int] = None,
             end: Optional[int] = None, max_points: Optional[int] = None,
             limit: Optional[int] = None) -> Buckets:
        con = self._rcon()
        kid = self._key_id(con, key, create=False)
        if kid is None:
            return Buckets.empty()
        con.execute("BEGIN")  # one snapshot for all queries below
        try:
            return self._read(con, run_id, kid, start, end, max_points, limit)
        finally:
            con.execute("COMMIT")

    def _raw_limited(self, con, run_id, kid, lo, hi, limit):
        """First ``limit`` points in [lo, hi), decoding only the chunks needed."""
        ss, vs, ws, n = [], [], [], 0
        cut = hi
        for (blob,) in con.execute(
                "SELECT data FROM chunks WHERE run_id=? AND key_id=? AND level=0 AND last>=? AND start<? "
                "ORDER BY start", (run_id, kid, lo, hi)):
            s, v, w = decode(blob, RAW_DT, RAW_DELTA)
            a, b = np.searchsorted(s, [lo, hi])
            ss.append(s[a:b]); vs.append(v[a:b]); ws.append(w[a:b])
            n += b - a
            if n >= limit:
                cut = int(s[b - 1]) + 1  # hot points beyond this can't be in the first `limit`
                break
        sealed = (np.concatenate(ss), np.concatenate(vs), np.concatenate(ws)) if ss else \
            (np.zeros(0, I64), np.zeros(0, F64), np.zeros(0, F64))
        s, v, w = merge_points(sealed, self._hot_points(con, run_id, kid, lo, cut))
        return s[:limit], v[:limit], w[:limit]

    def _read(self, con, run_id, kid, start, end, max_points, limit=None) -> Buckets:
        row = self._series_row(con, run_id, kid)
        if row is None:
            return Buckets.empty()
        sealed, first, last_step, _ = row
        lo = -(2 ** 63) if start is None else int(start)
        hi = 2 ** 63 - 1 if end is None else int(end)
        if max(lo, first) > min(hi - 1, last_step):
            return Buckets.empty()
        if max_points is not None:
            eff_lo = lo if start is not None else first
            eff_hi = hi if end is not None else last_step + 1
            w = bucket_width(eff_lo, eff_hi, max_points)
        else:
            w = 1
        lo, hi = max(lo, first), min(hi, last_step + 1)

        hs, hv, hw = self._hot_points(con, run_id, kid, lo, hi)
        if w == 1:
            if limit is not None and max_points is None:
                s, v, wt = self._raw_limited(con, run_id, kid, lo, hi, int(limit))
            else:
                ss, sv, sw = self._sealed_points(con, run_id, kid, lo, hi)
                s, v, wt = merge_points((ss, sv, sw), (hs, hv, hw))
            if len(s) == 0:
                return Buckets.empty()
            return Buckets.from_raw(s, v, None if np.isnan(wt).all() else wt)

        level = 0
        while level < MAX_LEVEL and FANOUT ** (level + 1) <= w:
            level += 1
        W = FANOUT ** level
        w = -(-w // W) * W  # round up to a multiple of the level width
        complete = {L: ((sealed + 1) // FANOUT ** L) * FANOUT ** L if sealed is not None else -(2 ** 63)
                    for L in range(0, level + 1)}
        complete[0] = sealed + 1 if sealed is not None else -(2 ** 63)

        parts: List[Agg] = []

        def serve(L: int, a: int, b: int):
            """Aggregate sealed data in [a, b) into the output grid (width w)."""
            if b <= a:
                return
            if L == 0:
                s, v, wt = self._sealed_points(con, run_id, kid, a, b)
                parts.append(agg_points(s, v, wt, w))
                return
            WL = FANOUT ** L
            a2 = -(-a // WL) * WL
            b2 = min((b // WL) * WL, complete[L])
            if b2 <= a2:
                serve(L - 1, a, b)
                return
            serve(L - 1, a, a2)
            buckets = self._level_buckets(con, run_id, kid, L, a2 // WL, b2 // WL)
            parts.append(reduce_aggs([buckets], w // WL))
            serve(L - 1, b2, b)

        sealed_hi = min(hi, complete[0])
        serve(level, lo, sealed_hi)
        parts.append(agg_points(hs, hv, hw, w))  # hot: unsealed tail + unmerged late points
        a = reduce_aggs(parts)
        if len(a) == 0:
            return Buckets.empty()
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(a.nn > 0, a.sum / np.maximum(a.nn, 1), np.nan)
        return Buckets(step=a.last_step, min=a.min, max=a.max, mean=mean, last=a.last, count=a.count,
                       wall_time=None if np.isnan(a.last_wall).all() else a.last_wall, width=int(w))

    def stats(self) -> StoreStats:
        size = sum(os.path.getsize(self.path + s) for s in ("", "-wal") if os.path.exists(self.path + s))
        con = self._rcon()
        sealed = con.execute("SELECT coalesce(sum(n), 0) FROM chunks WHERE level=0").fetchone()[0]
        hot = con.execute("SELECT count(*) FROM hot").fetchone()[0]
        return StoreStats(bytes_on_disk=size, points=sealed + hot)

    def close(self) -> None:
        if self._wcon is not None:
            self._wcon.close()
            self._wcon = None
        con = getattr(self._local, "con", None)
        if con is not None:
            con.close()
            self._local.con = None
