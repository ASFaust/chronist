"""Server core: data dir, the single writer thread, and read/aggregation logic.

Nothing here knows about HTTP. The Starlette app (``app.py``) and the local
read-back API both build on it.
"""
from __future__ import annotations

import json
import logging
import math
import os
import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..meta import MetadataStore
from ..store import Buckets, MetricStore, Series, bucket_width, open_store
from ..store.cache import CachingStore

log = logging.getLogger("chronist.server")

DATA_VERSION = 1


def default_data_dir() -> str:
    return os.environ.get("CHRONIST_DATA") or os.path.join(os.path.expanduser("~"), ".chronist", "data")


def open_data_dir(path: str, readonly: bool = False, metric_store: Optional[str] = None
                  ) -> Tuple[MetadataStore, MetricStore, dict]:
    path = os.path.abspath(os.path.expanduser(path))
    cfg_path = os.path.join(path, "chronist.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        if metric_store and metric_store != cfg["metric_store"]:
            raise ValueError(f"data dir uses metric store {cfg['metric_store']!r}, not {metric_store!r}")
    else:
        if readonly:
            raise FileNotFoundError(f"no chronist data dir at {path}")
        os.makedirs(path, exist_ok=True)
        cfg = {"version": DATA_VERSION, "metric_store": metric_store or "sqlite_rows"}
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
    meta = MetadataStore(os.path.join(path, "meta.sqlite"), readonly=readonly)
    store = open_store(cfg["metric_store"], os.path.join(path, f"metrics.{cfg['metric_store']}.sqlite"),
                       readonly=readonly)
    return meta, store, cfg


# ---------------------------------------------------------------- writer
class Writer:
    """The only thread that writes to storage. Everything else submits ops."""

    def __init__(self, on_error: Optional[Callable[[BaseException], None]] = None):
        self.q: "queue.Queue[Optional[Tuple[Callable, tuple, Future]]]" = queue.Queue()
        self.thread = threading.Thread(target=self._loop, name="chronist-writer", daemon=True)
        self.thread.start()

    def submit(self, fn: Callable, *args) -> Future:
        fut: Future = Future()
        self.q.put((fn, args, fut))
        return fut

    def _loop(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            fn, args, fut = item
            if not fut.set_running_or_notify_cancel():
                continue
            try:
                fut.set_result(fn(*args))
            except BaseException as e:  # noqa: BLE001 - report to the caller
                fut.set_exception(e)

    def stop(self, timeout: float = 30):
        self.q.put(None)
        self.thread.join(timeout)


# ---------------------------------------------------------------- events
class Broadcaster:
    """Fan-out of write events to SSE subscribers (thread-safe publish)."""

    def __init__(self):
        self._subs: List[Callable[[dict], None]] = []
        self._lock = threading.Lock()

    def subscribe(self, cb: Callable[[dict], None]) -> Callable[[], None]:
        with self._lock:
            self._subs.append(cb)

        def unsubscribe():
            with self._lock:
                if cb in self._subs:
                    self._subs.remove(cb)
        return unsubscribe

    def publish(self, ev: dict) -> None:
        with self._lock:
            subs = list(self._subs)
        for cb in subs:
            try:
                cb(ev)
            except Exception:  # pragma: no cover
                log.exception("subscriber failed")


# ---------------------------------------------------------------- payload decoding
def decode_points(payload: dict) -> Dict[str, Series]:
    """{"points": {key: {"step": [...], "value": [...], "wall": [...]?}}}"""
    pts = payload.get("points") or {}
    if not isinstance(pts, dict):
        raise ValueError("points must be an object keyed by metric name")
    out = {}
    for k, col in pts.items():
        if not isinstance(k, str) or not k:
            raise ValueError("metric keys must be non-empty strings")
        step = col.get("step")
        value = col.get("value")
        if step is None or value is None:
            raise ValueError(f"points[{k!r}] needs step and value")
        value = [math.nan if v is None else v for v in value]
        wall = col.get("wall")
        if wall is not None:
            wall = [math.nan if v is None else v for v in wall]
        out[k] = Series(step, value, wall)
    return out


# ---------------------------------------------------------------- core
class Core:
    def __init__(self, data_dir: str, metric_store: Optional[str] = None,
                 heartbeat_timeout: float = 180.0):
        self.data_dir = os.path.abspath(os.path.expanduser(data_dir))
        self.meta, store, self.cfg = open_data_dir(self.data_dir, metric_store=metric_store)
        self.store = CachingStore(store)
        self.writer = Writer()
        self.events = Broadcaster()
        self.heartbeat_timeout = heartbeat_timeout
        self.pool = ThreadPoolExecutor(max_workers=min(16, (os.cpu_count() or 4)), thread_name_prefix="chronist-read")
        self._last_maintain = time.time()

    def close(self):
        self.writer.submit(self.store.flush).result()
        self.writer.stop()
        self.pool.shutdown(wait=False)
        self.store.close()
        self.meta.close()

    # ---- write ops (run on the writer thread) -------------------------
    def _check_run(self, run_id: int) -> str:
        row = self.meta.w.execute(
            "SELECT p.name FROM runs r JOIN projects p ON p.id=r.project_id WHERE r.id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"run {run_id} not found")
        return row[0]

    def op_create_run(self, body: dict) -> dict:
        project = body.get("project") or "default"
        uid = body.get("uid")
        if not uid:
            raise ValueError("uid is required")
        run_id, new = self.meta.create_run(
            project, uid, name=body.get("name"), config=body.get("config") or {},
            tags=body.get("tags") or [], meta=body.get("meta") or {}, created=body.get("created"),
            notes=body.get("notes") or "")
        run = self.meta.get_runs([run_id], include=("tags",))[0]
        self.events.publish({"type": "run", "project": project, "run_id": run_id, "status": "running",
                             "new": new})
        return {"id": run_id, "uid": uid, "name": run["name"], "project": project, "created_new": new}

    def op_log(self, run_id: int, points: Dict[str, Series]) -> dict:
        project = self._check_run(run_id)
        self.store.write(run_id, points)
        last = {}
        n = 0
        for k, s in points.items():
            if len(s):
                i = int(np.argmax(s.step))
                last[k] = (int(s.step[i]), float(s.value[i]))
                n += len(s)
        self.meta.update_summary(run_id, last)
        if last:
            self.events.publish({"type": "points", "project": project, "run_id": run_id,
                                 "last": {k: v[0] for k, v in last.items()}})
        now = time.time()
        if now - self._last_maintain > 300:
            self._last_maintain = now
            self.store.maintain()
        return {"points": n}

    def op_heartbeat(self, run_id: int) -> dict:
        self._check_run(run_id)
        self.meta.heartbeat(run_id)
        return {"ok": True}

    def op_finish(self, run_id: int, status: str, when: Optional[float]) -> dict:
        project = self._check_run(run_id)
        self.meta.finish(run_id, status, when)
        self.events.publish({"type": "run", "project": project, "run_id": run_id, "status": status})
        return {"ok": True}

    def op_patch(self, run_id: int, body: dict) -> dict:
        project = self._check_run(run_id)
        allowed = {"name", "notes", "tags", "add_tags", "remove_tags"}
        self.meta.patch(run_id, **{k: v for k, v in body.items() if k in allowed})
        self.events.publish({"type": "run", "project": project, "run_id": run_id})
        return {"ok": True}

    def op_delete(self, run_id: int) -> dict:
        project = self._check_run(run_id)
        self.store.delete_run(run_id)
        self.meta.delete_run(run_id)
        self.events.publish({"type": "deleted", "project": project, "run_id": run_id})
        return {"ok": True}

    def op_mark_crashed(self) -> List[int]:
        ids = self.meta.mark_crashed(self.heartbeat_timeout)
        for i in ids:
            self.events.publish({"type": "run", "run_id": i, "status": "crashed"})
        return ids

    # ---- reads (any thread) -------------------------------------------
    def series(self, run_ids: List[int], keys: List[str], start: Optional[int] = None,
               end: Optional[int] = None, max_points: Optional[int] = 2000, x: str = "step",
               group_by: Optional[List[str]] = None, agg: str = "mean_std") -> dict:
        # Deliberately serial: concurrent SQLite reads in one process contend
        # (global allocator mutex) and measured ~8x slower than a plain loop.
        return series(self.meta, self.store, run_ids, keys, start, end, max_points, x, group_by, agg)


# ---------------------------------------------------------------- series reads + aggregation
def common_range(store: MetricStore, pairs, start, end) -> Tuple[Optional[int], Optional[int]]:
    """Fill open bounds from the data so all reads share one bucket grid."""
    if start is not None and end is not None:
        return start, end
    lo, hi = None, None
    for run, key in pairs:
        r = store.step_range(run, key)
        if r is None:
            continue
        lo = r[0] if lo is None else min(lo, r[0])
        hi = r[1] + 1 if hi is None else max(hi, r[1] + 1)
    if lo is None:
        return start, end
    return (lo if start is None else start), (hi if end is None else end)


def _bucket_payload(b: Buckets) -> dict:
    return b.to_dict()


def series(meta: MetadataStore, store: MetricStore, run_ids: List[int], keys: List[str],
           start: Optional[int] = None, end: Optional[int] = None, max_points: Optional[int] = 2000,
           x: str = "step", group_by: Optional[List[str]] = None, agg: str = "mean_std",
           pool: Optional[ThreadPoolExecutor] = None) -> dict:
    """Multi-run, multi-key fetch.

    x: "step", "wall", "relative" (seconds since run start) or another metric
       key (its value at the same bucket is used as x).
    group_by: config keys; runs with equal values are aggregated per bucket.
    agg: "mean_std" | "min_max" | "median_iqr".
    """
    runs = {r["id"]: r for r in meta.get_runs(run_ids, include=("config",))}
    run_ids = [r for r in run_ids if r in runs]
    xkey = x if x not in ("step", "wall", "relative") else None
    read_keys = list(dict.fromkeys(keys + ([xkey] if xkey else [])))
    pairs = [(r, k) for r in run_ids for k in read_keys]
    if max_points is not None:
        start, end = common_range(store, pairs, start, end)

    def read(pair):
        return pair, store.read(pair[0], pair[1], start, end, max_points)

    if pool is not None and len(pairs) > 1:
        results = dict(pool.map(read, pairs))
    else:
        results = dict(map(read, pairs))

    out_series = []
    for r in run_ids:
        run = runs[r]
        xb = results.get((r, xkey)) if xkey else None
        for k in keys:
            b = results[(r, k)]
            entry = {"run_id": r, "key": k, **_bucket_payload(b)}
            xs = _x_values(b, x, run, xb)
            if xs is not None:
                entry["x"] = _clean(xs)
            out_series.append(entry)

    resp: Dict[str, Any] = {"start": start, "end": end, "series": out_series}
    if group_by:
        resp["groups"] = aggregate(runs, run_ids, keys, results, group_by, agg, x, xkey)
    return resp


def _x_values(b: Buckets, x: str, run: dict, xb: Optional[Buckets]):
    if x == "step":
        return None
    if x in ("wall", "relative"):
        if b.wall_time is None:
            return np.full(len(b), np.nan)
        return b.wall_time - (run["created"] if x == "relative" else 0.0)
    # another metric as x: match buckets by grid index
    if xb is None or len(xb) == 0:
        return np.full(len(b), np.nan)
    w = max(b.width, 1)
    xmap = dict(zip((xb.step // max(xb.width, 1)).tolist(), xb.mean.tolist()))
    return np.array([xmap.get(s, math.nan) for s in (b.step // w).tolist()])


def _clean(a: np.ndarray) -> list:
    out = a.tolist()
    for i in np.flatnonzero(~np.isfinite(a)):
        out[i] = None
    return out


def group_key(run: dict, group_by: List[str]) -> Tuple:
    cfg = run.get("config", {})
    vals = []
    for g in group_by:
        if g in ("name", "status"):
            vals.append(run.get(g))
        else:
            vals.append(cfg.get(g[7:] if g.startswith("config.") else g))
    return tuple(vals)


def aggregate(runs: Dict[int, dict], run_ids: List[int], keys: List[str],
              results: Dict[Tuple[int, str], Buckets], group_by: List[str], agg: str,
              x: str, xkey: Optional[str]) -> List[dict]:
    groups: Dict[Tuple, List[int]] = {}
    for r in run_ids:
        groups.setdefault(group_key(runs[r], group_by), []).append(r)
    out = []
    for gk, members in groups.items():
        for k in keys:
            bs = [(r, results[(r, k)]) for r in members if len(results[(r, k)])]
            label = {g: _jsonable(v) for g, v in zip(group_by, gk)}
            if not bs:
                out.append({"group": label, "key": k, "runs": members, "step": [], "n": []})
                continue
            w = max(b.width for _, b in bs)
            idx = sorted(set().union(*[set((b.step // w).tolist()) for _, b in bs]))
            pos = {i: j for j, i in enumerate(idx)}
            m = np.full((len(bs), len(idx)), np.nan)
            st = np.full((len(bs), len(idx)), np.nan)
            xv = np.full((len(bs), len(idx)), np.nan)
            for row, (r, b) in enumerate(bs):
                cols = [pos[i] for i in (b.step // w).tolist()]
                m[row, cols] = b.mean
                st[row, cols] = b.step
                xs = _x_values(b, x, runs[r], results.get((r, xkey)) if xkey else None)
                if xs is not None:
                    xv[row, cols] = xs
            with np.errstate(all="ignore"), _quiet():
                n = np.sum(np.isfinite(m), axis=0)
                entry = {"group": label, "key": k, "runs": [r for r, _ in bs],
                         "step": np.round(np.nanmean(st, axis=0)).astype(np.int64).tolist(),
                         "n": n.tolist(), "width": int(w)}
                if x != "step":
                    entry["x"] = _clean(np.nanmean(xv, axis=0))
                if agg == "min_max":
                    entry["center"] = _clean(np.nanmean(m, axis=0))
                    entry["lo"] = _clean(np.nanmin(m, axis=0))
                    entry["hi"] = _clean(np.nanmax(m, axis=0))
                elif agg == "median_iqr":
                    q = np.nanpercentile(m, [25, 50, 75], axis=0)
                    entry["center"], entry["lo"], entry["hi"] = _clean(q[1]), _clean(q[0]), _clean(q[2])
                else:
                    mu = np.nanmean(m, axis=0)
                    sd = np.nanstd(m, axis=0)
                    entry["center"], entry["lo"], entry["hi"] = _clean(mu), _clean(mu - sd), _clean(mu + sd)
            out.append(entry)
    return out


class _quiet:
    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *a):
        self._w.__exit__(*a)


def _jsonable(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)
