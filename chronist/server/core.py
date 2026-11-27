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

from ..blobs import FsBlobStore, valid_hash
from ..meta import MetadataStore
from ..store import Buckets, MetricStore, Series, bucket_width, open_store
from ..store.cache import CachingStore

log = logging.getLogger("chronist.server")

DATA_VERSION = 1
DEFAULT_METRIC_STORE = "chunked"  # new data dirs; existing ones keep what chronist.json records


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
        cfg = {"version": DATA_VERSION, "metric_store": metric_store or DEFAULT_METRIC_STORE}
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
    meta = MetadataStore(os.path.join(path, "meta.sqlite"), readonly=readonly)
    store = open_store(cfg["metric_store"], os.path.join(path, f"metrics.{cfg['metric_store']}.sqlite"),
                       readonly=readonly)
    return meta, store, cfg


def lock_data_dir(path: str):
    """Exclusive lock on a data dir; raises if another process holds it."""
    import fcntl
    f = open(os.path.join(path, ".lock"), "a+")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.seek(0)
        who = f.read().strip()
        f.close()
        raise RuntimeError(f"{path} is in use by another chronist server{f' (pid {who})' if who else ''}; "
                           "only one process may write a data dir")
    f.seek(0)
    f.truncate()
    f.write(str(os.getpid()))
    f.flush()
    return f


def _dir_size(path: str) -> int:
    total = 0
    for name in os.listdir(path):
        p = os.path.join(path, name)
        if os.path.isfile(p):
            total += os.path.getsize(p)
    return total


def open_hist_store(data_dir: str, readonly: bool = False):
    from ..store.hist_store import SqliteHistStore
    path = os.path.join(os.path.abspath(os.path.expanduser(data_dir)), "hists.sqlite")
    if readonly and not os.path.exists(path):
        return None
    return SqliteHistStore(path, readonly=readonly)


class QuotaError(Exception):
    def __init__(self, status: int, msg: str):
        super().__init__(msg)
        self.status = status
        self.msg = msg


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
def decode_hists(payload: dict) -> Dict[str, list]:
    """{"hists": [{"key": k, <HistRecord wire fields>}, ...]} -> {key: [HistRecord]}"""
    from ..hist import HistRecord
    out: Dict[str, list] = {}
    for d in payload.get("hists") or []:
        if not isinstance(d, dict) or not isinstance(d.get("key"), str) or not d["key"]:
            raise ValueError("each histogram needs a non-empty key")
        counts = d.get("counts")
        if isinstance(counts, list):  # JSON clients may send plain lists
            import numpy as _np
            d = {**d, "counts": _np.asarray(counts, dtype="<u8").tobytes(), "c64": True}
        if not isinstance(d["counts"], (bytes, bytearray)):
            raise ValueError("histogram counts must be bytes (uint32/uint64 LE) or a list")
        if len(d["counts"]) > 8 * 1024 + 8:
            raise ValueError("too many histogram bins (max 1024)")
        out.setdefault(d["key"], []).append(HistRecord.from_wire(d))
    return out


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
                 heartbeat_timeout: float = 180.0, cache_mb: float = 64.0, blob_dir: Optional[str] = None,
                 max_blob_storage: Optional[int] = None, max_file_size: int = 20 * 10 ** 9):
        self.data_dir = os.path.abspath(os.path.expanduser(data_dir))
        os.makedirs(self.data_dir, exist_ok=True)
        self._lock = lock_data_dir(self.data_dir)  # the server is the only writer, enforced
        self.meta, store, self.cfg = open_data_dir(self.data_dir, metric_store=metric_store)
        self.store = CachingStore(store, max_mb=cache_mb)
        self.blobs = FsBlobStore(blob_dir or os.path.join(self.data_dir, "blobs"))
        self.max_blob_storage = max_blob_storage  # bytes; None = unlimited
        self.max_file_size = max_file_size
        self._blob_usage: Optional[int] = None
        self._quota_warned = False
        self.hists = open_hist_store(self.data_dir)
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
        self.hists.close()
        self.meta.close()
        if self._lock is not None:
            self._lock.close()
            self._lock = None

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

    def op_log(self, run_id: int, points: Dict[str, Series], hists: Optional[Dict[str, list]] = None) -> dict:
        project = self._check_run(run_id)
        self.store.write(run_id, points)
        if hists:
            self.hists.write(run_id, hists)
            hlast = {k: max(r.step for r in rs) for k, rs in hists.items() if rs}
            self.meta.update_hist_keys(run_id, hlast)
            self.events.publish({"type": "hist", "project": project, "run_id": run_id, "last": hlast})
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

    def op_rewind(self, run_id: int, step: int) -> dict:
        """Discard every point at step >= ``step`` (resume from an earlier checkpoint)."""
        project = self._check_run(run_id)
        step = int(step)
        self.store.delete_range(run_id, step)
        self.hists.delete_range(run_id, step)
        keys = self.store.keys(run_id)
        self.meta.replace_summary(run_id, self.store.last(run_id, keys))
        hk = {}
        for k in self.hists.keys(run_id):
            r = self.hists.last(run_id, k)
            if r is not None:
                hk[k] = r.step
        self.meta.update_hist_keys(run_id, hk, replace=True)
        self.events.publish({"type": "rewind", "project": project, "run_id": run_id, "step": step})
        return {"ok": True, "step": step}

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
        allowed = {"name", "notes", "tags", "add_tags", "remove_tags", "meta"}
        if "meta" in body and not isinstance(body["meta"], dict):
            raise ValueError("meta must be an object")
        self.meta.patch(run_id, **{k: v for k, v in body.items() if k in allowed})
        self.events.publish({"type": "run", "project": project, "run_id": run_id})
        return {"ok": True}

    def op_delete(self, run_id: int) -> dict:
        project = self._check_run(run_id)
        self.store.delete_run(run_id)
        self.hists.delete_run(run_id)
        self.meta.delete_run(run_id)
        self.events.publish({"type": "deleted", "project": project, "run_id": run_id})
        return {"ok": True}

    def op_put_blob(self, h: str, data: bytes) -> dict:
        existed = self.blobs.has(h)
        self.blobs.put(h, data)  # verifies the hash
        if not existed and self._blob_usage is not None:
            self._blob_usage += len(data)
        return {"ok": True}

    # ---- blob storage quota ----
    def blob_usage(self) -> int:
        if self._blob_usage is None:
            total = 0
            if os.path.isdir(self.blobs.path):
                for root, _, names in os.walk(self.blobs.path):
                    for n in names:
                        try:
                            total += os.path.getsize(os.path.join(root, n))
                        except OSError:
                            pass
            self._blob_usage = total
        return self._blob_usage

    def check_file_upload(self, size: int) -> None:
        """Raise QuotaError if a new file of ``size`` bytes may not be stored."""
        if size > self.max_file_size:
            raise QuotaError(413, f"file is {size / 1e9:.2f} GB; the server limit is {self.max_file_size / 1e9:.2f} GB")
        if self.max_blob_storage:
            used = self.blob_usage()
            if used + size > self.max_blob_storage:
                raise QuotaError(507, f"blob storage quota reached ({used / 1e9:.2f} of "
                                      f"{self.max_blob_storage / 1e9:.2f} GB); file uploads are paused, metrics still work")
            if used > 0.8 * self.max_blob_storage and not self._quota_warned:
                self._quota_warned = True
                log.warning("blob storage at %.0f%% of quota", 100 * used / self.max_blob_storage)

    # ---- files ----
    def op_register_file(self, run_id: int, manifest: str, local_id: str, retention: Optional[dict]) -> dict:
        project = self._check_run(run_id)
        if not valid_hash(manifest):
            raise ValueError("bad manifest hash")
        if not isinstance(local_id, str) or not local_id:
            raise ValueError("local_id is required")
        raw = self.blobs.read(manifest)
        if raw is None:
            raise ValueError("manifest blob not uploaded")
        m = json.loads(raw)
        if m.get("kind") != "chronist.file/1":
            raise ValueError("not a file manifest")
        if m["size"] > self.max_file_size:
            raise QuotaError(413, "file exceeds the server's max file size")
        missing = self.blobs.missing(m["chunks"])
        if missing:
            raise ValueError(f"{len(missing)} chunks of {m['name']!r} are missing")
        vid, new = self.meta.add_file_version(run_id, local_id, manifest, m)
        dropped: List[int] = []
        if retention:
            kl, ke = retention.get("keep_last"), retention.get("keep_every")
            for v in (kl, ke):
                if v is not None and (not isinstance(v, int) or isinstance(v, bool) or v < 1):
                    raise ValueError("retention values must be positive ints")
            self.meta.set_file_policy(run_id, m["name"], kl, ke)
        victims = self.meta.retention_victims(run_id, m["name"])
        if victims:
            self.meta.delete_file_versions(run_id, victims)
            dropped = victims
        self.events.publish({"type": "files", "project": project, "run_id": run_id})
        return {"version_id": vid, "new": new, "dropped": dropped}

    def op_delete_file(self, run_id: int, version_id: int) -> dict:
        project = self._check_run(run_id)
        f = self.meta.get_file(version_id)
        if f is None or f["run_id"] != run_id:
            raise KeyError(f"no file version {version_id} in run {run_id}")
        self.meta.delete_file_versions(run_id, [version_id])
        self.events.publish({"type": "files", "project": project, "run_id": run_id})
        return {"ok": True}

    def op_set_aliases(self, run_id: int, name: str, set_: dict, remove: list) -> dict:
        project = self._check_run(run_id)
        self.meta.set_file_aliases(run_id, name, set_ or {}, remove or [])
        self.events.publish({"type": "files", "project": project, "run_id": run_id})
        return {"ok": True}

    def file_chunk(self, version_id: int, i: int) -> Optional[bytes]:
        """Stored (codec-tagged) bytes of chunk i of a file version."""
        f = self.meta.get_file(version_id)
        if f is None:
            return None
        m = json.loads(self.blobs.read(f["manifest"]))
        if not 0 <= i < len(m["chunks"]):
            return None
        return self.blobs.get(m["chunks"][i])

    def op_set_code(self, run_id: int, manifest: str) -> dict:
        project = self._check_run(run_id)
        if not valid_hash(manifest):
            raise ValueError("bad manifest hash")
        raw = self.blobs.read(manifest)
        if raw is None:
            raise ValueError("manifest blob not uploaded")
        from ..client.code import manifest_blobs
        missing = self.blobs.missing(manifest_blobs(json.loads(raw)))
        if missing:
            raise ValueError(f"{len(missing)} blobs referenced by the manifest are missing")
        self.meta.set_code(run_id, manifest)
        self.events.publish({"type": "run", "project": project, "run_id": run_id})
        return {"ok": True}

    def manifest(self, run_id: int) -> Optional[dict]:
        h = self.meta.get_code(run_id)
        if h is None:
            return None
        raw = self.blobs.read(h)
        return None if raw is None else {"hash": h, **json.loads(raw)}

    def op_gc(self, vacuum: bool = False, grace_s: float = 86400.0) -> dict:
        """Delete blobs no run references (older than ``grace_s``: uploads in
        flight are not attached yet), then give free pages back to the OS."""
        from ..client.code import manifest_blobs
        refs = set()
        for mh in self.meta.manifests():  # code snapshots
            refs.add(mh)
            raw = self.blobs.read(mh)
            if raw is not None:
                refs.update(manifest_blobs(json.loads(raw)))
        for mh in self.meta.file_manifests():  # file versions
            refs.add(mh)
            raw = self.blobs.read(mh)
            if raw is not None:
                refs.update(json.loads(raw)["chunks"])
        cutoff = time.time() - grace_s
        removed = freed = 0
        if os.path.isdir(self.blobs.path):
            for sub in os.listdir(self.blobs.path):
                d = os.path.join(self.blobs.path, sub)
                if not os.path.isdir(d):
                    continue
                for name in os.listdir(d):
                    p = os.path.join(d, name)
                    if sub + name in refs:
                        continue
                    try:
                        st = os.stat(p)
                        if st.st_mtime > cutoff:
                            continue
                        os.remove(p)
                        removed += 1
                        freed += st.st_size
                    except FileNotFoundError:
                        pass
        if self._blob_usage is not None:
            self._blob_usage = max(0, self._blob_usage - freed)
        before = _dir_size(self.data_dir)
        self.meta.vacuum(full=vacuum)
        if hasattr(self.store, "vacuum"):
            self.store.vacuum(full=vacuum)
        self.hists.vacuum(full=vacuum)
        after = _dir_size(self.data_dir)
        return {"blobs_removed": removed, "blob_bytes_freed": freed,
                "db_bytes_freed": max(0, before - after), "vacuum": "full" if vacuum else "incremental"}

    def render_hist(self, run_ids: List[int], key: str, start=None, end=None, max_steps: int = 300,
                    rows: int = 128, y_range=None, mode: str = "heatmap", step=None, robust: bool = True) -> dict:
        return render_hist(self.hists, run_ids, key, start, end, max_steps, rows, y_range, mode, step, robust)

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
    runs = {r["id"]: r for r in meta.get_runs(run_ids, include=("config", "tags"))}
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
    """Group values for a run. Fields use query-language names: config.<key>,
    or the run fields name / status / tags."""
    from ..query import field_from_name
    cfg = run.get("config", {})
    vals = []
    for g in group_by:
        f = field_from_name(g)
        if f is None or f.kind == "last":
            raise ValueError(f"cannot group by {g!r}: use config.<key> or name / status / tags")
        if f.kind == "config":
            vals.append(cfg.get(f.name))
        elif f.name == "tags":
            vals.append(",".join(run.get("tags") or []))
        else:
            vals.append(run.get(f.name if f.name != "run_id" else "id"))
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


# ---------------------------------------------------------------- histogram rendering
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def _stats(recs) -> dict:
    from ..hist import _f
    cols = ("count", "nan_count", "posinf_count", "neginf_count", "min", "max", "mean", "std",
            "underflow", "overflow")
    return {c: [(_f(getattr(r, c)) if isinstance(getattr(r, c), float) else getattr(r, c)) for r in recs]
            for c in cols}


def render_hist(hists, run_ids: List[int], key: str, start=None, end=None, max_steps: int = 300,
                rows: int = 128, y_range=None, mode: str = "heatmap", step=None, robust: bool = True) -> dict:
    """Server-side histogram views for one key over several runs.

    heatmap:   per run a steps x rows density matrix on one shared value grid
    quantiles: per run 5/25/50/75/95% quantiles per step
    step:      per run the record at (or before) ``step``, re-binned on a shared grid
    All modes include exact per-step stats (mean/std/min/max/count/NaN/Inf).
    """
    from .. import hist as H
    max_steps = max(1, min(int(max_steps or 300), 5000))
    rows = max(1, min(int(rows or 128), 1024))
    out: Dict[str, Any] = {"key": key, "mode": mode, "runs": []}
    if mode == "step":
        recs = {}
        for rid in run_ids:
            r = hists.at(rid, key, int(step)) if step is not None else hists.last(rid, key)
            if r is not None:
                recs[rid] = r
        lo, hi = y_range or H.common_range(list(recs.values()))
        edges = np.linspace(lo, hi, rows + 1)
        out["edges"] = edges.tolist()
        for rid, r in recs.items():
            out["runs"].append({"run_id": rid, "step": r.step, "density": H.rebin(r, edges).tolist(),
                                **{k: v[0] for k, v in _stats([r]).items()}})
        return out
    per_run = {rid: hists.read(rid, key, start, end, max_steps) for rid in run_ids}
    allrecs = [r for rs in per_run.values() for r in rs]
    if mode == "heatmap":
        lo, hi = y_range or (H.robust_range(allrecs) if robust else H.common_range(allrecs))
        out["y_range"] = [lo, hi]
        out["rows"] = rows
        vmax = 0.0
        for rid, rs in per_run.items():
            m, _ = H.to_dense(rs, rows=rows, y_range=(lo, hi))
            vmax = max(vmax, float(m.max()) if m.size else 0.0)
            outside = [float(r.count - row.sum()) if r.count else 0.0 for r, row in zip(rs, m)]
            out["runs"].append({"run_id": rid, "steps": [r.step for r in rs], "outside": outside,
                                "matrix": np.round(m, 3).ravel().tolist(), **_stats(rs)})
        out["vmax"] = vmax
    elif mode == "quantiles":
        out["quantiles"] = list(QUANTILES)
        for rid, rs in per_run.items():
            qs = np.array([H.quantiles(r, QUANTILES) for r in rs]).reshape(len(rs), len(QUANTILES))
            q = {f"q{int(p * 100):02d}": [None if not np.isfinite(v) else float(v) for v in qs[:, i]]
                 for i, p in enumerate(QUANTILES)}
            out["runs"].append({"run_id": rid, "steps": [r.step for r in rs], **q, **_stats(rs)})
    else:
        raise ValueError(f"unknown histogram render mode {mode!r}")
    return out
