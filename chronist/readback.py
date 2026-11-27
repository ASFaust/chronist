"""Python read-back API.

    import chronist
    runs = chronist.runs('env = "seaquest" and model.arch = "lstm"', project="bptt-free")
    runs.config()                                 # one row per run, flattened config
    runs.history(["loss", "return"])              # long: run_id, run, key, step, value, ...
    runs.history(["loss"], max_points=2000)       # downsampled (value = bucket mean)

Source selection: ``url=`` / ``CHRONIST_URL`` -> REST; otherwise the local
data dir (``data_dir=`` / ``CHRONIST_DATA`` / ``~/.chronist/data``), opened
read-only, so it is safe while a server is running on it.
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import quote

import numpy as np


def long_table(meta, store, run_ids: List[int], keys: Optional[List[str]] = None,
               max_points: Optional[int] = None, start: Optional[int] = None,
               end: Optional[int] = None) -> Dict[str, list]:
    """Columns of a long table straight from the stores (local mode, export)."""
    runs = {r["id"]: r for r in meta.get_runs(run_ids, include=())}
    cols: Dict[str, list] = {c: [] for c in ("run_id", "run", "key", "step", "value", "wall_time")}
    if max_points is not None:
        for c in ("min", "max", "count"):
            cols[c] = []
    for rid in run_ids:
        if rid not in runs:
            continue
        for k in (keys if keys is not None else store.keys(rid)):
            b = store.read(rid, k, start, end, max_points)
            n = len(b)
            if n == 0:
                continue
            _append_buckets(cols, rid, runs[rid]["name"], k, b.to_dict(), raw=max_points is None)
    return cols


def _append_buckets(cols: Dict[str, list], rid: int, name: str, key: str, b: dict, raw: bool):
    n = len(b["step"])
    cols["run_id"].extend([rid] * n)
    cols["run"].extend([name] * n)
    cols["key"].extend([key] * n)
    cols["step"].extend(b["step"])
    cols["value"].extend(b["last"] if raw else b["mean"])
    cols["wall_time"].extend(b.get("wall_time") or [None] * n)
    if "min" in cols:
        cols["min"].extend(b["min"])
        cols["max"].extend(b["max"])
        cols["count"].extend(b["count"])


def _frame(cols: Dict[str, list], backend: str):
    if backend == "dict":
        return cols
    if backend == "polars":
        import polars as pl
        return pl.DataFrame(cols, strict=False)
    import pandas as pd
    df = pd.DataFrame(cols)
    for c in ("value", "wall_time", "min", "max"):
        if c in df:
            df[c] = df[c].astype("float64")
    return df


def _default_backend() -> str:
    b = os.environ.get("CHRONIST_DATAFRAME")
    if b:
        return b
    try:
        import pandas  # noqa: F401
        return "pandas"
    except ImportError:
        try:
            import polars  # noqa: F401
            return "polars"
        except ImportError:
            return "dict"


# ---------------------------------------------------------------- sources
class _LocalSource:
    def __init__(self, data_dir: str):
        from .server.core import open_data_dir
        self.meta, self.store, _ = open_data_dir(data_dir, readonly=True)
        from .blobs import FsBlobStore
        self.blobs = FsBlobStore(os.path.join(os.path.abspath(os.path.expanduser(data_dir)), "blobs"))

    def query(self, q: str, project: Optional[str]) -> List[dict]:
        ids = self.meta.query_ids(project, q)
        return self.meta.get_runs(ids)

    def history(self, run_ids, keys, max_points, start, end) -> Dict[str, list]:
        return long_table(self.meta, self.store, run_ids, keys, max_points, start, end)

    def keys(self, run_id: int) -> List[str]:
        return self.store.keys(run_id)

    def get_run(self, ref) -> Optional[dict]:
        rid = _as_id(ref)
        if rid is None:
            rid = self.meta.run_id_by_uid(str(ref))
        if rid is None:
            return None
        r = self.meta.get_run(rid)
        if r is not None:
            r["keys"] = self.store.keys(rid)
        return r

    def manifest(self, run_id: int) -> Optional[dict]:
        import json
        h = self.meta.get_code(run_id)
        raw = None if h is None else self.blobs.read(h)
        return None if raw is None else {"hash": h, **json.loads(raw)}

    def blob(self, h: str) -> Optional[bytes]:
        return self.blobs.read(h)

    def close(self):
        self.store.close()
        self.meta.close()


class _RemoteSource:
    def __init__(self, url: str, token: Optional[str]):
        import httpx
        headers = {"Accept": "application/msgpack"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.http = httpx.Client(base_url=url.rstrip("/"), headers=headers, timeout=120)

    def _get(self, path, **params):
        import msgpack
        r = self.http.get(path, params={k: v for k, v in params.items() if v is not None})
        _raise(r)
        return msgpack.unpackb(r.content, raw=False, strict_map_key=False)

    def _post(self, path, body):
        import msgpack
        r = self.http.post(path, json=body)
        _raise(r)
        return msgpack.unpackb(r.content, raw=False, strict_map_key=False)

    def query(self, q: str, project: Optional[str]) -> List[dict]:
        return self._get("/api/runs", q=q, project=project)["runs"]

    def history(self, run_ids, keys, max_points, start, end) -> Dict[str, list]:
        cols: Dict[str, list] = {c: [] for c in ("run_id", "run", "key", "step", "value", "wall_time")}
        if max_points is not None:
            for c in ("min", "max", "count"):
                cols[c] = []
        names = {}
        for rid in run_ids:
            if keys is None:
                run = self._get(f"/api/runs/{rid}")
                names[rid] = run["name"]
                rkeys = run["keys"]
            else:
                rkeys = keys
            for k in rkeys:
                params = {"start": start, "end": end, "max_points": max_points}
                b = self._get(f"/api/runs/{rid}/metrics/{quote(k, safe='/')}", **params)
                if b["step"]:
                    _append_buckets(cols, rid, names.get(rid, ""), k, b, raw=max_points is None)
        return cols

    def keys(self, run_id: int) -> List[str]:
        return self._get(f"/api/runs/{run_id}")["keys"]

    def get_run(self, ref) -> Optional[dict]:
        rid = _as_id(ref)
        try:
            if rid is None:
                rid = self._get(f"/api/runs/by-uid/{quote(str(ref), safe='')}")["id"]
            return self._get(f"/api/runs/{rid}")
        except RuntimeError as e:
            if "404" in str(e):
                return None
            raise

    def manifest(self, run_id: int) -> Optional[dict]:
        r = self.http.get(f"/api/runs/{run_id}/code")
        if r.status_code == 404:
            return None
        _raise(r)
        import msgpack
        return msgpack.unpackb(r.content, raw=False, strict_map_key=False)

    def blob(self, h: str) -> Optional[bytes]:
        from .codec import decompress
        r = self.http.get(f"/api/blobs/{h}")
        if r.status_code == 404:
            return None
        _raise(r)
        return decompress(r.content)

    def close(self):
        self.http.close()


def _as_id(ref) -> Optional[int]:
    if isinstance(ref, int) and not isinstance(ref, bool):
        return ref
    if isinstance(ref, str) and ref.isdigit():
        return int(ref)
    return None


def _raise(r):
    if r.status_code >= 400:
        try:
            msg = r.json().get("error")
        except Exception:
            msg = r.text
        raise RuntimeError(f"chronist server error {r.status_code}: {msg}")


def _open_source(url=None, token=None, data_dir=None):
    url = url or (None if data_dir else os.environ.get("CHRONIST_URL"))
    if url:
        return _RemoteSource(url, token or os.environ.get("CHRONIST_TOKEN"))
    from .server.core import default_data_dir
    return _LocalSource(data_dir or default_data_dir())


# ---------------------------------------------------------------- public API
class RunSet:
    """A queried set of runs. Iterable over run dicts."""

    def __init__(self, source, runs: List[dict], dataframe: Optional[str] = None):
        self._src = source
        self.runs = runs
        self._df = dataframe or _default_backend()

    def __len__(self):
        return len(self.runs)

    def __iter__(self):
        return iter(self.runs)

    def __getitem__(self, i):
        return self.runs[i]

    def __repr__(self):
        return f"<RunSet {len(self.runs)} runs>"

    @property
    def ids(self) -> List[int]:
        return [r["id"] for r in self.runs]

    def config(self, summary: bool = False):
        """One row per run: id, name, status, tags, steps + flattened config
        (+ ``last(<key>)`` columns if summary=True)."""
        keys: List[str] = []
        for r in self.runs:
            for k in r["config"]:
                if k not in keys:
                    keys.append(k)
        cols: Dict[str, list] = {c: [] for c in ("run_id", "name", "status", "tags", "steps", "created")}
        for k in keys:
            cols[k] = []
        skeys: List[str] = sorted({k for r in self.runs for k in r.get("summary", {})}) if summary else []
        for k in skeys:
            cols[f"last({k})"] = []
        for r in self.runs:
            cols["run_id"].append(r["id"])
            cols["name"].append(r["name"])
            cols["status"].append(r["status"])
            cols["tags"].append(",".join(r.get("tags", [])))
            cols["steps"].append(r.get("steps"))
            cols["created"].append(r.get("created"))
            for k in keys:
                v = r["config"].get(k)
                cols[k].append(str(v) if isinstance(v, list) else v)
            for k in skeys:
                s = r["summary"].get(k)
                cols[f"last({k})"].append(None if s is None else s["value"])
        return _frame(cols, self._df)

    def history(self, keys: Optional[Iterable[str]] = None, max_points: Optional[int] = None,
                start: Optional[int] = None, end: Optional[int] = None):
        """Long table: run_id, run, key, step, value, wall_time (+ min, max,
        count when downsampled). keys=None -> every key of every run."""
        if isinstance(keys, str):
            keys = [keys]
        cols = self._src.history(self.ids, list(keys) if keys is not None else None, max_points, start, end)
        names = {r["id"]: r["name"] for r in self.runs}
        cols["run"] = [names.get(i, "") for i in cols["run_id"]]
        return _frame(cols, self._df)

    def keys(self) -> List[str]:
        """Union of metric keys over the runs."""
        out = set()
        for r in self.runs:
            out.update(r.get("summary", {}).keys())
        return sorted(out)

    def close(self):
        self._src.close()


def runs(query: str = "", project: Optional[str] = None, *, url: Optional[str] = None,
         token: Optional[str] = None, data_dir: Optional[str] = None,
         dataframe: Optional[str] = None) -> RunSet:
    """Query runs. ``dataframe`` = "pandas" | "polars" | "dict"."""
    src = _open_source(url, token, data_dir)
    return RunSet(src, src.query(query, project), dataframe)


# ---------------------------------------------------------------- single runs + code
class CodeSnapshot:
    """A run's code snapshot (manifest + content-addressed blobs)."""

    def __init__(self, src, manifest: dict, run: dict):
        self._src = src
        self.manifest = manifest
        self.run = run
        self._by_path = {f["path"]: f for f in manifest["files"]}

    @property
    def hash(self) -> str:
        return self.manifest["hash"]

    @property
    def root(self) -> dict:
        return self.manifest["root"]

    @property
    def skipped(self) -> List[dict]:
        return self.manifest["skipped"]

    def files(self) -> List[str]:
        return [f["path"] for f in self.manifest["files"]]

    def read_bytes(self, path: str) -> bytes:
        f = self._by_path.get(path)
        if f is None:
            raise FileNotFoundError(path)
        if "symlink" in f:
            return f["symlink"].encode()
        data = self._src.blob(f["sha256"])
        if data is None:
            raise FileNotFoundError(f"blob for {path} missing")
        return data

    def read(self, path: str) -> str:
        return self.read_bytes(path).decode("utf-8", errors="replace")

    def env(self) -> dict:
        import json
        return json.loads(self._src.blob(self.manifest["env"]))

    def packages(self) -> List[str]:
        return self._src.blob(self.manifest["packages"]).decode().split()

    def checkout(self, dest: str, force: bool = False) -> str:
        """Restore the file tree into ``dest``; also writes CHRONIST_RUN.json."""
        import json

        from .client.code import write_tree
        write_tree(self.manifest, self._src.blob, dest, force=force)
        meta = self.run.get("meta", {})
        info = {
            "run": {k: self.run.get(k) for k in ("id", "uid", "project", "name", "status", "created")},
            "config": self.run.get("config"),
            "command": meta.get("command"),
            "code": {k: v for k, v in (meta.get("code") or {}).items()},
            "manifest": self.hash,
            "env": self.env(),
            "skipped": self.skipped,
        }
        with open(os.path.join(dest, "CHRONIST_RUN.json"), "w") as f:
            json.dump(info, f, indent=2, default=str)
        return os.path.abspath(dest)


class RunRef:
    """One run: ``chronist.run(id_or_uid)``."""

    def __init__(self, src, data: dict, dataframe: Optional[str] = None):
        self._src = src
        self.data = data
        self._df = dataframe
        self._code: Any = False

    def __getitem__(self, k):
        return self.data[k]

    def __repr__(self):
        return f"<chronist run {self.data['id']} {self.data['name']!r}>"

    id = property(lambda self: self.data["id"])
    uid = property(lambda self: self.data["uid"])
    name = property(lambda self: self.data["name"])
    config = property(lambda self: self.data["config"])
    summary = property(lambda self: self.data["summary"])
    meta = property(lambda self: self.data["meta"])

    @property
    def code(self) -> Optional[CodeSnapshot]:
        if self._code is False:
            m = self._src.manifest(self.id)
            self._code = None if m is None else CodeSnapshot(self._src, m, self.data)
        return self._code

    def history(self, keys=None, max_points=None, start=None, end=None):
        return RunSet(self._src, [self.data], self._df).history(keys, max_points, start, end)


def run(ref, *, url: Optional[str] = None, token: Optional[str] = None, data_dir: Optional[str] = None,
        dataframe: Optional[str] = None) -> RunRef:
    """A single run by server id or uid."""
    src = _open_source(url, token, data_dir)
    data = src.get_run(ref)
    if data is None:
        raise KeyError(f"run {ref!r} not found")
    return RunRef(src, data, dataframe)


def code_diff(a, b, *, url: Optional[str] = None, token: Optional[str] = None,
              data_dir: Optional[str] = None, unified: bool = True) -> dict:
    """Diff the code snapshots and configs of two runs.

    Returns {"added", "removed", "changed": [paths], "diffs": {path: unified diff},
             "config": {"added", "removed", "changed": {key: [a, b]}}}."""
    import difflib

    from .client.code import diff_manifests
    ra = a if isinstance(a, RunRef) else run(a, url=url, token=token, data_dir=data_dir)
    rb = b if isinstance(b, RunRef) else run(b, url=url, token=token, data_dir=data_dir)
    out: Dict[str, Any] = {"added": [], "removed": [], "changed": [], "diffs": {}}
    if ra.code is not None and rb.code is not None:
        out.update(diff_manifests(ra.code.manifest, rb.code.manifest))
        if unified:
            for p in out["changed"]:
                da, db = ra.code.read_bytes(p), rb.code.read_bytes(p)
                if b"\0" in da[:8192] or b"\0" in db[:8192]:
                    out["diffs"][p] = "binary files differ"
                    continue
                out["diffs"][p] = "".join(difflib.unified_diff(
                    da.decode(errors="replace").splitlines(True), db.decode(errors="replace").splitlines(True),
                    f"a/{p}", f"b/{p}"))
    ca, cb = ra.config, rb.config
    out["config"] = {"added": sorted(set(cb) - set(ca)), "removed": sorted(set(ca) - set(cb)),
                     "changed": {k: [ca[k], cb[k]] for k in sorted(set(ca) & set(cb)) if ca[k] != cb[k]}}
    return out
