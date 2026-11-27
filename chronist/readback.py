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


PAGE = 1_000_000  # raw points per read: bounds transient memory


def _empty_cols(downsampled: bool) -> Dict[str, list]:
    cols = {c: [] for c in ("run_id", "run", "key", "step", "value", "wall_time")}
    if downsampled:
        for c in ("min", "max", "count"):
            cols[c] = []
    return cols


def _page_cols(rid: int, name: str, key: str, b, raw: bool) -> Dict[str, np.ndarray]:
    """One read (Buckets) as numpy columns."""
    n = len(b)
    out = {
        "run_id": np.full(n, rid, dtype=np.int64),
        "run": np.full(n, name, dtype=object),
        "key": np.full(n, key, dtype=object),
        "step": np.asarray(b.step, dtype=np.int64),
        "value": np.asarray(b.last if raw else b.mean, dtype=np.float64),
        "wall_time": (np.asarray(b.wall_time, dtype=np.float64) if b.wall_time is not None
                      else np.full(n, np.nan)),
    }
    if not raw:
        out["min"] = np.asarray(b.min, dtype=np.float64)
        out["max"] = np.asarray(b.max, dtype=np.float64)
        out["count"] = np.asarray(b.count, dtype=np.int64)
    return out


def iter_pages(meta, store, run_ids: List[int], keys: Optional[List[str]] = None,
               max_points: Optional[int] = None, start: Optional[int] = None,
               end: Optional[int] = None, page: int = PAGE):
    """Yield numpy column pages of a long table straight from the stores.
    Raw reads are paged so no single read holds more than ``page`` points."""
    names = {r["id"]: r["name"] for r in meta.get_runs(run_ids, include=())}
    for rid in run_ids:
        if rid not in names:
            continue
        for k in (keys if keys is not None else store.keys(rid)):
            if max_points is not None:
                b = store.read(rid, k, start, end, max_points)
                if len(b):
                    yield _page_cols(rid, names[rid], k, b, raw=False)
                continue
            s = start
            while True:
                b = store.read(rid, k, s, end, None, limit=page)
                if len(b) == 0:
                    break
                yield _page_cols(rid, names[rid], k, b, raw=True)
                if len(b) < page:
                    break
                s = int(b.step[-1]) + 1


def _concat(pages, downsampled: bool) -> Dict[str, np.ndarray]:
    cols = _empty_cols(downsampled)
    for p in pages:
        for c in cols:
            cols[c].append(p[c])
    dtypes = {"run_id": np.int64, "step": np.int64, "count": np.int64, "run": object, "key": object}
    return {c: (np.concatenate(v) if v else np.zeros(0, dtypes.get(c, np.float64))) for c, v in cols.items()}


def long_table(meta, store, run_ids: List[int], keys: Optional[List[str]] = None,
               max_points: Optional[int] = None, start: Optional[int] = None,
               end: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Columns (numpy arrays) of a long table straight from the stores."""
    return _concat(iter_pages(meta, store, run_ids, keys, max_points, start, end), max_points is not None)


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
        self._data_dir = data_dir
        self._hists = False  # opened lazily
        from .blobs import FsBlobStore
        self.blobs = FsBlobStore(os.path.join(os.path.abspath(os.path.expanduser(data_dir)), "blobs"))

    def query(self, q: str, project: Optional[str]) -> List[dict]:
        ids = self.meta.query_ids(project, q)
        return self.meta.get_runs(ids)

    def history(self, run_ids, keys, max_points, start, end):
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

    def hist_records(self, run_id, key, start=None, end=None, max_steps=None, at=None):
        from .server.core import open_hist_store
        if self._hists is False:
            self._hists = open_hist_store(self._data_dir, readonly=True)
        if self._hists is None:
            return []
        if at is not None:
            r = self._hists.at(run_id, key, at)
            return [] if r is None else [r]
        return self._hists.read(run_id, key, start, end, max_steps)

    def manifest(self, run_id: int) -> Optional[dict]:
        import json
        h = self.meta.get_code(run_id)
        raw = None if h is None else self.blobs.read(h)
        return None if raw is None else {"hash": h, **json.loads(raw)}

    def blob(self, h: str) -> Optional[bytes]:
        return self.blobs.read(h)

    def list_files(self, run_id: int) -> List[dict]:
        return self.meta.list_files(run_id)

    def file_info(self, version_id: int) -> dict:
        import json
        f = self.meta.get_file(version_id)
        if f is None:
            raise KeyError(f"file version {version_id} not found")
        m = json.loads(self.blobs.read(f["manifest"]))
        return {**f, "chunk_hashes": m["chunks"], "chunk_size": m["chunk_size"]}

    def file_chunk(self, version_id: int, i: int, h: str) -> bytes:
        data = self.blobs.get(h)
        if data is None:
            raise RuntimeError(f"chunk {h} missing from the local data dir")
        return data

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

    def history(self, run_ids, keys, max_points, start, end):
        from .store import Buckets
        pages = []
        for rid in run_ids:
            run = self._get(f"/api/runs/{rid}")
            for k in (keys if keys is not None else run["keys"]):
                s = start
                while True:
                    params = {"start": s, "end": end, "max_points": max_points}
                    d = self._get(f"/api/runs/{rid}/metrics/{quote(k, safe='/')}", **params)
                    if d["step"]:
                        f = lambda name: np.array([np.nan if v is None else v for v in d[name]], dtype=np.float64)
                        wt = f("wall_time") if d.get("wall_time") else None
                        if d.get("raw"):
                            b = Buckets.from_raw(np.asarray(d["step"], np.int64), f("value"), wt)
                        else:
                            b = Buckets(np.asarray(d["step"], np.int64), f("min"), f("max"), f("mean"), f("last"),
                                        np.asarray(d["count"], np.int64), wt)
                        pages.append(_page_cols(rid, run["name"], k, b, raw=max_points is None))
                    if d.get("next_start") is None:
                        break
                    s = d["next_start"]
        return _concat(pages, max_points is not None)

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

    def hist_records(self, run_id, key, start=None, end=None, max_steps=None, at=None):
        from .hist import HistRecord
        d = self._get(f"/api/runs/{run_id}/hist/{quote(key, safe='/')}", start=start, end=end,
                      max_steps=max_steps, at=at)
        return [HistRecord.from_json(r) for r in d["records"]]

    def manifest(self, run_id: int) -> Optional[dict]:
        r = self.http.get(f"/api/runs/{run_id}/code")
        if r.status_code == 404:
            return None
        if r.status_code == 403 and "code not shared" in r.text:
            raise PermissionError("code snapshots are not shared with this token")
        _raise(r)
        import msgpack
        return msgpack.unpackb(r.content, raw=False, strict_map_key=False)

    def list_files(self, run_id: int) -> List[dict]:
        return self._get(f"/api/runs/{run_id}/files")["files"]

    def file_info(self, version_id: int) -> dict:
        return self._get(f"/api/files/{version_id}")

    def file_chunk(self, version_id: int, i: int, h: str) -> bytes:
        for attempt in range(5):
            try:
                r = self.http.get(f"/api/files/{version_id}/chunk/{i}")
                _raise(r)
                return r.content
            except Exception:
                if attempt == 4:
                    raise
                import time as _t
                _t.sleep(0.5 * 2 ** attempt)

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
        from .client.auth import resolve_token
        return _RemoteSource(url, resolve_token(url, token))
    from .server.core import default_data_dir
    return _LocalSource(data_dir or default_data_dir())


# ---------------------------------------------------------------- public API
RUN_COLUMNS = ("run_id", "name", "status", "tags", "steps", "created")



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
        """One row per run: run_id, name, status, tags, steps, created, then
        every config key as ``config.<key>`` (+ ``metric.<key>``, each metric's
        latest value, if summary=True). Column names are the query-language
        names, so they never clash and can be pasted into queries."""
        keys: List[str] = []
        for r in self.runs:
            for k in r["config"]:
                if k not in keys:
                    keys.append(k)
        cols: Dict[str, list] = {c: [] for c in RUN_COLUMNS}
        for k in keys:
            cols[f"config.{k}"] = []
        skeys: List[str] = sorted({k for r in self.runs for k in r.get("summary", {})}) if summary else []
        for k in skeys:
            cols[f"metric.{k}"] = []
        for r in self.runs:
            cols["run_id"].append(r["id"])
            cols["name"].append(r["name"])
            cols["status"].append(r["status"])
            cols["tags"].append(",".join(r.get("tags", [])))
            cols["steps"].append(r.get("steps"))
            cols["created"].append(r.get("created"))
            for k in keys:
                v = r["config"].get(k)
                cols[f"config.{k}"].append(str(v) if isinstance(v, list) else v)
            for k in skeys:
                s = r["summary"].get(k)
                cols[f"metric.{k}"].append(None if s is None else s["value"])
        return _frame(cols, self._df)

    def history(self, keys: Optional[Iterable[str]] = None, max_points: Optional[int] = None,
                start: Optional[int] = None, end: Optional[int] = None):
        """Long table: run_id, run, key, step, value, wall_time (+ min, max,
        count when downsampled). keys=None -> every key of every run."""
        if isinstance(keys, str):
            keys = [keys]
        cols = self._src.history(self.ids, list(keys) if keys is not None else None, max_points, start, end)
        return _frame(cols, self._df)

    def histogram_records(self, key: str, max_steps: Optional[int] = None, start: Optional[int] = None,
                          end: Optional[int] = None) -> Dict[int, Any]:
        """{run_id: HistList} for one histogram key (``.to_dense(rows)`` on each list)."""
        from .hist import HistList
        return {r["id"]: HistList(self._src.hist_records(r["id"], key, start, end, max_steps)) for r in self.runs}

    def histograms(self, key: str, max_steps: Optional[int] = None, start: Optional[int] = None,
                   end: Optional[int] = None):
        """Long table, one row per (run, step): stats + ``edges`` and ``counts`` arrays."""
        cols: Dict[str, list] = {c: [] for c in (
            "run_id", "run", "step", "wall_time", "count", "nan_count", "posinf_count", "neginf_count",
            "min", "max", "mean", "std", "lo", "hi", "underflow", "overflow", "edges", "counts")}
        names = {r["id"]: r["name"] for r in self.runs}
        for rid, recs in self.histogram_records(key, max_steps, start, end).items():
            for rec in recs:
                cols["run_id"].append(rid)
                cols["run"].append(names[rid])
                for c in list(cols)[2:]:
                    if c not in ("edges", "counts"):
                        cols[c].append(getattr(rec, c))
                cols["edges"].append(rec.edges)
                cols["counts"].append(rec.counts)
        if self._df == "polars":  # polars wants lists for nested columns
            cols["edges"] = [e.tolist() for e in cols["edges"]]
            cols["counts"] = [c.tolist() for c in cols["counts"]]
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

    # ---- files ----
    def files(self, raw: bool = False):
        """All file versions: name, version_id, step, kind, size, aliases, metadata, created."""
        rows = self._src.list_files(self.id)
        if raw:
            return rows
        cols = {c: [r.get(c) for r in rows] for c in ("name", "version_id", "step", "kind", "size", "aliases",
                                                       "metadata", "created", "sha256", "filename")}
        return _frame(cols, self._df or _default_backend())

    def _resolve_file(self, name: str, alias: Optional[str], step: Optional[int], version: Optional[int]) -> dict:
        rows = [r for r in self._src.list_files(self.id) if r["name"] == name]
        if not rows:
            raise KeyError(f"run {self.id} has no file named {name!r}")
        if version is not None:
            hit = [r for r in rows if r["version_id"] == version]
        elif step is not None:
            hit = sorted([r for r in rows if r["step"] == step], key=lambda r: r["created"])[-1:]
        else:
            alias = alias or "latest"
            hit = [r for r in rows if alias in r["aliases"]]
        if not hit:
            raise KeyError(f"no version of {name!r} matches (alias={alias}, step={step}, version={version})")
        return hit[0]

    def download(self, name: str, alias: Optional[str] = None, step: Optional[int] = None,
                 version: Optional[int] = None, dest: str = ".") -> str:
        """Download a file version (default: alias "latest"); returns the local path.
        Chunks are cached by hash (``$CHRONIST_CACHE``, default ~/.cache/chronist),
        so repeated or interrupted downloads only fetch what's missing; the
        whole-file sha256 is verified."""
        f = self._resolve_file(name, alias, step, version)
        as_dir = dest.endswith(("/", os.sep))  # check before abspath strips it
        dest = os.path.abspath(os.path.expanduser(dest))
        if os.path.isdir(dest) or as_dir:
            os.makedirs(dest, exist_ok=True)
            dest = os.path.join(dest, f.get("filename") or f["name"].replace("/", "_"))
        return _download(self._src, f, dest)

    def load_object(self, name: str, alias: Optional[str] = None, step: Optional[int] = None,
                    version: Optional[int] = None, **load_kwargs):
        """Inverse of save_object: torch.load or pickle.load of the version."""
        f = self._resolve_file(name, alias, step, version)
        path = _download(self._src, f, os.path.join(cache_dir(), "files", f["sha256"]))
        if f.get("serializer") == "torch":
            import torch
            return torch.load(path, **load_kwargs)
        import pickle
        with open(path, "rb") as fh:
            return pickle.load(fh, **load_kwargs)

    def histograms(self, key: str, max_steps: Optional[int] = None, start: Optional[int] = None,
                   end: Optional[int] = None):
        """HistList of records (``.to_dense(rows)``)."""
        from .hist import HistList
        return HistList(self._src.hist_records(self.id, key, start, end, max_steps))

    def histogram(self, key: str, step: Optional[int] = None):
        """The record at ``step`` (or the nearest one before it); the last one if step is None."""
        recs = self._src.hist_records(self.id, key, at=2 ** 62 if step is None else int(step))
        return recs[0] if recs else None


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


def cache_dir() -> str:
    return os.environ.get("CHRONIST_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "chronist")


def _sha256_file(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(buf)
    return h.hexdigest()


def _download(src, f: dict, dest: str) -> str:
    """Assemble a file version at ``dest`` via the chunk cache; verify sha256."""
    import hashlib

    from .blobs import FsBlobStore
    from .codec import decompress
    if os.path.exists(dest) and os.path.getsize(dest) == f["size"] and _sha256_file(dest) == f["sha256"]:
        return dest  # already here
    info = src.file_info(f["version_id"])
    cache = FsBlobStore(os.path.join(cache_dir(), "blobs"))
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = dest + ".part"
    whole = hashlib.sha256()
    with open(tmp, "wb") as out:
        for i, h in enumerate(info["chunk_hashes"]):
            data = cache.read(h)
            if data is None:
                stored = src.file_chunk(f["version_id"], i, h)
                data = decompress(stored)
                if hashlib.sha256(data).hexdigest() != h:
                    raise IOError(f"chunk {i} of {f['name']!r} failed verification")
                cache.put(h, stored, verify=False)
            whole.update(data)
            out.write(data)
    if whole.hexdigest() != f["sha256"]:
        os.remove(tmp)
        raise IOError(f"{f['name']!r}: sha256 mismatch after download")
    os.replace(tmp, dest)
    return dest
