"""The logging client: ``chronist.init()`` -> Run."""
from __future__ import annotations

import atexit
import dataclasses
import hashlib
import logging
import math
import os
import platform
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Dict, Iterable, Optional

from ..blobs import FsBlobStore
from ..hist import DEFAULT_BINS, HistConfig, Histogram
from ..hist import compute as compute_hist
from ..hist import numel as hist_numel
from .code import CodeError, code_opts_from_env_and_config, outside_imports, take_snapshot
from .rundir import RunDir, RunWriter, default_dir, new_uid, write_json_atomic
from .uploader import BackgroundUploader

_logger = logging.getLogger("chronist")

_active: Optional["Run"] = None
_crashed = False
_orig_excepthook = sys.excepthook


def _excepthook(tp, val, tb):
    global _crashed
    if not issubclass(tp, KeyboardInterrupt):
        _crashed = True
    _orig_excepthook(tp, val, tb)


sys.excepthook = _excepthook


# ---------------------------------------------------------------- metadata capture
def _git(*args: str, cwd: Optional[str] = None) -> Optional[str]:
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.decode(errors="replace")


def capture_meta(rundir: Optional[str] = None) -> dict:
    meta: Dict[str, Any] = {
        "command": [sys.executable] + sys.argv,
        "cwd": os.getcwd(),
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pid": os.getpid(),
        "start_time": time.time(),
    }
    commit = _git("rev-parse", "HEAD")
    if commit:
        meta["git_commit"] = commit.strip()
        meta["git_branch"] = (_git("rev-parse", "--abbrev-ref", "HEAD") or "").strip()
        remote = _git("config", "--get", "remote.origin.url")
        if remote:
            meta["git_remote"] = remote.strip()
        status = _git("status", "--porcelain", "--untracked-files=no") or ""
        meta["git_dirty"] = bool(status.strip())
        if meta["git_dirty"]:
            diff = _git("diff", "HEAD") or ""
            meta["git_diff_hash"] = hashlib.sha256(diff.encode()).hexdigest()[:16]
            if rundir:
                with open(os.path.join(rundir, "diff.patch"), "w") as f:
                    f.write(diff)
    return meta


def _config_to_dict(config: Any) -> dict:
    if config is None:
        return {}
    if isinstance(config, dict):
        return config
    if dataclasses.is_dataclass(config):
        return dataclasses.asdict(config)
    if hasattr(config, "__dict__"):  # argparse.Namespace & co.
        return dict(vars(config))
    try:  # omegaconf / hydra
        from omegaconf import OmegaConf  # type: ignore
        return OmegaConf.to_container(config, resolve=True)
    except Exception:
        raise TypeError(f"cannot use {type(config).__name__} as config; pass a dict")


def _scalar(v: Any) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    if v is None:
        return math.nan
    if hasattr(v, "item"):  # numpy / torch / jax scalars and 1-element arrays
        return float(v.item())
    return float(v)


def _split_values(d: dict, auto_hist: bool, prefix: str = "", scalars=None, hists=None):
    """Flatten nested dicts with '/' and sort values into scalars and histograms."""
    scalars = {} if scalars is None else scalars
    hists = {} if hists is None else hists
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            _split_values(v, auto_hist, key + "/", scalars, hists)
        elif isinstance(v, Histogram):
            hists[key] = v
        else:
            n = hist_numel(v)
            if n is None or n == 1:
                scalars[key] = _scalar(v) if n is None else _scalar(_first(v))
            elif not auto_hist:
                raise TypeError(f"{key!r}: got {n} values; log a scalar or pass chronist.Histogram(...) "
                                f"(auto_histograms is off)")
            elif n == 0:
                continue  # empty vector: nothing to record
            else:
                hists[key] = Histogram(v)  # default settings for this key
    return scalars, hists


def _first(v):
    if isinstance(v, (list, tuple)):
        return v[0] if not isinstance(v[0], (list, tuple)) else _first(v[0])
    return v.reshape(-1)[0] if hasattr(v, "reshape") else v


# ---------------------------------------------------------------- Run
class Run:
    def __init__(self, rundir: RunDir, info: dict, url: Optional[str], token: Optional[str],
                 resume: bool, start_step: int, rewind: Optional[int] = None,
                 histogram_bins: int = DEFAULT_BINS, auto_histograms: bool = True):
        self._hist_cfg = HistConfig(histogram_bins)
        self._files = None
        self._file_policies: Dict[str, dict] = {}
        self._auto_hist = auto_histograms
        self.dir = rundir
        self._info = info
        self.uid: str = info["uid"]
        self.project: str = info["project"]
        self.name: Optional[str] = info.get("name")
        self.config: dict = info.get("config", {})
        self.url = url
        self._writer = RunWriter(rundir)
        if resume:
            self._writer.append(["r", time.time()])
        if rewind is not None:
            self._writer.append(["w", int(rewind)])
        self._step = start_step
        # On a plain resume, steps below this were already logged and new
        # values there would be dropped (first write wins): warn once.
        self._floor = start_step if resume and rewind is None else None
        self._warned_floor = False
        self._last_time: Dict[str, float] = {}
        self._finished = False
        self._lock = threading.Lock()
        self._uploader: Optional[BackgroundUploader] = None
        if url:
            self._uploader = BackgroundUploader(rundir, self._writer, url, token, resume=resume)
            if self._uploader.lock_handle is None:
                _logger.warning("chronist: another process is uploading run %s; this one only writes locally", self.uid)
            self._uploader.start()
        else:
            self._flusher = threading.Thread(target=self._flush_loop, daemon=True, name="chronist-flush")
            self._flusher.start()
        atexit.register(self._atexit)

    def __repr__(self):
        return f"<chronist.Run {self.project}/{self.uid} {'online' if self.url else 'offline'}>"

    @property
    def id(self) -> Optional[int]:
        """Server-side run id (None while offline or not yet created)."""
        return self._uploader.up.run_id if self._uploader else None

    @property
    def step(self) -> int:
        """The step the next ``log()`` without explicit step will use."""
        return self._step

    def _flush_loop(self):
        while not self._finished:
            time.sleep(2.0)
            self._writer.flush()

    def log(self, data: Dict[str, Any], step: Optional[int] = None, *,
            every: Optional[int] = None, interval: Optional[float] = None) -> None:
        """Log scalars and histograms. Never blocks on the network.

        Values with more than one element (lists, numpy arrays, torch/jax
        tensors) become histograms, as does chronist.Histogram(x, ...).

        step: explicit step (int). Default: one past the previous step.
        every: only log when ``step % every == 0``.
        interval: only log if at least this many seconds passed since the
            last logged call with the same set of keys.
        Nested dicts are flattened with '/': {"train": {"loss": x}} -> "train/loss".
        """
        if self._finished:
            raise RuntimeError("run is finished")
        with self._lock:
            s = self._step if step is None else int(step)
            if every is not None and s % every != 0:
                if step is None:
                    self._step += 1
                return
            now = time.time()
            if interval is not None:
                sig = ",".join(sorted(data))
                if now - self._last_time.get(sig, -math.inf) < interval:
                    if step is None:
                        self._step += 1
                    return
                self._last_time[sig] = now
            values, hists = _split_values(data, self._auto_hist)
            if self._floor is not None and s < self._floor and not self._warned_floor:
                self._warned_floor = True
                _logger.warning(
                    "chronist: logging step %d, but this resumed run already has data up to step %d. "
                    "Values at already-logged steps are ignored (first write wins). To resume from an "
                    "earlier checkpoint, use chronist.init(resume=%r, rewind=%d).",
                    s, self._floor - 1, self.uid, s)
            if values:
                self._writer.append(["l", s, now, values])
            if hists:
                recs = {}
                for key, h in hists.items():
                    o = self._hist_cfg.resolve(key, h)
                    recs[key] = compute_hist(h.values, s, bins=o["bins"], range=o["range"], raw=o["raw"],
                                             wall_time=now).to_wire()
                self._writer.append(["h", s, now, recs])
            self._step = max(self._step, s + 1)

    # ---- files ----
    def _file_worker(self):
        if self._files is None:
            from .files import FileWorker
            self._files = FileWorker(self.dir)
            if self._uploader is not None:
                self._files.on_ready = self._uploader.wake
        return self._files

    def save_file(self, path, name: Optional[str] = None, step: Optional[int] = None, kind: str = "other",
                  aliases: Iterable[str] = (), metadata: Optional[dict] = None, move: bool = False,
                  link: bool = False) -> dict:
        """Attach a file (checkpoint, plot, output). The content is captured now:
        overwriting ``path`` afterwards does not change this version. Chunking,
        hashing and upload happen in the background.

        name: logical name grouping versions (default: the file's basename)
        aliases: e.g. ["best"]; "latest" is maintained automatically
        move: move instead of copy; link: hard-link (only if you never modify
        the source file in place afterwards)."""
        from .files import stage_file
        if self._finished:
            raise RuntimeError("run is finished")
        e = stage_file(self.dir, path, name, step, kind, aliases, metadata, move=move, link=link)
        self._attach_policy(e)
        self._file_worker().submit(e)
        return {k: e[k] for k in ("local_id", "name", "step", "kind", "size")}

    def save_object(self, obj: Any, name: str, step: Optional[int] = None, kind: str = "checkpoint",
                    aliases: Iterable[str] = (), metadata: Optional[dict] = None) -> dict:
        """Serialize ``obj`` (torch.save if torch is imported, else pickle) as a file version."""
        from .files import stage_object
        if self._finished:
            raise RuntimeError("run is finished")
        e = stage_object(self.dir, obj, name, step, kind, aliases, metadata)
        self._attach_policy(e)
        self._file_worker().submit(e)
        return {k: e[k] for k in ("local_id", "name", "step", "kind", "size")}

    def configure_files(self, name: str, keep_last: Optional[int] = None, keep_every: Optional[int] = None) -> None:
        """Retention for versions of ``name`` (applied on the server): keep the
        newest ``keep_last`` versions plus the first version in every
        ``keep_every``-step window. Versions with an alias are always kept."""
        for v in (keep_last, keep_every):
            if v is not None and (isinstance(v, bool) or not isinstance(v, int) or v < 1):
                raise ValueError("keep_last / keep_every must be positive ints")
        self._file_policies[name] = {"keep_last": keep_last, "keep_every": keep_every}

    def _attach_policy(self, e: dict) -> None:
        pol = self._file_policies.get(e["name"])
        if pol:
            from .rundir import write_json_atomic
            meta = {k: v for k, v in e.items() if k != "dir"}
            meta["retention"] = pol
            write_json_atomic(os.path.join(e["dir"], "meta.json"), meta)
            e["retention"] = pol

    def configure_histograms(self, pattern: str, bins: Optional[int] = None,
                             range: Optional[tuple] = None, raw: Optional[bool] = None) -> None:
        """Defaults for histogram keys matching a glob (``"grad/*"``). Precedence:
        chronist.Histogram(...) arguments > the last matching pattern > init(histogram_bins=) > 64."""
        self._hist_cfg.configure(pattern, bins=bins, range=range, raw=raw)

    def finish(self, status: str = "finished", timeout: Optional[float] = None) -> None:
        """Mark the run done and upload what's left (waits at most `timeout`
        seconds, default $CHRONIST_FINISH_TIMEOUT or 60)."""
        global _active
        if self._finished:
            return
        self._finished = True
        if self._files is not None:
            self._files.drain()  # every saved file chunked before the final upload
            self._files.stop()
        self._recheck_imports()
        self._writer.append(["f", status, time.time()])
        self._writer.close()
        if self._uploader is not None:
            if timeout is None:
                timeout = float(os.environ.get("CHRONIST_FINISH_TIMEOUT", "60"))
            ok = self._uploader.stop(timeout)
            if ok:
                from .prune import auto_prune
                try:
                    if auto_prune(self.dir, self.url.rstrip("/")):
                        _logger.info("chronist: run %s uploaded; local buffer removed", self.uid)
                except Exception as e:  # pragma: no cover - cleanup must never fail a run
                    _logger.debug("chronist: prune failed: %s", e)
            if not ok:
                _logger.warning("chronist: could not upload everything for run %s; it is buffered in %s. "
                            "Upload later with: chronist sync %s", self.uid, self.dir.path, self.dir.path)
        if _active is self:
            _active = None
        try:
            atexit.unregister(self._atexit)
        except Exception:  # pragma: no cover
            pass

    def _recheck_imports(self):
        """Imports can happen after init: re-run the outside-import check."""
        snap = self._info.get("meta", {}).get("code") or {}
        if not snap.get("root") or "manifest" not in snap:
            return
        try:
            now = outside_imports(snap["root"])
        except Exception:  # pragma: no cover
            return
        if now != snap.get("outside_imports"):
            new = [m for m in now if m not in (snap.get("outside_imports") or [])]
            for imp in new[:5]:
                _logger.warning("chronist: imported local module %r from %s, outside code root %s",
                                imp["module"], os.path.dirname(imp["file"]), snap["root"])
            self._writer.append(["m", {"code": {"outside_imports": now}}])

    def _atexit(self):
        if not self._finished:
            self.finish("crashed" if _crashed else "finished")

    def __enter__(self):
        return self

    def __exit__(self, tp, val, tb):
        self.finish("crashed" if tp is not None and not issubclass(tp, KeyboardInterrupt) else "finished")
        return False


def init(project: Optional[str] = None, name: Optional[str] = None, config: Any = None,
         tags: Iterable[str] = (), notes: str = "", *, resume: Optional[str] = None,
         url: Optional[str] = None, token: Optional[str] = None, dir: Optional[str] = None,
         offline: bool = False, capture: bool = True, code: Any = "auto",
         rewind: Optional[int] = None, histogram_bins: int = DEFAULT_BINS,
         auto_histograms: bool = True) -> Run:
    """Start (or resume) a run.

    code: code snapshot. "auto" (default) | "git" | a path | None/False (off)
        | chronist.Code(...). Relative paths resolve against the entry
        script's directory. See docs/CODE_SNAPSHOTS.md.
    project: defaults to $CHRONIST_PROJECT or "default".
    resume: uid of a run to re-attach to and continue logging.
    histogram_bins: default bins for histograms (64); auto_histograms=False
        makes non-scalar values raise TypeError instead of becoming histograms.
    rewind: with resume, discard everything the run logged at step >= rewind
        (e.g. resuming from an earlier checkpoint); logging continues there.
    url/token: server; default $CHRONIST_URL / $CHRONIST_TOKEN. No URL (or
        offline=True) -> only local files; upload later with `chronist sync`.
    dir: local buffer root; default $CHRONIST_DIR or ~/.chronist.
    """
    global _active
    url = None if offline else (url or os.environ.get("CHRONIST_URL") or None)
    token = token or os.environ.get("CHRONIST_TOKEN") or None
    base = os.path.abspath(os.path.expanduser(dir or default_dir()))  # absolute once; chdir-proof
    root = os.path.join(base, "runs")
    project = project or os.environ.get("CHRONIST_PROJECT") or "default"
    cfg = _config_to_dict(config)
    code_opts = code_opts_from_env_and_config(code)  # raises ValueError on bad input
    if rewind is not None:
        if not resume:
            raise ValueError("rewind= needs resume=<run uid>")
        if isinstance(rewind, bool) or not isinstance(rewind, int) or rewind < 0:
            raise ValueError("rewind must be a non-negative integer step")

    def snapshot() -> Optional[dict]:
        if code_opts is None:
            return None
        try:
            return take_snapshot(code_opts, FsBlobStore(os.path.join(base, "blobs")))
        except CodeError:
            raise
        except Exception as e:  # never kill a run over a snapshot problem
            _logger.warning("chronist: code snapshot failed: %s", e)
            return {"aborted": str(e)}

    if resume:
        rd = RunDir(os.path.join(root, resume))
        if rd.exists():
            info = rd.info()
            if cfg:
                info["config"] = {**info.get("config", {}), **cfg}
                write_json_atomic(rd.info_path, info)
            last = rd.last_step()
            start = 0 if last is None else last + 1
            if rewind is not None:
                start = min(start, rewind)
        else:  # run created elsewhere: attach by uid
            snap = snapshot()
            os.makedirs(rd.path, exist_ok=True)
            info = {"uid": resume, "project": project, "name": name, "config": cfg,
                    "tags": list(tags), "notes": notes, "created": time.time(),
                    "meta": capture_meta(rd.path) if capture else {}}
            if snap:
                info["meta"]["code"] = snap
            write_json_atomic(rd.info_path, info)
            start = _remote_next_step(url, token, resume) if url else 0
            if rewind is not None:
                start = min(start, rewind)
        run = Run(rd, info, url, token, resume=True, start_step=start, rewind=rewind,
                  histogram_bins=histogram_bins, auto_histograms=auto_histograms)
    else:
        snap = snapshot()
        uid = new_uid()
        rd = RunDir(os.path.join(root, uid))
        os.makedirs(rd.path, exist_ok=True)
        info = {"uid": uid, "project": project, "name": name, "config": cfg, "tags": list(tags),
                "notes": notes, "created": time.time(),
                "meta": capture_meta(rd.path) if capture else {}}
        if snap:
            info["meta"]["code"] = snap
        write_json_atomic(rd.info_path, info)
        run = Run(rd, info, url, token, resume=False, start_step=0,
                  histogram_bins=histogram_bins, auto_histograms=auto_histograms)
    _active = run
    return run


def _remote_next_step(url: str, token: Optional[str], uid: str) -> int:
    import httpx
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        r = httpx.get(f"{url.rstrip('/')}/api/runs/by-uid/{uid}", headers=headers, timeout=10)
        if r.status_code != 200:
            return 0
        rid = r.json()["id"]
        run = httpx.get(f"{url.rstrip('/')}/api/runs/{rid}", headers=headers, timeout=10).json()
        return 0 if run.get("steps") is None else run["steps"] + 1
    except Exception:
        return 0


def active() -> Optional[Run]:
    return _active


def log(data: Dict[str, Any], step: Optional[int] = None, **kw) -> None:
    """Log to the active run (the last one from ``init``)."""
    if _active is None:
        raise RuntimeError("call chronist.init() first")
    _active.log(data, step, **kw)


def finish(status: str = "finished") -> None:
    if _active is not None:
        _active.finish(status)
