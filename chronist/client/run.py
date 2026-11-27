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


def _scalar(v: Any) -> Optional[float]:
    if isinstance(v, (int, float)):
        return float(v)
    if hasattr(v, "item"):  # numpy / torch / jax scalars
        try:
            v = v.item()
        except (ValueError, RuntimeError):
            raise TypeError("only scalars can be logged (histograms are not supported yet)")
        return float(v)
    if v is None:
        return math.nan
    return float(v)


def _flatten_metrics(d: dict, prefix: str = "", out: Optional[dict] = None) -> dict:
    out = {} if out is None else out
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            _flatten_metrics(v, key + "/", out)
        else:
            out[key] = _scalar(v)
    return out


# ---------------------------------------------------------------- Run
class Run:
    def __init__(self, rundir: RunDir, info: dict, url: Optional[str], token: Optional[str],
                 resume: bool, start_step: int):
        self.dir = rundir
        self.uid: str = info["uid"]
        self.project: str = info["project"]
        self.name: Optional[str] = info.get("name")
        self.config: dict = info.get("config", {})
        self.url = url
        self._writer = RunWriter(rundir)
        if resume:
            self._writer.append(["r", time.time()])
        self._step = start_step
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
        """Log scalars. Never blocks on the network.

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
            values = _flatten_metrics(data)
            if values:
                self._writer.append(["l", s, now, values])
            self._step = max(self._step, s + 1)

    def finish(self, status: str = "finished", timeout: Optional[float] = None) -> None:
        """Mark the run done and upload what's left (waits at most `timeout`
        seconds, default $CHRONIST_FINISH_TIMEOUT or 60)."""
        global _active
        if self._finished:
            return
        self._finished = True
        self._writer.append(["f", status, time.time()])
        self._writer.close()
        if self._uploader is not None:
            if timeout is None:
                timeout = float(os.environ.get("CHRONIST_FINISH_TIMEOUT", "60"))
            ok = self._uploader.stop(timeout)
            if not ok:
                _logger.warning("chronist: could not upload everything for run %s; it is buffered in %s. "
                            "Upload later with: chronist sync %s", self.uid, self.dir.path, self.dir.path)
        if _active is self:
            _active = None
        try:
            atexit.unregister(self._atexit)
        except Exception:  # pragma: no cover
            pass

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
         offline: bool = False, capture: bool = True) -> Run:
    """Start (or resume) a run.

    project: defaults to $CHRONIST_PROJECT or "default".
    resume: uid of a run to re-attach to and continue logging.
    url/token: server; default $CHRONIST_URL / $CHRONIST_TOKEN. No URL (or
        offline=True) -> only local files; upload later with `chronist sync`.
    dir: local buffer root; default $CHRONIST_DIR or ~/.chronist.
    """
    global _active
    url = None if offline else (url or os.environ.get("CHRONIST_URL") or None)
    token = token or os.environ.get("CHRONIST_TOKEN") or None
    root = os.path.join(dir or default_dir(), "runs")
    project = project or os.environ.get("CHRONIST_PROJECT") or "default"
    cfg = _config_to_dict(config)

    if resume:
        rd = RunDir(os.path.join(root, resume))
        if rd.exists():
            info = rd.info()
            if cfg:
                info["config"] = {**info.get("config", {}), **cfg}
                write_json_atomic(rd.info_path, info)
            last = rd.last_step()
            start = 0 if last is None else last + 1
        else:  # run created elsewhere: attach by uid
            os.makedirs(rd.path, exist_ok=True)
            info = {"uid": resume, "project": project, "name": name, "config": cfg,
                    "tags": list(tags), "notes": notes, "created": time.time(),
                    "meta": capture_meta(rd.path) if capture else {}}
            write_json_atomic(rd.info_path, info)
            start = _remote_next_step(url, token, resume) if url else 0
        run = Run(rd, info, url, token, resume=True, start_step=start)
    else:
        uid = new_uid()
        rd = RunDir(os.path.join(root, uid))
        os.makedirs(rd.path, exist_ok=True)
        info = {"uid": uid, "project": project, "name": name, "config": cfg, "tags": list(tags),
                "notes": notes, "created": time.time(),
                "meta": capture_meta(rd.path) if capture else {}}
        write_json_atomic(rd.info_path, info)
        run = Run(rd, info, url, token, resume=False, start_step=0)
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
