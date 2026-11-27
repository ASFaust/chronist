"""`chronist sync`: upload local run buffers. Same path as live upload."""
from __future__ import annotations

import logging
import os
import socket
import time
from typing import Iterable, List, Optional

from .rundir import RunDir, default_dir
from .uploader import ServerGone, Uploader, pid_alive

log = logging.getLogger("chronist")


def find_runs(paths: Iterable[str]) -> List[RunDir]:
    """Run dirs from paths that are run dirs, a `runs/` folder, or a CHRONIST_DIR root."""
    out = []
    for p in paths:
        p = os.path.abspath(os.path.expanduser(p))
        if os.path.exists(os.path.join(p, "run.json")):
            out.append(RunDir(p))
            continue
        base = os.path.join(p, "runs") if os.path.isdir(os.path.join(p, "runs")) else p
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            rd = RunDir(os.path.join(base, name))
            if rd.exists():
                out.append(rd)
    return out


def sync_run(rd: RunDir, url: str, token: Optional[str] = None, mark_dead_crashed: bool = True) -> dict:
    """Upload one run dir. Returns a small status dict."""
    lock = rd.try_lock()
    if lock is None:
        return {"uid": os.path.basename(rd.path), "skipped": "uploading elsewhere (live run)"}
    up = Uploader(rd, url, token)
    try:
        if up.prog.get("gone"):
            return {"uid": rd.info()["uid"], "skipped": "deleted on server"}
        sent = up.upload_pending()
        info = rd.info()
        status = None
        if mark_dead_crashed and rd.finished() is None:
            meta = info.get("meta", {})
            same_host = meta.get("hostname") == socket.gethostname()
            if same_host and meta.get("pid") and not pid_alive(meta["pid"]) and not up.prog.get("crash_sent"):
                # process is gone without finishing
                up._post(f"/api/runs/{up.run_id}/finish", {"status": "crashed", "when": time.time()})
                up.prog["crash_sent"] = True
                up._save()
                status = "crashed"
        return {"uid": info["uid"], "run_id": up.run_id, "points": sent, "status": status}
    except ServerGone:
        up.prog["gone"] = True
        up._save()
        return {"uid": os.path.basename(rd.path), "skipped": "deleted on server"}
    finally:
        up.close()
        lock.close()


def sync(paths: Optional[Iterable[str]] = None, url: Optional[str] = None, token: Optional[str] = None,
         verbose: bool = True) -> List[dict]:
    url = url or os.environ.get("CHRONIST_URL")
    if not url:
        raise SystemExit("no server: pass --url or set CHRONIST_URL")
    token = token or os.environ.get("CHRONIST_TOKEN")
    results = []
    for rd in find_runs(paths or [default_dir()]):
        try:
            res = sync_run(rd, url, token)
        except Exception as e:  # keep going with other runs
            res = {"uid": os.path.basename(rd.path), "error": str(e)}
        results.append(res)
        if verbose:
            if "error" in res:
                print(f"  {res['uid']}: ERROR {res['error']}")
            elif "skipped" in res:
                print(f"  {res['uid']}: skipped ({res['skipped']})")
            elif res["points"] or res.get("status"):
                print(f"  {res['uid']}: run {res['run_id']}, {res['points']} points"
                      + (f", marked {res['status']}" if res.get("status") else ""))
    return results
