"""Local cleanup: delete run buffers that are safely on a server, and local
blobs no remaining run needs.

A run dir is prunable only when, for some server URL in its sync.json, the
uploader recorded: code uploaded, the finish record sent, and progress at the
exact end of the last segment. Pruning removes the whole directory; resuming
such a run later works through the server (``init(resume=uid)``).
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import time
from typing import Iterable, List, Optional, Set

from ..blobs import FsBlobStore
from .rundir import RunDir, default_dir

log = logging.getLogger("chronist")

BLOB_GRACE_S = 3600  # init() writes blobs before run.json exists: never touch fresh ones


def keep_local() -> bool:
    return os.environ.get("CHRONIST_KEEP_LOCAL", "").lower() in ("1", "true", "yes")


def _all_progress(rd: RunDir) -> dict:
    p = os.path.join(rd.path, "sync.json")
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)


def uploaded_to(rd: RunDir) -> List[str]:
    """Server URLs that hold everything this run dir contains."""
    if not rd.exists():
        return []
    finished = rd.finished() is not None
    segs = rd.segments()
    if not segs:
        return []
    from .files import staged
    files = staged(rd)
    last, size = segs[-1], os.path.getsize(rd.segment_path(segs[-1]))
    out = []
    for url, prog in _all_progress(rd).items():
        if prog.get("gone"):
            continue
        done = (finished and prog.get("finish_sent")) or prog.get("crash_sent")  # crash_sent: sync marked it
        files_ok = all(e.get("state") == "ready" and e["local_id"] in set(prog.get("files_done", []))
                       for e in files)
        if done and files_ok and prog.get("code_done") and prog.get("seg") == last and prog.get("offset") == size:
            out.append(url)
    return out


def _referenced_blobs(rundirs: Iterable[RunDir], store: FsBlobStore) -> Set[str]:
    from .code import manifest_blobs
    refs: Set[str] = set()
    from .files import file_manifest_hashes
    for rd in rundirs:
        try:
            mh = ((rd.info().get("meta") or {}).get("code") or {}).get("manifest")
        except (OSError, ValueError):
            continue
        if mh:
            refs.add(mh)
            raw = store.read(mh)
            if raw is not None:
                refs.update(manifest_blobs(json.loads(raw)))
        for fh in file_manifest_hashes(rd):  # file chunks
            refs.add(fh)
            raw = store.read(fh)
            if raw is not None:
                refs.update(json.loads(raw).get("chunks", []))
    return refs


def gc_local_blobs(root: str, dry_run: bool = False, ignore_grace: Iterable[str] = ()) -> int:
    """Delete blobs in <root>/blobs that no remaining run dir references.
    ``ignore_grace``: hashes known to be uploaded (a just-pruned run's), which
    may go even if they are newer than the grace period."""
    ignore_grace = set(ignore_grace)
    store = FsBlobStore(os.path.join(root, "blobs"))
    if not os.path.isdir(store.path):
        return 0
    runs_dir = os.path.join(root, "runs")
    rundirs = [RunDir(os.path.join(runs_dir, n)) for n in os.listdir(runs_dir)] if os.path.isdir(runs_dir) else []
    refs = _referenced_blobs(rundirs, store)
    cutoff = time.time() - BLOB_GRACE_S
    n = 0
    for sub in os.listdir(store.path):
        d = os.path.join(store.path, sub)
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            h = sub + name
            p = os.path.join(d, name)
            if h in refs or ".tmp" in name:
                continue
            try:
                if h not in ignore_grace and os.path.getmtime(p) > cutoff:
                    continue
                if not dry_run:
                    os.remove(p)
                n += 1
            except FileNotFoundError:
                pass
    return n


def prune_run(rd: RunDir, url: Optional[str] = None, dry_run: bool = False) -> bool:
    """Delete a run dir if it is fully uploaded (to ``url`` if given)."""
    urls = uploaded_to(rd)
    if not urls or (url is not None and url.rstrip("/") not in urls):
        return False
    lock = rd.try_lock()  # never delete while something is uploading it
    if lock is None:
        return False
    try:
        if not dry_run:
            shutil.rmtree(rd.path, ignore_errors=True)
    finally:
        lock.close()
    return True


def prune(paths: Optional[Iterable[str]] = None, url: Optional[str] = None, older_than_days: float = 0,
          dry_run: bool = False, verbose: bool = True) -> dict:
    from .sync import find_runs
    roots = list(paths or [default_dir()])
    cutoff = time.time() - older_than_days * 86400
    pruned, kept = 0, 0
    for rd in find_runs(roots):
        try:
            created = rd.info().get("created", 0)
        except (OSError, ValueError):
            continue
        if created > cutoff:
            kept += 1
            continue
        if prune_run(rd, url, dry_run):
            pruned += 1
            if verbose:
                print(f"  {'would prune' if dry_run else 'pruned'} {os.path.basename(rd.path)}")
        else:
            kept += 1
    blobs = 0
    for r in roots:
        r = os.path.abspath(os.path.expanduser(r))
        if os.path.isdir(os.path.join(r, "blobs")):
            blobs += gc_local_blobs(r, dry_run)
    return {"pruned": pruned, "kept": kept, "blobs": blobs}


def auto_prune(rd: RunDir, url: str) -> bool:
    """After a successful upload: drop the local copy and its now-unneeded blobs."""
    if keep_local():
        return False
    root = os.path.dirname(os.path.dirname(os.path.abspath(rd.path)))
    store = FsBlobStore(os.path.join(root, "blobs"))
    mine = _referenced_blobs([rd], store)
    if not prune_run(rd, url):
        return False
    try:
        gc_local_blobs(root, ignore_grace=mine)
    except OSError as e:  # pragma: no cover
        log.debug("blob gc failed: %s", e)
    return True
