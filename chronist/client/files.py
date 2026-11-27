"""Files attached to runs (checkpoints, plots, outputs): capture + chunking.

save_file() captures content synchronously into the run buffer
(``<run>/files/<local_id>/data`` + ``meta.json``), so later overwrites of the
source path don't matter. A background worker then splits it into
content-addressed chunks in the shared local blob store, writes a file
manifest blob, and marks the staged file ``ready``; the uploader sends only
chunks the server lacks. Staged files a dead process never processed are
handled by ``process_staged`` (called by ``chronist sync``).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import queue
import secrets
import shutil
import sys
import threading
import time
from typing import Any, Dict, Iterable, List, Optional

from ..blobs import FsBlobStore
from ..codec import compress_maybe
from .rundir import RunDir, write_json_atomic

log = logging.getLogger("chronist")

CHUNK_SIZE = 8 * 1024 * 1024
KINDS = ("checkpoint", "plot", "data", "output", "other")
MANIFEST_KIND = "chronist.file/1"


def files_dir(rd: RunDir) -> str:
    return os.path.join(rd.path, "files")


def staged(rd: RunDir) -> List[dict]:
    """All staged file entries (meta.json contents + 'dir'), oldest first."""
    d = files_dir(rd)
    if not os.path.isdir(d):
        return []
    out = []
    for lid in os.listdir(d):
        p = os.path.join(d, lid, "meta.json")
        try:
            with open(p) as f:
                m = json.load(f)
        except (OSError, ValueError):
            continue
        m["dir"] = os.path.join(d, lid)
        out.append(m)
    return sorted(out, key=lambda m: m.get("created", 0))


def _check_meta(name: str, kind: str, aliases, metadata, step) -> None:
    if not isinstance(name, str) or not name or "/" in name and name.strip("/") != name:
        raise ValueError(f"bad file name {name!r}")
    if not isinstance(kind, str) or not kind:
        raise ValueError("kind must be a non-empty string")
    for a in aliases:
        if not isinstance(a, str) or not a or a == "latest":
            raise ValueError(f"bad alias {a!r} ('latest' is maintained automatically)")
    if metadata is not None:
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a dict")
        if len(json.dumps(metadata, default=str)) > 64 * 1024:
            raise ValueError("metadata must be small (<= 64 KB as JSON)")
    if step is not None and (isinstance(step, bool) or not isinstance(step, int)):
        raise ValueError("step must be an int")


def stage_file(rd: RunDir, path: str, name: Optional[str] = None, step: Optional[int] = None,
               kind: str = "other", aliases: Iterable[str] = (), metadata: Optional[dict] = None,
               move: bool = False, link: bool = False, serializer: Optional[str] = None) -> dict:
    """Capture ``path`` into the run buffer now. Returns the staged entry."""
    src = os.path.abspath(os.path.expanduser(os.fspath(path)))
    if not os.path.isfile(src):
        raise FileNotFoundError(src)
    name = name or os.path.basename(src)
    aliases = list(aliases)
    _check_meta(name, kind, aliases, metadata, step)
    lid = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(4)
    d = os.path.join(files_dir(rd), lid)
    os.makedirs(d)
    dst = os.path.join(d, "data")
    if move:
        try:
            os.replace(src, dst)  # same filesystem: instant
        except OSError:
            shutil.copyfile(src, dst)
            os.remove(src)
    elif link:
        # Only safe if the caller writes new files instead of modifying this one in place.
        try:
            os.link(src, dst)
        except OSError:
            shutil.copyfile(src, dst)
    else:
        shutil.copyfile(src, dst)  # sendfile/copy_file_range on Linux
    meta = {"local_id": lid, "name": name, "step": step, "kind": kind, "aliases": aliases,
            "metadata": metadata or {}, "created": time.time(), "filename": os.path.basename(src),
            "source": src, "size": os.path.getsize(dst), "state": "staged", "serializer": serializer}
    write_json_atomic(os.path.join(d, "meta.json"), meta)
    return {**meta, "dir": d}


def stage_object(rd: RunDir, obj: Any, name: str, step: Optional[int] = None, kind: str = "checkpoint",
                 aliases: Iterable[str] = (), metadata: Optional[dict] = None) -> dict:
    """Serialize straight into the run buffer (torch.save if torch is loaded, else pickle)."""
    aliases = list(aliases)
    _check_meta(name, kind, aliases, metadata, step)
    lid = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(4)
    d = os.path.join(files_dir(rd), lid)
    os.makedirs(d)
    dst = os.path.join(d, "data")
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.save(obj, dst)
        ser = "torch"
    else:
        with open(dst, "wb") as f:
            pickle.dump(obj, f, protocol=4)
        ser = "pickle"
    meta = {"local_id": lid, "name": name, "step": step, "kind": kind, "aliases": aliases,
            "metadata": metadata or {}, "created": time.time(), "filename": f"{name}.{'pt' if ser == 'torch' else 'pkl'}",
            "source": None, "size": os.path.getsize(dst), "state": "staged", "serializer": ser}
    write_json_atomic(os.path.join(d, "meta.json"), meta)
    return {**meta, "dir": d}


def process_staged_entry(entry: dict, blobs: FsBlobStore, chunk_size: Optional[int] = None) -> dict:
    """Chunk + hash a staged file into the blob store and write its manifest."""
    chunk_size = chunk_size or CHUNK_SIZE
    d = entry["dir"]
    data = os.path.join(d, "data")
    chunks: List[str] = []
    whole = hashlib.sha256()
    size = 0
    with open(data, "rb") as f:
        while True:
            buf = f.read(chunk_size)
            if not buf:
                break
            whole.update(buf)
            size += len(buf)
            h = hashlib.sha256(buf).hexdigest()
            if not blobs.has(h):
                blobs.put(h, compress_maybe(buf), verify=False)
            chunks.append(h)
    manifest = {"kind": MANIFEST_KIND, "name": entry["name"], "step": entry["step"], "file_kind": entry["kind"],
                "size": size, "sha256": whole.hexdigest(), "chunk_size": chunk_size, "chunks": chunks,
                "aliases": entry["aliases"], "metadata": entry["metadata"], "created": entry["created"],
                "filename": entry["filename"], "serializer": entry.get("serializer")}
    mhash = blobs.put_raw(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())
    meta = {k: v for k, v in entry.items() if k != "dir"}
    meta.update(state="ready", manifest=mhash, sha256=manifest["sha256"], size=size)
    write_json_atomic(os.path.join(d, "meta.json"), meta)
    os.remove(data)  # content now lives in the (deduplicating) blob store
    return {**meta, "dir": d}


def process_staged(rd: RunDir) -> int:
    """Chunk every staged-but-unprocessed file of a run dir (e.g. the process died)."""
    blobs = FsBlobStore(rd.blobs_dir)
    n = 0
    for e in staged(rd):
        if e.get("state") == "staged" and os.path.exists(os.path.join(e["dir"], "data")):
            process_staged_entry(e, blobs)
            n += 1
    return n


def file_manifest_hashes(rd: RunDir) -> List[str]:
    return [e["manifest"] for e in staged(rd) if e.get("manifest")]


class FileWorker(threading.Thread):
    """Background chunking so save_file returns as soon as the copy is done."""

    def __init__(self, rd: RunDir):
        super().__init__(name="chronist-files", daemon=True)
        self.rd = rd
        self.q: "queue.Queue[Optional[dict]]" = queue.Queue()
        self.blobs = FsBlobStore(rd.blobs_dir)
        self.on_ready = None  # callback, e.g. wake the uploader
        self.start()

    def submit(self, entry: dict) -> None:
        self.q.put(entry)

    def run(self):
        while True:
            e = self.q.get()
            if e is None:
                return
            try:
                process_staged_entry(e, self.blobs)
                if self.on_ready:
                    self.on_ready()
            except Exception as ex:  # keep the staged copy; sync retries
                log.warning("chronist: could not process file %s: %s", e.get("name"), ex)
            finally:
                self.q.task_done()

    def drain(self):
        self.q.join()

    def stop(self):
        self.q.put(None)
        self.join(60)
