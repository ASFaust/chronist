"""Uploads a RunDir to a server. Used both live (background thread) and by
``chronist sync`` — the exact same code path."""
from __future__ import annotations

import gzip
import logging
import os
import socket
import threading
import time
from typing import Dict, List, Optional

import msgpack

from .rundir import RunDir

log = logging.getLogger("chronist")

MAX_POINTS_PER_REQUEST = 200_000


class ServerGone(Exception):
    """The run was deleted on the server; stop uploading it."""


class Uploader:
    def __init__(self, rundir: RunDir, url: str, token: Optional[str] = None, timeout: float = 30.0):
        import httpx
        self.rd = rundir
        self.url = url.rstrip("/")
        headers = {"User-Agent": "chronist-client"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.http = httpx.Client(base_url=self.url, headers=headers, timeout=timeout)
        self.prog = rundir.progress(self.url)
        self.run_id: Optional[int] = self.prog.get("run_id")
        self.last_contact = 0.0

    def close(self):
        self.http.close()

    # ---- protocol ----
    def _post(self, path: str, body: dict, packed: bool = False):
        if packed:
            raw = gzip.compress(msgpack.packb(body, use_bin_type=True), compresslevel=3)
            r = self.http.post(path, content=raw, headers={"Content-Type": "application/msgpack",
                                                           "Content-Encoding": "gzip"})
        else:
            r = self.http.post(path, json=body)
        if r.status_code == 404 and path != "/api/runs":
            raise ServerGone(r.text)
        if r.status_code == 401:
            raise PermissionError(f"{self.url} rejected the login (no or revoked token); "
                                  f"run: chronist login {self.url}")
        r.raise_for_status()
        self.last_contact = time.time()
        return r.json()

    def ensure_run(self, resume: bool = False) -> int:
        if self.run_id is not None and not resume:
            return self.run_id
        info = self.rd.info()
        res = self._post("/api/runs", {k: info.get(k) for k in
                                       ("project", "uid", "name", "config", "tags", "notes", "meta", "created")})
        self.run_id = res["id"]
        if self.prog.get("run_id") != self.run_id:
            self.prog = {"run_id": self.run_id, "seg": self.prog.get("seg", 0), "offset": self.prog.get("offset", 0)}
            self._save()
        return self.run_id

    def _save(self):
        self.rd.save_progress(self.url, self.prog)

    def heartbeat(self):
        if self.run_id is not None:
            self._post(f"/api/runs/{self.run_id}/heartbeat", {})

    def upload_code(self) -> None:
        """Content-addressed code snapshot upload: only blobs the server lacks."""
        if self.prog.get("code_done"):
            return
        snap = (self.rd.info().get("meta") or {}).get("code") or {}
        mhash = snap.get("manifest")
        if not mhash:
            self.prog["code_done"] = True
            self._save()
            return
        import json as _json

        from ..blobs import FsBlobStore
        from .code import manifest_blobs
        store = FsBlobStore(self.rd.blobs_dir)
        raw = store.read(mhash)
        if raw is None:
            log.warning("chronist: code manifest %s missing from %s; skipping code upload", mhash, store.path)
            self.prog["code_done"] = True
            self._save()
            return
        hashes = manifest_blobs(_json.loads(raw)) + [mhash]
        missing = self._post("/api/blobs/missing", {"hashes": hashes})["missing"]
        for h in missing:
            data = store.get(h)
            if data is None:
                raise RuntimeError(f"blob {h} missing locally")
            r = self.http.put(f"/api/blobs/{h}", content=data, headers={"Content-Type": "application/octet-stream"})
            r.raise_for_status()
        self._post(f"/api/runs/{self.run_id}/code", {"manifest": mhash})
        self.prog["code_done"] = True
        self.prog["code_blobs_sent"] = len(missing)
        self._save()

    def upload_files(self, budget_s: Optional[float] = None) -> int:
        """Upload ready files (missing chunks only), oldest first. With a time
        budget, returns early so metric uploads keep priority; progress is
        per chunk on the server, so the next call resumes where this stopped."""
        from ..blobs import FsBlobStore
        from .files import staged
        done = set(self.prog.get("files_done", []))
        todo = [e for e in staged(self.rd) if e.get("state") == "ready" and e["local_id"] not in done]
        if not todo:
            return 0
        store = FsBlobStore(self.rd.blobs_dir)
        t0 = time.time()
        rate = float(os.environ.get("CHRONIST_UPLOAD_MBPS", "0") or 0) * 1e6 / 8  # bytes/s
        sent_bytes, n = 0, 0
        for e in todo:
            raw = store.read(e["manifest"])
            if raw is None:
                log.warning("chronist: manifest of file %s missing locally; skipping", e["name"])
                continue
            import json as _json
            chunks = _json.loads(raw)["chunks"]
            r = self.http.post("/api/blobs/missing", json={"hashes": list(dict.fromkeys(chunks + [e["manifest"]])),
                                                           "purpose": "file", "size": e["size"]})
            if r.status_code == 507 or r.status_code == 413:
                msg = r.json().get("error", r.text)
                if not self.prog.get("file_refused_warned"):
                    log.warning("chronist: server refused file %s: %s (metrics continue)", e["name"], msg)
                    self.prog["file_refused_warned"] = True
                continue
            r.raise_for_status()
            for h in r.json()["missing"]:
                data = store.get(h)
                if data is None:
                    raise RuntimeError(f"chunk {h} of {e['name']} missing locally")
                t = time.time()
                self.http.put(f"/api/blobs/{h}", content=data,
                              headers={"Content-Type": "application/octet-stream"}).raise_for_status()
                sent_bytes += len(data)
                if rate:
                    wait = sent_bytes / rate - (time.time() - t0)
                    if wait > 0:
                        time.sleep(wait)
                if budget_s is not None and time.time() - t0 > budget_s:
                    return n  # resume next round, after metrics
            res = self._post(f"/api/runs/{self.run_id}/files",
                             {"manifest": e["manifest"], "local_id": e["local_id"],
                              "retention": e.get("retention")})
            self.prog.setdefault("files_done", []).append(e["local_id"])
            self.prog.setdefault("file_versions", {})[e["local_id"]] = res.get("version_id")
            self._save()
            self.last_contact = time.time()
            n += 1
            if budget_s is not None and time.time() - t0 > budget_s:
                break
        return n

    def upload_pending(self) -> int:
        """Upload everything currently on disk. Returns number of points sent."""
        if self.prog.get("gone"):
            return 0
        rid = self.ensure_run()
        self.upload_code()
        sent = 0
        while True:
            seg, off = self.prog.get("seg", 0), self.prog.get("offset", 0)
            recs, ends, nseg, noff = self.rd.read_records(seg, off)
            if not recs:
                if (nseg, noff) != (seg, off):
                    self.prog.update(seg=nseg, offset=noff)
                    self._save()
                return sent
            batch: Dict[str, Dict[str, list]] = {}
            hists: list = []
            n = 0

            def flush_batch():
                nonlocal batch, hists, n, sent
                if not batch and not hists:
                    return
                body = {"points": batch}
                if hists:
                    body["hists"] = hists
                self._post(f"/api/runs/{rid}/log", body, packed=True)
                sent += n
                batch, hists, n = {}, [], 0

            def commit(i):
                # progress = right after record i (records come from segment nseg)
                self.prog.update(seg=nseg, offset=ends[i])
                self._save()

            for i, rec in enumerate(recs):
                kind = rec[0]
                if kind == "l":
                    _, step, wall, values = rec
                    for k, v in values.items():
                        col = batch.get(k)
                        if col is None:
                            col = batch[k] = {"step": [], "value": [], "wall": []}
                        col["step"].append(step)
                        col["value"].append(None if v is None or v != v else v)
                        col["wall"].append(wall)
                        n += 1
                    if n >= MAX_POINTS_PER_REQUEST:
                        flush_batch()
                        commit(i)
                elif kind == "h":
                    _, _step, _wall, hrecs = rec
                    for k, r in hrecs.items():
                        hists.append({"key": k, **r})
                        n += 1 + len(r.get("counts", b"")) // 64
                    if n >= MAX_POINTS_PER_REQUEST:
                        flush_batch()
                        commit(i)
                elif kind == "w":
                    flush_batch()
                    self._post(f"/api/runs/{rid}/rewind", {"step": int(rec[1])})
                    commit(i)
                elif kind == "m":
                    flush_batch()
                    r = self.http.patch(f"/api/runs/{rid}", json={"meta": rec[1]})
                    if r.status_code == 404:
                        raise ServerGone(r.text)
                    r.raise_for_status()
                    commit(i)
                elif kind in ("f", "r"):
                    # points before the status change go first
                    flush_batch()
                    if kind == "f":
                        self._post(f"/api/runs/{rid}/finish", {"status": rec[1], "when": rec[2]})
                        self.prog["finish_sent"] = True
                    else:
                        self.ensure_run(resume=True)
                        self.prog["finish_sent"] = False
                    commit(i)
                # unknown record kinds (from newer clients) are skipped
            flush_batch()
            commit(len(recs) - 1)


class BackgroundUploader(threading.Thread):
    """Live upload loop: flush the local buffer, upload, heartbeat, back off on errors."""

    def __init__(self, rundir: RunDir, writer, url: str, token: Optional[str],
                 interval: float = 2.0, heartbeat_every: float = 30.0, resume: bool = False):
        super().__init__(name="chronist-uploader", daemon=True)
        self.rd = rundir
        self.writer = writer
        self.up = Uploader(rundir, url, token)
        self.interval = interval
        self.heartbeat_every = heartbeat_every
        self.resume = resume
        self._stopping = threading.Event()
        self._wake = threading.Event()
        self.lock_handle = rundir.try_lock()
        self.error: Optional[BaseException] = None
        self.idle = threading.Event()
        self._warned = False

    def run(self):
        backoff = self.interval
        created = False
        while True:
            stopping = self._stopping.is_set()
            try:
                self.writer.flush()
                if self.lock_handle is not None:
                    if not created:
                        self.up.ensure_run(resume=self.resume)
                        created = True
                    self.up.upload_pending()
                    self.up.upload_files(budget_s=None if stopping else 5.0)
                    if time.time() - self.up.last_contact > self.heartbeat_every:
                        self.up.heartbeat()
                self.error = None
                backoff = self.interval
                self.idle.set()
                if self._warned:
                    log.warning("chronist: connection to %s restored", self.up.url)
                    self._warned = False
            except ServerGone:
                log.warning("chronist: run was deleted on the server; local logging continues")
                self.up.prog["gone"] = True
                self.up._save()
                self.lock_handle = None
            except Exception as e:  # network down, server error: keep buffering
                self.error = e
                if not self._warned:
                    log.warning("chronist: upload to %s failed (%s); buffering locally and retrying",
                                self.up.url, e)
                    self._warned = True
                backoff = min(backoff * 2, 60.0)
                if stopping:
                    return
            if stopping:
                return
            self._wake.wait(backoff)
            self._wake.clear()

    def wake(self):
        self._wake.set()

    def stop(self, timeout: float):
        """Final upload attempt, waiting at most `timeout` seconds."""
        self._stopping.set()
        self._wake.set()
        self.join(timeout)
        ok = not self.is_alive() and self.error is None
        if not self.is_alive():
            self.up.close()
        if self.lock_handle is not None:
            self.lock_handle.close()  # releases the flock so `chronist sync` can take over
            self.lock_handle = None
        return ok


def host() -> str:
    return socket.gethostname()


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover
        return True
