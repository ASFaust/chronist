"""Local per-run buffer: the on-disk format shared by live logging and sync.

Layout of ``$CHRONIST_DIR/runs/<uid>/``::

    run.json               init info: uid, project, name, config, tags, meta, created
    events-000000.msgpack  append-only msgpack records (rotated at SEGMENT_BYTES)
    sync.json              upload progress, per server URL
    sync.lock              held (flock) by whoever is uploading this run

Records::

    ["l", step, wall_time, {key: value, ...}]   logged scalars
    ["f", status, when]                         finish
    ["r", when]                                 resumed (new process attached)
    ["m", {meta}]                               metadata update (merged into run meta)
    ["w", step]                                 rewind: discard everything at step >= step
"""
from __future__ import annotations

import fcntl
import json
import os
import secrets
import threading
import time
from typing import Iterator, List, Optional, Tuple

import msgpack

SEGMENT_BYTES = 64 * 1024 * 1024


def default_dir() -> str:
    return os.environ.get("CHRONIST_DIR") or os.path.join(os.path.expanduser("~"), ".chronist")


def new_uid() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(4)


def _seg_name(n: int) -> str:
    return f"events-{n:06d}.msgpack"


def write_json_atomic(path: str, data: dict) -> None:
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1, default=str)
    os.replace(tmp, path)


class RunDir:
    def __init__(self, path: str):
        self.path = path

    # ---- info ----
    @property
    def info_path(self) -> str:
        return os.path.join(self.path, "run.json")

    def info(self) -> dict:
        with open(self.info_path) as f:
            return json.load(f)

    @property
    def blobs_dir(self) -> str:
        """Shared content-addressed store: <CHRONIST_DIR>/blobs (runs are <CHRONIST_DIR>/runs/<uid>)."""
        return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(self.path))), "blobs")

    def exists(self) -> bool:
        return os.path.exists(self.info_path)

    def segments(self) -> List[int]:
        out = []
        for name in os.listdir(self.path):
            if name.startswith("events-") and name.endswith(".msgpack"):
                out.append(int(name[7:13]))
        return sorted(out)

    def segment_path(self, n: int) -> str:
        return os.path.join(self.path, _seg_name(n))

    # ---- progress ----
    def progress(self, url: str) -> dict:
        p = os.path.join(self.path, "sync.json")
        if not os.path.exists(p):
            return {}
        with open(p) as f:
            return json.load(f).get(url, {})

    def save_progress(self, url: str, prog: dict) -> None:
        p = os.path.join(self.path, "sync.json")
        allp = {}
        if os.path.exists(p):
            with open(p) as f:
                allp = json.load(f)
        allp[url] = prog
        write_json_atomic(p, allp)

    def try_lock(self):
        """Exclusive upload lock; returns a file handle (keep it) or None."""
        f = open(os.path.join(self.path, "sync.lock"), "a")
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return f
        except OSError:
            f.close()
            return None

    # ---- reading ----
    def read_records(self, seg: int, offset: int, max_bytes: int = 8 * 1024 * 1024
                     ) -> Tuple[List[list], List[int], int, int]:
        """Complete records from (seg, offset). Returns (records, ends, seg,
        offset): ``ends[i]`` is the byte offset right after record i inside
        segment ``seg`` (the segment the records come from), and (seg, offset)
        is the position after the last complete record. Moves to the next
        segment when the current one is exhausted and a newer one exists."""
        while True:
            path = self.segment_path(seg)
            if not os.path.exists(path):
                return [], [], seg, offset
            with open(path, "rb") as f:
                f.seek(offset)
                data = f.read(max_bytes)
            recs: List[list] = []
            ends: List[int] = []
            if data:
                up = msgpack.Unpacker(raw=False, strict_map_key=False)
                up.feed(data)
                consumed = 0
                try:
                    for rec in up:
                        recs.append(rec)
                        consumed = up.tell()
                        ends.append(offset + consumed)
                except (msgpack.ExtraData, ValueError):  # pragma: no cover - corrupt tail
                    pass
                if recs:
                    return recs, ends, seg, offset + consumed
                if len(data) == max_bytes:  # pragma: no cover - one huge record
                    max_bytes *= 4
                    continue
            # nothing complete here: if a newer segment exists, this one is done
            if (seg + 1) in self.segments():
                seg, offset = seg + 1, 0
                continue
            return [], [], seg, offset

    def all_records(self) -> Iterator[list]:
        for seg in self.segments():
            with open(self.segment_path(seg), "rb") as f:
                up = msgpack.Unpacker(f, raw=False, strict_map_key=False)
                try:
                    yield from up
                except ValueError:  # pragma: no cover - truncated tail
                    return

    def last_step(self) -> Optional[int]:
        last = None
        for rec in self.all_records():
            if rec[0] == "l":
                last = rec[1] if last is None else max(last, rec[1])
            elif rec[0] == "w":  # rewind: everything >= rec[1] is gone
                last = None if last is None or rec[1] <= 0 else min(last, rec[1] - 1)
        return last

    def finished(self) -> Optional[list]:
        fin = None
        for rec in self.all_records():
            if rec[0] == "f":
                fin = rec
            elif rec[0] == "r":
                fin = None
        return fin


class RunWriter:
    """Appends records. Thread-safe; ``log`` never touches the network."""

    def __init__(self, rundir: RunDir):
        self.rd = rundir
        segs = rundir.segments()
        self.seg = segs[-1] + 1 if segs else 0  # every process starts a fresh segment
        self._f = open(rundir.segment_path(self.seg), "ab", buffering=1024 * 1024)
        self._size = 0
        self._lock = threading.Lock()
        self._packer = msgpack.Packer(use_bin_type=True)

    def append(self, rec: list) -> None:
        b = self._packer.pack(rec)
        with self._lock:
            self._f.write(b)
            self._size += len(b)
            if self._size >= SEGMENT_BYTES:
                self._f.close()
                self.seg += 1
                self._f = open(self.rd.segment_path(self.seg), "ab", buffering=1024 * 1024)
                self._size = 0

    def flush(self, fsync: bool = False) -> None:
        with self._lock:
            if not self._f.closed:
                self._f.flush()
                if fsync:
                    os.fsync(self._f.fileno())

    def close(self) -> None:
        with self._lock:
            if not self._f.closed:
                self._f.flush()
                os.fsync(self._f.fileno())
                self._f.close()
