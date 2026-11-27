"""Content-addressed blob storage (code snapshots, environment info).

Blobs are keyed by the sha256 of their *uncompressed* content and stored
compressed (``codec``). ``FsBlobStore`` lays them out like git objects:
``<dir>/ab/cdef...``. Writes are atomic (tmp + rename), so a reader never
sees a partial blob.
"""
from __future__ import annotations

import hashlib
import os
import re
from typing import Iterable, List, Optional

try:
    from typing import Protocol
except ImportError:  # pragma: no cover
    Protocol = object  # type: ignore

from .codec import compress, decompress

HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def valid_hash(h: str) -> bool:
    return bool(HASH_RE.match(h))


class BlobStore(Protocol):
    def missing(self, hashes: Iterable[str]) -> List[str]: ...
    def put(self, h: str, compressed: bytes) -> None: ...      # verifies the hash
    def get(self, h: str) -> Optional[bytes]: ...              # compressed bytes
    def read(self, h: str) -> Optional[bytes]: ...             # uncompressed content


class FsBlobStore:
    def __init__(self, path: str):
        self.path = os.path.abspath(path)

    def _p(self, h: str) -> str:
        if not valid_hash(h):
            raise ValueError(f"bad blob hash {h!r}")
        return os.path.join(self.path, h[:2], h[2:])

    def has(self, h: str) -> bool:
        return os.path.exists(self._p(h))

    def missing(self, hashes: Iterable[str]) -> List[str]:
        return [h for h in dict.fromkeys(hashes) if not self.has(h)]

    def put(self, h: str, compressed: bytes, verify: bool = True) -> None:
        if self.has(h):
            return
        if verify and sha256(decompress(compressed)) != h:
            raise ValueError(f"blob content does not match hash {h}")
        p = self._p(h)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = f"{p}.tmp{os.getpid()}.{id(compressed)}"
        with open(tmp, "wb") as f:
            f.write(compressed)
        os.replace(tmp, p)

    def put_raw(self, data: bytes) -> str:
        """Store uncompressed content; returns its hash."""
        h = sha256(data)
        if not self.has(h):
            self.put(h, compress(data), verify=False)
        return h

    def get(self, h: str) -> Optional[bytes]:
        try:
            with open(self._p(h), "rb") as f:
                return f.read()
        except FileNotFoundError:
            return None

    def read(self, h: str) -> Optional[bytes]:
        c = self.get(h)
        return None if c is None else decompress(c)
