"""Compression shared by chunk storage and blobs: zstd if available, else zlib.
The first byte tags the codec so either side can decode both."""
from __future__ import annotations

import zlib

try:
    import zstandard as _zstd
    _ZC = _zstd.ZstdCompressor(level=3)
    _ZD = _zstd.ZstdDecompressor()
except ImportError:  # pragma: no cover
    _zstd = None


def compress(raw: bytes) -> bytes:
    if _zstd is not None:
        return b"Z" + _ZC.compress(raw)
    return b"L" + zlib.compress(raw, 6)


def compress_maybe(raw: bytes, min_saving: float = 0.05) -> bytes:
    """Compress unless it saves less than ``min_saving`` (checkpoints, random
    data): then store as-is behind the raw tag."""
    c = compress(raw)
    if len(c) > (1 - min_saving) * (len(raw) + 1):
        return b"R" + raw
    return c


def decompress(blob: bytes) -> bytes:
    tag = blob[:1]
    if tag == b"R":
        return bytes(blob[1:])
    if tag == b"Z":
        if _zstd is None:  # pragma: no cover
            raise RuntimeError("data is zstd-compressed; pip install zstandard")
        return _ZD.decompress(blob[1:], max_output_size=1 << 31)
    if tag == b"L":
        return zlib.decompress(blob[1:])
    raise ValueError("unknown compression tag")
