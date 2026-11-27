"""Read cache in front of any MetricStore.

Downsampled reads of finished runs are immutable, and plots re-request the
same windows constantly (panel re-renders, grouping, several viewers). The
cache keys on (run, key, start, end, max_points), is bounded by the number of
cached buckets, and is invalidated per run on every write/delete. A per-run
version counter keeps a read that raced with a write from caching stale data.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

from .base import Buckets, MetricStore, Points, StoreStats


class CachingStore:
    BYTES_PER_BUCKET = 64  # 7-8 float64/int64 arrays

    def __init__(self, inner: MetricStore, max_mb: float = 64.0):
        max_buckets = int(max_mb * 1024 * 1024 / self.BYTES_PER_BUCKET)
        self.inner = inner
        self.max_buckets = max_buckets
        self._lru: "OrderedDict[tuple, Buckets]" = OrderedDict()
        self._size = 0
        self._ver: Dict[int, int] = {}
        self._by_run: Dict[int, set] = {}
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    def __getattr__(self, name):  # name, path, anything backend-specific
        return getattr(self.inner, name)

    def _invalidate(self, run_id: int) -> None:
        with self._lock:
            self._ver[run_id] = self._ver.get(run_id, 0) + 1
            for k in self._by_run.pop(run_id, ()):
                b = self._lru.pop(k, None)
                if b is not None:
                    self._size -= len(b)

    def write(self, run_id: int, points: Points) -> None:
        self.inner.write(run_id, points)
        self._invalidate(run_id)

    def delete_run(self, run_id: int) -> None:
        self.inner.delete_run(run_id)
        self._invalidate(run_id)

    def delete_range(self, run_id: int, start: int) -> None:
        self.inner.delete_range(run_id, start)
        self._invalidate(run_id)

    def read(self, run_id: int, key: str, start: Optional[int] = None,
             end: Optional[int] = None, max_points: Optional[int] = None,
             limit: Optional[int] = None) -> Buckets:
        if max_points is None:  # raw reads are for export/read-back: don't cache
            return self.inner.read(run_id, key, start, end, None, limit)
        ck = (run_id, key, start, end, max_points)
        with self._lock:
            b = self._lru.get(ck)
            if b is not None:
                self._lru.move_to_end(ck)
                self.hits += 1
                return b
            ver = self._ver.get(run_id, 0)
        self.misses += 1
        b = self.inner.read(run_id, key, start, end, max_points)
        if len(b) > self.max_buckets // 8:
            return b  # don't let one huge raw read flush the cache
        with self._lock:
            if self._ver.get(run_id, 0) == ver and ck not in self._lru:
                self._lru[ck] = b
                self._by_run.setdefault(run_id, set()).add(ck)
                self._size += len(b)
                while self._size > self.max_buckets and self._lru:
                    old_k, old_b = self._lru.popitem(last=False)
                    self._size -= len(old_b)
                    s = self._by_run.get(old_k[0])
                    if s is not None:
                        s.discard(old_k)
        return b

    # -- pass-through --
    def keys(self, run_id: int) -> List[str]:
        return self.inner.keys(run_id)

    def last(self, run_id: int, keys: List[str]) -> Dict[str, Tuple[int, float]]:
        return self.inner.last(run_id, keys)

    def step_range(self, run_id: int, key: str):
        return self.inner.step_range(run_id, key)

    def flush(self) -> None:
        self.inner.flush()

    def maintain(self) -> None:
        self.inner.maintain()

    def stats(self) -> StoreStats:
        return self.inner.stats()

    def close(self) -> None:
        self.inner.close()
