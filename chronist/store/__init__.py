"""Metric storage backends. See ``base.MetricStore`` for the interface."""
from .base import Buckets, MetricStore, Points, Series, StoreStats, bucket_width, points_from_rows
from .sqlite_rows import SqliteRowStore

BACKENDS = {"sqlite_rows": SqliteRowStore}


def open_store(kind: str, path: str, readonly: bool = False) -> MetricStore:
    try:
        cls = BACKENDS[kind]
    except KeyError:
        raise ValueError(f"unknown metric store {kind!r}; have {sorted(BACKENDS)}")
    return cls(path, readonly=readonly)


__all__ = ["Buckets", "MetricStore", "Points", "Series", "StoreStats", "SqliteRowStore",
           "bucket_width", "points_from_rows", "open_store", "BACKENDS"]
