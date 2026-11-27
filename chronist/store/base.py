"""MetricStore protocol and the shared data types.

Range convention for every backend: ``start`` is inclusive, ``end`` is
exclusive, i.e. a read covers steps ``start <= step < end``. ``None`` means
unbounded on that side.

Bucketing convention: buckets lie on an *absolute* grid, bucket index is
``step // width``. Because the grid does not depend on the requested range,
bucket boundaries stay stable across zooms and are aligned across runs, which
is what grouping/aggregation across runs relies on. ``Buckets.step`` is the
step of the last point inside each bucket (so a raw read, ``count == 1``,
reports the point's own step).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np

try:  # Python 3.8+: typing.Protocol
    from typing import Protocol
except ImportError:  # pragma: no cover
    Protocol = object  # type: ignore


@dataclass
class Series:
    """Points of a single key: parallel arrays."""

    step: np.ndarray  # int64
    value: np.ndarray  # float64
    wall_time: Optional[np.ndarray] = None  # float64 unix seconds, or None

    def __post_init__(self):
        self.step = np.asarray(self.step, dtype=np.int64)
        self.value = np.asarray(self.value, dtype=np.float64)
        if self.wall_time is not None:
            self.wall_time = np.asarray(self.wall_time, dtype=np.float64)
            if len(self.wall_time) != len(self.step):
                raise ValueError("wall_time length mismatch")
        if len(self.step) != len(self.value):
            raise ValueError("step/value length mismatch")

    def __len__(self) -> int:
        return len(self.step)


# A batch of points to write: key -> Series.
Points = Mapping[str, Series]


def points_from_rows(rows) -> Dict[str, Series]:
    """Build Points from an iterable of (key, step, value[, wall_time])."""
    acc: Dict[str, list] = {}
    for r in rows:
        acc.setdefault(r[0], []).append(r[1:])
    out = {}
    for k, lst in acc.items():
        steps = [r[0] for r in lst]
        vals = [r[1] for r in lst]
        walls = [r[2] if len(r) > 2 else math.nan for r in lst]
        has_wall = any(len(r) > 2 for r in lst)
        out[k] = Series(steps, vals, walls if has_wall else None)
    return out


@dataclass
class Buckets:
    """The one format UI and aggregation code handle. Raw reads have count==1."""

    step: np.ndarray
    min: np.ndarray
    max: np.ndarray
    mean: np.ndarray
    last: np.ndarray
    count: np.ndarray
    wall_time: Optional[np.ndarray] = None  # wall time of the `last` point
    width: int = 1  # bucket width in steps (1 for a raw read)

    def __len__(self) -> int:
        return len(self.step)

    @classmethod
    def empty(cls) -> "Buckets":
        z = np.zeros(0)
        return cls(np.zeros(0, np.int64), z, z, z, z, np.zeros(0, np.int64))

    @classmethod
    def from_raw(cls, step, value, wall_time=None) -> "Buckets":
        step = np.asarray(step, np.int64)
        value = np.asarray(value, np.float64)
        return cls(step, value, value, value, value,
                   np.ones(len(step), np.int64),
                   None if wall_time is None else np.asarray(wall_time, np.float64), 1)

    def to_dict(self) -> dict:
        d = {
            "step": self.step.tolist(),
            "min": _nan_list(self.min),
            "max": _nan_list(self.max),
            "mean": _nan_list(self.mean),
            "last": _nan_list(self.last),
            "count": self.count.tolist(),
            "width": int(self.width),
        }
        if self.wall_time is not None:
            d["wall_time"] = _nan_list(self.wall_time)
        return d


def _nan_list(a: np.ndarray) -> list:
    """JSON has no NaN/inf: map non-finite values to None."""
    out = a.tolist()
    if not np.isfinite(a).all():
        bad = ~np.isfinite(a)
        for i in np.flatnonzero(bad):
            out[i] = None
    return out


@dataclass
class StoreStats:
    bytes_on_disk: int
    points: int
    extra: dict = field(default_factory=dict)


def bucket_width(start: int, end: int, max_points: int) -> int:
    """Bucket width so that the half-open range [start, end) touches at most
    ``max_points`` buckets of the absolute grid ``step // width``."""
    span = max(end - start, 1)
    if max_points <= 1:
        return max(span + abs(start), 1) * 2  # everything in one bucket
    if span <= max_points:
        return 1
    return max(1, math.ceil(span / (max_points - 1)))


class MetricStore(Protocol):
    def write(self, run_id: int, points: Points) -> None:
        """Write a batch. Idempotent: duplicate (run, key, step) is ignored."""

    def read(self, run_id: int, key: str, start: Optional[int] = None,
             end: Optional[int] = None, max_points: Optional[int] = None) -> Buckets:
        """max_points=None -> raw points; otherwise <= max_points buckets."""

    def keys(self, run_id: int) -> List[str]: ...

    def last(self, run_id: int, keys: List[str]) -> Dict[str, Tuple[int, float]]: ...

    def step_range(self, run_id: int, key: str) -> Optional[Tuple[int, int]]:
        """(min_step, max_step) of a key, or None if it has no points."""

    def delete_run(self, run_id: int) -> None: ...

    def flush(self) -> None: ...

    def maintain(self) -> None: ...

    def stats(self) -> StoreStats: ...

    def close(self) -> None: ...
