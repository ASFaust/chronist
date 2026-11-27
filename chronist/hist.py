"""Histograms: client-side computation, records, re-binning and quantiles.

Shared by the client (computing records, ``to_dense``) and the server
(rendering). See docs/HISTOGRAMS_FILES.md.

A record summarizes one (run, key, step): exact stats over the finite values,
NaN/Inf counts, and ``counts`` over ``bins`` uniform bins on ``[lo, hi]``.
Adaptive mode (default) uses this step's min/max as the range; fixed mode
uses a user range plus underflow/overflow counters. Constant input is a
single-bin record with ``lo == hi`` (all mass at that value). ``raw`` keeps
the exact values (float32) for small vectors.
"""
from __future__ import annotations

import fnmatch
import math
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_BINS = 64
MAX_BINS = 1024
MAX_RAW = 10_000


class Histogram:
    """Mark a value to be logged as a histogram.

    ``chronist.Histogram(x, bins=None, range=None, raw=False)`` where ``x`` is a
    list, numpy array, torch tensor (any device) or jax array of any shape.
    """

    __slots__ = ("values", "bins", "range", "raw")

    def __init__(self, values: Any, bins: Optional[int] = None, range: Optional[Tuple[float, float]] = None,
                 raw: bool = False):
        if bins is not None:
            _check_bins(bins)
        if range is not None:
            lo, hi = float(range[0]), float(range[1])
            if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
                raise ValueError(f"histogram range must be finite with lo < hi, got {range!r}")
            range = (lo, hi)
        self.values = values
        self.bins = bins
        self.range = range
        self.raw = bool(raw)


def _check_bins(bins: int) -> None:
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise ValueError(f"bins must be a positive int, got {bins!r}")
    if bins > MAX_BINS:
        raise ValueError(f"bins={bins} exceeds the maximum of {MAX_BINS}")


@dataclass
class HistRecord:
    step: int
    count: int  # finite values
    nan_count: int
    posinf_count: int
    neginf_count: int
    min: float
    max: float
    mean: float
    std: float
    lo: float
    hi: float
    counts: np.ndarray  # int64 in memory; uint32/uint64 on the wire / disk
    underflow: int = 0
    overflow: int = 0
    fixed: bool = False
    raw: Optional[np.ndarray] = None  # float32
    wall_time: Optional[float] = None

    @property
    def bins(self) -> int:
        return len(self.counts)

    @property
    def edges(self) -> np.ndarray:
        if self.lo == self.hi:
            return np.array([self.lo, self.hi])
        return np.linspace(self.lo, self.hi, self.bins + 1)

    # ---- wire format (msgpack-friendly dict) ----
    def to_wire(self) -> dict:
        big = self.counts.size and int(self.counts.max()) >= 2 ** 32
        d = {"step": int(self.step), "count": int(self.count), "nan": int(self.nan_count),
             "pinf": int(self.posinf_count), "ninf": int(self.neginf_count),
             "min": _f(self.min), "max": _f(self.max), "mean": _f(self.mean), "std": _f(self.std),
             "lo": _f(self.lo), "hi": _f(self.hi),
             "counts": self.counts.astype("<u8" if big else "<u4").tobytes(), "c64": bool(big),
             "under": int(self.underflow), "over": int(self.overflow), "fixed": bool(self.fixed)}
        if self.raw is not None:
            d["raw"] = self.raw.astype("<f4").tobytes()
        if self.wall_time is not None:
            d["wall"] = float(self.wall_time)
        return d

    @classmethod
    def from_wire(cls, d: dict) -> "HistRecord":
        counts = np.frombuffer(d["counts"], dtype="<u8" if d.get("c64") else "<u4").astype(np.int64)
        raw = d.get("raw")
        return cls(step=int(d["step"]), count=int(d["count"]), nan_count=int(d.get("nan", 0)),
                   posinf_count=int(d.get("pinf", 0)), neginf_count=int(d.get("ninf", 0)),
                   min=_nf(d.get("min")), max=_nf(d.get("max")), mean=_nf(d.get("mean")), std=_nf(d.get("std")),
                   lo=_nf(d.get("lo")), hi=_nf(d.get("hi")), counts=counts,
                   underflow=int(d.get("under", 0)), overflow=int(d.get("over", 0)), fixed=bool(d.get("fixed")),
                   raw=None if raw is None else np.frombuffer(raw, dtype="<f4").copy(),
                   wall_time=d.get("wall"))

    @classmethod
    def from_json(cls, d: dict) -> "HistRecord":
        n = lambda v: math.nan if v is None else float(v)  # noqa: E731
        return cls(step=int(d["step"]), count=int(d["count"]), nan_count=int(d["nan_count"]),
                   posinf_count=int(d["posinf_count"]), neginf_count=int(d["neginf_count"]),
                   min=n(d["min"]), max=n(d["max"]), mean=n(d["mean"]), std=n(d["std"]),
                   lo=n(d["lo"]), hi=n(d["hi"]), counts=np.asarray(d["counts"], dtype=np.int64),
                   underflow=int(d.get("underflow", 0)), overflow=int(d.get("overflow", 0)),
                   fixed=bool(d.get("fixed")),
                   raw=None if d.get("raw") is None else np.asarray(d["raw"], dtype=np.float32),
                   wall_time=d.get("wall_time"))

    def quantiles(self, qs=(0.05, 0.25, 0.5, 0.75, 0.95)) -> List[float]:
        return quantiles(self, qs)

    def to_json(self) -> dict:
        """JSON-friendly dict (API responses)."""
        d = {"step": self.step, "count": self.count, "nan_count": self.nan_count,
             "posinf_count": self.posinf_count, "neginf_count": self.neginf_count,
             "min": _f(self.min), "max": _f(self.max), "mean": _f(self.mean), "std": _f(self.std),
             "lo": _f(self.lo), "hi": _f(self.hi), "counts": self.counts.tolist(),
             "underflow": self.underflow, "overflow": self.overflow, "fixed": self.fixed,
             "wall_time": self.wall_time}
        if self.raw is not None:
            d["raw"] = [float(x) for x in self.raw]
        return d


def _f(x):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else float(x)


def _nf(x):
    return math.nan if x is None else float(x)


# ---------------------------------------------------------------- detection
def is_tensor_like(v: Any) -> bool:
    return hasattr(v, "shape") and hasattr(v, "dtype")


def numel(v: Any) -> Optional[int]:
    """Element count for arrays/tensors/sequences, None for scalars."""
    if isinstance(v, (list, tuple)):
        return len(v) if not v or not isinstance(v[0], (list, tuple)) else int(np.asarray(v).size)
    if is_tensor_like(v):
        try:
            n = v.numel() if hasattr(v, "numel") and callable(v.numel) else int(np.prod(v.shape))
            return int(n)
        except Exception:
            return None
    return None


# ---------------------------------------------------------------- computation
def compute(values: Any, step: int, bins: int = DEFAULT_BINS, range: Optional[Tuple[float, float]] = None,
            raw: bool = False, wall_time: Optional[float] = None) -> HistRecord:
    """Summarize ``values`` into a record. Torch tensors are reduced on their
    own device; only the stats and ``bins`` counts come back to the host."""
    _check_bins(bins)
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(values, torch.Tensor):
        return _compute_torch(torch, values, step, bins, range, raw, wall_time)
    x = np.asarray(values, dtype=np.float64).ravel()
    return _compute_numpy(x, step, bins, range, raw, wall_time)


def _compute_numpy(x, step, bins, range, raw, wall_time) -> HistRecord:
    nan = int(np.isnan(x).sum())
    pinf = int(np.isposinf(x).sum())
    ninf = int(np.isneginf(x).sum())
    f = x[np.isfinite(x)]
    n = int(f.size)
    if raw:
        if x.size > MAX_RAW:
            raise ValueError(f"raw histograms hold at most {MAX_RAW} values (got {x.size}); use bins instead")
    if n == 0:
        return HistRecord(step, 0, nan, pinf, ninf, math.nan, math.nan, math.nan, math.nan,
                          math.nan, math.nan, np.zeros(0, np.int64), 0, 0, range is not None,
                          x.astype(np.float32) if raw else None, wall_time)
    mn, mx = float(f.min()), float(f.max())
    mean = float(f.mean())
    std = float(f.std())
    under = over = 0
    if range is not None:
        lo, hi = range
        under = int((f < lo).sum())
        over = int((f > hi).sum())
        inside = f[(f >= lo) & (f <= hi)]
        counts = np.histogram(inside, bins=bins, range=(lo, hi))[0].astype(np.int64)
    elif mn == mx:
        lo = hi = mn
        counts = np.array([n], dtype=np.int64)
    else:
        lo, hi = mn, mx
        counts = np.histogram(f, bins=bins, range=(lo, hi))[0].astype(np.int64)
    return HistRecord(step, n, nan, pinf, ninf, mn, mx, mean, std, float(lo), float(hi), counts,
                      under, over, range is not None, x.astype(np.float32) if raw else None, wall_time)


def _compute_torch(torch, t, step, bins, range, raw, wall_time) -> HistRecord:
    with torch.no_grad():
        x = t.detach().reshape(-1)
        if not x.is_floating_point():
            x = x.float()
        if raw:
            if x.numel() > MAX_RAW:
                raise ValueError(f"raw histograms hold at most {MAX_RAW} values (got {x.numel()}); use bins instead")
            return _compute_numpy(x.double().cpu().numpy(), step, bins, range, True, wall_time)
        nan_m = torch.isnan(x)
        pinf_m = torch.isposinf(x)
        ninf_m = torch.isneginf(x)
        fin = x[~(nan_m | pinf_m | ninf_m)]
        nan, pinf, ninf = int(nan_m.sum()), int(pinf_m.sum()), int(ninf_m.sum())
        n = int(fin.numel())
        if n == 0:
            return HistRecord(step, 0, nan, pinf, ninf, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan,
                              np.zeros(0, np.int64), 0, 0, range is not None, None, wall_time)
        f64 = fin.double()  # exact-enough stats; stays on device
        mn_t, mx_t = torch.aminmax(fin)
        mn, mx = float(mn_t), float(mx_t)
        mean = float(f64.mean())
        std = float(f64.std(unbiased=False)) if n > 1 else 0.0
        under = over = 0
        if range is not None:
            lo, hi = range
            under = int((fin < lo).sum())
            over = int((fin > hi).sum())
            counts_t = torch.histc(fin.float(), bins=bins, min=lo, max=hi)  # ignores values outside
        elif mn == mx:
            lo = hi = mn
            counts_t = None
        else:
            lo, hi = mn, mx
            counts_t = torch.histc(fin.float(), bins=bins, min=lo, max=hi)
        counts = (np.array([n], np.int64) if counts_t is None
                  else counts_t.to("cpu").numpy().round().astype(np.int64))
    return HistRecord(step, n, nan, pinf, ninf, mn, mx, mean, std, float(lo), float(hi), counts,
                      under, over, range is not None, None, wall_time)


# ---------------------------------------------------------------- config precedence
class HistConfig:
    """call > key pattern (last matching rule wins) > run default > 64."""

    def __init__(self, default_bins: int = DEFAULT_BINS):
        _check_bins(default_bins)
        self.default_bins = default_bins
        self.rules: List[Tuple[str, dict]] = []

    def configure(self, pattern: str, bins: Optional[int] = None, range: Optional[Tuple[float, float]] = None,
                  raw: Optional[bool] = None) -> None:
        if bins is not None:
            _check_bins(bins)
        opts = {k: v for k, v in (("bins", bins), ("range", range), ("raw", raw)) if v is not None}
        if "range" in opts:
            opts["range"] = Histogram([0], range=opts["range"]).range  # validate
        self.rules.append((pattern, opts))

    def resolve(self, key: str, h: Optional[Histogram]) -> dict:
        out = {"bins": self.default_bins, "range": None, "raw": False}
        for pattern, opts in self.rules:
            if fnmatch.fnmatchcase(key, pattern):
                out.update(opts)
        if h is not None:
            if h.bins is not None:
                out["bins"] = h.bins
            if h.range is not None:
                out["range"] = h.range
            if h.raw:
                out["raw"] = True
        return out


# ---------------------------------------------------------------- re-binning / quantiles
def record_cdf_points(r: HistRecord) -> Tuple[np.ndarray, np.ndarray]:
    """(x, cumulative mass) knots of the piecewise-linear CDF of the in-range mass."""
    if r.lo == r.hi:
        return np.array([r.lo, r.lo]), np.array([0.0, float(r.counts.sum())])
    return r.edges, np.concatenate([[0.0], np.cumsum(r.counts, dtype=np.float64)])


def rebin(r: HistRecord, edges: np.ndarray) -> np.ndarray:
    """Mass of ``r`` in each target bin (overlap-proportional; exact for raw).
    Mass outside the target range is dropped. Sums to the in-range mass."""
    rows = len(edges) - 1
    if r.count == 0 or rows < 1:
        return np.zeros(max(rows, 0))
    if r.raw is not None:
        v = r.raw[np.isfinite(r.raw)].astype(np.float64)
        return np.histogram(v, bins=edges)[0].astype(np.float64)
    if r.lo == r.hi:  # point mass
        out = np.zeros(rows)
        if edges[0] <= r.lo <= edges[-1]:
            i = min(int(np.searchsorted(edges, r.lo, side="right")) - 1, rows - 1)
            out[max(i, 0)] = float(r.counts.sum())
        return out
    xs, cum = record_cdf_points(r)
    at = np.interp(edges, xs, cum, left=0.0, right=cum[-1])
    return np.diff(at)


def common_range(records: Sequence[HistRecord]) -> Tuple[float, float]:
    los = [r.lo for r in records if r.count and math.isfinite(r.lo)]
    his = [r.hi for r in records if r.count and math.isfinite(r.hi)]
    if not los:
        return 0.0, 1.0
    lo, hi = min(los), max(his)
    if lo == hi:
        pad = abs(lo) * 0.05 or 0.5
        lo, hi = lo - pad, hi + pad
    return lo, hi


def to_dense(records: Sequence[HistRecord], rows: int = 128,
             y_range: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Re-bin records onto a common value grid: (steps x rows matrix, edges)."""
    lo, hi = y_range if y_range is not None else common_range(records)
    edges = np.linspace(lo, hi, rows + 1)
    m = np.zeros((len(records), rows), dtype=np.float32)
    for i, r in enumerate(records):
        m[i] = rebin(r, edges)
    return m, edges


def quantiles(r: HistRecord, qs: Sequence[float]) -> List[float]:
    """Quantiles of the finite values: exact for raw records, otherwise
    linear interpolation within bins (error <= one bin width). Underflow /
    overflow mass (fixed mode) counts as being at the range edges."""
    if r.count == 0:
        return [math.nan] * len(qs)
    if r.raw is not None:
        v = r.raw[np.isfinite(r.raw)].astype(np.float64)
        return [float(x) for x in np.quantile(v, qs)]
    xs, cum = record_cdf_points(r)
    cum = cum + r.underflow
    xs = np.concatenate([[r.lo], xs, [r.hi]])
    cum = np.concatenate([[0.0], cum, [cum[-1] + r.overflow]])
    total = cum[-1]
    out = []
    for q in qs:
        t = q * total
        j = int(np.searchsorted(cum, t, side="left"))
        j = min(max(j, 1), len(cum) - 1)
        c0, c1 = cum[j - 1], cum[j]
        x0, x1 = xs[j - 1], xs[j]
        out.append(float(x0 if c1 == c0 else x0 + (t - c0) / (c1 - c0) * (x1 - x0)))
    return out


def subsample_indices(n: int, max_steps: Optional[int]) -> np.ndarray:
    """Evenly strided indices into n items, always including first and last."""
    if max_steps is None or n <= max_steps:
        return np.arange(n)
    if max_steps <= 1:
        return np.array([n - 1])
    return np.unique(np.round(np.linspace(0, n - 1, max_steps)).astype(np.int64))


class HistList(list):
    """List of HistRecords (one run, one key) with ``to_dense``."""

    def to_dense(self, rows: int = 128, y_range: Optional[Tuple[float, float]] = None):
        """(steps x rows matrix, edges, steps): same re-binning as the server."""
        m, edges = to_dense(self, rows, y_range)
        return m, edges, np.array([r.step for r in self], dtype=np.int64)
