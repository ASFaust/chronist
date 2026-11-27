"""Histogram core: computation, config precedence, re-binning, quantiles."""
import math

import numpy as np
import pytest

from chronist import hist as H


def test_numpy_adaptive_stats_and_counts():
    x = np.random.default_rng(0).normal(size=10_000)
    r = H.compute(x, step=3, bins=50)
    assert r.count == 10_000 and r.bins == 50 and r.counts.sum() == 10_000
    assert r.lo == x.min() and r.hi == x.max()
    assert r.mean == pytest.approx(x.mean()) and r.std == pytest.approx(x.std())
    assert not r.fixed and r.underflow == r.overflow == 0


def test_nan_inf_excluded_and_counted():
    x = np.array([1.0, 2.0, np.nan, np.inf, -np.inf, -np.inf, 3.0])
    r = H.compute(x, 0, bins=4)
    assert (r.count, r.nan_count, r.posinf_count, r.neginf_count) == (3, 1, 1, 2)
    assert r.counts.sum() == 3 and r.min == 1.0 and r.max == 3.0


def test_all_nan_and_constant():
    r = H.compute([np.nan, np.nan], 0)
    assert r.count == 0 and r.nan_count == 2 and r.bins == 0 and math.isnan(r.mean)
    r = H.compute([5.0] * 10, 0)
    assert r.lo == r.hi == 5.0 and r.counts.tolist() == [10] and r.std == 0.0
    m, edges = H.to_dense([r], rows=8)
    assert m.sum() == 10 and edges[0] < 5.0 < edges[-1]


def test_fixed_range_under_overflow():
    x = np.array([-5.0, -1.0, 0.0, 0.5, 1.0, 9.0, 10.0])
    r = H.compute(x, 0, bins=4, range=(-1.0, 1.0))
    assert r.fixed and r.underflow == 1 and r.overflow == 2
    assert r.counts.sum() == 4 and (r.lo, r.hi) == (-1.0, 1.0)


def test_raw_mode_and_cap():
    r = H.compute([3.0, 1.0, 2.0], 0, raw=True)
    assert r.raw.tolist() == [3.0, 1.0, 2.0] and r.counts.sum() == 3
    assert H.quantiles(r, [0.5]) == [2.0]
    with pytest.raises(ValueError, match="at most"):
        H.compute(np.zeros(H.MAX_RAW + 1), 0, raw=True)


def test_bins_cap_and_validation():
    with pytest.raises(ValueError):
        H.Histogram([1, 2], bins=H.MAX_BINS + 1)
    with pytest.raises(ValueError):
        H.Histogram([1, 2], bins=0)
    with pytest.raises(ValueError):
        H.Histogram([1, 2], range=(1, 1))


def test_config_precedence():
    c = H.HistConfig(default_bins=32)
    assert c.resolve("grad/a", None)["bins"] == 32
    c.configure("grad/*", bins=128)
    c.configure("weights/*", range=(-1, 1))
    assert c.resolve("grad/a", None)["bins"] == 128
    assert c.resolve("grad/a", H.Histogram([1], bins=8))["bins"] == 8  # call wins
    assert c.resolve("weights/w", None)["range"] == (-1.0, 1.0)
    assert c.resolve("other", None) == {"bins": 32, "range": None, "raw": False}
    assert H.HistConfig().resolve("x", None)["bins"] == 64
    with pytest.raises(ValueError):
        c.configure("x", bins=2000)


def test_wire_roundtrip():
    r = H.compute(np.arange(100.0), 7, bins=10, wall_time=123.5)
    r2 = H.HistRecord.from_wire(r.to_wire())
    assert r2.step == 7 and r2.counts.tolist() == r.counts.tolist() and r2.wall_time == 123.5
    big = H.HistRecord(0, 2 ** 33, 0, 0, 0, 0.0, 1.0, 0.5, 0.1, 0.0, 1.0, np.array([2 ** 33], np.int64))
    assert H.HistRecord.from_wire(big.to_wire()).counts[0] == 2 ** 33  # uint64 when needed


@pytest.mark.parametrize("rows", [7, 64, 300])
def test_rebin_conserves_mass(rows):
    rng = np.random.default_rng(1)
    recs = [H.compute(rng.normal(loc=i, scale=1 + i / 5, size=5000), i, bins=37) for i in range(10)]
    m, edges = H.to_dense(recs, rows=rows)
    assert m.shape == (10, rows)
    np.testing.assert_allclose(m.sum(axis=1), [r.count for r in recs], rtol=1e-5)
    # explicit narrower y-range: column sum = mass within the range (uniform-within-bin model)
    lo, hi = -1.0, 2.0
    m2, e2 = H.to_dense(recs, rows=rows, y_range=(lo, hi))
    for r, col in zip(recs, m2):
        xs, cum = H.record_cdf_points(r)
        want = np.interp(hi, xs, cum) - np.interp(lo, xs, cum)
        assert col.sum() == pytest.approx(want, rel=1e-5, abs=1e-3)


def test_quantiles_within_one_bin():
    x = np.random.default_rng(2).exponential(size=20_000)
    r = H.compute(x, 0, bins=64)
    width = (r.hi - r.lo) / r.bins
    for q, got in zip([0.05, 0.25, 0.5, 0.75, 0.95], H.quantiles(r, [0.05, 0.25, 0.5, 0.75, 0.95])):
        assert abs(got - np.quantile(x, q)) <= width
    raw = H.compute(x[:1000], 0, raw=True)
    assert H.quantiles(raw, [0.3]) == pytest.approx([np.quantile(x[:1000].astype(np.float32), 0.3)])


def test_subsample_indices():
    assert H.subsample_indices(5, None).tolist() == [0, 1, 2, 3, 4]
    idx = H.subsample_indices(1000, 7)
    assert len(idx) <= 7 and idx[0] == 0 and idx[-1] == 999
    assert H.subsample_indices(10, 1).tolist() == [9]


def test_detection():
    assert H.numel(3.0) is None
    assert H.numel([1, 2, 3]) == 3 and H.numel([[1, 2], [3, 4]]) == 4
    assert H.numel(np.zeros((2, 3))) == 6 and H.numel(np.float32(1)) == 1


torch = pytest.importorskip("torch")


def test_torch_matches_numpy_and_stays_on_device(monkeypatch):
    t = torch.randn(4, 1000, dtype=torch.float32)
    t[0, 0] = float("nan")
    t[0, 1] = float("inf")
    r = H.compute(t, 0, bins=32)
    x = t.double().numpy().ravel()
    f = x[np.isfinite(x)]
    assert (r.count, r.nan_count, r.posinf_count) == (f.size, 1, 1)
    assert r.mean == pytest.approx(f.mean()) and r.std == pytest.approx(f.std())
    assert r.counts.sum() == f.size
    # only bins come back to the host: Tensor.cpu / .numpy are never called on the full tensor
    calls = []
    orig_to = torch.Tensor.to

    def spy_to(self, *a, **k):
        calls.append(self.numel())
        return orig_to(self, *a, **k)
    monkeypatch.setattr(torch.Tensor, "to", spy_to)
    H.compute(torch.randn(100_000), 0, bins=16)
    assert calls and max(calls) == 16
    r = H.compute(torch.tensor([2.0, 2.0]), 0)
    assert r.lo == r.hi == 2.0 and r.counts.tolist() == [2]
    r = H.compute(torch.arange(10), 0, bins=5, range=(2, 6))  # int tensor, fixed range
    assert r.underflow == 2 and r.overflow == 3 and r.counts.sum() == 5
