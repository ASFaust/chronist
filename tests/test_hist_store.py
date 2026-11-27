"""HistStore conformance (every backend in HIST_BACKENDS)."""
import numpy as np
import pytest

from chronist import hist as H
from chronist.store.hist_store import HIST_BACKENDS


@pytest.fixture(params=sorted(HIST_BACKENDS))
def store(request, tmp_path):
    s = HIST_BACKENDS[request.param](str(tmp_path / "h.db"))
    yield s
    s.close()


def recs(steps, seed=0, **kw):
    rng = np.random.default_rng(seed)
    return [H.compute(rng.normal(size=500), s, wall_time=1000.0 + s, **kw) for s in steps]


def test_roundtrip_exact(store):
    r = recs([5], bins=20)[0]
    raw = H.compute([1.0, 2.0, np.nan], 6, raw=True)
    fixed = H.compute(np.arange(10.0), 7, bins=4, range=(2, 6))
    const = H.compute([3.0, 3.0], 8)
    empty = H.compute([np.nan], 9)
    store.write(1, {"w": [r, raw, fixed, const, empty]})
    got = store.read(1, "w")
    assert [g.step for g in got] == [5, 6, 7, 8, 9]
    g = got[0]
    assert g.counts.tolist() == r.counts.tolist() and g.mean == r.mean and g.wall_time == 1005.0
    assert got[1].raw[:2].tolist() == [1.0, 2.0] and np.isnan(got[1].raw[2]) and got[1].nan_count == 1
    assert got[2].underflow == 2 and got[2].overflow == 3 and got[2].fixed
    assert got[3].lo == got[3].hi == 3.0
    assert got[4].count == 0 and got[4].bins == 0


def test_idempotent(store):
    a = recs(range(10), seed=1)
    store.write(1, {"w": a})
    store.write(1, {"w": recs(range(5, 15), seed=2)})  # overlapping retry with other values
    got = store.read(1, "w")
    assert [g.step for g in got] == list(range(15))
    assert got[3].mean == a[3].mean  # first write wins
    assert store.stats().points == 15


def test_subsample_and_range(store):
    store.write(1, {"w": recs(range(0, 1000, 10))})
    got = store.read(1, "w", max_steps=7)
    assert len(got) <= 7 and got[0].step == 0 and got[-1].step == 990
    got = store.read(1, "w", 100, 200)
    assert [g.step for g in got] == list(range(100, 200, 10))
    got = store.read(1, "w", 100, 200, max_steps=3)
    assert got[0].step == 100 and got[-1].step == 190


def test_keys_at_last_delete(store):
    store.write(1, {"a": recs([0, 10, 20]), "b": recs([5])})
    store.write(2, {"a": recs([0])})
    assert store.keys(1) == ["a", "b"] and store.keys(2) == ["a"] and store.keys(3) == []
    assert store.at(1, "a", 15).step == 10 and store.at(1, "a", -1) is None
    assert store.last(1, "a").step == 20 and store.last(1, "zz") is None
    store.delete_range(1, 10)
    assert [g.step for g in store.read(1, "a")] == [0] and store.keys(1) == ["a", "b"]
    store.delete_run(1)
    assert store.keys(1) == [] and len(store.read(2, "a")) == 1
