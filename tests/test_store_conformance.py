"""MetricStore conformance suite. Every backend in BACKENDS must pass."""
import math

import numpy as np
import pytest

from chronist.store import BACKENDS, Series, open_store
from chronist.store.cache import CachingStore


@pytest.fixture(params=sorted(BACKENDS) + [f"{b}+cache" for b in sorted(BACKENDS)] + ["chunked-tiny"])
def store(request, tmp_path):
    kind, _, cached = request.param.partition("+")
    if kind == "chunked-tiny":  # tiny chunks: every test exercises sealing, pyramid, late merges
        from chronist.store.chunked import ChunkedStore
        s = ChunkedStore(str(tmp_path / "metrics.db"), chunk=8)
    else:
        s = open_store(kind, str(tmp_path / "metrics.db"))
    if cached:
        s = CachingStore(s)
    yield s
    s.close()


def pts(key, steps, values=None, wall=None):
    steps = np.asarray(steps)
    if values is None:
        values = steps * 0.5 + 1
    return {key: Series(steps, values, wall)}


def raw(store, run, key, start=None, end=None):
    b = store.read(run, key, start, end)
    assert (b.count == 1).all()
    return b.step, b.last


def test_roundtrip(store):
    store.write(1, pts("loss", [0, 1, 2], [3.0, 2.0, 1.0]))
    s, v = raw(store, 1, "loss")
    assert s.tolist() == [0, 1, 2]
    assert v.tolist() == [3.0, 2.0, 1.0]
    assert store.keys(1) == ["loss"]


def test_duplicates_ignored(store):
    store.write(1, pts("loss", [0, 1, 2], [1.0, 2.0, 3.0]))
    store.write(1, pts("loss", [1, 2, 3], [9.0, 9.0, 4.0]))  # retried overlap
    store.flush()
    store.write(1, pts("loss", [0, 3], [9.0, 9.0]))
    store.maintain()
    store.write(1, pts("loss", [2, 3, 4], [9.0, 9.0, 5.0]))
    s, v = raw(store, 1, "loss")
    assert s.tolist() == [0, 1, 2, 3, 4]
    assert v.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]  # first write wins
    assert store.stats().points == 5


def test_out_of_order(store):
    store.write(1, pts("x", [5, 3, 9], [5.0, 3.0, 9.0]))
    store.write(1, pts("x", [1, 7], [1.0, 7.0]))
    s, v = raw(store, 1, "x")
    assert s.tolist() == [1, 3, 5, 7, 9]
    assert v.tolist() == [1.0, 3.0, 5.0, 7.0, 9.0]
    assert store.last(1, ["x"]) == {"x": (9, 9.0)}
    assert store.step_range(1, "x") == (1, 9)


def test_runs_and_keys_isolated(store):
    store.write(1, {**pts("a", [0, 1]), **pts("b", [0])})
    store.write(2, pts("a", [0, 1, 2]))
    assert store.keys(1) == ["a", "b"]
    assert store.keys(2) == ["a"]
    assert len(store.read(2, "a")) == 3
    assert len(store.read(2, "b")) == 0
    assert len(store.read(3, "a")) == 0
    assert len(store.read(1, "nope", max_points=10)) == 0


def test_range_boundaries(store):
    store.write(1, pts("x", range(10)))
    s, _ = raw(store, 1, "x", 3, 7)  # start inclusive, end exclusive
    assert s.tolist() == [3, 4, 5, 6]
    s, _ = raw(store, 1, "x", None, 2)
    assert s.tolist() == [0, 1]
    s, _ = raw(store, 1, "x", 8, None)
    assert s.tolist() == [8, 9]
    s, _ = raw(store, 1, "x", 5, 5)
    assert s.tolist() == []
    b = store.read(1, "x", 3, 7, max_points=2)
    assert b.count.sum() == 4
    assert b.min.min() == 3 * 0.5 + 1 and b.max.max() == 6 * 0.5 + 1


@pytest.mark.parametrize("n,max_points", [(1000, 10), (1000, 100), (997, 7), (50, 100), (10_000, 2), (5000, 1)])
def test_downsampling_matches_raw(store, n, max_points):
    rng = np.random.default_rng(0)
    steps = np.sort(rng.choice(n * 3, size=n, replace=False))
    vals = rng.normal(size=n)
    store.write(1, pts("x", steps, vals))
    b = store.read(1, "x", max_points=max_points)
    assert len(b) <= max_points
    assert b.count.sum() == n
    w = b.width
    idx = steps // w
    for i in range(len(b)):
        bi = b.step[i] // w
        m = idx == bi
        assert m.sum() == b.count[i]
        assert b.step[i] == steps[m].max()
        assert b.min[i] == pytest.approx(vals[m].min())
        assert b.max[i] == pytest.approx(vals[m].max())
        assert b.mean[i] == pytest.approx(vals[m].mean())
        assert b.last[i] == pytest.approx(vals[m][-1])


def test_downsample_window(store):
    store.write(1, pts("x", range(100_000)))
    for lo, hi, mp in [(1234, 56789, 300), (0, 100_000, 2000), (99_000, 99_500, 1000), (17, 18, 5)]:
        b = store.read(1, "x", lo, hi, max_points=mp)
        assert 0 < len(b) <= mp
        assert b.count.sum() == hi - lo
        assert b.step.min() >= lo and b.step.max() < hi


def test_wall_time(store):
    store.write(1, pts("x", [0, 1, 2], [1.0, 2.0, 3.0], [100.0, 101.0, 102.5]))
    b = store.read(1, "x")
    assert b.wall_time.tolist() == [100.0, 101.0, 102.5]
    b = store.read(1, "x", max_points=2)
    assert b.wall_time is not None and b.wall_time[-1] == 102.5


def test_nan_and_inf(store):
    store.write(1, pts("x", [0, 1, 2], [1.0, math.nan, math.inf]))
    _, v = raw(store, 1, "x")
    assert v[0] == 1.0 and math.isnan(v[1]) and v[2] == math.inf


def test_delete_run(store):
    store.write(1, pts("x", range(100)))
    store.write(2, pts("x", range(10)))
    store.delete_run(1)
    assert store.keys(1) == []
    assert len(store.read(1, "x")) == 0
    assert store.last(1, ["x"]) == {}
    assert store.stats().points == 10
    store.write(1, pts("x", [5]))  # run id reusable after delete
    assert raw(store, 1, "x")[0].tolist() == [5]


def test_last_matches_final_raw_point(store):
    rng = np.random.default_rng(1)
    for _ in range(5):
        steps = rng.choice(10_000, size=200, replace=False)
        store.write(1, pts("a", steps, rng.normal(size=200)))
    s, v = raw(store, 1, "a")
    assert store.last(1, ["a", "missing"]) == {"a": (int(s[-1]), float(v[-1]))}


def test_readonly_reader_sees_commits(store, tmp_path):
    store.write(1, pts("x", range(10)))
    reader = open_store(store.name, store.path, readonly=True)
    try:
        assert len(reader.read(1, "x")) == 10
        store.write(1, pts("x", range(10, 20)))
        assert len(reader.read(1, "x")) == 20
        with pytest.raises(Exception):
            reader.write(1, pts("x", [100]))
    finally:
        reader.close()


def test_explicit_range_shares_grid(store):
    store.write(1, pts("a", range(0, 50_000)))
    store.write(1, pts("b", range(30_000, 100_000)))
    store.write(2, pts("a", range(0, 7_000)))
    widths = {store.read(r, k, 0, 100_000, max_points=500).width for r, k in [(1, "a"), (1, "b"), (2, "a")]}
    assert len(widths) == 1
    b = store.read(1, "b", 0, 100_000, max_points=500)
    assert len(b) <= 500 and b.count.sum() == 70_000


def test_cache_invalidation(tmp_path):
    s = CachingStore(open_store("sqlite_rows", str(tmp_path / "m.db")))
    s.write(1, pts("x", range(100)))
    a = s.read(1, "x", max_points=10)
    assert s.read(1, "x", max_points=10) is a and s.hits == 1
    s.write(1, pts("x", range(100, 200)))
    b = s.read(1, "x", max_points=10)
    assert b.count.sum() == 200
    s.write(2, pts("x", [1]))  # other run: run 1 stays cached
    assert s.read(1, "x", max_points=10) is b
    s.delete_run(1)
    assert len(s.read(1, "x", max_points=10)) == 0
    s.close()
