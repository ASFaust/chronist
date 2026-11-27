"""Randomized differential test: every backend must agree with SqliteRowStore."""
import numpy as np
import pytest

from chronist.store import BACKENDS, Series, SqliteRowStore
from chronist.store.chunked import ChunkedStore


def candidates(tmp_path):
    out = [ChunkedStore(str(tmp_path / "c8.db"), chunk=8), ChunkedStore(str(tmp_path / "c64.db"), chunk=64)]
    for name, cls in BACKENDS.items():
        if cls is not SqliteRowStore:
            out.append(cls(str(tmp_path / f"{name}.db")))
    return out


def assert_same(a, b, ctx):
    assert len(a) == len(b), ctx
    if not len(a):
        return
    assert a.width == b.width, ctx
    np.testing.assert_array_equal(a.step, b.step, err_msg=ctx)
    np.testing.assert_array_equal(a.count, b.count, err_msg=ctx)
    for f in ("min", "max", "mean", "last"):
        np.testing.assert_allclose(getattr(a, f), getattr(b, f), rtol=1e-9, atol=1e-12, equal_nan=True, err_msg=f"{ctx} {f}")


@pytest.mark.parametrize("seed", range(int(__import__("os").environ.get("CHRONIST_DIFF_SEEDS", "6"))))
def test_random_against_oracle(tmp_path, seed):
    rng = np.random.default_rng(seed)
    oracle = SqliteRowStore(str(tmp_path / "oracle.db"))
    cands = candidates(tmp_path)
    sent = []
    n_steps = int(rng.choice([300, 3000, 20000]))
    pos = 0
    for it in range(60):
        r = rng.random()
        if r < 0.1 and sent:  # retry an old batch (with different values: first write must win)
            run, key, s, v = sent[int(rng.integers(len(sent)))]
            batch = (run, key, s, v + 1.0)
        elif r < 0.2:  # genuine out-of-order points far in the past
            run, key = int(rng.integers(1, 3)), "ab"[int(rng.integers(2))]
            s = np.unique(rng.integers(0, max(pos, 10), size=int(rng.integers(1, 20))))
            batch = (run, key, s, rng.normal(size=len(s)))
        else:  # in-order append, sometimes sparse
            run, key = int(rng.integers(1, 3)), "ab"[int(rng.integers(2))]
            stride = int(rng.choice([1, 1, 3, 17]))
            n = int(rng.integers(1, n_steps // 10 + 2))
            s = pos + np.arange(n) * stride
            pos = int(s[-1]) + 1
            v = rng.normal(size=n)
            v[rng.random(n) < 0.02] = np.nan
            batch = (run, key, s, v)
        sent.append(batch)
        run, key, s, v = batch
        wall = 1000.0 + s * 0.5
        for st in [oracle] + cands:
            st.write(run, {key: Series(s, v, wall)})
        if rng.random() < 0.1:
            for st in cands:
                st.maintain()
        if rng.random() < 0.04 and pos > 10:  # rewind a run
            rr, cut = int(rng.integers(1, 3)), int(rng.integers(0, pos))
            for st in [oracle] + cands:
                st.delete_range(rr, cut)
    for st in cands:
        assert st.stats().points == oracle.stats().points
    for run in (1, 2):
        for key in "ab":
            rngs = oracle.step_range(run, key)
            if rngs is None:
                continue
            lo_, hi_ = rngs
            reads = [(None, None, None), (None, None, 1000), (None, None, 37), (None, None, 3)]
            for _ in range(15):
                a = int(rng.integers(lo_ - 5, hi_ + 5))
                b = a + int(rng.integers(1, hi_ - lo_ + 20))
                reads.append((a, b, int(rng.choice([2, 7, 50, 500, 5000]))))
                reads.append((a, b, None))
            for a, b, mp in reads:
                want = oracle.read(run, key, a, b, mp)
                for st in cands:
                    got = st.read(run, key, a, b, mp)
                    ctx = f"{type(st).__name__}(chunk={getattr(st, 'chunk', '')}) run={run} key={key} [{a},{b}) mp={mp}"
                    if mp is None or want.width == got.width:
                        assert_same(got, want, ctx)
                    else:  # pyramid backends may use a coarser, aligned width: compare totals
                        assert got.width % 1 == 0 and got.width >= want.width, ctx
                        assert len(got) <= mp, ctx
                        assert got.count.sum() == want.count.sum(), ctx
                        np.testing.assert_allclose(np.nanmin(got.min), np.nanmin(want.min), err_msg=ctx)
                        np.testing.assert_allclose(np.nanmax(got.max), np.nanmax(want.max), err_msg=ctx)
                        assert got.step[-1] == want.step[-1], ctx
            for st in cands:
                assert st.last(run, [key]) == oracle.last(run, [key])
                assert st.step_range(run, key) == oracle.step_range(run, key)
                lim = int(rng.integers(1, 300))
                a = oracle.read(run, key, None, None, None, limit=lim)
                assert_same(st.read(run, key, None, None, None, limit=lim), a, f"limit={lim}")
    for st in [oracle] + cands:
        st.close()
