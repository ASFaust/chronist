"""HistStore benchmark: 10 keys x N logged steps x 64 bins, then reads.

    python bench/bench_hist.py [--steps 10000]
"""
import argparse
import os
import shutil
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from chronist import hist as H  # noqa: E402
from chronist.store.hist_store import HIST_BACKENDS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10_000)
    ap.add_argument("--keys", type=int, default=10)
    ap.add_argument("--bins", type=int, default=64)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    print("| backend | records | B/record | write rec/s | compute rec/s | read 300 steps ms | heatmap 300x128 ms |")
    print("|---|---|---|---|---|---|---|")
    for name, cls in HIST_BACKENDS.items():
        d = tempfile.mkdtemp()
        try:
            st = cls(os.path.join(d, "h.db"))
            t_comp = t_write = 0.0
            for lo in range(0, a.steps, 500):
                batch = {}
                t = time.perf_counter()
                for k in range(a.keys):
                    batch[f"k{k}"] = [H.compute(rng.normal(scale=1 + s / a.steps, size=4096), s, bins=a.bins)
                                      for s in range(lo, min(lo + 500, a.steps))]
                t_comp += time.perf_counter() - t
                t = time.perf_counter()
                st.write(1, batch)
                t_write += time.perf_counter() - t
            n = a.steps * a.keys
            lat, hm = [], []
            for _ in range(10):
                t = time.perf_counter()
                recs = st.read(1, "k3", max_steps=300)
                lat.append(time.perf_counter() - t)
                t = time.perf_counter()
                H.to_dense(recs, rows=128)
                hm.append(time.perf_counter() - t)
            print(f"| {name} | {n:,} | {st.stats().bytes_on_disk / n:.0f} | {n / t_write:,.0f} | "
                  f"{n / t_comp:,.0f} | {np.median(lat) * 1000:.1f} | {np.median(hm) * 1000:.1f} |")
            st.close()
        finally:
            shutil.rmtree(d)


if __name__ == "__main__":
    main()
