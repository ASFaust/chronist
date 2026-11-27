"""MetricStore benchmark harness.

Workload per scenario: `runs` runs x `metrics` metrics x `steps` steps of noisy
learning curves, produced by `writers` parallel producer threads that hand
batches to a single writer thread (the same single-writer shape the server
uses). A fraction `dup` of batches is re-sent, simulating client retries.

Each (backend, scenario) runs in a fresh subprocess so peak RAM is meaningful.

    python bench/bench_store.py                    # quick scenarios
    python bench/bench_store.py --scenario 10M     # one big run
    python bench/bench_store.py --backends sqlite_rows --scenario 100k,grouped
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import resource
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from chronist.store import BACKENDS, Series, open_store  # noqa: E402

SCENARIOS = {
    # name: (runs, steps, metrics, batch_steps)
    "100k": (8, 100_000, 20, 1000),
    "1M": (1, 1_000_000, 20, 5000),
    "10M": (1, 10_000_000, 20, 10_000),
    "grouped": (50, 20_000, 4, 1000),  # many runs for the 50-run grouped read
}


def curve(rng, steps: np.ndarray, metric: int) -> np.ndarray:
    scale = 1 + metric % 5
    base = scale * np.exp(-steps / (steps[-1] * 0.3 + 1)) + 0.05 * metric
    return base + rng.normal(0, 0.05 * scale, size=len(steps))


def produce(run_ids, steps, metrics, batch_steps, dup, out: queue.Queue, seed):
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for run in run_ids:
        all_steps = np.arange(steps, dtype=np.int64)
        vals = {f"m{m:02d}": curve(rng, all_steps, m) for m in range(metrics)}
        for lo in range(0, steps, batch_steps):
            s = all_steps[lo:lo + batch_steps]
            batch = {k: Series(s, v[lo:lo + batch_steps], t0 + s * 0.01) for k, v in vals.items()}
            out.put((run, batch))
            if rng.random() < dup:
                out.put((run, batch))  # retried duplicate
    out.put(None)


def pct(xs, p):
    return float(np.percentile(np.array(xs) * 1000, p))


def run_one(backend: str, scenario: str, writers: int, dup: float, workdir: str) -> dict:
    runs, steps, metrics, batch_steps = SCENARIOS[scenario]
    path = os.path.join(workdir, "metrics.db")
    store = open_store(backend, path)
    q: queue.Queue = queue.Queue(maxsize=64)
    run_ids = list(range(1, runs + 1))
    nw = min(writers, runs) if runs > 1 else 1
    shards = [run_ids[i::nw] for i in range(nw)]
    threads = [threading.Thread(target=produce, args=(sh, steps, metrics, batch_steps, dup, q, i), daemon=True)
               for i, sh in enumerate(shards)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    done = 0
    sent = 0
    while done < len(threads):
        item = q.get()
        if item is None:
            done += 1
            continue
        run, batch = item
        store.write(run, batch)
        sent += sum(len(s) for s in batch.values())
    store.flush()
    store.maintain()
    ingest_s = time.perf_counter() - t0
    st = store.stats()
    expected = runs * steps * metrics
    assert st.points == expected, (st.points, expected)

    rng = np.random.default_rng(42)
    lat_full, lat_zoom, lat_group = [], [], []
    for _ in range(20):
        run = int(rng.integers(1, runs + 1))
        key = f"m{int(rng.integers(metrics)):02d}"
        t = time.perf_counter()
        b = store.read(run, key, max_points=2000)
        lat_full.append(time.perf_counter() - t)
        assert len(b) <= 2000
        lo = int(rng.integers(0, steps - steps // 100))
        t = time.perf_counter()
        store.read(run, key, lo, lo + steps // 100, max_points=2000)
        lat_zoom.append(time.perf_counter() - t)
    if runs >= 50:
        for _ in range(5):
            key = f"m{int(rng.integers(metrics)):02d}"
            t = time.perf_counter()
            for run in range(1, 51):
                store.read(run, key, max_points=2000)
            lat_group.append(time.perf_counter() - t)
    store.close()
    return {
        "backend": backend,
        "scenario": scenario,
        "points": expected,
        "bytes_per_point": st.bytes_on_disk / expected,
        "ingest_pts_per_s": sent / ingest_s,
        "full_p50_ms": pct(lat_full, 50), "full_p95_ms": pct(lat_full, 95),
        "zoom_p50_ms": pct(lat_zoom, 50), "zoom_p95_ms": pct(lat_zoom, 95),
        "group50_p50_ms": pct(lat_group, 50) if lat_group else None,
        "group50_p95_ms": pct(lat_group, 95) if lat_group else None,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }


def fmt(v, digits=1):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:,.{digits}f}"
    return f"{v:,}" if isinstance(v, int) else str(v)


def table(results) -> str:
    cols = [("backend", "backend"), ("scenario", "scenario"), ("points", "points"),
            ("B/pt", "bytes_per_point"), ("ingest pt/s", "ingest_pts_per_s"),
            ("full p50/p95 ms", ("full_p50_ms", "full_p95_ms")),
            ("zoom p50/p95 ms", ("zoom_p50_ms", "zoom_p95_ms")),
            ("50-run p50/p95 ms", ("group50_p50_ms", "group50_p95_ms")),
            ("peak RSS MB", "peak_rss_mb")]
    lines = ["| " + " | ".join(c[0] for c in cols) + " |", "|" + "---|" * len(cols)]
    for r in results:
        cells = []
        for _, k in cols:
            if isinstance(k, tuple):
                cells.append(f"{fmt(r[k[0]])} / {fmt(r[k[1]])}" if r[k[0]] is not None else "-")
            elif k == "ingest_pts_per_s":
                cells.append(fmt(r[k], 0))
            else:
                cells.append(fmt(r[k]))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", default=",".join(sorted(BACKENDS)))
    ap.add_argument("--scenario", default="100k,grouped")
    ap.add_argument("--writers", type=int, default=8)
    ap.add_argument("--dup", type=float, default=0.05)
    ap.add_argument("--workdir", default=None, help="where to put the databases (default: temp)")
    ap.add_argument("--out", default=None, help="write markdown table here too")
    ap.add_argument("--_child", nargs=2, help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a._child:
        backend, scenario = a._child
        wd = tempfile.mkdtemp(dir=a.workdir)
        try:
            print(json.dumps(run_one(backend, scenario, a.writers, a.dup, wd)))
        finally:
            shutil.rmtree(wd, ignore_errors=True)
        return

    results = []
    for scenario in a.scenario.split(","):
        for backend in a.backends.split(","):
            print(f"... {backend} / {scenario}", file=sys.stderr, flush=True)
            cmd = [sys.executable, __file__, "--_child", backend, scenario,
                   "--writers", str(a.writers), "--dup", str(a.dup)]
            if a.workdir:
                cmd += ["--workdir", a.workdir]
            out = subprocess.run(cmd, capture_output=True, text=True)
            if out.returncode != 0:
                print(out.stderr, file=sys.stderr)
                continue
            results.append(json.loads(out.stdout.strip().splitlines()[-1]))
    t = table(results)
    print(t)
    if a.out:
        with open(a.out, "w") as f:
            f.write(t + "\n")


if __name__ == "__main__":
    main()
