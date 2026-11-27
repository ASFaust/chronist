# MetricStore benchmark

`python bench/bench_store.py --scenario 100k,grouped,1M` (and `--scenario 10M`).
Synthetic noisy curves, 20 metrics (grouped: 4), 8 producer threads into one
writer, 5% retried duplicate batches. Reads use max_points=2000; "zoom" is a 1%
window; "50-run" is one key from 50 runs, read serially. 256-core Linux box,
SQLite 3.31, Python 3.9. Values are float64 Gaussian noise, which is the worst
case for compression.

| backend | scenario | points | B/pt | ingest pt/s | full p50/p95 ms | zoom p50/p95 ms | 50-run p50/p95 ms | peak RSS MB |
|---|---|---|---|---|---|---|---|---|
| sqlite_rows | 100k | 16,000,000 | 32.9 | 618,912 | 34.2 / 36.1 | 0.6 / 0.7 | - | 244.7 |
| chunked | 100k | 16,000,000 | 10.9 | 364,944 | 2.7 / 3.0 | 0.5 / 0.7 | - | 257.5 |
| sqlite_rows | grouped | 4,000,000 | 33.1 | 719,925 | 10.0 / 10.4 | 0.2 / 0.2 | 507.2 / 508.7 | 180.8 |
| chunked | grouped | 4,000,000 | 16.1 | 414,448 | 3.5 / 3.8 | 0.2 / 0.3 | 175.0 / 182.8 | 181.4 |
| sqlite_rows | 1M | 20,000,000 | 32.5 | 718,024 | 323.5 / 333.2 | 7.1 / 7.1 | - | 423.3 |
| chunked | 1M | 20,000,000 | 9.8 | 451,133 | 1.7 / 2.0 | 3.2 / 7.1 | - | 389.5 |
| sqlite_rows | 10M | 200,000,000 | 32.2 | 714,126 | 3,754.3 / 3,811.9 | 35.2 / 37.7 | - | 2,096.6 |

Notes
- Concurrent SQLite reads in one process were ~8x *slower* than serial on this
  machine (likely the global allocator mutex), so the server reads serially and
  caches immutable results instead (`store/cache.py`).
- chunked 10M not measured yet.
