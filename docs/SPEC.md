# Chronist — Specification for Implementation

Chronist is a lightweight, self-hostable experiment tracker for ML research (a wandb/Aim alternative). Primary use: tracking training runs, comparing learning curves, and reading data back in Python to make custom plots. Single user, runs locally or on a small VPS (e.g. a small Hetzner instance).

Motivation / lessons from existing tools:
- **wandb**: limited plotting, self-hosting only on enterprise.
- **Aim**: good query/grouping UX, but hangs — backend and UI contend over the same index/lockfiles. **Chronist must never have two processes writing the same storage.** The server is the only writer.
- **TensorBoard**: slow with many runs, no filtering/grouping by config.

---

## 1. Architecture

Three components, shipped as **one pip package** (`pip install chronist`):

| Component | Tech | Role |
|---|---|---|
| Client | Python | `chronist.init()/log()/finish()`, local-first buffer, background upload, offline mode, read-back API |
| Server | Python (Starlette/FastAPI + uvicorn, orjson/msgpack, numpy) | REST API, ingestion, storage, queries, downsampling, serves the UI |
| UI | Static SPA (built once, bundled into the wheel) | Run table, search, plots. Served by the server as static files |

There is **no separate viewer**: the viewer *is* the server.

- **Local:** `chronist ui` → starts the server on `localhost`, no auth, opens the browser.
- **Remote:** `chronist serve --token ...` behind a reverse proxy (Caddy) that terminates TLS on 443. Chronist itself does not implement TLS. SSH-tunnel access (`ssh -L`) works automatically with a localhost-bound server.

Python is chosen for the server deliberately: the bottleneck is storage layout and query design, not request handling, and staying in Python keeps the single-`pip install` story. Only rewrite hot paths if profiling demands it.

---

## 2. Scale targets

- Low thousands of runs per project.
- Typical run: ~100k steps. Some runs: up to ~10M steps.
- ~20 metrics per run (more possible).
- Rough worst case: billions of points total → storage format matters (see §5).
- Must run comfortably on a small VPS: low idle CPU, modest RAM.

---

## 3. Data model

- **Project** → many **Runs**.
- **Run**: id, name, project, status (`running` / `finished` / `crashed`), created/updated timestamps, heartbeat timestamp, tags, notes.
- **Config**: arbitrary nested dict, **flattened to dotted keys** for storage and querying (`model.arch = "lstm"`, `optim.lr = 3e-4`). Values typed (str / int / float / bool / null).
- **Auto-captured metadata**: git commit, dirty flag + diff hash, command line, hostname, Python version, start time.
- **Metrics**: scalar time series `(key, step, value)`. Optional wall-time per point.
- **Summary**: last value (and step) per metric — usable in the run table and in queries.
- **Histograms** (post-v1, see §9): vector per step for a key.

---

## 4. Storage: interface first, backends swappable

Storage is split into two parts:

### 4.1 MetadataStore (fixed, SQLite)
Runs, flattened config, tags, notes, status, autocomplete index. Small relational data — always SQLite (WAL mode). The query language (§7) compiles to SQL against it. Not swappable.

### 4.2 MetricStore (swappable interface)
Time-series storage behind an interface so backends can be benchmarked and replaced later. **Downsampling is the backend's responsibility** — this is what lets both a naive backend and a pre-aggregated pyramid backend fit behind the same API.

```python
class MetricStore(Protocol):
    def write(self, run_id: int, points: Points) -> None: ...
        # batch of (key, step, value[, wall_time]). Idempotent:
        # duplicate (run, key, step) is ignored, never duplicated.

    def read(self, run_id: int, key: str,
             start: int | None = None, end: int | None = None,
             max_points: int | None = None) -> Buckets: ...
        # max_points=None -> raw points. Otherwise <= max_points buckets.

    def keys(self, run_id: int) -> list[str]: ...
    def last(self, run_id: int, keys: list[str]) -> dict[str, tuple[int, float]]: ...
    def delete_run(self, run_id: int) -> None: ...
    def flush(self) -> None: ...          # may be a no-op
    def maintain(self) -> None: ...       # compaction hook, may be a no-op
    def stats(self) -> StoreStats: ...    # bytes on disk, point count
```

`Buckets` is always the same shape: numpy arrays `step, min, max, mean, last, count`. A raw read is buckets with `count == 1`. UI and aggregation code only ever handle this one format.

### 4.3 Backends (in order)
1. **`SqliteRowStore`** — v1 default. One row per point, `WITHOUT ROWID`, PK `(run_id, key_id, step)`, `INSERT OR IGNORE`. Downsampling via `GROUP BY step / bucket_width` in SQL. Target ~100 lines.
2. **`ChunkedStore`** — later, when 10M-step runs hurt. See §5.
3. Optional experiments: Parquet-per-run, DuckDB, different codecs/chunk sizes.

### 4.4 Conformance test suite (every backend must pass)
- Duplicates ignored, including across `flush()` / `maintain()`.
- Out-of-order writes read back correctly.
- `read(max_points=N)` returns ≤ N buckets; min/max/mean/last/count are correct versus a raw read.
- Range boundaries (`start`, `end`) are inclusive/exclusive as documented and correct.
- `delete_run` removes everything.
- `last()` matches the final raw point.

### 4.5 Benchmark harness
Synthetic workloads: noisy learning curves at 100k and 10M steps, 20 metrics, 8 parallel writers, 5% retried duplicate batches. Report per backend:
- bytes per point on disk
- ingest throughput (points/s)
- read latency p50/p95: full-range plot, zoomed window, 50-run grouped plot
- peak RAM

Output: one comparison table.

---

## 5. Future backend design: ChunkedStore (not v1, for reference)

- `hot(run, key, step, value)`: recent tail, row-per-point, cheap appends.
- `chunks(run, key, level, start_step, end_step, blob)`: sealed blocks of ~4096 points. Steps delta-encoded; values byte-shuffled + zstd. Expected ~3–6 bytes/point.
- Background compactor (`maintain()`) seals full blocks from `hot` into chunks and builds a **downsampling pyramid**: each level aggregates 16 buckets of the level below into (min, max, mean, last, count). 10M steps → 625k → 39k → 2.4k.
- Read: choose the coarsest level giving ≥ ~2× requested `max_points` in the visible range; raw points only for narrow zooms.
- Idempotency: per (run, key) "sealed up to step" watermark; retried points below it are dropped. Rare genuine out-of-order points go to `hot` and get merged at next compaction.

---

## 6. Ingestion & client

### Client
- API: `run = chronist.init(project=..., name=..., config=..., tags=...)`, `run.log({...}, step=...)`, `run.finish()`. Also usable as context manager. Optional helpers to log every N steps or by time interval — never required.
- **Local-first:** every `log()` appends to a local per-run buffer (append-only file, e.g. msgpack/JSONL segments) before anything is sent. Never blocks training.
- Background thread uploads **batched, compressed** chunks. If the server is unreachable: keep buffering, retry with backoff.
- Steps are sent in increasing order per run.
- **Offline mode:** no server configured → only writes local files. `chronist sync [path]` uploads later. **Backlog upload uses this exact path.**
- Config via env vars / args: `CHRONIST_URL`, `CHRONIST_TOKEN`, `CHRONIST_DIR`.
- Heartbeat while running; `finish()` sets status; crash detection on server via missing heartbeat.
- Resume: re-attach to an existing run id and continue logging.

### Server ingestion
- Many concurrent client connections supported.
- All writes go through **one writer queue** (single writer to storage).
- Idempotent on `(run, key, step)` — retries and re-syncs are always safe.
- Auth: bearer tokens. Two kinds: **write** (clients) and **read-only** (sharing the UI). Local mode: auth off.

### REST API (sketch)
- `POST /api/runs` — create/resume run (config, metadata)
- `POST /api/runs/{id}/log` — batch of points
- `POST /api/runs/{id}/heartbeat`, `POST /api/runs/{id}/finish`
- `PATCH /api/runs/{id}` — tags, notes, name
- `DELETE /api/runs/{id}`
- `GET /api/runs?q=...` — filtered run list
- `GET /api/suggest?prefix=...` — autocomplete for keys/values
- `GET /api/runs/{id}/metrics/{key}?start&end&max_points` — downsampled series
- `POST /api/series` — batched multi-run/multi-key fetch, optional grouping/aggregation
- `GET /api/stream?...` — SSE for live updates (new points since step)
- `GET /api/export?q=...` — bulk export (parquet / arrow)

---

## 7. Query language

**No eval of Python expressions** (Aim does this; unsafe on a networked server). A small parser that compiles to parameterized SQL.

Example:
```
env = "seaquest" and model.arch = "lstm" and steps >= 20000
```

- Fields: any flattened config key; run fields (`name`, `status`, `tags`, `created`, `steps`); metric summaries (`last(loss) < 0.1`).
- Operators: `= != < <= > >=`, `in [...]`, `contains`, `exists`, `and`, `or`, `not`, parentheses.
- Same language in UI and Python read-back API.
- **Autocomplete**: index of all config key names, and distinct values per key. Suggest values only for low-cardinality keys (skip seeds, continuous floats beyond a threshold). Suggestions while typing both keys and values.

---

## 8. UI

Responsiveness is a hard requirement — used for fast experiment iteration; plot updates must feel instant.

### Performance rules
- **Server-side downsampling always**: request ≈ 2× pixel width of buckets, never raw 10M points.
- **Incremental live updates**: fetch only `since=last_step`, pushed via SSE.
- **Fast chart lib**: uPlot (or comparable canvas/WebGL). Not Plotly for the main line plots.
- Re-query on zoom with the new range (fetch finer resolution).

### Run table
- Search bar with the query language + autocomplete.
- Columns: name, status, steps, selectable config keys, selectable last-metric values. Sortable.
- Select runs, tags, notes, delete.

### Panels (the core improvement over wandb)
- Any set of metrics from any set of runs on **one** plot.
- X-axis selectable: step, wall time, relative time, or any other logged metric.
- **Group by** config key(s) → aggregate across runs (mean ± std, or min/max band, or median + quantiles). Typical: group by architecture, aggregate over seeds.
- Smoothing (EMA), log-scale x/y, min/max band from buckets.
- Color by config key.

### Saved views
- Entire view state (query, panels, grouping, axes) encoded in the URL → bookmarkable/shareable.

---

## 9. Histograms (right after v1)

- Logging a vector for a key at a step → histogram series.
- **Fixed bins per key** (chosen by client or fixed from first steps), so counts can be summed for downsampling. Store small vectors raw is an option; large tensors are binned client-side (don't upload full weight tensors every step).
- Separate interface methods (`write_hist` / `read_hist`) — do not bend the scalar `MetricStore` API.
- Display: step × value heatmap, ridgeline (TensorBoard-style), or quantile bands over time.

---

## 10. Python read-back API (first-class)

Main use is reading data back to make custom plots.

```python
import chronist
runs = chronist.runs('env = "seaquest" and model.arch = "lstm"', project="bptt-free")
runs.config()                               # DataFrame: one row per run, flattened config
df = runs.history(["loss", "return"])       # long DataFrame: run_id, key, step, value
df = runs.history(["loss"], max_points=2000)  # downsampled
```
- Returns polars (and/or pandas) DataFrames.
- Works against a remote server (via REST) or directly on a local data dir.

---

## 11. v1 scope

**In:**
- Client: init/log/finish, local buffer, background batched upload, offline mode, `chronist sync`, heartbeat, resume, auto git/command capture.
- Server: REST API, single-writer ingestion, idempotent writes, tokens (write + read-only), MetadataStore, `SqliteRowStore`, query language + autocomplete index, downsampled reads, SSE.
- UI: run table with search + autocomplete; line panels with multi-run/multi-metric overlay, selectable x-axis, grouping + aggregation, smoothing, log scale; URL-encoded views; tags/notes/delete.
- Python read-back API.
- MetricStore conformance tests + benchmark harness.
- CLI: `chronist ui`, `chronist serve`, `chronist sync`.

**Right after v1:** histograms; `ChunkedStore`.

**Out (for now):** importers from wandb/TensorBoard, media (images/audio), artifacts/checkpoints, multi-user accounts, alerts, sweeps.

---

## 12. Suggested build order

1. Package skeleton, `MetricStore` protocol, `SqliteRowStore`, conformance tests.
2. Benchmark harness.
3. MetadataStore + query language parser → SQL + autocomplete index.
4. Server: ingestion endpoints, single writer, auth, read endpoints with downsampling.
5. Client: local buffer, uploader, offline mode, `sync`.
6. Python read-back API.
7. UI: run table + search, then panels, grouping, URL state, SSE live updates.
