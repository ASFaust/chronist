# chronist

A lightweight, self-hostable experiment tracker for ML research: log training
runs, compare learning curves, and read data back in Python for custom plots.
One `pip install`, single user, runs on a laptop or on a small VPS.

The design rules come from problems with other tools (see [docs/SPEC.md](docs/SPEC.md)):
the server is the **only** process that writes storage, plots are always
downsampled on the server, and the query language never evals Python.

```
pip install -e .            # from this repo; extras: [pandas], [polars], [export], [dev]
```

## Logging

```python
import chronist

run = chronist.init(project="bptt-free", name="lstm-seed0",
                    config={"model": {"arch": "lstm"}, "optim": {"lr": 3e-4}, "seed": 0},
                    tags=["baseline"])
for step in range(100_000):
    loss = ...
    run.log({"loss": loss, "train": {"acc": acc}}, step=step)   # -> keys "loss", "train/acc"
    run.log({"eval/return": ret}, step=step, every=1000)         # optional throttling
run.finish()                                                     # or: with chronist.init(...) as run:
```

- `log()` only appends to a local file (`$CHRONIST_DIR/runs/<uid>/`, default
  `~/.chronist`) and never touches the network, so it cannot block training.
  A background thread uploads batched, gzip'd msgpack chunks. If the server
  is down, data stays buffered and the uploader retries with backoff.
- No server configured → offline: only local files. Upload later with
  `chronist sync` (same code path as the live uploader).
- Config: `CHRONIST_URL`, `CHRONIST_TOKEN`, `CHRONIST_DIR`, `CHRONIST_PROJECT`.
- Captured automatically: git commit/branch/remote, dirty flag + diff hash
  (the diff itself is kept in the run dir as `diff.patch`), command line, cwd,
  hostname, Python version, start time.
- Heartbeats while running; `finish()` sets the status; an uncaught exception
  marks the run `crashed`; a run that stops heartbeating is marked `crashed` by
  the server (default after 180 s).
- Resume: `chronist.init(resume="<uid>")` re-attaches and continues at the next step.

## Viewing

**Local:** `chronist ui` starts the server on `localhost:7070` (no auth) and
opens the browser. It also watches `~/.chronist` and live-syncs offline runs,
so locally you don't need `CHRONIST_URL` at all.

**Remote:** on the VPS:

```
chronist serve --host 127.0.0.1 --port 7070 --token <write-token> --read-token <read-token>
```

Put Caddy in front for TLS (`Caddyfile`):

```
chronist.example.org {
    reverse_proxy 127.0.0.1:7070
}
```

Without `--token` on a non-localhost bind, the server generates a write and a
read token, stores them in `<data>/tokens.json` and prints them
(`chronist tokens --new read` makes more). A localhost-bound server without
tokens is reachable through `ssh -L 7070:localhost:7070 vps`. Share a
read-only view as `https://chronist.example.org/?token=<read-token>#...`.

Clients: `export CHRONIST_URL=https://chronist.example.org CHRONIST_TOKEN=<write-token>`.

### UI

- **Run table:** query bar with autocomplete for keys and values, sortable
  columns, any config key or `last(metric)` as a column, shift-click range
  select, bulk tag/delete, and a run drawer (config, summary, git/command
  metadata, tags, notes, rename, delete).
- **Panels:** any metrics from any runs on one plot. X axis: step, wall time,
  relative time, or another metric. Group by config keys with mean ± std,
  min–max, or median + IQR bands. EMA smoothing, log x/y, min/max band of each
  downsampled bucket, color by config key. Drag to zoom (re-queries at finer
  resolution), double-click to reset.
- **Live:** running runs stream in over SSE; panels fetch only the new tail.
- **Views:** the whole view state (project, query, selection, columns, panels,
  zoom) is in the URL, so you can bookmark or share it.

With nothing selected, panels plot the first 20 matching runs. Grouped panels
aggregate over all matching runs (up to 1000).

## Query language

```
env = "seaquest" and model.arch = "lstm" and steps >= 20000
(optim.lr < 1e-3 or tags contains "baseline") and not model.dropout exists
last(loss) < 0.1 and status in [running, finished] and created > 2026-11-01
```

- Fields: any flattened config key; run fields `id name status tags notes
  created updated heartbeat steps`; metric summaries `last(<key>)`. Use
  backticks for odd key names or keys that collide with run fields: `` `name` = "x" ``.
- Operators: `= != < <= > >=`, `in [...]`, `not in [...]`, `contains`, `exists`,
  `and`, `or`, `not`, parentheses. Bare words are strings (`arch = lstm`).
- `a != x` means `not (a = x)`, so it also matches runs without `a`.

It compiles to parameterized SQL. Nothing is evaluated.

## Reading data back

```python
import chronist
runs = chronist.runs('env = "seaquest" and model.arch = "lstm"', project="bptt-free")
runs.config()                                  # one row per run: id, name, status, ..., flattened config
runs.config(summary=True)                      # + last(<metric>) columns
df = runs.history(["loss", "return"])          # long: run_id, run, key, step, value, wall_time
df = runs.history(["loss"], max_points=2000)   # downsampled: value = bucket mean, + min, max, count
```

Returns pandas by default (`dataframe="polars"` or `"dict"` for the others).
It reads from `CHRONIST_URL` if set, otherwise directly from the local data
dir (`~/.chronist/data`, read-only, safe while a server is running). Pass
`url=` or `data_dir=` to choose. Bulk export over HTTP:
`GET /api/export?project=..&q=..&format=parquet|arrow|csv`.

## CLI

| command | |
|---|---|
| `chronist ui [--data DIR] [--port N] [--no-browser] [--no-watch]` | local server + UI, no auth |
| `chronist serve [--host H] [--port N] [--token T] [--read-token R] [--no-auth]` | server for remote use |
| `chronist sync [PATH...] [--url U] [--token T]` | upload offline/backlogged runs |
| `chronist tokens [--new write\|read] [--revoke T]` | manage server tokens |

## Architecture

```
client (training process)                 server (the only writer)
  run.log() ─► local msgpack segments        Starlette + uvicorn
                 │                              ├─ REST /api/*, SSE /api/stream, static UI
  uploader thread ─► POST /api/runs/{id}/log ──►├─ single writer thread (queue)
  chronist sync  ─┘   (gzip msgpack, retried)   │     ├─ MetadataStore  (SQLite: runs, config EAV, tags, summaries)
                                                │     └─ MetricStore    (swappable; v1: SqliteRowStore)
                                                └─ readers: per-thread read-only SQLite connections (WAL)
```

- `chronist/store/` holds the `MetricStore` protocol and backends. Every
  backend must pass `tests/test_store_conformance.py`. Downsampling is the
  backend's job. Buckets lie on an absolute grid (`step // width`), so runs and
  zoom levels line up.
- `chronist/meta.py`: runs, flattened typed config, tags, summaries, autocomplete.
- `chronist/query.py`: query parser → SQL, completion context.
- `chronist/server/`: `core.py` (writer, events, series/grouping), `app.py` (HTTP).
- `chronist/client/`: local buffer format (`rundir.py`), uploader, `init/log/finish`, sync.
- `chronist/ui/static/`: the SPA. Preact + htm and uPlot are vendored as ES
  modules, so there is no build step.

## Development

```
uv venv -p 3.9 .venv && uv pip install -p .venv -e ".[dev]"
.venv/bin/pytest                      # store conformance, query, server, client end-to-end
.venv/bin/python bench/bench_store.py --scenario 100k,grouped,1M
```

Benchmark results are in [bench/RESULTS.md](bench/RESULTS.md).
