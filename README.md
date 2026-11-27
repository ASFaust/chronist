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
  Resuming from an earlier checkpoint:
  `chronist.init(resume="<uid>", rewind=5000)` discards everything the run
  logged at step ≥ 5000 and continues there. Without `rewind`, values at
  already-logged steps are ignored (first write wins), and the client warns
  the first time you log one.
- Cleanup: once a finished run is fully on the server (points, finish record,
  code), its local buffer and now-unneeded local blobs are deleted, after
  `finish()` or `chronist sync`. Set `CHRONIST_KEEP_LOCAL=1` to keep them;
  `chronist prune` removes older ones.

## Histograms

```python
run.log({"weights/layer1": layer.weight}, step=t)               # >1 element -> histogram (64 bins)
run.log({"grad/l1": chronist.Histogram(g, bins=128)}, step=t)  # explicit
run.log({"returns": chronist.Histogram(r, raw=True)}, step=t)  # exact values (<= 10k)
run.configure_histograms("weights/*", range=(-1, 1))           # fixed range + under/overflow
chronist.init(histogram_bins=32, auto_histograms=False)         # run default / strict mode
```

- Each record holds the exact stats over finite values (count, mean, std,
  min, max), NaN/±Inf counts, and bin counts. The bin range is adaptive (this
  step's min/max) by default, or a fixed range. Torch tensors are reduced on
  their own device; only the stats and bins come back to the host.
- Bins: `Histogram(..., bins=)` > the last matching `configure_histograms`
  pattern > `init(histogram_bins=)` > 64 (max 1024).
- About 300 B per record: log them every N steps (`run.log(..., every=100)`).
- Views: a heatmap (step × value) with a robust value range (0.5–99.5%
  quantiles, so heavy tails don't flatten it; "full range" toggles min–max),
  quantile bands, stats lines, and a single step with a slider. Over many
  steps the server subsamples whole records evenly (never merges them).
- Read-back: `runs.histograms("grad/l1", max_steps=200)` (DataFrame with
  `edges`/`counts` arrays), `run.histogram("grad/l1", step=5000)` (nearest step
  at or before), `run.histograms(key).to_dense(rows=128)`.
- A rewind also discards histogram records.

## Files (checkpoints, plots, outputs)

```python
run.save_file("checkpoints/model.pt", step=t, aliases=["best"], metadata={"val_loss": 0.21})
run.save_file("plots/curve.png", kind="plot")
run.save_file("ckpt.pt", step=t, move=True)              # or link=True (hard link; never modify in place)
run.save_object(model.state_dict(), name="model", step=t) # torch.save if torch is loaded, else pickle
run.configure_files("model", keep_last=3, keep_every=100_000)

r = chronist.run(42)
r.files()                                           # name, version_id, step, kind, size, aliases, metadata
r.download("model", alias="best", dest="./ckpt/")   # default alias: latest
r.load_object("model", step=200_000)
```
```
chronist pull <run> <name> [--alias A | --step N | --version V] [--dest DIR]
```

- Content is captured when you call `save_file`, so overwriting `model.pt`
  afterwards is fine. A background thread splits it into 8 MiB
  content-addressed chunks (compressed only if that saves ≥ 5%). Identical
  files and chunks are stored once.
- Uploads only send chunks the server lacks, are resumable, and run after
  metrics in each upload round, so checkpoints never delay metrics.
  `CHRONIST_UPLOAD_MBPS` caps the bandwidth. Offline runs upload with
  `chronist sync`.
- `latest` moves automatically; other aliases move when reassigned (UI or
  `PATCH .../files/<name>/aliases`). Retention keeps the newest `keep_last`
  versions plus the first in each `keep_every`-step window, and aliased
  versions are always kept. Dropped versions' chunks are freed by GC.
- Server limits: `--max-blob-storage 50GB` (warns at 80%; at 100% new file
  uploads are refused, but metrics keep working), `--max-file-size` (default
  20 GB), `--blob-dir` to put blobs on another volume.
- Downloads verify sha256 and cache chunks in `~/.cache/chronist`
  (`CHRONIST_CACHE`), so repeated or interrupted downloads only fetch what's
  missing. `GET /api/files/<version>/download` serves the whole file with
  HTTP Range support. Read-only share tokens can download files.
- Reverse proxies: chunk uploads are ≤ 8 MiB plus a few bytes; with nginx, set
  `client_max_body_size 16m` (Caddy has no default limit).
- The UI's run drawer has a **Files** tab: versions grouped by name, aliases,
  metadata, download, and for owners set/move alias and delete.

Differences from the spec in `docs/HISTOGRAMS_FILES.md`:
- Blob GC is the server's mark-and-sweep (daily, `chronist gc`) over code and
  file manifests. It isn't in `MetricStore.maintain()`, and nothing deletes
  blobs directly.
- Quota is checked when a file upload starts (`blobs/missing` with
  `purpose: "file"`), not on every blob PUT, because blob uploads are shared
  with code snapshots.
- File chunks are served via `/api/files/<version>/chunk/<i>`, not the blob
  endpoint, so read-only share links work without code access.

## Code snapshots

Every run records exactly which code ran, including uncommitted and untracked
files (details: [docs/CODE_SNAPSHOTS.md](docs/CODE_SNAPSHOTS.md)).

```python
chronist.init(code="auto")      # default: git root, else nearest pyproject/setup.py, else the script's dir
chronist.init(code="git")       # git root; error if not in a repo
chronist.init(code="../..")     # relative to the entry script's directory (not the cwd)
chronist.init(code=None)        # off
chronist.init(code=chronist.Code(root=2, exclude=["data/"], max_total_mb=50, env_vars=["CUDA_VISIBLE_DEVICES"]))
```

- Precedence: the `code=` argument, then `CHRONIST_CODE`, then
  `[tool.chronist] code = ".."` in `pyproject.toml` or `.chronist.toml`
  (relative to that file), then auto.
- Selection: `git ls-files` (tracked + untracked-not-ignored, including
  submodules), `.gitignore` via pathspec outside git, `.chronistignore`, and a
  built-in deny list. Checkpoints, data, venvs and secrets like `.env`, `*.key`
  and `id_rsa` are never included. Files over 1 MB, binary files, and anything
  past 50 MB total are skipped, and every skip is recorded with its reason.
- Content is hashed and copied into `~/.chronist/blobs` (content-addressed,
  compressed) inside `init()`, so later edits or `os.chdir` don't affect it.
  Uploads send only the blobs the server lacks, so 1000 runs of the same repo
  cost almost nothing.
- Also stored: the package list, Python, platform, host, GPU, torch/CUDA
  versions, the command line, and git info. Env vars are recorded only if listed
  in `env_vars`.
- Modules imported from outside the code root (e.g. via `sys.path.append`) are
  reported at init and finish and flagged ⚠ in the UI.

```
chronist checkout <run> [dir]        # restore the tree (+ CHRONIST_RUN.json)
chronist code diff <run_a> <run_b>   # changed files (unified diff) + config diff
chronist snapshot [--code ..] [--run ID]
```
```python
r = chronist.run(42)                 # or a uid
r.code.files(); r.code.read("src/model.py"); r.code.checkout("/tmp/x")
chronist.code_diff(41, 42)
```

The UI's run drawer has a **Code** tab: a file browser with a
syntax-highlighted viewer, root/mode info, the skip report, and environment
and package lists.

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

Without `--token` on a non-localhost bind, the server generates a write
token, stores it in `<data>/tokens.json` and prints it. A localhost-bound
server without tokens is reachable through `ssh -L 7070:localhost:7070 vps`.

**Sharing:** the UI's **Share** button (shown to write-token holders) creates
a read-only link to the current view. **Share code as well** is off by
default. Without it, the link's token can't read code snapshots (manifests,
files, blobs, snapshot metadata) through the UI, the API or Python. Links are
listed and revocable in the same dialog and take effect immediately. CLI
equivalent: `chronist tokens --new read [--code] [--label L]`, `chronist tokens`,
`chronist tokens --revoke <id>`. A server in local no-auth mode can't share,
because anyone who can reach it sees everything.

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
`GET /api/export?project=..&q=..&format=parquet|arrow|csv` (streamed in
100k-point pages; server memory stays flat regardless of size). Raw reads
over HTTP are paged (≤ 250k points per response, `next_start` for the next
page); read-back handles the paging for you.

## CLI

| command | |
|---|---|
| `chronist ui [--data DIR] [--port N] [--no-browser] [--no-watch]` | local server + UI, no auth |
| `chronist serve [--host H] [--port N] [--token T] [--read-token R] [--no-auth]` | server for remote use |
| `chronist sync [PATH...] [--url U] [--token T]` | upload offline/backlogged runs |
| `chronist tokens [--new write\|read] [--code] [--label L] [--revoke ID]` | manage server/share tokens |
| `chronist checkout <run> [dir] [--force]` | restore a run's code snapshot |
| `chronist code diff <a> <b> [--stat]` | code + config diff of two runs |
| `chronist snapshot [--code C] [--run ID]` | standalone snapshot (optionally attach to a run) |
| `chronist prune [PATH...] [--url U] [--older-than DAYS] [--dry-run]` | delete local buffers already on a server |
| `chronist gc [--url U \| --data DIR] [--vacuum]` | server: drop unreferenced blobs, vacuum (also runs daily) |

The server takes an exclusive lock on its data dir, so a second server, or
`chronist gc --data` while a server is running, refuses to start. Use
`chronist gc --url` to run GC inside the server.

## Architecture

```
client (training process)                 server (the only writer)
  run.log() ─► local msgpack segments        Starlette + uvicorn
                 │                              ├─ REST /api/*, SSE /api/stream, static UI
  uploader thread ─► POST /api/runs/{id}/log ──►├─ single writer thread (queue)
  chronist sync  ─┘   (gzip msgpack, retried)   │     ├─ MetadataStore  (SQLite: runs, config EAV, tags, summaries)
                                                │     └─ MetricStore    (swappable; default: ChunkedStore)
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
