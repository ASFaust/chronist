# chronist: notes for working on this repo

Self-hosted, single-user experiment tracker for ML research (a wandb/Aim
alternative), published on PyPI as `chronist`. The user-facing docs are in
`README.md`; this file holds what you need to *change* the code safely. Open
work is listed in `docs/BACKLOG.md`.

## Workflow (important)

- **Work on `dev`.** Commit and push there; `ci.yml` runs the tests on Python
  3.9–3.13 for every push to a non-main branch and every PR.
- **Every push to `main` publishes a release to PyPI**
  (`.github/workflows/publish.yml`): tests, next version, build, upload via
  trusted publishing (environment `pypi`), then tag `vX.Y.Z` and create a
  GitHub Release.
  - The bump is patch by default. `[minor]` / `[major]` in a commit's
    **subject line** bump more, and `[skip release]` there releases nothing.
    Markers in commit bodies are ignored on purpose.
  - Release = `git checkout main && git merge --ff-only dev && git push origin main && git checkout dev`.
    Only do this when the user asks for a release, and say that it publishes.
- The version comes from git tags (`setuptools-scm` writes `chronist/_version.py`,
  which is gitignored); never edit a version number by hand. Breaking changes:
  use `[minor]` while at 0.x.
- `gh` here can watch runs (`gh run watch <id> --repo ASFaust/chronist`) and read
  logs, but **cannot** create releases, re-run jobs or read branch protection
  (403). Pushes go over SSH.
- Commit messages end with the attribution lines the session gives you.

## Environment

- Python env: `.venv` in the repo (Python 3.9, made with `uv`; 3.9 matches the
  user's research venv). Set it up with:
  `uv venv -p 3.9 .venv && uv pip install -p .venv -e ".[dev]"`
  (+ `uv pip install -p .venv torch --index-url https://download.pytorch.org/whl/cpu`
  for the torch tests, which skip without it).
- Tests: `.venv/bin/pytest -q -n 16 tests` (~12 s; xdist). `tests/conftest.py`
  isolates `CHRONIST_CREDENTIALS`/`CHRONIST_TOKEN`; tests never touch real
  user files.
- Local build: `uv build` (plain `python -m build` fails in uv venvs: no pip).
- Don't `pkill -f "...chronist serve..."` with a pattern that also matches your
  own shell command line; it kills the shell (exit 144). Kill by PID, e.g.
  `ps aux | grep "port 7881" | grep -v grep | awk '{print $2}' | xargs -r kill`.

## Checking the UI (no build step; vanilla ES modules)

The UI is static files in `chronist/ui/static/` (Preact via `htm`, uPlot,
highlight.js, all vendored in `vendor/`). Edit and reload; there is no bundler.
To look at it, run a server on a scratch data dir with demo runs and use
Playwright **1.49.1** from a throwaway venv (newer versions don't support this
Ubuntu 20.04 host):

```
uv venv <scratch>/pw && uv pip install -p <scratch>/pw playwright==1.49.1 && <scratch>/pw/bin/playwright install chromium
chronist serve --data <scratch>/data --port 7881 --no-auth &
```

Create the page with `locale="en-US"`: the headless shell has an empty
`navigator.language`, and uPlot's `Intl.NumberFormat` throws without it. Log
`pageerror` and console errors. Screenshot and look at the result before
claiming a UI change works.

## Architecture

```
client (training process)                    server (the ONLY writer of the data dir)
 chronist.init/log -> ~/.chronist/runs/<uid>/  Starlette + uvicorn (chronist/server/app.py)
   msgpack event segments (rundir.py)           single writer thread: every write op (core.py)
 uploader thread / chronist sync ----HTTP---->  MetadataStore  meta.sqlite   (runs, config EAV, tags,
   (uploader.py; same path for both)                                          summaries, files, hist keys)
                                                MetricStore    metrics.chunked.sqlite (store/chunked.py)
                                                HistStore      hists.sqlite  (store/hist_store.py)
                                                FsBlobStore    blobs/ab/cd…  (blobs.py: code + file chunks)
```

- **Single writer**: the server takes an flock on `<data>/.lock`; all
  writes go through `Core.writer`. Reads use per-thread read-only SQLite
  connections. Reads are deliberately **serial**: concurrent SQLite reads in
  one process were ~8× slower here (allocator mutex). `store/cache.py` caches
  downsampled reads (`--cache-mb`).
- **MetricStore** (`store/base.py`): `write/read/keys/last/step_range/
  delete_range/delete_run/...`; downsampling is the backend's job; buckets lie on
  an absolute grid `step // width`. Backends: `chunked` (default: zstd chunks
  plus a downsampling pyramid) and `sqlite_rows` (simple, the test oracle). Any
  backend must pass `tests/test_store_conformance.py` and the randomized
  `tests/test_store_differential.py` (`CHRONIST_DIFF_SEEDS=60` for a deep run).
- **Idempotency**: first write of (run, key, step) wins; retries are always
  safe. `rewind=` (`delete_range`) is the only way to overwrite history.
- **Client buffer**: `log()` only appends to local msgpack segments. Record
  kinds: `l` (scalars), `h` (histograms), `f` (finish), `r` (resume),
  `m` (meta patch), `w` (rewind). Uploads send metrics first, then files on a
  time budget. A finished run is pruned locally once fully uploaded
  (`client/prune.py`; `CHRONIST_KEEP_LOCAL=1` keeps it).
- **Blobs** are content-addressed (sha256 of the uncompressed content;
  `codec.py` tags: `Z` zstd, `L` zlib, `R` raw). Code snapshots and file chunks
  share them. They are freed only by mark-and-sweep GC (`Core.op_gc`: daily, via
  `chronist gc` or `POST /api/admin/gc`, with a 24 h grace for uploads in
  flight). Never delete blobs directly (e.g. in delete_run).

## Design decisions to keep (agreed with the user)

- **Field names (0.2.0):** every data source except run fields has a prefix:
  `config.<key>`, `metric.<key>` (latest value); run fields are bare (`name`,
  `status`, `tags`, `steps`, `created`, `run_id`, ...). The same names are used
  in queries, `RunSet.config()` columns, UI columns, sorting, grouping and
  autocomplete. A sub-config called `config` is `config.config.<key>`.
  Unprefixed unknown names are **errors with a hint**, never silent empty
  results. Don't reintroduce rename-on-collision or per-run guessing (both were
  tried and removed).
- **Query language** (`query.py`) compiles to parameterized SQL and never
  evaluates Python. `contains` is a case-insensitive substring test on strings
  and an element test on lists.
- **Auth** (`server/auth.py`, `server/tokens.py`, `client/auth.py`):
  `chronist serve` requires a single password (scrypt hash in
  `<data>/auth.json`; prompted on first start). A login issues a per-device
  write token. Share links are read tokens, with code access optional
  (`code` flag). `chronist ui` is local and has no auth. Clients look up a token
  in this order: argument, `CHRONIST_TOKEN`, `~/.config/chronist/credentials.json`,
  then an interactive prompt. `tokens.json` is reloaded on change and written
  under a lock.
- **Code snapshots** (`client/code.py`): the root is resolved against the entry
  script's directory, not the cwd; file content is captured at `init()`.
- **Histograms**: records are subsampled over steps, never merged; the heatmap
  defaults to a robust value range (0.5–99.5%).
- **UI selection**: only selected runs visible under the current filter
  count; if there are none, panels plot the filtered runs.

## Specs and where things are

- `docs/SPEC.md` (original spec; the README is authoritative where they
  differ), `docs/CODE_SNAPSHOTS.md`, `docs/HISTOGRAMS_FILES.md` (both addenda;
  the README lists the deviations).
- Benchmarks: `bench/bench_store.py`, `bench/bench_hist.py`, results in
  `bench/RESULTS.md`.
- The user's downstream project ("rollmann") uses chronist through
  `chronist.runs(...).config()/history()` and a Hetzner server; breaking
  API changes affect it, so call them out.
