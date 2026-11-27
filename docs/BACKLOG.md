# Backlog

Open items as of 0.2.1, roughly by priority. Remove an item when it's done.

## UI polish (the user has just started using the UI on real data)

- Continue collecting and fixing UI feedback from real use (rollmann project,
  ~3800 runs on a Hetzner instance). Fixed in 0.2.1: metric options filtered to
  the filtered runs; hidden selection no longer blanks the plots; long run names.
- Performance with thousands of runs: every live update reloads the whole run
  list (throttled to 2 s), and `/api/runs` returns config + summary for all
  runs. Consider paging, or loading only the visible columns.
- Config keys in the Group/Color/column pickers are still project-wide; they
  could be filtered to the filtered runs like the metrics are.
- Run-vs-run code and config diff view (the CLI and Python have it:
  `chronist code diff`, `chronist.code_diff`).
- Checkpoint markers in line panels (spec: optional), ridgeline histogram
  view, file previews.
- Only tested in headless Chromium; check Firefox and Safari.

## Robustness / ops

- Real-workload testing: PyTorch DDP (every rank calls `init()`; there is no
  rank awareness yet), DataLoader workers forking after `init()`, notebooks.
- Schema migrations: `chronist.json` has a data version but nothing checks it
  yet; new tables are created with IF NOT EXISTS, but changed columns would
  need migrations.
- A safe backup command (SQLite online backup, while the server runs).
- Read-only share links still see the run's command line, cwd and git commit in
  the Meta tab (not counted as "code"); hide them if wanted.
- S3-compatible BlobStore backend (large checkpoints off the small server disk).
- `include_outside_imports` for code snapshots (spec addendum, optional).
- Multi-user accounts (explicitly deferred; single password for now).

## Known limitations

- The chunk size and pyramid fanout of the `chunked` store are constants;
  10M-step reads are ~2 ms full range, ~24 ms zoomed (bench/RESULTS.md).
- Peak server RSS is ~200–250 MB during exports (SQLite caches + pyarrow).
