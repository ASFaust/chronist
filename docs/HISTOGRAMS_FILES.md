# Chronist — Histograms & Files Specification

Addendum to the main Chronist spec and the code-snapshot spec. Adds two logged data types:

1. **Histograms** — distributions of vectors/tensors over time (weights, gradients, activations, per-episode returns, ...).
2. **Files** — explicit files attached to a run, e.g. model weight checkpoints, saved plots, eval outputs.

> **Supersedes §9 of the main spec** ("fixed bins per key"). Histograms now use per-step adaptive ranges; downsampling over steps is done by step subsampling instead of summing bins (see §1.4).

---

# Part 1 — Histograms

## 1.1 Client API

```python
# explicit
run.log({"weights/layer1": chronist.Histogram(w)}, step=t)
run.log({"grad/layer1": chronist.Histogram(g, bins=128)}, step=t)

# auto-detection: arrays/tensors with more than one element become histograms
run.log({"loss": loss, "returns/per_env": returns_np}, step=t)
```

Accepted inputs: Python lists/tuples of numbers, NumPy arrays, PyTorch tensors (any device), JAX arrays. Arbitrary shape — flattened.

**Auto-detection rules** in `log()`:
- Python number, 0-d array/tensor, or array/tensor with exactly 1 element → scalar.
- Array/tensor/list with > 1 element → histogram with the default settings for that key.
- Can be disabled: `chronist.init(auto_histograms=False)` → non-scalars raise `TypeError` instead.

Mixed scalars and histograms in one `log()` call at the same step is allowed.

## 1.2 Bin configuration

**Default: 64 bins.** Changeable at three levels (most specific wins):

```python
chronist.init(histogram_bins=64)                       # run-wide default
run.configure_histograms("grad/*", bins=128)           # per key pattern (glob)
run.configure_histograms("weights/*", range=(-1, 1))   # fixed range, see below
chronist.Histogram(x, bins=32)                         # per call
```

`chronist.Histogram(values, bins=None, range=None, raw=False)`:
- `bins`: int, 1..1024 (hard cap 1024; `ValueError` above).
- `range`: `None` (default, adaptive) or `(lo, hi)` fixed.
- `raw`: store the raw values instead of bins (see §1.3).

**Binning modes:**
- **Adaptive (default):** uniform bins over `[min, max]` of *this step's* finite values. Handles drifting distributions (weights growing, gradients shrinking) without losing resolution.
- **Fixed range:** uniform bins over `[lo, hi]`, plus **underflow and overflow counters** for values outside. Useful when the user wants identical edges across steps/runs.
- Constant input (`min == max`): single-value record (all mass at that value), no division by zero.

## 1.3 What is recorded per histogram step

Computed **client-side**, on the tensor's device where possible (`torch.histc` / `torch.aminmax` on GPU), so large tensors are never copied to CPU or sent over the network.

Per (run, key, step) record:
| Field | Notes |
|---|---|
| `step`, `wall_time` | |
| `count` | number of finite values |
| `nan_count`, `inf_count` (`+inf`, `-inf` separate) | excluded from bins and stats |
| `min`, `max`, `mean`, `std` | exact, over finite values |
| `lo`, `hi` | bin range (= min/max in adaptive mode) |
| `counts` | `bins` counts (uint32; uint64 if count exceeds 2^32) |
| `underflow`, `overflow` | fixed-range mode only |
| `raw` | optional, raw float32 values, only if `raw=True` |

**Raw mode** (`raw=True`): exact values stored (float32), capped at 10,000 values per step (`ValueError` above, with a hint to use bins). Useful for small exact vectors like per-environment returns. Stats and bins are still computed so all views work uniformly.

## 1.4 Downsampling & rendering (server-side)

- **Over steps: subsample, don't merge.** If a range contains more histogram steps than requested (`max_steps`), return an evenly strided subset (always including the first and last step). Each returned record is exact; no approximate merging needed.
- **Rendering a heatmap:** histograms at different steps have different edges (adaptive mode). The server re-bins each returned record onto a **common value grid** for the view: `[global_lo, global_hi]` of the visible records (or user-set y-range) with `H` rows (≈ panel pixel height, default 128). Re-binning assumes uniform mass within each source bin (overlap-proportional splitting). Result: a dense `steps × H` matrix (float32) → sent to the UI.
- **Quantiles** (for quantile-band view): approximated per step from the CDF of the bins (linear interpolation within bins). Exact for raw records.
- Summary stats (`mean`, `std`, `min`, `max`, `nan_count`) are exact and can be plotted as line series directly from the histogram records — **no need to also log them as scalars**.

## 1.5 Storage interface

Separate swappable interface (do not bend `MetricStore`):

```python
class HistStore(Protocol):
    def write(self, run_id: int, records: list[HistRecord]) -> None: ...
        # idempotent on (run, key, step)
    def read(self, run_id: int, key: str,
             start: int | None = None, end: int | None = None,
             max_steps: int | None = None) -> list[HistRecord]: ...
        # strided subsample if more than max_steps
    def keys(self, run_id: int) -> list[str]: ...
    def last(self, run_id: int, key: str) -> HistRecord | None: ...
    def delete_run(self, run_id: int) -> None: ...
    def stats(self) -> StoreStats: ...
```

**v1 backend `SqliteHistStore`**: table `hist(run_id, key_id, step, wall_time, count, nan_count, posinf_count, neginf_count, min, max, mean, std, lo, hi, underflow, overflow, counts BLOB, raw BLOB NULL)`, `WITHOUT ROWID`, PK `(run_id, key_id, step)`, `INSERT OR IGNORE`. `counts` as little-endian uint32 array, zstd-compressed if beneficial.

Size estimate: 64 bins → ~256 B counts + ~100 B header ≈ 300–400 B/step before compression. 10 keys × 10k logged steps ≈ 40 MB per run — users should log histograms every N steps (document this; the client helper `log_every` from the main spec applies).

Add `HistStore` to the conformance-test and benchmark harnesses.

## 1.6 Ingestion & API

- Histograms travel in the same batched log upload as scalars (msgpack), as a separate record type. Same idempotency and offline/sync path.
- `GET /api/runs/{id}/hist/{key}?start&end&max_steps` → records.
- `POST /api/hist/render` → `{runs, key, start, end, max_steps, rows, y_range?, mode: "heatmap"|"quantiles"}` → rendered matrix / quantile series. Supports multiple runs (for side-by-side panels).

## 1.7 UI

Histogram panel modes:
- **Heatmap** (default): x = step, y = value, color = density. Linear/log color scale. Overflow/underflow and NaN/Inf shown as marker rows if non-zero.
- **Ridgeline** (TensorBoard-style offset curves) for a subset of steps.
- **Quantile bands**: median line + shaded bands (e.g. 5–95, 25–75), overlayable across runs and groupable like line panels.
- **Single step**: bar histogram at a selected step with a step slider; overlay several runs at the same step.
- **Stats lines**: mean/std/min/max as line series.

Zoom re-queries with the new range (like line panels).

## 1.8 Python read-back

```python
runs.histograms("grad/layer1")                 # list/long DataFrame of records
runs.histograms("grad/layer1", max_steps=200)  # subsampled
run.histogram("grad/layer1", step=5000)        # nearest step at or before
```
Returns records with `edges` (computed from `lo/hi/bins`) and `counts` as NumPy arrays; a convenience `to_dense(rows=128)` returns the re-binned `steps × rows` matrix (same algorithm as the server).

---

# Part 2 — Files

## 2.1 Client API

```python
run.save_file("checkpoints/model.pt", step=t)                 # name defaults to basename
run.save_file("model.pt", name="model", step=t,
              aliases=["best"], metadata={"val_loss": 0.21})
run.save_file("plots/curve.png", kind="plot")
run.save_file("ckpt.pt", step=t, move=True)                   # move instead of copy
run.save_object(model.state_dict(), name="model", step=t)     # serialize directly (torch.save / pickle)
```

Files are **not** passed through `log()` (they're large and have different semantics).

Parameters:
- `name`: logical name, groups versions (default: file basename). A run can have many versions of the same name at different steps.
- `step`: optional int.
- `kind`: free-form string, suggested values `checkpoint`, `plot`, `data`, `output`, `other` (default `other`, or `checkpoint` for `save_object`).
- `aliases`: list of strings, e.g. `best`. Each alias points to exactly one version per (run, name); assigning moves it. **`latest` is maintained automatically.**
- `metadata`: small JSON dict (e.g. eval metrics at that checkpoint).
- `move`: move the file into the local buffer instead of copying (zero extra disk, file disappears from original path).

## 2.2 Capture semantics

The user often overwrites the same path (`model.pt`) every N steps, so **content must be captured at call time**:
- `save_file` copies (or moves) the file into the local run buffer **synchronously**, then returns. Hashing and upload happen in the background.
- For large files, use a fast copy (`os.link` hardlink if same filesystem and user opts in via `link=True` — only safe if the user writes new files rather than modifying in place; document the caveat; default is a real copy, `shutil.copyfile` with `copy_file_range`/`sendfile`).
- `save_object` serializes straight into the buffer (no temp file in the working tree).
- Paths are resolved to absolute at call time (cwd-robustness, as in the code spec).

## 2.3 Storage model

Files reuse the content-addressed **`BlobStore`** from the code-snapshot spec, extended with chunking:

- A file is split into **fixed-size chunks (default 8 MiB)**. Each chunk is a blob keyed by `sha256` (stored zstd-compressed if it compresses well; checkpoints usually don't — store raw if compression saves < 5%).
- A **file manifest**: `{name, step, kind, size, sha256 (whole file), chunks: [hash, ...], aliases, metadata, created}`.
- Identical files (and identical chunks) are stored once.
- File index lives in the MetadataStore: table `files(run_id, name, version_id, step, kind, size, sha256, manifest_hash, metadata JSON, created)` and `file_aliases(run_id, name, alias, version_id)`.

`BlobStore` backends:
- v1: local filesystem directory (`blobs/ab/cdef...`), configurable path (`--blob-dir`) so it can live on a separate volume.
- Later: S3-compatible backend (e.g. Hetzner Object Storage / Storage Box), so large checkpoints don't fill the small server's disk.

## 2.4 Upload protocol

Reuses the code-snapshot blob protocol:
1. Client hashes chunks in the background after capture.
2. `POST /api/blobs/missing` → server returns missing chunk hashes.
3. `PUT /api/blobs/{hash}` per missing chunk (server verifies hash; rejects mismatch).
4. `POST /api/runs/{id}/files` with the file manifest → server checks all chunks exist, registers the version, updates aliases.
- **Resumable**: interrupted uploads continue with the missing chunks only.
- Chunks are well below typical reverse-proxy body limits (configure Caddy accordingly; document it).
- Upload bandwidth limit option on the client (`CHRONIST_UPLOAD_MBPS`) so checkpoint uploads don't starve metric uploads; metrics always have priority in the uploader queue.
- Offline mode: chunks + manifest stay in the local run dir; `chronist sync` uploads them.

## 2.5 Retention

Checkpoints accumulate fast; the server disk is small.

- Per run / per name policy, set by client or server default:
  ```python
  run.configure_files("model", keep_last=3, keep_every=100_000)  # steps
  ```
  Always kept: versions with any alias (`latest`, `best`, user aliases).
- Server-side: `--max-blob-storage 50GB` quota → warn at 80%, reject new file uploads at 100% (metrics keep working).
- Pruned versions are removed from the index; blobs are freed by **mark-and-sweep GC** in `maintain()` (collect all chunk/blob hashes referenced by remaining file manifests and code manifests; delete the rest). Never delete blobs inside `delete_run` or prune directly.
- `chronist gc [--dry-run]` CLI to run GC manually and report reclaimable space.

## 2.6 Download / read-back

```python
run = chronist.run("run-id")
run.files()                                   # DataFrame: name, step, kind, size, aliases, metadata
path = run.download("model", alias="best", dest="./ckpt/")
path = run.download("model", step=200_000)
obj  = run.load_object("model", alias="latest")   # inverse of save_object
```
- CLI: `chronist pull <run_id> <name> [--step N | --alias A] [--dest DIR]`.
- Downloads verify the whole-file `sha256`. Chunked range download, resumable.
- Local cache by `sha256` (`~/.cache/chronist/blobs`) to avoid re-downloading.

## 2.7 Server API

- `GET /api/runs/{id}/files` → versions
- `POST /api/runs/{id}/files` → register version (manifest)
- `PATCH /api/runs/{id}/files/{name}/aliases` → set/move aliases
- `DELETE /api/runs/{id}/files/{name}/{version}` → remove version (blobs freed by GC)
- `GET /api/files/{version_id}/download` → streamed file (assembled from chunks), supports HTTP `Range`
- Blob endpoints shared with code snapshots.
- Auth: download requires a read token; upload/delete a write token. Server-side max file size (`--max-file-size`, default 20 GB).

## 2.8 UI

- Run page "Files" tab: list grouped by name → versions with step, size, kind, aliases, metadata; download button; delete; set alias.
- In line panels: optional markers at steps where a checkpoint exists (hover → metadata, click → download).
- Previews (later, overlaps with "media" which is out of v1 scope): images (`png/jpg/svg`), text/JSON/YAML, CSV head.

---

# Scope

**v1 (this addendum):**
- Histograms: `chronist.Histogram`, auto-detection, default 64 bins configurable (run / key pattern / call), adaptive + fixed range, NaN/Inf counting, raw mode, client-side GPU computation, `HistStore` + `SqliteHistStore`, render endpoint, heatmap + quantile + single-step + stats views, read-back API.
- Files: `save_file` / `save_object`, call-time capture, chunked content-addressed storage on filesystem `BlobStore`, resumable upload/download, aliases (`latest` automatic), metadata, retention policy + quota, mark-and-sweep GC, Files tab, `chronist pull`, `chronist gc`.

**Later:** S3-compatible BlobStore, file previews, ridgeline view polish, symlog bin scale for histograms.

---

# Tests

**Histograms**
- Auto-detection: scalar vs 1-element vs multi-element; `auto_histograms=False` raises.
- Bin config precedence: call > key pattern > run default > 64; cap at 1024.
- Adaptive vs fixed range; underflow/overflow counts; constant input; all-NaN input; mixed NaN/Inf counted and excluded.
- GPU tensor stays on device (no full host copy) — check via mocking / memory.
- Idempotency on (run, key, step); retries don't duplicate.
- Subsampling returns ≤ max_steps, includes first and last.
- Re-binning conserves total mass (sum of rendered column = count within y-range).
- Quantiles exact for raw mode, within one bin width for binned mode.
- Offline log + sync produces identical records.

**Files**
- Overwriting the source file right after `save_file` does not change the stored version.
- `move=True` removes source; `save_object` round-trips via `load_object`.
- Chunking: file of size exactly N×chunk, N×chunk+1, 0 bytes.
- Dedup: saving the same file twice stores chunks once.
- Interrupted upload resumes with only missing chunks; server rejects chunk with wrong hash.
- Aliases: `latest` auto-moves; `best` moves on reassignment; aliased versions survive retention.
- Retention `keep_last` / `keep_every` correct; GC frees exactly unreferenced blobs and never blobs still referenced by code snapshots or other runs.
- Quota: uploads rejected at 100%, metric ingestion unaffected.
- Download verifies sha256; Range requests work; cache hit avoids network.
