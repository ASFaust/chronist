# Chronist — Code Snapshot Feature Specification

Addendum to the main Chronist spec. Adds automatic capture of the code state (and environment) of each run for reproducibility, plus restore and diff.

Goals:
- Every run records exactly which code ran — including uncommitted changes and untracked files, which a git commit hash alone misses.
- Snapshots are cheap: deduplicated across runs, so 1000 runs of the same repo cost almost nothing.
- Predictable root selection, robust against scripts that change the working directory.
- Restore any run's code (`chronist checkout`) and diff code + config between runs.

---

## 1. Client API

A single parameter `code` on `init()` controls both whether a snapshot is taken and where its root is:

```python
chronist.init(code="auto")       # default: snapshot, root chosen automatically
chronist.init(code=None)         # no snapshot (False accepted as alias)
chronist.init(code="git")        # snapshot, git repo root; error if not in a repo
chronist.init(code="../..")      # snapshot, explicit relative root
chronist.init(code="/abs/path")  # snapshot, explicit absolute root
```

Advanced options via a config object:

```python
chronist.init(code=chronist.Code(
    root="git",              # "auto" | "git" | path string | int (parent levels)
    exclude=["data/", "*.csv"],
    include=[],              # force-include patterns that would otherwise be skipped
    max_file_mb=1.0,
    max_total_mb=50.0,
    env_vars=["CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS"],  # opt-in allowlist
    recurse_submodules=True,
    force=False,             # allow dangerous roots (/, $HOME)
))
```

Rules:
- `None` / `False` → off. Any other value → on.
- **Bare integers are NOT accepted as the shorthand** (`code=2`), because `True == 1` in Python makes `code=1` ambiguous. Integers are only valid as `Code(root=2)` (= go up 2 directory levels from the reference point).
- `code=True` is accepted as an alias for `"auto"`.
- Invalid values → `ValueError` with a clear message.

CLI equivalents:
- `--code auto|git|none|<path>` on any command that starts a run; env var `CHRONIST_CODE`.
- Config file: `[tool.chronist] code = ".."` in `pyproject.toml`, or `.chronist.toml`.

---

## 2. Root resolution

### 2.1 Modes
- **`"git"`**: root = git repo root (`git rev-parse --show-toplevel`, run with `-C <script_dir>`). If not inside a repo → raise an error. Never silently fall back.
- **path string**: absolute paths used as-is; relative paths resolved against the reference point (§2.2). Then `realpath`.
- **int (only via `Code(root=n)`)**: go up `n` levels from the reference point.
- **`"auto"`** (default): first match of the precedence chain below.

### 2.2 Reference point for relative paths / ints
Relative paths are resolved against the **entry script's directory**, NOT the current working directory. Otherwise `python train.py` and `python experiments/rl/train.py` would snapshot different trees.

- Script directory = `dirname(sys.modules["__main__"].__file__)`. Since Python 3.9 this is absolute and unaffected by `os.chdir`. Works for `python -m pkg.module` too.
- **Do not use `sys.argv[0]`** (relative to launch cwd, breaks after `chdir`).
- No `__main__.__file__` (notebook, REPL, embedded interpreter) → reference point = launch cwd (§3), log a warning.
- Paths from a config file (`pyproject.toml` / `.chronist.toml`) are resolved relative to **that config file's directory**.

### 2.3 Precedence chain
1. Explicit `init(code=...)` argument / CLI flag
2. `CHRONIST_CODE` env var
3. Config file (`[tool.chronist] code`), searched upward from the script directory
4. *(auto only from here)* git repo root
5. Nearest ancestor containing a marker: `pyproject.toml`, `setup.py`, `setup.cfg`, `.chronist-root`
6. Script directory

### 2.4 Guardrails
- Refuse root `/` and `$HOME` unless `force=True`.
- Before hashing: quick pre-scan with a file-count/size budget. If limits are exceeded, warn and list the largest directories; do not hang on huge trees. Abort snapshot (not the run) if the scan itself exceeds a hard cap (e.g. 100k files) and log why.
- Record in run metadata: resolved absolute root, mode, and how it was chosen, e.g. `{"root": "/home/a/research/bptt", "mode": "explicit", "spec": "../..", "reference": "/home/a/research/bptt/exp/rl/train.py"}`. Show this in the UI.

---

## 3. Working-directory robustness

Scripts often `os.chdir()` (e.g. to import from sibling dirs). Rule: **resolve everything to absolute paths once, inside `init()`, from cwd-independent sources. Never use relative paths afterwards.**

- Script path: `sys.modules["__main__"].__file__` (see §2.2).
- Launch directory: `os.environ.get("PWD")` — set by the shell at launch, not updated by `os.chdir()`. Record both `launch_cwd` (from PWD, if present) and `init_cwd` (`os.getcwd()` at init) in metadata.
- After `init()`, all of these are absolute and fixed: snapshot root, local buffer dir (`CHRONIST_DIR`), manifest paths. Git commands always run as `git -C <abs_root> ...`.
- The background upload thread must only use absolute paths.
- Snapshot hashing happens synchronously in `init()` → later `chdir` or file edits during the run do not affect it.

---

## 4. File selection

Applied within the resolved root:

1. **Git-aware listing** wherever a git repo is present:
   `git -C <root> ls-files --cached --others --exclude-standard [--recurse-submodules]`
   → tracked files + untracked-but-not-ignored. Git itself applies `.gitignore` semantics (nested files, negations, global excludes). **Do not reimplement gitignore parsing.**
   - With an explicit path root that contains one or more git repos (or is inside one), use git listing for the parts covered by a repo.
   - Note: `--recurse-submodules` cannot be combined with `--others` in all git versions; if so, list submodules separately.
2. **Non-git**: walk the tree and apply all `.gitignore` files found, using the `pathspec` library (gitwildmatch).
3. **`.chronistignore`** (gitignore syntax) at the root: additional exclusions on top, also applied to git listings.
4. `Code.exclude` / `Code.include` patterns.
5. **Built-in deny list** (always, unless force-included):
   - Dirs: `.git/`, `__pycache__/`, `.venv/`, `venv/`, `env/`, `node_modules/`, `.mypy_cache/`, `.pytest_cache/`, `.ipynb_checkpoints/`, `wandb/`, `chronist-runs/` (local buffer dir), `.tox/`
   - Artifacts/data: `*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`, `*.npy`, `*.npz`, `*.h5`, `*.hdf5`, `*.pkl`, `*.parquet`, `*.zip`, `*.tar*`
   - **Secrets**: `.env`, `.env.*`, `*.pem`, `*.key`, `id_rsa*`, `*credentials*`, `*secret*`, `.netrc`, `.pypirc`
6. **Limits**: skip files > `max_file_mb`; skip binary files (null byte in first 8 KB); stop at `max_total_mb`.
7. Record a **skip report** in metadata: each skipped path with reason (`size`, `binary`, `deny`, `secret`, `budget`). Log a one-line summary at init.

Symlinks: store as symlinks (target string) in the manifest; do not follow symlinks outside the root.

---

## 5. Environment capture

Stored as small blobs alongside the code (same content-addressed store):
- Dependency state: prefer lockfile if present in root (`uv.lock`, `poetry.lock`, `requirements*.txt` are captured as normal files anyway); always also store `pip freeze` output (or `uv pip freeze`).
- Python version + executable path, platform/OS, hostname.
- torch / CUDA / cuDNN versions if importable; GPU model(s) (via torch or `nvidia-smi --query-gpu=name --format=csv`), failing silently if unavailable.
- Full command line: `sys.executable`, `sys.argv`, `-m` module if used.
- Env vars: **only** those in the `env_vars` allowlist. Never dump the whole environment (tokens live there).
- Git metadata (already in main spec): commit, branch, dirty flag, remote URL (strip credentials from URL).

---

## 6. Local-import check

Catch code the run uses but the snapshot misses (e.g. via `sys.path.append("../utils")`):

- At `init()`: iterate `sys.modules`, collect modules with a `__file__` that is **outside the snapshot root** and **not** under stdlib (`sysconfig.get_paths()["stdlib"]`), site-packages / dist-packages, or the active venv/prefix.
- If any: warn, e.g. `imported local module 'utils.helpers' from /home/a/utils, outside code root /home/a/research/exp; consider code=".."`.
- Re-check at `finish()` (imports can happen after init). Store the list as `code.outside_imports` in run metadata.
- UI flags runs with a non-empty list as "snapshot may be incomplete".
- Optional (later): `Code(include_outside_imports=True)` adds those files to the snapshot under a separate `__outside__/<abs-path-hash>/` prefix.

---

## 7. Storage model (content-addressed, like git objects)

- **Blob**: file content, keyed by `sha256`. Stored compressed (zstd). Deduplicated globally per server/project.
- **Manifest** per run: list of `{path (relative to root, posix), sha256, mode (exec bit), size, symlink_target?}` + root metadata + env blob references. The manifest itself is stored as a blob; the run references the manifest hash.
- Server side: blob table or blob directory (`blobs/ab/cdef...`), plus `run_code(run_id, manifest_hash)`. Keep this behind a small `BlobStore` interface (`has(hashes) -> missing`, `put(hash, bytes)`, `get(hash)`), similar to `MetricStore`, so backends can change later.

### Upload protocol
1. Client hashes files at `init()` (synchronous, fast — streaming sha256).
2. Background uploader: `POST /api/blobs/missing` with list of hashes → server returns the ones it lacks.
3. Client uploads only missing blobs (`PUT /api/blobs/{hash}`, compressed body; server verifies hash).
4. Client uploads manifest; `POST /api/runs/{id}/code {manifest_hash}`.
- Idempotent and resumable; fits the existing local-first buffer / retry / offline-sync machinery. In offline mode, blobs + manifest go into the local run directory and are uploaded by `chronist sync`.
- Since content is captured at init but uploaded later, the client must **copy content into the local buffer at init** (or verify the hash on upload and fail that file loudly if it changed) — do not read the working tree lazily during upload.

---

## 8. Commands

- `chronist snapshot [--code ...] [--project ...]` — take a standalone snapshot (prints manifest hash; optionally attach to a run with `--run`).
- **`chronist checkout <run_id> [dir]`** — restore the exact file tree of a run into `dir` (default `./chronist-checkout-<run_id>`). Refuse to write into a non-empty dir unless `--force`. Also writes `CHRONIST_RUN.json` with command line, env info, git metadata, and skip report.
- `chronist code diff <run_a> <run_b>` — CLI diff of two runs' snapshots (changed/added/removed files, unified diff for text).

Python API:
```python
run = chronist.run("run-id")
run.code.files()                 # list of paths
run.code.read("src/model.py")    # file content
run.code.checkout("/tmp/x")
chronist.code_diff("run-a", "run-b")
```

---

## 9. UI

- Run page: "Code" tab — file browser of the snapshot, syntax-highlighted viewer, root/mode info, skip report, outside-imports warning, environment info.
- **Diff two runs** (right after v1): select two runs → side-by-side **code diff + config diff**. Changed files found by comparing manifests (hash mismatch), so it's instant. This answers "why did run B do better than run A".

---

## 10. Scope

**v1:** `code=` API + `chronist.Code`, root resolution (all modes, precedence, guardrails), cwd robustness, file selection with git/pathspec/.chronistignore/deny list/limits, skip report, environment capture, local-import check (warn + metadata), content-addressed upload, `chronist checkout`, Code tab in UI (browse/view).

**Right after v1:** run-vs-run code + config diff (UI and CLI), `include_outside_imports`.

---

## 11. Tests

- Root resolution: each mode; relative path resolved against script dir not cwd (test by `chdir` before `init`); config-file-relative paths; `"git"` outside a repo raises; `/` and `$HOME` refused without `force`; `code=1` shorthand rejected; `code=True`/`False` aliases.
- `python -m pkg.mod` and plain script both resolve the same script dir.
- `os.chdir()` before and after `init()` doesn't change the snapshot or break uploads.
- File selection: nested `.gitignore` with negation, untracked files included, ignored excluded, submodule included, `.chronistignore` applied, secrets excluded, size/binary limits, skip report contents.
- Dedup: two runs with one changed file → second run uploads exactly one file blob + manifest.
- Upload resumable after interruption; offline snapshot + `sync` produces the same manifest hash.
- File modified after `init()` → snapshot reflects init-time content.
- `checkout` round-trip: restored tree hashes match the manifest; exec bits and symlinks preserved.
- Local-import check: module imported via `sys.path.append("../sibling")` is reported; stdlib/site-packages are not.
