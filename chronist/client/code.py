"""Code snapshots: which code (and environment) a run used.

See docs/CODE_SNAPSHOTS.md. Everything is resolved to absolute paths once,
synchronously, inside ``take_snapshot`` (called from ``init``). File content
is copied into the local content-addressed blob store right away, so later
edits or ``os.chdir`` cannot change what gets uploaded.
"""
from __future__ import annotations

import dataclasses
import fnmatch
import json
import logging
import os
import platform
import re
import site
import socket
import stat
import subprocess
import sys
import sysconfig
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from ..blobs import FsBlobStore, sha256

log = logging.getLogger("chronist")

MANIFEST_VERSION = 1
HARD_FILE_CAP = 100_000  # abort the snapshot (not the run) beyond this many candidate files

DENY_DIRS = [".git/", "__pycache__/", ".venv/", "venv/", "env/", "node_modules/", ".mypy_cache/",
             ".pytest_cache/", ".ipynb_checkpoints/", "wandb/", "chronist-runs/", ".chronist/", ".tox/"]
DENY_FILES = ["*.pt", "*.pth", "*.ckpt", "*.safetensors", "*.npy", "*.npz", "*.h5", "*.hdf5", "*.pkl",
              "*.parquet", "*.zip", "*.tar", "*.tar.*", "*.tgz"]
SECRET_FILES = [".env", ".env.*", "*.pem", "*.key", "id_rsa*", "*credentials*", "*secret*", ".netrc", ".pypirc"]
ROOT_MARKERS = ["pyproject.toml", "setup.py", "setup.cfg", ".chronist-root"]


class CodeError(ValueError):
    pass


@dataclass
class Code:
    """Advanced snapshot options: ``chronist.init(code=chronist.Code(...))``."""

    root: Union[str, int] = "auto"  # "auto" | "git" | path | int (parent levels)
    exclude: List[str] = field(default_factory=list)
    include: List[str] = field(default_factory=list)
    max_file_mb: float = 1.0
    max_total_mb: float = 50.0
    env_vars: List[str] = field(default_factory=list)
    recurse_submodules: bool = True
    force: bool = False


def parse_code_arg(code: Any) -> Optional[Code]:
    """Normalize the ``code=`` argument. None -> snapshot off."""
    if code is None or code is False:
        return None
    if code is True:
        return Code()
    if isinstance(code, Code):
        if isinstance(code.root, bool) or not isinstance(code.root, (str, int)):
            raise CodeError(f"Code(root=...) must be 'auto', 'git', a path or an int, got {code.root!r}")
        if isinstance(code.root, int) and code.root < 0:
            raise CodeError("Code(root=n) needs n >= 0")
        return code
    if isinstance(code, int):
        raise CodeError(f"code={code!r}: bare integers are not accepted (code=1 is ambiguous with True); "
                        f"use chronist.Code(root={code}) to go up {code} directory levels")
    if isinstance(code, (str, os.PathLike)):
        s = os.fspath(code)
        if s.lower() in ("none", "off", "false", ""):
            return None
        return Code(root=s)
    raise CodeError(f"invalid code={code!r}: use 'auto', 'git', a path, None or chronist.Code(...)")


# ---------------------------------------------------------------- reference points
def script_path() -> Optional[str]:
    main = sys.modules.get("__main__")
    f = getattr(main, "__file__", None)
    if not f:
        return None
    return os.path.realpath(os.path.abspath(f))


def launch_cwd() -> str:
    pwd = os.environ.get("PWD")
    return pwd if pwd and os.path.isdir(pwd) else os.getcwd()


_REFERENCE_OVERRIDE: Optional[str] = None  # set by CLI commands: resolve against cwd


def reference_dir() -> Tuple[str, Optional[str]]:
    """(directory relative roots resolve against, script path or None)."""
    if _REFERENCE_OVERRIDE:
        return _REFERENCE_OVERRIDE, None
    sp = script_path()
    if sp:
        return os.path.dirname(sp), sp
    log.warning("chronist: no __main__.__file__ (notebook/REPL); code paths resolve against the launch directory")
    return launch_cwd(), None


# ---------------------------------------------------------------- config files
def _load_toml(path: str) -> dict:
    try:
        import tomllib  # type: ignore
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore
        except ImportError:
            log.warning("chronist: cannot read %s (pip install tomli)", path)
            return {}
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except Exception as e:
        log.warning("chronist: cannot parse %s: %s", path, e)
        return {}


def find_config(start: str) -> Optional[Tuple[str, dict]]:
    """Nearest .chronist.toml or pyproject.toml with [tool.chronist], searching upward."""
    d = start
    while True:
        p = os.path.join(d, ".chronist.toml")
        if os.path.isfile(p):
            return p, _load_toml(p)
        p = os.path.join(d, "pyproject.toml")
        if os.path.isfile(p):
            cfg = _load_toml(p).get("tool", {}).get("chronist")
            if cfg is not None:
                return p, cfg
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


# ---------------------------------------------------------------- root resolution
def _git_toplevel(d: str) -> Optional[str]:
    try:
        out = subprocess.run(["git", "-C", d, "rev-parse", "--show-toplevel"], capture_output=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return os.path.realpath(out.stdout.decode().strip())


def _resolve_spec(spec: Union[str, int], base: str) -> Tuple[str, str]:
    """Resolve an explicit spec against ``base``. Returns (root, mode)."""
    if isinstance(spec, int):
        root = base
        for _ in range(spec):
            root = os.path.dirname(root)
        return os.path.realpath(root), "levels"
    if spec == "git":
        top = _git_toplevel(base)
        if top is None:
            raise CodeError(f"code='git' but {base} is not inside a git repository")
        return top, "git"
    p = os.path.expanduser(spec)
    if not os.path.isabs(p):
        p = os.path.join(base, p)
    p = os.path.realpath(p)
    if not os.path.isdir(p):
        raise CodeError(f"code root {p} (from {spec!r}) is not a directory")
    return p, "explicit"


def resolve_root(opts: Code) -> dict:
    """Pick the snapshot root (spec section 2). Returns root metadata."""
    ref_dir, script = reference_dir()
    info: Dict[str, Any] = {"reference": script or ref_dir, "launch_cwd": launch_cwd(), "init_cwd": os.getcwd()}
    spec = opts.root
    if spec != "auto":
        root, mode = _resolve_spec(spec, ref_dir)
        info.update(root=root, mode=mode, chosen_by="argument", spec=spec)
    else:
        env = os.environ.get("CHRONIST_CODE")
        cfg = find_config(ref_dir)
        cfg_code = cfg[1].get("code") if cfg else None
        if env and env.lower() != "auto" and parse_code_arg(env) is not None:
            root, mode = _resolve_spec(env, ref_dir)
            info.update(root=root, mode=mode, chosen_by="env", spec=env)
        elif cfg_code not in (None, "auto", True):
            if isinstance(cfg_code, bool) or not isinstance(cfg_code, (str, int)):
                raise CodeError(f"{cfg[0]}: code must be a string or int")
            root, mode = _resolve_spec(cfg_code, os.path.dirname(cfg[0]))
            info.update(root=root, mode=mode, chosen_by="config", spec=cfg_code, config=cfg[0])
        else:
            top = _git_toplevel(ref_dir)
            if top:
                info.update(root=top, mode="git", chosen_by="auto")
            else:
                root = None
                d = ref_dir
                while True:
                    if any(os.path.exists(os.path.join(d, m)) for m in ROOT_MARKERS):
                        root = d
                        break
                    parent = os.path.dirname(d)
                    if parent == d:
                        break
                    d = parent
                if root:
                    info.update(root=os.path.realpath(root), mode="marker", chosen_by="auto")
                else:
                    info.update(root=os.path.realpath(ref_dir), mode="script_dir", chosen_by="auto")
    root = info["root"]
    if not opts.force and root in ("/", os.path.realpath(os.path.expanduser("~"))):
        raise CodeError(f"refusing to snapshot {root!r} (use chronist.Code(root=..., force=True) if you mean it)")
    return info


def code_opts_from_env_and_config(code: Any) -> Optional[Code]:
    """Effective options. An explicit argument wins; with the default "auto",
    CHRONIST_CODE and then the config file may turn snapshots off or add options."""
    opts = parse_code_arg(code)
    if opts is None or opts.root != "auto":
        return opts
    env = os.environ.get("CHRONIST_CODE")
    if env is not None:
        return None if parse_code_arg(env) is None else opts
    cfg = find_config(reference_dir()[0])
    if cfg:
        c = cfg[1].get("code", "auto")
        if c is False or (isinstance(c, str) and c.lower() in ("none", "off", "false")):
            return None
        for k in ("exclude", "include", "env_vars"):
            if isinstance(cfg[1].get(k), list) and not getattr(opts, k):
                setattr(opts, k, list(cfg[1][k]))
        for k in ("max_file_mb", "max_total_mb"):
            if isinstance(cfg[1].get(k), (int, float)) and not isinstance(cfg[1].get(k), bool):
                setattr(opts, k, float(cfg[1][k]))
    return opts


# ---------------------------------------------------------------- file selection
def _spec(patterns: List[str]):
    import pathspec
    if hasattr(pathspec, "GitIgnoreSpec"):  # pathspec >= 0.10: exact gitignore semantics
        return pathspec.GitIgnoreSpec.from_lines(patterns)
    return pathspec.PathSpec.from_lines("gitwildmatch", patterns)


def _git_list(repo_dir: str, recurse_submodules: bool, prefix: str = "") -> Optional[List[str]]:
    """Tracked + untracked-not-ignored files under repo_dir (relative to it),
    with prefix prepended. Submodules are listed separately (git <2.30 cannot
    combine --recurse-submodules with --others)."""
    try:
        out = subprocess.run(["git", "-C", repo_dir, "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                             capture_output=True, timeout=60)
        st = subprocess.run(["git", "-C", repo_dir, "ls-files", "-z", "-s"], capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    submodules = set()
    for entry in st.stdout.decode(errors="surrogateescape").split("\0"):
        if entry.startswith("160000 "):
            submodules.add(entry.split("\t", 1)[1])
    files = []
    for rel in dict.fromkeys(out.stdout.decode(errors="surrogateescape").split("\0")):
        if not rel:
            continue
        if rel in submodules or rel.rstrip("/") in submodules:
            if recurse_submodules:
                sub = os.path.join(repo_dir, rel.rstrip("/"))
                if os.path.exists(os.path.join(sub, ".git")):
                    files.extend(_git_list(sub, True, prefix + rel.rstrip("/") + "/") or [])
            continue
        files.append(prefix + rel)
    return files


def _walk_list(root: str, recurse_submodules: bool) -> List[str]:
    """Non-git listing with .gitignore files (pathspec); nested git repos use git."""
    out: List[str] = []
    specs: List[Tuple[str, Any]] = []  # (dir rel to root, spec)

    def ignored(rel: str, is_dir: bool) -> bool:
        path = rel + ("/" if is_dir else "")
        for base, sp in specs:
            if base and not path.startswith(base + "/"):
                continue
            sub = path[len(base) + 1:] if base else path
            if sp.match_file(sub):
                return True
        return False

    def walk(rel_dir: str):
        full = os.path.join(root, rel_dir) if rel_dir else root
        if rel_dir and os.path.exists(os.path.join(full, ".git")):
            out.extend(_git_list(full, recurse_submodules, rel_dir + "/") or [])
            return
        gi = os.path.join(full, ".gitignore")
        pushed = False
        if os.path.isfile(gi):
            try:
                with open(gi, errors="replace") as f:
                    specs.append((rel_dir, _spec(f.read().splitlines())))
                    pushed = True
            except OSError:
                pass
        try:
            entries = sorted(os.scandir(full), key=lambda e: e.name)
        except OSError:
            entries = []
        for e in entries:
            rel = f"{rel_dir}/{e.name}" if rel_dir else e.name
            if e.name == ".git":
                continue
            is_dir = e.is_dir(follow_symlinks=False)
            if ignored(rel, is_dir):
                continue
            if is_dir:
                # deny-listed dirs are pruned here to avoid walking huge trees
                if (e.name + "/") in DENY_DIRS:
                    out.append(rel + "/")  # recorded as skipped below
                    continue
                walk(rel)
            else:
                out.append(rel)
            if len(out) > HARD_FILE_CAP:
                raise _TooMany()
        if pushed:
            specs.pop()

    walk("")
    return out


class _TooMany(Exception):
    pass


def list_candidates(root: str, opts: Code) -> List[str]:
    if _git_toplevel(root):
        files = _git_list(root, opts.recurse_submodules)
        if files is not None:
            return files
    return _walk_list(root, opts.recurse_submodules)


def _is_binary(data: bytes) -> bool:
    return b"\0" in data[:8192]


def select_and_store(root: str, opts: Code, blobs: FsBlobStore) -> Tuple[List[dict], List[dict], dict]:
    """Pick files, hash them and copy their content into ``blobs``.
    Returns (manifest entries, skip report, totals)."""
    candidates = list_candidates(root, opts)
    if len(candidates) > HARD_FILE_CAP:
        raise _TooMany()
    chronistignore = os.path.join(root, ".chronistignore")
    user_ignore = None
    if os.path.isfile(chronistignore):
        with open(chronistignore, errors="replace") as f:
            user_ignore = _spec(f.read().splitlines())
    exclude = _spec(opts.exclude) if opts.exclude else None
    include = _spec(opts.include) if opts.include else None
    deny_dirs = _spec(DENY_DIRS)
    deny_files = _spec(DENY_FILES)
    secrets = _spec(SECRET_FILES)

    entries, skipped = [], []
    total = 0
    budget = opts.max_total_mb * 1024 * 1024
    max_file = opts.max_file_mb * 1024 * 1024
    dir_sizes: Dict[str, int] = {}
    for rel in sorted(candidates):
        rel = rel.replace(os.sep, "/")
        forced = include is not None and include.match_file(rel)
        if rel.endswith("/"):  # pruned deny-listed directory from the walker
            if not forced:
                skipped.append({"path": rel, "reason": "deny"})
            continue
        if not forced:
            if user_ignore is not None and user_ignore.match_file(rel):
                skipped.append({"path": rel, "reason": "chronistignore"})
                continue
            if exclude is not None and exclude.match_file(rel):
                skipped.append({"path": rel, "reason": "exclude"})
                continue
            if secrets.match_file(rel):
                skipped.append({"path": rel, "reason": "secret"})
                continue
            if deny_dirs.match_file(rel) or deny_files.match_file(rel):
                skipped.append({"path": rel, "reason": "deny"})
                continue
        full = os.path.join(root, rel)
        try:
            st = os.lstat(full)
        except OSError:
            continue  # tracked but deleted in the working tree
        if stat.S_ISLNK(st.st_mode):
            entries.append({"path": rel, "symlink": os.readlink(full), "mode": 0o120000})
            continue
        if not stat.S_ISREG(st.st_mode):
            continue
        top = rel.split("/", 1)[0] if "/" in rel else "."
        dir_sizes[top] = dir_sizes.get(top, 0) + st.st_size
        if st.st_size > max_file and not forced:
            skipped.append({"path": rel, "reason": "size", "size": st.st_size})
            continue
        if total + st.st_size > budget:
            skipped.append({"path": rel, "reason": "budget", "size": st.st_size})
            continue
        try:
            with open(full, "rb") as f:
                data = f.read()
        except OSError as e:
            skipped.append({"path": rel, "reason": f"unreadable: {e.strerror}"})
            continue
        if _is_binary(data) and not forced:
            skipped.append({"path": rel, "reason": "binary", "size": len(data)})
            continue
        h = blobs.put_raw(data)
        total += len(data)
        entries.append({"path": rel, "sha256": h, "size": len(data),
                        "mode": 0o100755 if st.st_mode & 0o111 else 0o100644})
    totals = {"files": len(entries), "bytes": total, "skipped": len(skipped)}
    if any(s["reason"] == "budget" for s in skipped):
        biggest = sorted(dir_sizes.items(), key=lambda kv: -kv[1])[:5]
        log.warning("chronist: code snapshot hit max_total_mb=%s; largest top-level entries: %s",
                    opts.max_total_mb, ", ".join(f"{d} ({n / 1e6:.1f} MB)" for d, n in biggest))
    return entries, skipped, totals


# ---------------------------------------------------------------- environment
def _strip_credentials(url: str) -> str:
    return re.sub(r"(?<=://)[^/@]+@", "", url)


def _run(cmd: List[str], cwd: Optional[str] = None, timeout: float = 5) -> Optional[str]:
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.decode(errors="replace") if out.returncode == 0 else None


def capture_env(root: str, opts: Code) -> Tuple[dict, str]:
    """(small env info dict, package list text)."""
    try:
        from importlib import metadata as im
        pkgs = sorted({f"{d.metadata['Name']}=={d.version}" for d in im.distributions() if d.metadata["Name"]},
                      key=str.lower)
    except Exception:  # pragma: no cover
        pkgs = []
    main = sys.modules.get("__main__")
    spec = getattr(main, "__spec__", None)
    env: Dict[str, Any] = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "argv": list(sys.argv),
        "module": spec.name if spec is not None else None,
        "env_vars": {k: os.environ[k] for k in opts.env_vars if k in os.environ},
    }
    torch = sys.modules.get("torch")  # importing torch just for this would cost seconds
    if torch is not None:
        try:
            env["torch"] = torch.__version__
            env["cuda"] = torch.version.cuda
            env["cudnn"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
            if torch.cuda.is_available():
                env["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        except Exception:
            pass
    if "gpus" not in env:
        smi = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], timeout=5)
        if smi:
            env["gpus"] = [l.strip() for l in smi.splitlines() if l.strip()]
    commit = _run(["git", "-C", root, "rev-parse", "HEAD"])
    if commit:
        env["git"] = {
            "commit": commit.strip(),
            "branch": (_run(["git", "-C", root, "rev-parse", "--abbrev-ref", "HEAD"]) or "").strip(),
            "dirty": bool((_run(["git", "-C", root, "status", "--porcelain", "--untracked-files=no"]) or "").strip()),
            "remote": _strip_credentials((_run(["git", "-C", root, "config", "--get", "remote.origin.url"]) or "").strip()),
        }
    return env, "\n".join(pkgs) + "\n"


# ---------------------------------------------------------------- outside imports
def _system_prefixes() -> List[str]:
    paths = set()
    for k in ("stdlib", "platstdlib", "purelib", "platlib"):
        p = sysconfig.get_paths().get(k)
        if p:
            paths.add(os.path.realpath(p))
    for p in (sys.prefix, sys.base_prefix, sys.exec_prefix, getattr(sys, "base_exec_prefix", sys.prefix)):
        paths.add(os.path.realpath(p))
    try:
        for p in site.getsitepackages() + [site.getusersitepackages()]:
            paths.add(os.path.realpath(p))
    except Exception:  # pragma: no cover
        pass
    return [p.rstrip("/") + "/" for p in paths]


def outside_imports(root: str) -> List[dict]:
    """Modules loaded from files outside the code root that are not stdlib or
    installed packages (e.g. via sys.path.append("../utils"))."""
    root_p = root.rstrip("/") + "/"
    sysp = _system_prefixes()
    here = os.path.realpath(os.path.dirname(os.path.dirname(__file__))).rstrip("/") + "/"
    out = []
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if not f or not isinstance(f, str):
            continue
        try:
            f = os.path.realpath(f)
        except Exception:
            continue
        if f.startswith(root_p) or f.startswith(here) or any(f.startswith(p) for p in sysp):
            continue
        if "site-packages/" in f or "dist-packages/" in f:
            continue
        out.append({"module": name, "file": f})
    return sorted(out, key=lambda d: d["module"])


# ---------------------------------------------------------------- snapshot
def take_snapshot(opts: Code, blobs: FsBlobStore) -> Optional[dict]:
    """Snapshot the code root into ``blobs``. Returns run metadata for the
    snapshot (with ``manifest`` = manifest blob hash), or None if aborted."""
    t0 = time.time()
    root_info = resolve_root(opts)
    root = root_info["root"]
    try:
        entries, skipped, totals = select_and_store(root, opts, blobs)
    except _TooMany:
        log.warning("chronist: code snapshot aborted: more than %d files under %s; "
                    "pick a narrower root with code=... or add a .chronistignore", HARD_FILE_CAP, root)
        return {**root_info, "aborted": f"more than {HARD_FILE_CAP} files"}
    env, packages = capture_env(root, opts)
    env_hash = blobs.put_raw(json.dumps(env, sort_keys=True, indent=1).encode())
    pkg_hash = blobs.put_raw(packages.encode())
    manifest = {
        "version": MANIFEST_VERSION,
        "root": {k: root_info[k] for k in ("root", "mode", "chosen_by", "spec") if k in root_info},
        "files": entries,
        "skipped": skipped,
        "env": env_hash,
        "packages": pkg_hash,
    }
    mhash = blobs.put_raw(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())
    imports = outside_imports(root)
    for imp in imports[:5]:
        log.warning("chronist: imported local module %r from %s, outside code root %s; consider code=\"..\"",
                    imp["module"], os.path.dirname(imp["file"]), root)
    by_reason: Dict[str, int] = {}
    for s in skipped:
        by_reason[s["reason"]] = by_reason.get(s["reason"], 0) + 1
    log.info("chronist: code snapshot of %s: %d files, %.1f KB%s (%.2fs)", root, totals["files"], totals["bytes"] / 1024,
             (", skipped " + ", ".join(f"{n} {r}" for r, n in sorted(by_reason.items()))) if skipped else "",
             time.time() - t0)
    return {**root_info, "manifest": mhash, **totals, "skipped_by_reason": by_reason,
            "outside_imports": imports}


def manifest_blobs(manifest: dict) -> List[str]:
    """Every blob a manifest references (not including itself)."""
    hs = [f["sha256"] for f in manifest["files"] if "sha256" in f]
    return hs + [manifest["env"], manifest["packages"]]


# ---------------------------------------------------------------- restore / diff
def write_tree(manifest: dict, read_blob, dest: str, force: bool = False) -> int:
    """Restore a manifest's files into ``dest``. Returns number of files written."""
    dest = os.path.abspath(dest)
    if os.path.exists(dest) and os.listdir(dest) and not force:
        raise CodeError(f"{dest} is not empty (use --force)")
    os.makedirs(dest, exist_ok=True)
    n = 0
    for f in manifest["files"]:
        rel = f["path"]
        if rel.startswith("/") or ".." in rel.split("/"):
            raise CodeError(f"refusing unsafe path in manifest: {rel!r}")
        p = os.path.join(dest, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        if "symlink" in f:
            if os.path.lexists(p):
                os.remove(p)
            os.symlink(f["symlink"], p)
        else:
            data = read_blob(f["sha256"])
            if data is None:
                raise CodeError(f"missing blob for {rel}")
            with open(p, "wb") as fh:
                fh.write(data)
            os.chmod(p, 0o755 if f.get("mode") == 0o100755 else 0o644)
        n += 1
    return n


def diff_manifests(a: dict, b: dict) -> Dict[str, List[str]]:
    fa = {f["path"]: f.get("sha256", "link:" + f.get("symlink", "")) for f in a["files"]}
    fb = {f["path"]: f.get("sha256", "link:" + f.get("symlink", "")) for f in b["files"]}
    return {
        "added": sorted(set(fb) - set(fa)),
        "removed": sorted(set(fa) - set(fb)),
        "changed": sorted(p for p in set(fa) & set(fb) if fa[p] != fb[p]),
    }
