"""Code snapshots (docs/CODE_SNAPSHOTS.md section 11)."""
import json
import os
import subprocess
import sys

import pytest

import chronist
from chronist.blobs import FsBlobStore, sha256
from chronist.client import code as C
from chronist.client.rundir import RunDir

from test_client import LiveServer, free_port  # noqa: F401  (fixtures/helpers)

PY = sys.executable


def git(cwd, *args):
    subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True,
                   env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                        "GIT_COMMITTER_EMAIL": "t@t"})


def write(p, text="x\n", mode=None):
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(text, bytes):
        p.write_bytes(text)
    else:
        p.write_text(text)
    if mode:
        os.chmod(p, mode)


TRAIN = """
import os, sys, json
import chronist
{pre}
run = chronist.init(project="p", offline=True, code={code})
{post}
run.finish()
print("SNAP=" + json.dumps(run._info["meta"].get("code")))
"""


@pytest.fixture
def proj(tmp_path):
    """A git repo with nested ignores, untracked files, secrets, a submodule."""
    p = tmp_path / "proj"
    write(p / "pyproject.toml", "[project]\nname='x'\n")
    write(p / "exp" / "__init__.py", "")
    write(p / "exp" / "rl" / "__init__.py", "")
    write(p / "exp" / "rl" / "train.py", TRAIN.format(pre="", post="", code='"auto"'))
    write(p / "src" / "model.py", "class Model: pass\n")
    write(p / "run.sh", "#!/bin/sh\necho hi\n", mode=0o755)
    write(p / ".gitignore", "*.log\nlogs/\n")
    write(p / "data" / ".gitignore", "*\n!keep.txt\n!.gitignore\n")  # nested with negation
    write(p / "data" / "keep.txt", "kept\n")
    write(p / "data" / "big.bin", "ignored by nested gitignore\n")
    write(p / "debug.log", "ignored\n")
    write(p / "logs" / "a.txt", "ignored dir\n")
    write(p / ".env", "TOKEN=secret\n")
    write(p / "weights.pt", "fake checkpoint\n")
    write(p / "notes" / "untracked.md", "untracked but not ignored\n")
    write(p / "blob.dat", b"\x00\x01binary\x00")
    write(p / "huge.txt", "y" * (2 * 1024 * 1024))
    write(p / "scratch" / "tmp.txt", "excluded by .chronistignore\n")
    write(p / ".chronistignore", "scratch/\n")
    os.symlink("src/model.py", p / "model_link.py")
    git(p, "init", "-q")
    # submodule
    sub = tmp_path / "subsrc"
    write(sub / "lib.py", "LIB = 1\n")
    git(sub, "init", "-q")
    git(sub, "add", ".")
    git(sub, "commit", "-qm", "sub")
    git(p, "-c", "protocol.file.allow=always", "submodule", "add", "-q", str(sub), "vendor/sub")
    git(p, "add", "pyproject.toml", "exp", "src", "run.sh", ".gitignore", "data/.gitignore", "data/keep.txt",
        "model_link.py", ".chronistignore")
    git(p, "commit", "-qm", "init")
    return p


def cdir_for(script):
    p = script
    while p.name != "proj":
        p = p.parent
    return p.parent / ".cdir"


def run_script(script, cwd, env_extra=None, args=None):
    env = {**os.environ, "CHRONIST_DIR": str(cdir_for(script)), **(env_extra or {})}
    env.pop("CHRONIST_URL", None)
    env.pop("CHRONIST_CODE", None)
    env["PYTHONPATH"] = os.pathsep.join([os.path.dirname(os.path.dirname(chronist.__file__))])
    out = subprocess.run(args or [PY, str(script)], cwd=cwd, env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    line = [l for l in out.stdout.splitlines() if l.startswith("SNAP=")][-1]
    return json.loads(line[5:]), out.stderr


def manifest_of(snap, cdir):
    return json.loads(FsBlobStore(os.path.join(cdir, "blobs")).read(snap["manifest"]))


# ---------------------------------------------------------------- argument parsing
def test_parse_code_arg():
    assert C.parse_code_arg(None) is None
    assert C.parse_code_arg(False) is None
    assert C.parse_code_arg("none") is None
    assert C.parse_code_arg(True).root == "auto"
    assert C.parse_code_arg("git").root == "git"
    assert C.parse_code_arg("../..").root == "../.."
    assert C.parse_code_arg(C.Code(root=2)).root == 2
    for bad in (1, 2, 0, 3.5, ["x"], C.Code(root=True), C.Code(root=-1)):
        with pytest.raises(ValueError):
            C.parse_code_arg(bad)


# ---------------------------------------------------------------- root resolution
def test_auto_uses_git_root_and_selection(proj, tmp_path):
    snap, err = run_script(proj / "exp" / "rl" / "train.py", cwd=tmp_path)
    assert snap["root"] == str(proj) and snap["mode"] == "git" and snap["chosen_by"] == "auto"
    assert snap["reference"] == str(proj / "exp" / "rl" / "train.py")
    m = manifest_of(snap, tmp_path / ".cdir")
    files = {f["path"]: f for f in m["files"]}
    assert "src/model.py" in files and "exp/rl/train.py" in files
    assert "notes/untracked.md" in files  # untracked, not ignored
    assert "data/keep.txt" in files and "data/big.bin" not in files  # nested negation
    assert "debug.log" not in files and "logs/a.txt" not in files
    assert "vendor/sub/lib.py" in files  # submodule
    assert files["run.sh"]["mode"] == 0o100755
    assert files["model_link.py"]["symlink"] == "src/model.py"
    skipped = {s["path"]: s["reason"] for s in m["skipped"]}
    assert skipped[".env"] == "secret"
    assert skipped["weights.pt"] == "deny"
    assert skipped["blob.dat"] == "binary"
    assert skipped["huge.txt"] == "size"
    assert skipped["scratch/tmp.txt"] == "chronistignore"
    assert ".env" not in files and "scratch/tmp.txt" not in files


def test_relative_root_against_script_dir_not_cwd(proj, tmp_path):
    script = proj / "exp" / "rl" / "train.py"
    script.write_text(TRAIN.format(pre="", post="", code='"../.."'))
    a, _ = run_script(script, cwd=tmp_path)
    b, _ = run_script(script, cwd=proj / "src")
    assert a["root"] == b["root"] == str(proj) and a["mode"] == "explicit"
    assert a["manifest"] == b["manifest"]


def test_chdir_before_and_after_init(proj, tmp_path):
    script = proj / "exp" / "rl" / "train.py"
    script.write_text(TRAIN.format(pre='os.chdir("/")', post='os.chdir("/tmp")', code='"../.."'))
    snap, _ = run_script(script, cwd=proj)
    assert snap["root"] == str(proj)
    assert snap["init_cwd"] == "/"


def test_module_and_script_resolve_same(proj, tmp_path):
    a, _ = run_script(proj / "exp" / "rl" / "train.py", cwd=proj)
    b, _ = run_script(proj / "exp" / "rl" / "train.py", cwd=proj, args=[PY, "-m", "exp.rl.train"])
    assert a["reference"] == b["reference"] and a["root"] == b["root"]
    ma, mb = manifest_of(a, proj.parent / ".cdir"), manifest_of(b, proj.parent / ".cdir")
    assert ma["files"] == mb["files"]


def test_git_mode_outside_repo_raises(tmp_path):
    script = tmp_path / "nogit" / "train.py"
    write(script, TRAIN.format(pre="", post="", code='"git"'))
    env = {**os.environ, "CHRONIST_DIR": str(tmp_path / "cd"),
           "PYTHONPATH": os.path.dirname(os.path.dirname(chronist.__file__)),
           "GIT_CEILING_DIRECTORIES": str(tmp_path)}
    out = subprocess.run([PY, str(script)], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert out.returncode != 0 and "not inside a git repository" in out.stderr


def test_auto_non_git_marker_and_script_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    monkeypatch.delenv("CHRONIST_CODE", raising=False)
    write(tmp_path / "a" / "setup.cfg", "")
    write(tmp_path / "a" / "b" / "train.py", "")
    monkeypatch.setattr(C, "script_path", lambda: str(tmp_path / "a" / "b" / "train.py"))
    info = C.resolve_root(C.Code())
    assert info["root"] == str(tmp_path / "a") and info["mode"] == "marker"
    os.remove(tmp_path / "a" / "setup.cfg")
    info = C.resolve_root(C.Code())
    assert info["root"] == str(tmp_path / "a" / "b") and info["mode"] == "script_dir"
    info = C.resolve_root(C.Code(root=1))
    assert info["root"] == str(tmp_path / "a") and info["mode"] == "levels"


def test_env_and_config_precedence(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    write(tmp_path / "r" / "pyproject.toml", '[tool.chronist]\ncode = "sub"\nexclude = ["*.csv"]\n')
    write(tmp_path / "r" / "sub" / "x.py", "")
    write(tmp_path / "r" / "deep" / "train.py", "")
    monkeypatch.setattr(C, "script_path", lambda: str(tmp_path / "r" / "deep" / "train.py"))
    monkeypatch.delenv("CHRONIST_CODE", raising=False)
    info = C.resolve_root(C.Code())
    assert info["root"] == str(tmp_path / "r" / "sub") and info["chosen_by"] == "config"  # relative to config file
    assert C.code_opts_from_env_and_config("auto").exclude == ["*.csv"]
    monkeypatch.setenv("CHRONIST_CODE", "..")
    info = C.resolve_root(C.Code())
    assert info["root"] == str(tmp_path / "r") and info["chosen_by"] == "env"
    info = C.resolve_root(C.Code(root="."))  # explicit argument wins
    assert info["root"] == str(tmp_path / "r" / "deep") and info["chosen_by"] == "argument"
    monkeypatch.setenv("CHRONIST_CODE", "none")
    assert C.code_opts_from_env_and_config("auto") is None
    assert C.code_opts_from_env_and_config(".") is not None  # explicit beats env


def test_dangerous_roots_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(ValueError, match="refusing"):
        C.resolve_root(C.Code(root=str(tmp_path)))
    with pytest.raises(ValueError, match="refusing"):
        C.resolve_root(C.Code(root="/"))
    assert C.resolve_root(C.Code(root=str(tmp_path), force=True))["root"] == str(tmp_path)


def test_non_git_walk_with_gitignore(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    r = tmp_path / "r"
    write(r / ".gitignore", "*.tmp\nbuild/\n")
    write(r / "a.py")
    write(r / "b.tmp")
    write(r / "build" / "x.py")
    write(r / "pkg" / ".gitignore", "*\n!keep.py\n!.gitignore\n")
    write(r / "pkg" / "keep.py")
    write(r / "pkg" / "drop.py")
    write(r / "node_modules" / "m.js")
    write(r / "id_rsa", "key")
    store = FsBlobStore(str(tmp_path / "blobs"))
    entries, skipped, _ = C.select_and_store(str(r), C.Code(), store)
    paths = {e["path"] for e in entries}
    assert paths == {".gitignore", "a.py", "pkg/.gitignore", "pkg/keep.py"}
    sk = {s["path"]: s["reason"] for s in skipped}
    assert sk["node_modules/"] == "deny" and sk["id_rsa"] == "secret"


def test_include_exclude_budget(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    r = tmp_path / "r"
    for i in range(5):
        write(r / f"f{i}.py", "z" * 300_000)
    write(r / "data.csv", "1,2\n")
    write(r / "model.pt", "forced")
    store = FsBlobStore(str(tmp_path / "blobs"))
    entries, skipped, totals = C.select_and_store(
        str(r), C.Code(exclude=["*.csv"], include=["model.pt"], max_total_mb=1.0), store)
    paths = {e["path"] for e in entries}
    assert "model.pt" in paths and "data.csv" not in paths
    sk = {s["path"]: s["reason"] for s in skipped}
    assert sk["data.csv"] == "exclude"
    assert sum(1 for v in sk.values() if v == "budget") == 2 and totals["bytes"] <= 1024 * 1024


def test_outside_imports(tmp_path, monkeypatch):
    sib = tmp_path / "sibling"
    write(sib / "helpers_xyz.py", "X = 1\n")
    monkeypatch.syspath_prepend(str(sib))
    import helpers_xyz  # noqa: F401
    root = tmp_path / "root"
    root.mkdir()
    found = {m["module"]: m["file"] for m in C.outside_imports(str(root))}
    assert "helpers_xyz" in found
    assert "json" not in found and "numpy" not in found and "pytest" not in found
    assert not any(k.startswith("chronist") for k in found)
    sys.modules.pop("helpers_xyz", None)


# ---------------------------------------------------------------- upload, dedup, restore
@pytest.fixture
def server(tmp_path):
    s = LiveServer(str(tmp_path / "data")).start()
    yield s
    s.stop()


@pytest.fixture
def cdir(tmp_path, monkeypatch):
    d = tmp_path / "client"
    monkeypatch.setenv("CHRONIST_DIR", str(d))
    monkeypatch.delenv("CHRONIST_URL", raising=False)
    monkeypatch.delenv("CHRONIST_CODE", raising=False)
    return d


def small_proj(tmp_path):
    r = tmp_path / "small"
    write(r / "train.py", "print('train')\n")
    write(r / "model.py", "A = 1\n")
    write(r / "tool.sh", "#!/bin/sh\n", mode=0o755)
    os.symlink("model.py", r / "alias.py")
    return r


def test_dedup_init_time_content_and_checkout(server, cdir, tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    monkeypatch.setenv("CHRONIST_KEEP_LOCAL", "1")  # we inspect upload progress afterwards
    r = small_proj(tmp_path)
    run1 = chronist.init(project="p", url=server.url, code=str(r), config={"lr": 1})
    (r / "model.py").write_text("A = 2  # edited during the run\n")  # after init: must not leak in
    run1.finish()
    p1 = RunDir(run1.dir.path).progress(server.url)
    assert p1["code_done"] and p1["code_blobs_sent"] >= 5  # 4 files... + env + packages + manifest
    ref1 = chronist.run(run1.id, url=server.url)
    assert ref1.code.read("model.py") == "A = 1\n"

    run2 = chronist.init(project="p", url=server.url, code=str(r), config={"lr": 2})  # model.py changed
    run2.finish()
    p2 = RunDir(run2.dir.path).progress(server.url)
    assert p2["code_blobs_sent"] == 2  # exactly: the changed file + the new manifest

    d = chronist.code_diff(run1.id, run2.id, url=server.url)
    assert d["changed"] == ["model.py"] and d["added"] == [] and d["removed"] == []
    assert "-A = 1" in d["diffs"]["model.py"] and "+A = 2" in d["diffs"]["model.py"]
    assert d["config"]["changed"] == {"lr": [1, 2]}

    dest = tmp_path / "restored"
    ref2 = chronist.run(run2.uid, url=server.url)
    ref2.code.checkout(str(dest))
    for f in ref2.code.manifest["files"]:
        p = dest / f["path"]
        if "symlink" in f:
            assert os.readlink(p) == f["symlink"]
        else:
            assert sha256(p.read_bytes()) == f["sha256"]
            assert bool(os.stat(p).st_mode & 0o111) == (f["mode"] == 0o100755)
    info = json.loads((dest / "CHRONIST_RUN.json").read_text())
    assert info["run"]["id"] == run2.id and info["manifest"] == ref2.code.hash
    with pytest.raises(ValueError):
        ref2.code.checkout(str(dest))  # non-empty without force
    ref2.code.checkout(str(dest), force=True)
    # local data dir read-back works too
    loc = chronist.run(run2.id, data_dir=server.data_dir)
    assert loc.code.files() == ref2.code.files()


def test_offline_then_sync_same_manifest(server, cdir, tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    r = small_proj(tmp_path)
    run = chronist.init(project="p", code=str(r))
    run.finish()
    local_hash = run._info["meta"]["code"]["manifest"]
    chronist.sync(url=server.url, verbose=False)
    ref = chronist.run(run.uid, url=server.url)
    assert ref.code.hash == local_hash
    assert ref.meta["code"]["root"] == str(r)


def test_upload_resumes_after_interruption(cdir, tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    monkeypatch.setenv("CHRONIST_KEEP_LOCAL", "1")
    r = small_proj(tmp_path)
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    run = chronist.init(project="p", url=url, code=str(r))
    run.finish(timeout=0.3)  # server down: nothing uploaded
    srv = LiveServer(str(tmp_path / "data"), port=port).start()
    try:
        # simulate a partial earlier upload: some blobs already on the server
        snap = run._info["meta"]["code"]
        local = FsBlobStore(str(cdir / "blobs"))
        m = json.loads(local.read(snap["manifest"]))
        first = next(f["sha256"] for f in m["files"] if "sha256" in f)
        srv.core.blobs.put(first, local.get(first))
        chronist.sync(url=url, verbose=False)
        ref = chronist.run(run.uid, url=url)
        assert ref.code.hash == snap["manifest"]
        expected = len(set(C.manifest_blobs(m)) | {snap["manifest"]}) - 1  # all but the pre-uploaded one
        assert RunDir(run.dir.path).progress(url)["code_blobs_sent"] == expected
    finally:
        srv.stop()


def test_code_off_and_bad_values(cdir):
    run = chronist.init(project="p", offline=True, code=None)
    assert "code" not in run._info["meta"]
    run.finish()
    with pytest.raises(ValueError):
        chronist.init(project="p", offline=True, code=1)


def test_server_rejects_bad_blobs(server):
    import httpx
    from chronist.codec import compress
    h = sha256(b"hello")
    r = httpx.put(f"{server.url}/api/blobs/{h}", content=compress(b"not hello"))
    assert r.status_code == 400
    r = httpx.put(f"{server.url}/api/blobs/{h}", content=compress(b"hello"))
    assert r.status_code == 200
    assert httpx.post(f"{server.url}/api/blobs/missing", json={"hashes": [h, sha256(b"x")]}).json()["missing"] == [sha256(b"x")]
    rid = httpx.post(f"{server.url}/api/runs", json={"uid": "u", "project": "p"}).json()["id"]
    assert httpx.post(f"{server.url}/api/runs/{rid}/code", json={"manifest": sha256(b"nope")}).status_code == 400
    assert httpx.get(f"{server.url}/api/runs/{rid}/code").status_code == 404


# ---------------------------------------------------------------- sharing without code
def test_share_tokens_hide_code(cdir, tmp_path, monkeypatch):
    import httpx
    from chronist.server.tokens import TokenStore
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    store = TokenStore(str(tmp_path / "tokens.json"))
    store.add("w", "write")
    srv = LiveServer(str(tmp_path / "data"), tokens=store).start()
    try:
        r = small_proj(tmp_path)
        run = chronist.init(project="p", url=srv.url, token="w", code=str(r))
        run.finish()
        W = {"Authorization": "Bearer w"}
        mk = lambda code: httpx.post(f"{srv.url}/api/tokens", headers=W, json={"code": code, "label": "x"}).json()
        no_code, with_code = mk(False)["token"], mk(True)["token"]
        assert json.loads((tmp_path / "tokens.json").read_text())[no_code]["code"] is False  # persisted
        h = lambda t: {"Authorization": f"Bearer {t}"}
        man = httpx.get(f"{srv.url}/api/runs/{run.id}/code", headers=W).json()
        fh = man["files"][0].get("sha256") or man["files"][1]["sha256"]
        for path in (f"/api/runs/{run.id}/code", f"/api/runs/{run.id}/code/file?path=model.py", f"/api/blobs/{fh}"):
            rr = httpx.get(srv.url + path, headers=h(no_code))
            assert rr.status_code == 403 and rr.json()["error"] == "code not shared", path
            assert httpx.get(srv.url + path, headers=h(with_code)).status_code == 200, path
        # run records don't leak snapshot details either
        rec = httpx.get(f"{srv.url}/api/runs/{run.id}", headers=h(no_code)).json()
        assert "code" not in rec["meta"] and rec["code"] is None
        lst = httpx.get(f"{srv.url}/api/runs", headers=h(no_code)).json()["runs"][0]
        assert "code" not in lst["meta"]
        assert "code" in httpx.get(f"{srv.url}/api/runs/{run.id}", headers=h(with_code)).json()["meta"]
        # metrics still work; read tokens can't manage shares or upload blobs
        assert httpx.post(f"{srv.url}/api/series", headers=h(no_code), json={"runs": [run.id], "keys": ["x"]}).status_code == 200
        assert httpx.post(f"{srv.url}/api/tokens", headers=h(with_code), json={}).status_code == 403
        assert httpx.get(f"{srv.url}/api/tokens", headers=h(with_code)).status_code == 403
        assert httpx.post(f"{srv.url}/api/blobs/missing", headers=h(with_code), json={"hashes": []}).status_code == 403
        assert httpx.get(f"{srv.url}/api/whoami", headers=h(no_code)).json() == {"auth": True, "role": "read", "code": False, "password_login": False}
        # python read-back with a no-code token: run works, code access says why it fails
        ref = chronist.run(run.id, url=srv.url, token=no_code)
        assert ref.name == run._info.get("name") or ref.name
        with pytest.raises(PermissionError):
            ref.code
        # list + revoke
        listed = httpx.get(f"{srv.url}/api/tokens", headers=W).json()["tokens"]
        assert {t["code"] for t in listed} == {True, False} and all("token" not in t for t in listed)
        tid = next(t["id"] for t in listed if not t["code"])
        assert httpx.delete(f"{srv.url}/api/tokens/{tid}", headers=W).status_code == 200
        assert httpx.get(f"{srv.url}/api/runs", headers=h(no_code)).status_code == 401
    finally:
        srv.stop()


def test_cannot_create_share_tokens_without_auth(server):
    import httpx
    r = httpx.post(f"{server.url}/api/tokens", json={"code": False})
    assert r.status_code == 400 and "without authentication" in r.json()["error"]
    assert httpx.get(f"{server.url}/api/whoami").json() == {"auth": False, "role": "write", "code": True, "password_login": False}


def test_legacy_tokens_file(tmp_path):
    from chronist.server.tokens import TokenStore
    (tmp_path / "t.json").write_text(json.dumps({"a": "write", "b": "read"}))
    s = TokenStore(str(tmp_path / "t.json"))
    assert s.lookup("a")["code"] is True and s.lookup("b")["code"] is False and s.lookup("c") is None
