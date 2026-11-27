"""CLI smoke tests: every command parses, and the snapshot commands work end to end."""
import json
import os

import pytest

import chronist
from chronist.cli import main

from test_client import LiveServer


def run_cli(args, capsys):
    with pytest.raises(SystemExit) as e:
        main(args)
    out = capsys.readouterr()
    return e.value.code, out.out + out.err


@pytest.mark.parametrize("cmd", [["ui"], ["serve"], ["sync"], ["tokens"], ["checkout"], ["snapshot"],
                                 ["code", "diff"]])
def test_help(cmd, capsys):
    code, out = run_cli(cmd + ["--help"], capsys)
    assert code == 0 and "usage" in out


def test_tokens(tmp_path, capsys):
    d = str(tmp_path / "data")
    code, out = run_cli(["tokens", "--data", d, "--new", "read", "--code", "--label", "lab"], capsys)
    assert code == 0 and "can see code" in out
    code, out = run_cli(["tokens", "--data", d, "--new", "read"], capsys)
    code, out = run_cli(["tokens", "--data", d], capsys)
    lines = out.strip().splitlines()
    assert len(lines) == 2 and any("read+code" in l and "lab" in l for l in lines)
    tid = lines[0].split()[0]
    assert run_cli(["tokens", "--data", d, "--revoke", tid], capsys)[0] == 0
    assert len(run_cli(["tokens", "--data", d], capsys)[1].strip().splitlines()) == 1
    assert run_cli(["tokens", "--data", d, "--revoke", "nope"], capsys)[0] == 1


def test_checkout_and_diff(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CHRONIST_DIR", str(tmp_path / "client"))
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    monkeypatch.delenv("CHRONIST_URL", raising=False)
    monkeypatch.delenv("CHRONIST_CODE", raising=False)
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.py").write_text("A = 1\n")
    srv = LiveServer(str(tmp_path / "data")).start()
    try:
        r1 = chronist.init(project="p", url=srv.url, code=str(src), config={"lr": 1})
        r1.finish()
        (src / "a.py").write_text("A = 2\n")
        r2 = chronist.init(project="p", url=srv.url, code=str(src), config={"lr": 2})
        r2.finish()
        dest = tmp_path / "co"
        code, out = run_cli(["checkout", str(r1.id), str(dest), "--url", srv.url], capsys)
        assert code == 0 and (dest / "a.py").read_text() == "A = 1\n"
        assert json.loads((dest / "CHRONIST_RUN.json").read_text())["run"]["id"] == r1.id
        assert run_cli(["checkout", str(r1.id), str(dest), "--url", srv.url], capsys)[0] == 1  # non-empty
        code, out = run_cli(["code", "diff", str(r1.id), str(r2.id), "--url", srv.url], capsys)
        assert code == 0 and "M a.py" in out and "-A = 1" in out and "M config lr: 1 -> 2" in out
        code, out = run_cli(["code", "diff", str(r1.id), str(r2.id), "--data", srv.data_dir, "--stat"], capsys)
        assert code == 0 and "M a.py" in out and "-A = 1" not in out
        monkeypatch.chdir(src)
        code, out = run_cli(["snapshot", "--run", str(r2.id), "--url", srv.url], capsys)
        assert code == 0 and "attached to run" in out
    finally:
        srv.stop()


def test_prune_and_gc_cli(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CHRONIST_DIR", str(tmp_path / "client"))
    monkeypatch.setenv("CHRONIST_KEEP_LOCAL", "1")
    monkeypatch.delenv("CHRONIST_URL", raising=False)
    srv = LiveServer(str(tmp_path / "data")).start()
    try:
        run = chronist.init(project="p", url=srv.url, code=None)
        run.log({"x": 1.0})
        run.finish()
        code, out = run_cli(["prune", "--dry-run"], capsys)
        assert code == 0 and "would prune 1 runs" in out and os.path.exists(run.dir.path)
        code, out = run_cli(["prune", "--url", "http://elsewhere:1"], capsys)
        assert "pruned 0 runs" in out
        code, out = run_cli(["prune"], capsys)
        assert "pruned 1 runs" in out and not os.path.exists(run.dir.path)
        # gc: direct access refused while the server holds the data dir
        code, out = run_cli(["gc", "--data", srv.data_dir], capsys)
        assert code == 1 and "in use" in out
        code, out = run_cli(["gc", "--url", srv.url], capsys)
        assert code == 0 and "removed 0 blobs" in out
    finally:
        srv.stop()
    code, out = run_cli(["gc", "--data", str(tmp_path / "data"), "--vacuum"], capsys)
    assert code == 0 and "full vacuum" in out
