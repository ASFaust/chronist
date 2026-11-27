"""Single-user password login: server, CLI and client."""
import json
import os
import stat
import time

import httpx
import pytest

import chronist
from chronist.cli import main
from chronist.client import auth as CA
from chronist.server.auth import PasswordFile
from chronist.server.tokens import TokenStore

from test_client import LiveServer

PW = "correct horse battery"


@pytest.fixture
def pwserver(tmp_path, monkeypatch):
    monkeypatch.setattr(TokenStore, "RELOAD_EVERY", 0.0)
    data = tmp_path / "data"
    data.mkdir()
    pw = PasswordFile(str(data / "auth.json"))
    pw.set(PW)
    tokens = TokenStore(str(data / "tokens.json"))
    s = LiveServer(str(data), tokens=tokens, password=pw).start()
    yield s
    s.stop()


@pytest.fixture(autouse=True)
def client_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CHRONIST_DIR", str(tmp_path / "client"))
    monkeypatch.delenv("CHRONIST_URL", raising=False)


def run_cli(args, capsys):
    with pytest.raises(SystemExit) as e:
        main(args)
    out = capsys.readouterr()
    return e.value.code, out.out + out.err


def test_password_file(tmp_path):
    pw = PasswordFile(str(tmp_path / "auth.json"))
    assert not pw.is_set() and not pw.verify("anything")
    with pytest.raises(ValueError):
        pw.set("short")
    pw.set(PW)
    assert pw.verify(PW) and not pw.verify(PW + "x")
    raw = (tmp_path / "auth.json").read_text()
    assert PW not in raw and "scrypt" in raw
    assert stat.S_IMODE(os.stat(tmp_path / "auth.json").st_mode) == 0o600


def test_login_flow(pwserver):
    u = pwserver.url
    assert httpx.get(f"{u}/api/runs").status_code == 401
    assert httpx.get(f"{u}/api/whoami").status_code == 401
    assert httpx.post(f"{u}/api/login", json={"password": "nope"}).status_code == 401
    r = httpx.post(f"{u}/api/login", json={"password": PW, "label": "laptop"})
    tok = r.json()["token"]
    h = {"Authorization": f"Bearer {tok}"}
    assert httpx.get(f"{u}/api/runs", headers=h).status_code == 200
    assert httpx.post(f"{u}/api/runs", headers=h, json={"uid": "x"}).status_code == 200  # write access
    me = httpx.get(f"{u}/api/whoami", headers=h).json()
    assert me == {"auth": True, "role": "write", "code": True, "password_login": True}
    entry = next(t for t in pwserver.tokens.list() if t["label"].startswith("login: laptop"))
    assert entry["login"] and entry["role"] == "write"
    assert httpx.post(f"{u}/api/logout", headers=h).status_code == 200
    assert httpx.get(f"{u}/api/runs", headers=h).status_code == 401


def test_login_throttle(pwserver):
    u = pwserver.url
    codes = [httpx.post(f"{u}/api/login", json={"password": "bad"}).status_code for _ in range(4)]
    assert codes[:3] == [401, 401, 401]
    r = httpx.post(f"{u}/api/login", json={"password": PW})
    assert r.status_code == 429 and r.json()["retry_after"] >= 1  # even the right one waits now


def test_live_token_and_password_changes(pwserver, tmp_path, capsys):
    u = pwserver.url
    other = TokenStore(str(tmp_path / "data" / "tokens.json"))  # e.g. `chronist tokens` in another process
    t = other.create("read", label="cli-made")
    time.sleep(0.05)
    assert httpx.get(f"{u}/api/runs", headers={"Authorization": f"Bearer {t}"}).status_code == 200
    login_tok = httpx.post(f"{u}/api/login", json={"password": PW}).json()["token"]
    # change the password + revoke login sessions while the server runs; share token survives
    import io
    import sys
    sys_stdin = sys.stdin
    sys.stdin = io.StringIO("a brand new password\n")
    try:
        code, out = run_cli(["passwd", "--data", str(tmp_path / "data"), "--revoke-sessions"], capsys)
    finally:
        sys.stdin = sys_stdin
    assert code == 0 and "revoked 1 login sessions" in out
    time.sleep(0.05)
    assert httpx.get(f"{u}/api/runs", headers={"Authorization": f"Bearer {login_tok}"}).status_code == 401
    assert httpx.get(f"{u}/api/runs", headers={"Authorization": f"Bearer {t}"}).status_code == 200
    assert httpx.post(f"{u}/api/login", json={"password": PW}).status_code == 401
    assert httpx.post(f"{u}/api/login", json={"password": "a brand new password"}).status_code == 200


def test_serve_first_start(tmp_path, monkeypatch, capsys):
    import chronist.cli as cli
    started = []
    monkeypatch.setattr(cli, "_run_server", lambda core, *a, **k: (started.append(k), core.close()))
    d = str(tmp_path / "d")
    # no terminal, no password: refuse with instructions
    monkeypatch.setattr(cli, "_interactive", lambda: False)
    code, out = run_cli(["serve", "--data", d], capsys)
    assert code == 2 and "chronist passwd" in out and not started
    # interactive: asks twice, re-asks on mismatch / too short
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    answers = iter(["short", "long enough pw", "something else", "long enough pw", "long enough pw"])
    monkeypatch.setattr("getpass.getpass", lambda prompt="": next(answers))
    code, out = run_cli(["serve", "--data", d], capsys)
    assert code == 0 and started[-1]["require_auth"] is True
    assert PasswordFile(os.path.join(d, "auth.json")).verify("long enough pw")
    # next start: no prompt
    monkeypatch.setattr("getpass.getpass", lambda prompt="": pytest.fail("prompted again"))
    assert run_cli(["serve", "--data", d], capsys)[0] == 0
    # env var for containers
    d2 = str(tmp_path / "d2")
    monkeypatch.setattr(cli, "_interactive", lambda: False)
    monkeypatch.setenv("CHRONIST_PASSWORD", "from the environment")
    assert run_cli(["serve", "--data", d2], capsys)[0] == 0
    assert PasswordFile(os.path.join(d2, "auth.json")).verify("from the environment")
    # --no-auth: no password needed
    monkeypatch.delenv("CHRONIST_PASSWORD")
    assert run_cli(["serve", "--data", str(tmp_path / "d3"), "--no-auth"], capsys)[0] == 0
    assert started[-1]["require_auth"] is False


def test_client_prompts_once_and_saves(pwserver, monkeypatch):
    monkeypatch.setattr(CA, "interactive", lambda: True)
    prompts = []
    monkeypatch.setattr("getpass.getpass", lambda prompt="": prompts.append(prompt) or PW)
    monkeypatch.setenv("CHRONIST_URL", pwserver.url)
    run = chronist.init(project="p", code=None)
    run.log({"x": 1.0})
    run.finish()
    assert len(prompts) == 1 and CA.stored_token(pwserver.url)
    creds = os.environ["CHRONIST_CREDENTIALS"]
    assert stat.S_IMODE(os.stat(creds).st_mode) == 0o600 and PW not in open(creds).read()
    # later: no prompt, read-back too
    monkeypatch.setattr("getpass.getpass", lambda prompt="": pytest.fail("prompted again"))
    r2 = chronist.init(project="p", code=None)
    r2.finish()
    assert len(chronist.runs("", project="p")) == 2


def test_cli_login_logout(pwserver, monkeypatch, capsys):
    answers = iter(["wrong one", PW])
    monkeypatch.setattr("getpass.getpass", lambda prompt="": next(answers))
    code, out = run_cli(["login", pwserver.url], capsys)
    assert code == 0 and "wrong password" in out and "logged in" in out
    tok = CA.stored_token(pwserver.url)
    assert httpx.get(f"{pwserver.url}/api/runs", headers={"Authorization": f"Bearer {tok}"}).status_code == 200
    code, out = run_cli(["logout", pwserver.url], capsys)
    assert code == 0 and CA.stored_token(pwserver.url) is None
    assert httpx.get(f"{pwserver.url}/api/runs", headers={"Authorization": f"Bearer {tok}"}).status_code == 401


def test_batch_job_without_login_then_sync(pwserver, monkeypatch, caplog, capsys):
    monkeypatch.setattr(CA, "interactive", lambda: False)
    monkeypatch.setenv("CHRONIST_URL", pwserver.url)
    with caplog.at_level("WARNING", logger="chronist"):
        run = chronist.init(project="p", code=None)
        run.log({"x": 1.0})
        run.finish(timeout=2)
    assert "chronist login" in caplog.text
    assert os.path.exists(run.dir.path)  # nothing uploaded; kept locally
    monkeypatch.setattr("getpass.getpass", lambda prompt="": PW)
    assert run_cli(["login"], capsys)[0] == 0  # uses $CHRONIST_URL
    chronist.sync(verbose=False)
    assert chronist.runs("", project="p")[0]["summary"]["x"]["value"] == 1.0
