import socket
import threading
import time

import numpy as np
import pytest
import uvicorn

import chronist
from chronist.client.rundir import RunDir
from chronist.server.app import create_app
from chronist.server.core import Core


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class LiveServer:
    def __init__(self, data_dir, port=None, tokens=None):
        self.data_dir = data_dir
        self.port = port or free_port()
        self.tokens = tokens
        self.url = f"http://127.0.0.1:{self.port}"

    def start(self):
        self.core = Core(self.data_dir)
        cfg = uvicorn.Config(create_app(self.core, tokens=self.tokens), host="127.0.0.1",
                             port=self.port, log_level="error")
        self.server = uvicorn.Server(cfg)
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        while not self.server.started:
            time.sleep(0.02)
        return self

    def stop(self):
        self.server.should_exit = True
        self.thread.join(10)
        self.core.close()


@pytest.fixture
def server(tmp_path):
    s = LiveServer(str(tmp_path / "data")).start()
    yield s
    s.stop()


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("CHRONIST_DIR", str(tmp_path / "client"))
    monkeypatch.delenv("CHRONIST_URL", raising=False)
    monkeypatch.delenv("CHRONIST_TOKEN", raising=False)


def wait_for(cond, timeout=10):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.05)
    return False


def test_online_logging(server):
    run = chronist.init(project="p", name="r1", url=server.url, tags=["t"],
                        config={"model": {"arch": "lstm"}, "lr": 1e-3})
    for i in range(1000):
        run.log({"loss": 1.0 / (i + 1), "train": {"acc": np.float32(i / 1000)}})
    run.log({"eval": 0.5}, step=999)
    run.finish()
    rs = chronist.runs("model.arch = lstm", project="p", url=server.url)
    assert len(rs) == 1
    r = rs[0]
    assert r["status"] == "finished" and r["steps"] == 999 and r["tags"] == ["t"]
    assert r["meta"]["hostname"] == socket.gethostname()
    assert "git_commit" in r["meta"]
    df = rs.history(["loss", "train/acc"])
    assert len(df) == 2000 and set(df["key"]) == {"loss", "train/acc"}
    assert df[df.key == "loss"]["value"].iloc[0] == 1.0
    df = rs.history(["loss"], max_points=50)
    assert len(df) <= 50 and df["count"].sum() == 1000
    cfg = rs.config()
    assert cfg["model.arch"].tolist() == ["lstm"]


def test_context_manager_crash(server):
    with pytest.raises(ZeroDivisionError):
        with chronist.init(project="p", url=server.url) as run:
            run.log({"x": 1})
            1 / 0
    r = chronist.runs("", project="p", url=server.url)[0]
    assert r["status"] == "crashed"


def test_offline_then_sync(server, tmp_path):
    run = chronist.init(project="off", config={"a": 1})
    assert run.id is None
    for i in range(500):
        run.log({"loss": float(i)}, step=i * 10)
    run.finish()
    res = chronist.sync(url=server.url, verbose=False)
    assert res[0]["points"] == 500
    rs = chronist.runs("a = 1", project="off", url=server.url)
    assert rs[0]["status"] == "finished" and rs[0]["steps"] == 4990
    # re-sync uploads nothing new
    assert chronist.sync(url=server.url, verbose=False)[0]["points"] == 0


def test_server_down_then_up(tmp_path):
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    run = chronist.init(project="p", url=url)
    run._uploader.interval = 0.05
    for i in range(300):
        run.log({"x": float(i)})
    time.sleep(0.3)
    assert run._uploader.error is not None  # buffering, not crashing
    srv = LiveServer(str(tmp_path / "data"), port=port).start()
    try:
        run._uploader.wake()
        for i in range(300, 600):
            run.log({"x": float(i)})
        run.finish(timeout=20)
        rs = chronist.runs("", project="p", url=url)
        assert len(rs.history(["x"])) == 600
    finally:
        srv.stop()


def test_finish_while_server_down_then_sync(tmp_path):
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    run = chronist.init(project="p", url=url)
    for i in range(100):
        run.log({"x": float(i)})
    run.finish(timeout=0.5)
    srv = LiveServer(str(tmp_path / "data"), port=port).start()
    try:
        chronist.sync(url=url, verbose=False)
        rs = chronist.runs("", project="p", url=url)
        assert rs[0]["status"] == "finished" and len(rs.history("x")) == 100
    finally:
        srv.stop()


def test_resume(server):
    run = chronist.init(project="p", url=server.url)
    for i in range(10):
        run.log({"x": float(i)})
    run.finish()
    uid = run.uid
    run2 = chronist.init(project="p", url=server.url, resume=uid)
    assert run2.step == 10
    run2.log({"x": 10.0})
    assert wait_for(lambda: run2.id is not None)
    run2.finish()
    rs = chronist.runs("", project="p", url=server.url)
    assert len(rs) == 1 and rs[0]["steps"] == 10 and rs[0]["status"] == "finished"


def test_resume_from_other_machine(server, tmp_path, monkeypatch):
    run = chronist.init(project="p", url=server.url)
    run.log({"x": 1.0}, step=41)
    run.finish()
    monkeypatch.setenv("CHRONIST_DIR", str(tmp_path / "other"))
    run2 = chronist.init(project="p", url=server.url, resume=run.uid)
    assert run2.step == 42
    run2.log({"x": 2.0})
    run2.finish()
    df = chronist.runs("", project="p", url=server.url).history("x")
    assert df["step"].tolist() == [41, 42]


def test_every_and_interval(tmp_path):
    run = chronist.init(project="p", offline=True)
    for i in range(100):
        run.log({"a": 1.0}, every=10)
    for i in range(5):
        run.log({"b": 1.0}, interval=3600)
    run.finish()
    recs = [r for r in RunDir(run.dir.path).all_records() if r[0] == "l"]
    assert [r[1] for r in recs if "a" in r[3]] == list(range(0, 100, 10))
    assert len([r for r in recs if "b" in r[3]]) == 1


def test_local_readback(server):
    run = chronist.init(project="p", url=server.url, config={"k": "v"})
    for i in range(100):
        run.log({"m": float(i)})
    run.finish()
    rs = chronist.runs('k = "v"', project="p", data_dir=server.data_dir)
    assert len(rs) == 1
    df = rs.history(["m"], max_points=10)
    assert df["count"].sum() == 100
    pl_df = chronist.runs("", project="p", data_dir=server.data_dir, dataframe="polars").history()
    assert pl_df.height == 100


def test_sync_skips_live_uploader(server):
    run = chronist.init(project="p", url=server.url)
    run.log({"x": 1.0})
    res = chronist.sync(url=server.url, verbose=False)
    assert "skipped" in res[0]
    run.finish()


def test_auth_client(tmp_path):
    srv = LiveServer(str(tmp_path / "data"), tokens={"w": "write", "r": "read"}).start()
    try:
        run = chronist.init(project="p", url=srv.url, token="w")
        run.log({"x": 1.0})
        run.finish()
        assert len(chronist.runs("", project="p", url=srv.url, token="r")) == 1
        with pytest.raises(RuntimeError):
            chronist.runs("", project="p", url=srv.url, token="bad")
    finally:
        srv.stop()
