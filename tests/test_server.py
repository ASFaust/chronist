import gzip
import json
import threading

import msgpack
import pytest
from starlette.testclient import TestClient

from chronist.server.app import create_app
from chronist.server.core import Core


@pytest.fixture
def core(tmp_path):
    c = Core(str(tmp_path / "data"))
    yield c
    c.close()


@pytest.fixture
def client(core):
    with TestClient(create_app(core)) as c:
        yield c


def mkrun(client, uid="u1", project="p", **kw):
    r = client.post("/api/runs", json={"project": project, "uid": uid, **kw})
    assert r.status_code == 200, r.text
    return r.json()


def log(client, rid, points, **kw):
    body = {"points": {k: {"step": list(s), "value": list(v)} for k, (s, v) in points.items()}}
    r = client.post(f"/api/runs/{rid}/log", json=body, **kw)
    assert r.status_code == 200, r.text
    return r.json()


def test_create_log_read(client):
    run = mkrun(client, config={"model": {"arch": "lstm"}, "seed": 1}, tags=["a"], name="first")
    assert run["created_new"] and run["name"] == "first"
    again = mkrun(client)
    assert again["id"] == run["id"] and not again["created_new"]
    rid = run["id"]
    log(client, rid, {"loss": (range(100), [1.0 / (i + 1) for i in range(100)])})
    log(client, rid, {"loss": (range(50, 150), [9.0] * 100)})  # half duplicate
    r = client.get(f"/api/runs/{rid}/metrics/loss").json()
    assert len(r["step"]) == 150 and r["last"][0] == 1.0 and r["last"][99] == 0.01
    r = client.get(f"/api/runs/{rid}/metrics/loss", params={"max_points": 10}).json()
    assert len(r["step"]) <= 10 and sum(r["count"]) == 150
    runs = client.get("/api/runs", params={"project": "p", "q": "model.arch = lstm"}).json()
    assert runs["total"] == 1
    got = runs["runs"][0]
    assert got["steps"] == 149 and got["summary"]["loss"]["value"] == 9.0
    assert got["config"] == {"model.arch": "lstm", "seed": 1}
    assert client.get(f"/api/runs/{rid}").json()["keys"] == ["loss"]


def test_msgpack_gzip_ingest(client):
    rid = mkrun(client)["id"]
    body = {"points": {"train/loss": {"step": [0, 1, 2], "value": [1.0, float("nan"), 3.0], "wall": [1.0, 2.0, 3.0]}}}
    raw = gzip.compress(msgpack.packb(body))
    r = client.post(f"/api/runs/{rid}/log", content=raw,
                    headers={"Content-Type": "application/msgpack", "Content-Encoding": "gzip"})
    assert r.json() == {"points": 3}
    r = client.get(f"/api/runs/{rid}/metrics/train/loss")
    d = r.json()
    assert d["last"] == [1.0, None, 3.0] and d["wall_time"] == [1.0, 2.0, 3.0]
    r = client.get(f"/api/runs/{rid}/metrics/train/loss", headers={"Accept": "application/msgpack"})
    assert msgpack.unpackb(r.content)["step"] == [0, 1, 2]


def test_errors(client):
    assert client.post("/api/runs/999/log", json={"points": {"x": {"step": [1], "value": [1]}}}).status_code == 404
    rid = mkrun(client)["id"]
    assert client.post(f"/api/runs/{rid}/log", json={"points": {"x": {"step": [1, 2], "value": [1]}}}).status_code == 400
    assert client.post(f"/api/runs/{rid}/log", content=b"garbage",
                       headers={"Content-Type": "application/msgpack"}).status_code == 400
    r = client.get("/api/runs", params={"q": "env ="})
    assert r.status_code == 400 and "expected a value" in r.json()["error"]
    assert client.post("/api/runs", json={"project": "p"}).status_code == 400


def test_lifecycle(client, core):
    rid = mkrun(client)["id"]
    assert client.post(f"/api/runs/{rid}/heartbeat").status_code == 200
    client.patch(f"/api/runs/{rid}", json={"name": "renamed", "notes": "n", "add_tags": ["x", "y"]})
    client.patch(f"/api/runs/{rid}", json={"remove_tags": ["x"]})
    client.post(f"/api/runs/{rid}/finish", json={"status": "finished"})
    r = client.get(f"/api/runs/{rid}").json()
    assert (r["name"], r["notes"], r["tags"], r["status"]) == ("renamed", "n", ["y"], "finished")
    assert client.post(f"/api/runs/{rid}/finish", json={"status": "weird"}).status_code == 400
    log(client, rid, {"x": ([0], [1.0])})
    assert client.delete(f"/api/runs/{rid}").status_code == 200
    assert client.get(f"/api/runs/{rid}").status_code == 404
    assert core.store.keys(rid) == []


def test_crash_detection(client, core):
    rid = mkrun(client)["id"]
    core.meta.w.execute("UPDATE runs SET heartbeat = heartbeat - 10000 WHERE id=?", (rid,))
    assert core.writer.submit(core.op_mark_crashed).result() == [rid]
    assert client.get(f"/api/runs/{rid}").json()["status"] == "crashed"
    client.post(f"/api/runs/{rid}/heartbeat")
    assert client.get(f"/api/runs/{rid}").json()["status"] == "running"


def test_series_and_grouping(client):
    ids = {}
    for arch in ("lstm", "gru"):
        for seed in range(3):
            rid = mkrun(client, uid=f"{arch}{seed}", config={"arch": arch, "seed": seed})["id"]
            ids[(arch, seed)] = rid
            n = 1000 if arch == "lstm" else 500
            off = 0 if arch == "lstm" else 10
            log(client, rid, {"loss": (range(n), [off + seed + 0.0] * n), "acc": (range(n), [float(s) for s in range(n)])})
    r = client.post("/api/series", json={"q": "", "project": "p", "keys": ["loss"], "max_points": 100,
                                         "group_by": ["arch"]}).json()
    assert len(r["series"]) == 6
    assert all(len(s["step"]) <= 100 for s in r["series"])
    assert r["start"] == 0 and r["end"] == 1000
    groups = {g["group"]["arch"]: g for g in r["groups"]}
    assert groups["lstm"]["center"][0] == pytest.approx(1.0)
    assert groups["lstm"]["lo"][0] == pytest.approx(1 - (2 / 3) ** 0.5)
    assert groups["gru"]["center"][0] == pytest.approx(11.0)
    assert max(groups["gru"]["step"]) < 500 and groups["lstm"]["n"][0] == 3
    # x = another metric
    r = client.post("/api/series", json={"runs": [ids[("lstm", 0)]], "keys": ["loss"], "x": "acc",
                                         "max_points": None}).json()
    s = r["series"][0]
    assert s["x"][:3] == [0.0, 1.0, 2.0]
    r = client.post("/api/series", json={"runs": [ids[("lstm", 0)]], "keys": ["loss"], "agg": "median_iqr",
                                         "group_by": ["arch"], "x": "relative"}).json()
    assert r["groups"][0]["center"][0] == 0.0


def test_suggest_keys_projects(client):
    mkrun(client, uid="a", config={"model": {"arch": "lstm"}})
    mkrun(client, uid="b", config={"model": {"arch": "gru"}})
    r = client.get("/api/suggest", params={"project": "p", "text": "model.arch = "}).json()
    assert {i["label"] for i in r["items"]} == {"lstm", "gru"}
    k = client.get("/api/keys", params={"project": "p"}).json()
    assert k["config"][0]["key"] == "model.arch"
    assert client.get("/api/projects").json()["projects"][0]["runs"] == 2


def test_export(client):
    rid = mkrun(client, name="r1")["id"]
    log(client, rid, {"loss": (range(10), [float(i) for i in range(10)])})
    r = client.get("/api/export", params={"project": "p", "format": "csv", "keys": "loss"})
    lines = r.text.strip().splitlines()
    assert lines[0].startswith("run_id,run,key,step,value") and len(lines) == 11
    r = client.get("/api/export", params={"project": "p"})
    import io

    import pyarrow.parquet as pq
    t = pq.read_table(io.BytesIO(r.content))
    assert t.num_rows == 10 and t.column("run")[0].as_py() == "r1"


def test_auth(core):
    app = create_app(core, tokens={"w": "write", "r": "read"})
    with TestClient(app) as c:
        assert c.get("/api/runs").status_code == 401
        assert c.get("/api/runs", headers={"Authorization": "Bearer nope"}).status_code == 401
        assert c.get("/api/runs", headers={"Authorization": "Bearer r"}).status_code == 200
        assert c.get("/api/runs?token=r").status_code == 200
        assert c.post("/api/runs", json={"uid": "x"}, headers={"Authorization": "Bearer r"}).status_code == 403
        assert c.post("/api/runs", json={"uid": "x"}, headers={"Authorization": "Bearer w"}).status_code == 200
        assert c.post("/api/series", json={"runs": [1], "keys": ["a"]},
                      headers={"Authorization": "Bearer r"}).status_code == 200
        assert c.get("/api/health").status_code == 401
        assert c.get("/").status_code == 200  # static UI needs no token


def test_concurrent_writers(client, core):
    rids = [mkrun(client, uid=f"c{i}")["id"] for i in range(8)]
    errors = []

    def work(rid):
        try:
            for lo in range(0, 2000, 100):
                for _ in range(2):  # every batch retried
                    log(client, rid, {f"m{j}": (range(lo, lo + 100), [1.0] * 100) for j in range(5)})
        except Exception as e:  # pragma: no cover
            errors.append(e)

    ts = [threading.Thread(target=work, args=(r,)) for r in rids]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errors
    assert core.store.stats().points == 8 * 5 * 2000
