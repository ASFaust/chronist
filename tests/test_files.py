"""Files attached to runs (docs/HISTOGRAMS_FILES.md part 2)."""
import hashlib
import json
import os

import httpx
import pytest

import chronist
from chronist.client import files as F
from chronist.client.rundir import RunDir

from test_client import LiveServer, free_port

CHUNK = 1024


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("CHRONIST_DIR", str(tmp_path / "client"))
    monkeypatch.setenv("CHRONIST_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("CHRONIST_URL", raising=False)
    monkeypatch.delenv("CHRONIST_KEEP_LOCAL", raising=False)
    monkeypatch.setattr(F, "CHUNK_SIZE", CHUNK)


@pytest.fixture
def server(tmp_path):
    s = LiveServer(str(tmp_path / "data")).start()
    yield s
    s.stop()


def sha(b):
    return hashlib.sha256(b).hexdigest()


def blob_count(path):
    return sum(len(fs) for _, _, fs in os.walk(path)) if os.path.isdir(path) else 0


def test_capture_overwrite_move_chunking(server, tmp_path):
    src = tmp_path / "model.pt"
    run = chronist.init(project="p", url=server.url, code=None)
    sizes = {"exact": 3 * CHUNK, "plus1": 3 * CHUNK + 1, "empty": 0, "small": 10}
    contents = {}
    for name, n in sizes.items():
        data = os.urandom(n)
        src.write_bytes(data)
        run.save_file(src, name=name, step=n, kind="checkpoint", metadata={"n": n})
        src.write_bytes(b"overwritten right away")  # must not affect the saved version
        contents[name] = data
    mv = tmp_path / "moved.bin"
    mv.write_bytes(b"moved content")
    run.save_file(mv, move=True)
    assert not mv.exists()
    run.finish()
    r = chronist.run(run.id, url=server.url)
    df = r.files()
    assert set(df["name"]) == set(sizes) | {"moved.bin"}
    for name, data in contents.items():
        p = r.download(name, dest=str(tmp_path / "dl") + os.sep)
        assert open(p, "rb").read() == data, name
    assert open(r.download("moved.bin", dest=str(tmp_path / "x.bin")), "rb").read() == b"moved content"
    row = df[df.name == "exact"].iloc[0]
    assert row["kind"] == "checkpoint" and row["metadata"] == {"n": 3 * CHUNK} and row["aliases"] == ["latest"]


def test_save_object_roundtrip(server):
    torch = pytest.importorskip("torch")
    run = chronist.init(project="p", url=server.url, code=None)
    sd = {"w": torch.arange(6.0).reshape(2, 3), "step": 7}
    run.save_object(sd, name="model", step=7, aliases=["best"])
    run.finish()
    obj = chronist.run(run.id, url=server.url).load_object("model", alias="best")
    assert torch.equal(obj["w"], sd["w"]) and obj["step"] == 7


def test_pickle_object_without_torch(server, monkeypatch):
    import sys
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    run = chronist.init(project="p", url=server.url, code=None)
    run.save_object({"a": [1, 2, 3]}, name="state")
    run.finish()
    assert chronist.run(run.id, url=server.url).load_object("state") == {"a": [1, 2, 3]}


def test_dedup_and_aliases_and_retention(server, tmp_path):
    src = tmp_path / "ckpt.bin"
    run = chronist.init(project="p", url=server.url, code=None)
    run.configure_files("model", keep_last=2, keep_every=1000)
    same = os.urandom(5 * CHUNK)
    for step in range(0, 3000, 250):  # 12 versions
        src.write_bytes(same if step in (0, 250) else os.urandom(2 * CHUNK) + step.to_bytes(4, "little"))
        run.save_file(src, name="model", step=step, aliases=["best"] if step == 500 else [])
    run.finish()
    rows = chronist.run(run.id, url=server.url).files(raw=True)
    kept = sorted(r["step"] for r in rows)
    # keep_last=2 -> 2500, 2750; keep_every=1000 -> 0, 1000, 2000; alias best -> 500
    assert kept == [0, 500, 1000, 2000, 2500, 2750]
    by_alias = {a: r["step"] for r in rows for a in r["aliases"]}
    assert by_alias == {"best": 500, "latest": 2750}
    # move "best" to a newer version
    vid = next(r["version_id"] for r in rows if r["step"] == 2000)
    httpx.patch(f"{server.url}/api/runs/{run.id}/files/model/aliases", json={"set": {"best": vid}}).raise_for_status()
    rows = chronist.run(run.id, url=server.url).files(raw=True)
    assert {a: r["step"] for r in rows for a in r["aliases"]}["best"] == 2000
    assert httpx.patch(f"{server.url}/api/runs/{run.id}/files/model/aliases",
                       json={"set": {"latest": vid}}).status_code == 400
    # identical content saved twice -> its chunks are stored once
    prog = json.load(open(os.path.join(run.dir.path, "sync.json"))) if os.path.exists(run.dir.path) else None
    assert prog is None  # pruned after full upload (files included)
    # delete a version; latest moves back
    latest_vid = next(r["version_id"] for r in rows if "latest" in r["aliases"])
    httpx.delete(f"{server.url}/api/runs/{run.id}/files/model/{latest_vid}").raise_for_status()
    rows = chronist.run(run.id, url=server.url).files(raw=True)
    assert {a: r["step"] for r in rows for a in r["aliases"]}["latest"] == 2500


def test_dedup_counts(server, tmp_path, monkeypatch):
    monkeypatch.setenv("CHRONIST_KEEP_LOCAL", "1")
    src = tmp_path / "a.bin"
    data = os.urandom(4 * CHUNK)
    src.write_bytes(data)
    run = chronist.init(project="p", url=server.url, code=None)
    run.save_file(src, name="a")
    run.save_file(src, name="b")
    run.finish()
    before = blob_count(server.core.blobs.path)
    # 4 chunks + 2 manifests (names differ) -- not 8 chunks
    assert before == 6
    r2 = chronist.init(project="p", url=server.url, code=None)
    r2.save_file(src, name="a")
    r2.finish()
    assert blob_count(server.core.blobs.path) == before + 1  # only the new manifest


def test_offline_sync_resume_and_cache(tmp_path, monkeypatch):
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    src = tmp_path / "f.bin"
    data = os.urandom(6 * CHUNK + 5)
    src.write_bytes(data)
    run = chronist.init(project="p", url=url, code=None)
    run.save_file(src, name="weights", step=1)
    run.log({"x": 1.0})
    run.finish(timeout=0.3)  # server down: nothing uploaded, local copy kept
    assert os.path.exists(run.dir.path)
    srv = LiveServer(str(tmp_path / "data"), port=port).start()
    try:
        # half the chunks already on the server (interrupted earlier upload)
        local = chronist.blobs.FsBlobStore(RunDir(run.dir.path).blobs_dir)
        e = F.staged(RunDir(run.dir.path))[0]
        chunks = json.loads(local.read(e["manifest"]))["chunks"]
        for h in chunks[:3]:
            srv.core.blobs.put(h, local.get(h))
        n0 = blob_count(srv.core.blobs.path)
        chronist.sync(url=url, verbose=False)
        assert blob_count(srv.core.blobs.path) == n0 + (len(chunks) - 3) + 1  # rest + manifest
        assert not os.path.exists(run.dir.path)  # fully uploaded -> pruned
        r = chronist.run(run.uid, url=url)
        p = r.download("weights", dest=str(tmp_path / "out.bin"))
        assert open(p, "rb").read() == data
        # second download into a new place: served from the chunk cache, no network
        calls = []
        orig = r._src.file_chunk
        monkeypatch.setattr(r._src, "file_chunk", lambda *a: calls.append(a) or orig(*a))
        p2 = r.download("weights", dest=str(tmp_path / "again.bin"))
        assert open(p2, "rb").read() == data and calls == []
        # HTTP download with Range
        vid = r.files(raw=True)[0]["version_id"]
        full = httpx.get(f"{url}/api/files/{vid}/download")
        assert full.content == data and full.headers["accept-ranges"] == "bytes"
        part = httpx.get(f"{url}/api/files/{vid}/download", headers={"Range": f"bytes={CHUNK - 3}-{2 * CHUNK + 2}"})
        assert part.status_code == 206 and part.content == data[CHUNK - 3: 2 * CHUNK + 3]
        tail = httpx.get(f"{url}/api/files/{vid}/download", headers={"Range": "bytes=-7"})
        assert tail.content == data[-7:]
        assert httpx.get(f"{url}/api/files/{vid}/download", headers={"Range": "bytes=999999-"}).status_code == 416
        # CLI pull
        from chronist.cli import main
        with pytest.raises(SystemExit) as ex:
            main(["pull", run.uid, "weights", "--dest", str(tmp_path / "pulled") + os.sep, "--url", url])
        assert ex.value.code == 0 and open(tmp_path / "pulled" / "f.bin", "rb").read() == data
    finally:
        srv.stop()


def test_download_verifies(server, tmp_path):
    src = tmp_path / "f.bin"
    src.write_bytes(os.urandom(3 * CHUNK))
    run = chronist.init(project="p", url=server.url, code=None)
    run.save_file(src, name="f")
    run.finish()
    r = chronist.run(run.id, url=server.url)
    orig = r._src.file_chunk
    from chronist.codec import compress_maybe
    r._src.file_chunk = lambda vid, i, h: compress_maybe(b"tampered") if i == 1 else orig(vid, i, h)
    with pytest.raises(IOError, match="verification"):
        r.download("f", dest=str(tmp_path / "bad.bin"))
    assert not os.path.exists(tmp_path / "bad.bin")


def test_gc_respects_code_and_other_runs(server, tmp_path, monkeypatch):
    import time as _t
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    code = tmp_path / "code"
    code.mkdir()
    shared = os.urandom(2 * CHUNK)
    (code / "shared.bin.txt").write_bytes(b"x")  # code file
    src = tmp_path / "s.bin"
    src.write_bytes(shared)
    r1 = chronist.init(project="p", url=server.url, code=str(code))
    r1.save_file(src, name="m")
    r1.finish()
    r2 = chronist.init(project="p", url=server.url, code=str(code))  # same code snapshot
    r2.save_file(src, name="m")  # same chunks
    only1 = tmp_path / "only1.bin"
    only1.write_bytes(os.urandom(CHUNK))
    r2.finish()
    r3 = chronist.init(project="p", url=server.url, code=None)
    r3.save_file(only1, name="solo")
    r3.finish()
    old = _t.time() - 3 * 86400
    for root, _, names in os.walk(server.core.blobs.path):
        for n in names:
            os.utime(os.path.join(root, n), (old, old))
    httpx.delete(f"{server.url}/api/runs/{r1.id}").raise_for_status()
    httpx.delete(f"{server.url}/api/runs/{r3.id}").raise_for_status()
    res = httpx.post(f"{server.url}/api/admin/gc", json={}).json()
    assert res["blobs_removed"] == 3  # r1's file manifest, r3's chunk + manifest; shared chunks + code stay
    r = chronist.run(r2.id, url=server.url)
    assert open(r.download("m", dest=str(tmp_path / "back.bin")), "rb").read() == shared
    assert r.code.files() == ["shared.bin.txt"]


def test_quota_and_max_size(tmp_path):
    from chronist.server.core import Core
    from chronist.server.app import create_app
    from starlette.testclient import TestClient
    core = Core(str(tmp_path / "d"), max_blob_storage=10 * CHUNK, max_file_size=8 * CHUNK)
    try:
        c = TestClient(create_app(core))
        rid = c.post("/api/runs", json={"uid": "u", "project": "p"}).json()["id"]
        h = sha(b"x")
        r = c.post("/api/blobs/missing", json={"hashes": [h], "purpose": "file", "size": 9 * CHUNK})
        assert r.status_code == 413
        core._blob_usage = 9 * CHUNK
        r = c.post("/api/blobs/missing", json={"hashes": [h], "purpose": "file", "size": 2 * CHUNK})
        assert r.status_code == 507 and "metrics still work" in r.json()["error"]
        # code snapshots / plain blob checks and metrics are unaffected
        assert c.post("/api/blobs/missing", json={"hashes": [h]}).status_code == 200
        assert c.post(f"/api/runs/{rid}/log", json={"points": {"x": {"step": [0], "value": [1.0]}}}).status_code == 200
    finally:
        core.close()


def test_quota_client_keeps_logging(tmp_path, monkeypatch):
    monkeypatch.setenv("CHRONIST_KEEP_LOCAL", "1")
    srv = LiveServer(str(tmp_path / "data")).start()
    try:
        srv.core.max_blob_storage = 1  # full
        src = tmp_path / "f.bin"
        src.write_bytes(os.urandom(2 * CHUNK))
        run = chronist.init(project="p", url=srv.url, code=None)
        run.save_file(src, name="big")
        run.log({"x": 1.0})
        run.finish()
        r = chronist.run(run.id, url=srv.url)
        assert r.files(raw=True) == [] and r.summary["x"]["value"] == 1.0
        assert os.path.exists(run.dir.path)  # file not uploaded -> local copy kept
    finally:
        srv.stop()


def test_read_token_can_download_without_code(tmp_path):
    srv = LiveServer(str(tmp_path / "data"), tokens={"w": "write", "r": "read"}).start()
    try:
        src = tmp_path / "f.bin"
        src.write_bytes(b"hello file")
        run = chronist.init(project="p", url=srv.url, token="w", code=None)
        run.save_file(src, name="f")
        run.finish()
        r = chronist.run(run.id, url=srv.url, token="r")
        assert open(r.download("f", dest=str(tmp_path / "o")), "rb").read() == b"hello file"
        vid = r.files(raw=True)[0]["version_id"]
        assert httpx.delete(f"{srv.url}/api/runs/{run.id}/files/f/{vid}",
                            headers={"Authorization": "Bearer r"}).status_code == 403
    finally:
        srv.stop()
