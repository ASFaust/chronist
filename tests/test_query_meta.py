import time

import pytest

from chronist import query as Q
from chronist.meta import MetadataStore, flatten


@pytest.fixture
def meta(tmp_path):
    m = MetadataStore(str(tmp_path / "meta.db"))
    runs = [
        ("a", {"env": "seaquest", "model": {"arch": "lstm", "layers": 2}, "optim": {"lr": 3e-4}, "seed": 1, "fast": True}, ["baseline"]),
        ("b", {"env": "seaquest", "model": {"arch": "gru", "layers": 4}, "optim": {"lr": 1e-3}, "seed": 2, "fast": False}, []),
        ("c", {"env": "pong", "model": {"arch": "lstm", "layers": 2, "dropout": 0.1}, "optim": {"lr": 3e-4}, "seed": 3, "note": None}, ["baseline", "v2"]),
        ("d", {"env": "pong", "model": {"arch": "transformer"}, "seed": 4, "dims": [64, 64]}, []),
    ]
    for i, (name, cfg, tags) in enumerate(runs):
        rid, new = m.create_run("proj", f"uid-{name}", name=name, config=cfg, tags=tags)
        assert new
        m.update_summary(rid, {"loss": (1000 * (i + 1), 1.0 / (i + 1)), "return": (10 * (i + 1), 5.0 * i)})
    m.create_run("other", "uid-x", name="x", config={"env": "seaquest"})
    m.finish(2)
    yield m
    m.close()


def names(meta, q, project="proj"):
    ids = meta.query_ids(project, q, order="name", desc=False)
    return [r["name"] for r in meta.get_runs(ids)]


@pytest.mark.parametrize("q,expected", [
    ("", ["a", "b", "c", "d"]),
    ('env = "seaquest"', ["a", "b"]),
    ('env = seaquest and model.arch = "lstm"', ["a"]),
    ('env = "seaquest" or model.arch = "lstm"', ["a", "b", "c"]),
    ("model.layers >= 3", ["b"]),
    ("optim.lr < 5e-4", ["a", "c"]),
    ("optim.lr = 0.0003", ["a", "c"]),
    ("model.dropout exists", ["c"]),
    ("not model.dropout exists", ["a", "b", "d"]),
    ('model.arch != "lstm"', ["b", "d"]),
    ('model.arch in ["gru", "transformer"]', ["b", "d"]),
    ('model.arch not in [gru, transformer]', ["a", "c"]),
    ('model.arch contains "LST"', ["a", "c"]),
    ("fast = true", ["a"]),
    ("fast = false", ["b"]),
    ("note = null", ["c"]),
    ('tags contains "baseline"', ["a", "c"]),
    ('tags = v2', ["c"]),
    ("tags exists", ["a", "c"]),
    ("steps >= 2000", ["b", "c", "d"]),
    ("steps > 2000 and steps < 4000", ["c"]),
    ("last(loss) < 0.4", ["c", "d"]),
    ("last(return) >= 10", ["c", "d"]),
    ("status = finished", ["b"]),
    ("status in [running]", ["a", "c", "d"]),
    ('name = "a" or name contains "D"', ["a", "d"]),
    ("(env = pong or fast = true) and not (model.arch = transformer)", ["a", "c"]),
    ("id = 1", ["a"]),
    ('dims contains "64"', ["d"]),
    ("created > 2000-01-01", ["a", "b", "c", "d"]),
    ('created < "2000-01-01T00:00"', []),
])
def test_queries(meta, q, expected):
    assert names(meta, q) == expected


def test_project_filter(meta):
    assert names(meta, 'env = "seaquest"', project="other") == ["x"]
    assert names(meta, 'env = "seaquest"', project=None) == ["a", "b", "x"]


@pytest.mark.parametrize("bad", [
    "env =", "env seaquest", "(env = 1", "last(loss) < abc", "steps > foo",
    "env = 'unterminated", "and", "env = 1 2", "tags < 3", "fast < true", "env = [1]",
])
def test_errors(bad):
    with pytest.raises(Q.QueryError):
        Q.to_sql(bad)


def test_no_injection(meta):
    q = 'name = "x\\" OR 1=1 --"'
    assert names(meta, q) == []
    sql, params = Q.to_sql('`weird key"; DROP TABLE runs;` = 1')
    assert "DROP" not in sql and params[0] == 'weird key"; DROP TABLE runs;'


def test_flatten():
    assert flatten({"a": {"b": 1, "c": {"d": "x"}}, "e": [1, 2], "f": {}}) == \
        {"a.b": 1, "a.c.d": "x", "e": [1, 2], "f": {}}


def test_resume_and_status(meta):
    rid, new = meta.create_run("proj", "uid-b", config={"extra": 1})
    assert not new and rid == 2
    r = meta.get_run(2)
    assert r["status"] == "running" and r["config"]["extra"] == 1 and r["config"]["env"] == "seaquest"
    meta.w.execute("UPDATE runs SET heartbeat=? WHERE id=1", (time.time() - 1000,))
    assert meta.mark_crashed(300) == [1]
    assert meta.get_run(1)["status"] == "crashed"
    meta.heartbeat(1)
    assert meta.get_run(1)["status"] == "running"


def test_summary_keeps_latest(meta):
    meta.update_summary(1, {"loss": (500, 99.0)})  # older step, ignored
    assert meta.get_run(1)["summary"]["loss"] == {"step": 1000, "value": 1.0}
    meta.update_summary(1, {"loss": (5000, 0.5)})
    assert meta.get_run(1)["summary"]["loss"] == {"step": 5000, "value": 0.5}
    assert meta.get_run(1)["steps"] == 5000


def test_patch_delete(meta):
    meta.patch(1, name="renamed", notes="hello", add_tags=["x"], remove_tags=["baseline"])
    r = meta.get_run(1)
    assert (r["name"], r["notes"], r["tags"]) == ("renamed", "hello", ["x"])
    meta.delete_run(1)
    assert meta.get_run(1) is None
    assert names(meta, "") == ["b", "c", "d"]


def test_order(meta):
    ids = meta.query_ids("proj", "", order="last(loss)", desc=False)
    assert ids == [4, 3, 2, 1]
    ids = meta.query_ids("proj", "", order="model.layers", desc=True, limit=2)
    assert ids == [2, 3] or ids == [2, 1]


def labels(res):
    return [i["label"] for i in res["items"]]


def test_suggest_fields(meta):
    r = meta.suggest("proj", "mod")
    assert r["context"] == "field" and r["start"] == 0
    assert labels(r)[:4] == ["model.arch", "model.dropout", "model.layers", "model.dims"][:3] + labels(r)[3:4]
    r = meta.suggest("proj", 'env = "pong" and la')
    assert "last(loss)" in labels(r) and r["start"] == len('env = "pong" and ')


def test_suggest_values(meta):
    r = meta.suggest("proj", "model.arch = ")
    assert r["context"] == "value"
    assert set(labels(r)) == {"lstm", "gru", "transformer"}
    assert meta.suggest("proj", 'model.arch = "ls')["items"][0]["insert"] == '"lstm"'
    assert labels(meta.suggest("proj", "model.arch in [gru, "))[0] in {"lstm", "gru", "transformer"}
    assert set(labels(meta.suggest("proj", "status = "))) == {"running", "finished", "crashed"}
    assert set(labels(meta.suggest("proj", "tags contains "))) == {"baseline", "v2"}


def test_suggest_operator(meta):
    assert "=" in labels(meta.suggest("proj", "model.arch "))
    assert labels(meta.suggest("proj", 'model.arch = "x" ')) == ["and", "or"]
    assert labels(meta.suggest("proj", "model.arch = lstm ")) == ["and", "or"]


def test_high_cardinality_values_not_suggested(tmp_path):
    m = MetadataStore(str(tmp_path / "m.db"))
    for i in range(30):
        m.create_run("p", f"u{i}", config={"seed": i, "lr": i * 0.001, "arch": ["a", "b"][i % 2]})
    assert m.config_values("p", "seed") is None
    assert m.config_values("p", "lr") is None
    assert len(m.config_values("p", "arch")) == 2
