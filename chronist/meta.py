"""MetadataStore: runs, flattened config, tags, notes, status, summaries.

Always SQLite (WAL). Exactly one MetadataStore per data dir writes (the
server's writer thread); any number of read-only instances may read.
"""
from __future__ import annotations

import json
import math
import os
import random
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional, Tuple

from . import query as Q
from .store.sqlite_rows import connect

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id      INTEGER PRIMARY KEY,
    name    TEXT NOT NULL UNIQUE,
    created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id         INTEGER PRIMARY KEY,
    uid        TEXT NOT NULL UNIQUE,
    project_id INTEGER NOT NULL REFERENCES projects(id),
    name       TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'running',
    created    REAL NOT NULL,
    updated    REAL NOT NULL,
    heartbeat  REAL,
    finished   REAL,
    notes      TEXT NOT NULL DEFAULT '',
    steps      INTEGER,
    meta       TEXT NOT NULL DEFAULT '{}',
    config     TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS runs_project ON runs(project_id, created);
CREATE TABLE IF NOT EXISTS config (
    run_id INTEGER NOT NULL,
    key    TEXT NOT NULL,
    vtype  TEXT NOT NULL,
    v_text TEXT,
    v_num  REAL,
    PRIMARY KEY (run_id, key)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS config_key_text ON config(key, v_text);
CREATE INDEX IF NOT EXISTS config_key_num ON config(key, v_num);
CREATE TABLE IF NOT EXISTS tags (
    run_id INTEGER NOT NULL,
    tag    TEXT NOT NULL,
    PRIMARY KEY (run_id, tag)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS tags_tag ON tags(tag);
CREATE TABLE IF NOT EXISTS summary (
    run_id INTEGER NOT NULL,
    key    TEXT NOT NULL,
    step   INTEGER NOT NULL,
    value  REAL,
    PRIMARY KEY (run_id, key)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS summary_key ON summary(key, value);
CREATE TABLE IF NOT EXISTS hist_keys (
    run_id    INTEGER NOT NULL,
    key       TEXT NOT NULL,
    last_step INTEGER NOT NULL,
    PRIMARY KEY (run_id, key)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS files (
    version_id INTEGER PRIMARY KEY,
    run_id     INTEGER NOT NULL,
    local_id   TEXT NOT NULL,
    name       TEXT NOT NULL,
    step       INTEGER,
    kind       TEXT NOT NULL,
    size       INTEGER NOT NULL,
    sha256     TEXT NOT NULL,
    manifest   TEXT NOT NULL,
    filename   TEXT,
    serializer TEXT,
    metadata   TEXT NOT NULL DEFAULT '{}',
    created    REAL NOT NULL,
    UNIQUE (run_id, local_id)
);
CREATE INDEX IF NOT EXISTS files_run_name ON files(run_id, name);
CREATE TABLE IF NOT EXISTS file_aliases (
    run_id     INTEGER NOT NULL,
    name       TEXT NOT NULL,
    alias      TEXT NOT NULL,
    version_id INTEGER NOT NULL,
    PRIMARY KEY (run_id, name, alias)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS file_policies (
    run_id     INTEGER NOT NULL,
    name       TEXT NOT NULL,
    keep_last  INTEGER,
    keep_every INTEGER,
    PRIMARY KEY (run_id, name)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS run_code (
    run_id   INTEGER PRIMARY KEY,
    manifest TEXT NOT NULL
);
"""

STATUSES = ("running", "finished", "crashed")

# Suggest values only for low-cardinality keys.
MAX_SUGGEST_DISTINCT = 50
MAX_SUGGEST_FLOAT_DISTINCT = 12


# ---------------------------------------------------------------- config helpers
def flatten(d: Any, prefix: str = "") -> Dict[str, Any]:
    """Nested dict -> {dotted.key: value}. Lists stay values."""
    out: Dict[str, Any] = {}
    if not isinstance(d, dict):
        return {prefix or "value": d}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict) and v:
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out


def encode_value(v: Any) -> Tuple[str, Optional[str], Optional[float]]:
    if v is None:
        return "null", None, None
    if isinstance(v, bool):
        return "bool", "true" if v else "false", float(v)
    if isinstance(v, int):
        return "int", str(v), float(v)
    if isinstance(v, float):
        return "float", repr(v), None if math.isnan(v) else v
    if isinstance(v, str):
        return "str", v, None
    if isinstance(v, (list, tuple)):
        return "list", json.dumps(list(v), default=str), None
    if isinstance(v, dict):  # empty dict
        return "str", "{}", None
    # numpy scalars and friends
    try:
        import numpy as np
        if isinstance(v, np.generic):
            return encode_value(v.item())
    except ImportError:  # pragma: no cover
        pass
    return "str", str(v), None


def decode_value(vtype: str, v_text: Optional[str], v_num: Optional[float]) -> Any:
    if vtype == "null":
        return None
    if vtype == "bool":
        return bool(v_num)
    if vtype == "int":
        return int(v_text)
    if vtype == "float":
        return float(v_text)
    if vtype == "list":
        return json.loads(v_text)
    return v_text


ADJ = ("brisk calm dark eager fair gentle hazy icy jolly keen lively misty noble odd plucky quiet "
       "rapid shy tidy vivid wild young zesty amber bold cosmic dusty fuzzy golden").split()
NOUN = ("otter heron lynx moth newt owl panda quail raven seal tapir viper walrus yak zebra badger "
        "comet dune ember fjord glade harbor island jungle lagoon meadow nebula orchid").split()


def random_name(n: int) -> str:
    return f"{random.choice(ADJ)}-{random.choice(NOUN)}-{n}"


def _f(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else x


# ---------------------------------------------------------------- store
class MetadataStore:
    def __init__(self, path: str, readonly: bool = False):
        self.path = path
        self.readonly = readonly
        if not readonly:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            self._w = connect(path)
            self._w.execute("PRAGMA auto_vacuum=INCREMENTAL")  # only effective on a new file
            self._w.executescript(SCHEMA)
        else:
            self._w = None
        self._local = threading.local()

    # -- connections ---------------------------------------------------
    def _r(self) -> sqlite3.Connection:
        con = getattr(self._local, "con", None)
        if con is None:
            con = connect(self.path, readonly=True)
            self._local.con = con
        return con

    @property
    def w(self) -> sqlite3.Connection:
        if self._w is None:
            raise RuntimeError("metadata store opened read-only")
        return self._w

    @contextmanager
    def transaction(self):
        con = self.w
        if con.in_transaction:
            yield con
            return
        con.execute("BEGIN IMMEDIATE")
        try:
            yield con
            con.execute("COMMIT")
        except BaseException:
            con.execute("ROLLBACK")
            raise

    def vacuum(self, full: bool = False) -> None:
        self.w.execute("VACUUM" if full else "PRAGMA incremental_vacuum")
        self.w.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def manifests(self) -> List[str]:
        return [r[0] for r in self.w.execute("SELECT DISTINCT manifest FROM run_code")]

    def close(self):
        if self._w is not None:
            self._w.close()
            self._w = None
        con = getattr(self._local, "con", None)
        if con is not None:
            con.close()
            self._local.con = None

    # -- writes (writer thread only) -----------------------------------
    def project_id(self, name: str, create: bool = True) -> Optional[int]:
        con = self.w if create else self._r()
        row = con.execute("SELECT id FROM projects WHERE name=?", (name,)).fetchone()
        if row:
            return row[0]
        if not create:
            return None
        with self.transaction():
            con.execute("INSERT INTO projects(name, created) VALUES (?,?)", (name, time.time()))
        return con.execute("SELECT id FROM projects WHERE name=?", (name,)).fetchone()[0]

    def create_run(self, project: str, uid: str, name: Optional[str] = None,
                   config: Optional[dict] = None, tags: Iterable[str] = (),
                   meta: Optional[dict] = None, created: Optional[float] = None,
                   notes: str = "") -> Tuple[int, bool]:
        """Create a run, or re-attach to the run with this uid (resume).
        Returns (run_id, created_new)."""
        now = time.time()
        with self.transaction() as con:
            row = con.execute("SELECT id FROM runs WHERE uid=?", (uid,)).fetchone()
            if row:
                run_id = row[0]
                con.execute("UPDATE runs SET status='running', finished=NULL, heartbeat=?, updated=? "
                            "WHERE id=?", (now, now, run_id))
                if config:
                    self._set_config(con, run_id, config, merge=True)
                for t in tags:
                    con.execute("INSERT OR IGNORE INTO tags VALUES (?,?)", (run_id, t))
                return run_id, False
            pid = self.project_id(project)
            con.execute(
                "INSERT INTO runs(uid, project_id, name, status, created, updated, heartbeat, notes, meta) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (uid, pid, name or "", "running", created or now, now, now, notes or "",
                 json.dumps(meta or {}, default=str)))
            run_id = con.execute("SELECT id FROM runs WHERE uid=?", (uid,)).fetchone()[0]
            if not name:
                con.execute("UPDATE runs SET name=? WHERE id=?", (random_name(run_id), run_id))
            self._set_config(con, run_id, config or {}, merge=False)
            for t in tags:
                con.execute("INSERT OR IGNORE INTO tags VALUES (?,?)", (run_id, str(t)))
            return run_id, True

    def _set_config(self, con, run_id: int, config: dict, merge: bool):
        if merge:
            old = json.loads(con.execute("SELECT config FROM runs WHERE id=?", (run_id,)).fetchone()[0])
            old.update(config)
            config = old
        else:
            con.execute("DELETE FROM config WHERE run_id=?", (run_id,))
        con.execute("UPDATE runs SET config=? WHERE id=?", (json.dumps(config, default=str), run_id))
        rows = [(run_id, k, *encode_value(v)) for k, v in flatten(config).items()]
        con.executemany("INSERT OR REPLACE INTO config VALUES (?,?,?,?,?)", rows)

    def update_summary(self, run_id: int, last: Dict[str, Tuple[int, float]]) -> None:
        """Merge per-key (step, value) of the newest points in a batch."""
        if not last:
            return
        now = time.time()
        with self.transaction() as con:
            con.executemany(
                """INSERT INTO summary(run_id, key, step, value) VALUES (?,?,?,?)
                   ON CONFLICT(run_id, key) DO UPDATE SET
                     value = CASE WHEN excluded.step > step THEN excluded.value ELSE value END,
                     step = max(step, excluded.step)""",
                [(run_id, k, int(s), _f(v)) for k, (s, v) in last.items()])
            mx = max(int(s) for s, _ in last.values())
            con.execute("UPDATE runs SET steps = max(coalesce(steps, -1), ?), updated=?, heartbeat=?, "
                        "status = CASE WHEN status='crashed' THEN 'running' ELSE status END WHERE id=?",
                        (mx, now, now, run_id))

    def replace_summary(self, run_id: int, last: Dict[str, Tuple[int, float]]) -> None:
        """Set the summary exactly (after a rewind removed points)."""
        with self.transaction() as con:
            con.execute("DELETE FROM summary WHERE run_id=?", (run_id,))
            con.executemany("INSERT INTO summary(run_id, key, step, value) VALUES (?,?,?,?)",
                            [(run_id, k, int(st), _f(v)) for k, (st, v) in last.items()])
            steps = max((int(st) for st, _ in last.values()), default=None)
            con.execute("UPDATE runs SET steps=?, updated=? WHERE id=?", (steps, time.time(), run_id))

    def update_hist_keys(self, run_id: int, last: Dict[str, int], replace: bool = False) -> None:
        with self.transaction() as con:
            if replace:
                con.execute("DELETE FROM hist_keys WHERE run_id=?", (run_id,))
            con.executemany("""INSERT INTO hist_keys(run_id, key, last_step) VALUES (?,?,?)
                               ON CONFLICT(run_id, key) DO UPDATE SET last_step=max(last_step, excluded.last_step)""",
                            [(run_id, k, int(st)) for k, st in last.items()])
            if not replace and last:
                mx = max(int(v) for v in last.values())
                now = time.time()
                con.execute("UPDATE runs SET steps = max(coalesce(steps, -1), ?), updated=?, heartbeat=?, "
                            "status = CASE WHEN status='crashed' THEN 'running' ELSE status END WHERE id=?",
                            (mx, now, now, run_id))

    def hist_keys(self, project: Optional[str]) -> List[str]:
        sql = "SELECT DISTINCT h.key FROM hist_keys h"
        params: list = []
        if project is not None:
            sql += " JOIN runs r ON r.id=h.run_id JOIN projects p ON p.id=r.project_id WHERE p.name=?"
            params.append(project)
        return [r[0] for r in self._r().execute(sql + " ORDER BY h.key", params)]

    def heartbeat(self, run_id: int) -> None:
        now = time.time()
        with self.transaction() as con:
            con.execute("UPDATE runs SET heartbeat=?, "
                        "status = CASE WHEN status='crashed' THEN 'running' ELSE status END WHERE id=?",
                        (now, run_id))

    def finish(self, run_id: int, status: str = "finished", when: Optional[float] = None) -> None:
        if status not in STATUSES:
            raise ValueError(f"bad status {status!r}")
        now = time.time()
        with self.transaction() as con:
            con.execute("UPDATE runs SET status=?, finished=?, updated=? WHERE id=?",
                        (status, when or now, now, run_id))

    def patch(self, run_id: int, name: Optional[str] = None, notes: Optional[str] = None,
              tags: Optional[List[str]] = None, add_tags: Iterable[str] = (),
              remove_tags: Iterable[str] = (), config: Optional[dict] = None,
              meta: Optional[dict] = None) -> None:
        with self.transaction() as con:
            if meta:
                old = json.loads(con.execute("SELECT meta FROM runs WHERE id=?", (run_id,)).fetchone()[0])
                for k, v in meta.items():  # one level of dict merging
                    if isinstance(v, dict) and isinstance(old.get(k), dict):
                        old[k] = {**old[k], **v}
                    else:
                        old[k] = v
                con.execute("UPDATE runs SET meta=? WHERE id=?", (json.dumps(old, default=str), run_id))
            if name is not None:
                con.execute("UPDATE runs SET name=? WHERE id=?", (name, run_id))
            if notes is not None:
                con.execute("UPDATE runs SET notes=? WHERE id=?", (notes, run_id))
            if tags is not None:
                con.execute("DELETE FROM tags WHERE run_id=?", (run_id,))
                add_tags = list(tags) + list(add_tags)
            for t in add_tags:
                con.execute("INSERT OR IGNORE INTO tags VALUES (?,?)", (run_id, str(t)))
            for t in remove_tags:
                con.execute("DELETE FROM tags WHERE run_id=? AND tag=?", (run_id, t))
            if config is not None:
                self._set_config(con, run_id, config, merge=True)
            con.execute("UPDATE runs SET updated=? WHERE id=?", (time.time(), run_id))

    def delete_run(self, run_id: int) -> None:
        with self.transaction() as con:
            for t in ("config", "tags", "summary", "run_code", "hist_keys", "files", "file_aliases",
                      "file_policies"):
                con.execute(f"DELETE FROM {t} WHERE run_id=?", (run_id,))
            con.execute("DELETE FROM runs WHERE id=?", (run_id,))

    def set_code(self, run_id: int, manifest: str) -> None:
        with self.transaction() as con:
            con.execute("INSERT OR REPLACE INTO run_code VALUES (?,?)", (run_id, manifest))

    def get_code(self, run_id: int) -> Optional[str]:
        row = self._r().execute("SELECT manifest FROM run_code WHERE run_id=?", (run_id,)).fetchone()
        return row[0] if row else None

    # ---- files ----
    FILE_COLS = ("version_id, run_id, local_id, name, step, kind, size, sha256, manifest, filename, serializer, "
                 "metadata, created")

    def _file_row(self, row) -> dict:
        d = dict(zip([c.strip() for c in self.FILE_COLS.split(",")], row))
        d["metadata"] = json.loads(d["metadata"])
        return d

    def add_file_version(self, run_id: int, local_id: str, manifest_hash: str, m: dict) -> Tuple[int, bool]:
        """Register a version (idempotent per local_id). Returns (version_id, new)."""
        with self.transaction() as con:
            row = con.execute("SELECT version_id FROM files WHERE run_id=? AND local_id=?",
                              (run_id, local_id)).fetchone()
            if row:
                return row[0], False
            con.execute(f"INSERT INTO files({self.FILE_COLS}) VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (run_id, local_id, m["name"], m.get("step"), m.get("file_kind") or "other", int(m["size"]),
                         m["sha256"], manifest_hash, m.get("filename"), m.get("serializer"),
                         json.dumps(m.get("metadata") or {}, default=str), float(m.get("created") or time.time())))
            vid = con.execute("SELECT version_id FROM files WHERE run_id=? AND local_id=?",
                              (run_id, local_id)).fetchone()[0]
            for a in m.get("aliases") or []:
                self._set_alias(con, run_id, m["name"], a, vid)
            self._update_latest(con, run_id, m["name"])
            return vid, True

    @staticmethod
    def _set_alias(con, run_id, name, alias, vid):
        con.execute("INSERT OR REPLACE INTO file_aliases VALUES (?,?,?,?)", (run_id, name, alias, vid))

    @staticmethod
    def _update_latest(con, run_id, name):
        row = con.execute("SELECT version_id FROM files WHERE run_id=? AND name=? ORDER BY created DESC, "
                          "version_id DESC LIMIT 1", (run_id, name)).fetchone()
        if row is None:
            con.execute("DELETE FROM file_aliases WHERE run_id=? AND name=? AND alias='latest'", (run_id, name))
        else:
            con.execute("INSERT OR REPLACE INTO file_aliases VALUES (?,?,'latest',?)", (run_id, name, row[0]))

    def set_file_aliases(self, run_id: int, name: str, set_: Dict[str, int], remove: Iterable[str]) -> None:
        with self.transaction() as con:
            for alias, vid in set_.items():
                if alias == "latest":
                    raise ValueError("'latest' is maintained automatically")
                ok = con.execute("SELECT 1 FROM files WHERE version_id=? AND run_id=? AND name=?",
                                 (int(vid), run_id, name)).fetchone()
                if not ok:
                    raise KeyError(f"no version {vid} of {name!r} in run {run_id}")
                self._set_alias(con, run_id, name, alias, int(vid))
            for alias in remove:
                if alias == "latest":
                    raise ValueError("'latest' is maintained automatically")
                con.execute("DELETE FROM file_aliases WHERE run_id=? AND name=? AND alias=?", (run_id, name, alias))

    def delete_file_versions(self, run_id: int, version_ids: Iterable[int]) -> List[str]:
        """Remove versions (and aliases pointing at them). Returns affected names.
        Blobs are left for the mark-and-sweep GC."""
        names = set()
        with self.transaction() as con:
            for vid in version_ids:
                row = con.execute("SELECT name FROM files WHERE version_id=? AND run_id=?", (int(vid), run_id)).fetchone()
                if row is None:
                    continue
                names.add(row[0])
                con.execute("DELETE FROM files WHERE version_id=?", (int(vid),))
                con.execute("DELETE FROM file_aliases WHERE version_id=?", (int(vid),))
            for n in names:
                self._update_latest(con, run_id, n)
        return sorted(names)

    def set_file_policy(self, run_id: int, name: str, keep_last: Optional[int], keep_every: Optional[int]) -> None:
        with self.transaction() as con:
            con.execute("INSERT OR REPLACE INTO file_policies VALUES (?,?,?,?)", (run_id, name, keep_last, keep_every))

    def retention_victims(self, run_id: int, name: str) -> List[int]:
        """Versions the (run, name) policy says to drop. Aliased versions never are."""
        con = self.w
        pol = con.execute("SELECT keep_last, keep_every FROM file_policies WHERE run_id=? AND name=?",
                          (run_id, name)).fetchone()
        if pol is None or (pol[0] is None and pol[1] is None):
            return []
        keep_last, keep_every = pol
        vers = con.execute("SELECT version_id, step, created FROM files WHERE run_id=? AND name=? "
                           "ORDER BY created, version_id", (run_id, name)).fetchall()
        keep = {r[0] for r in con.execute("SELECT version_id FROM file_aliases WHERE run_id=? AND name=?",
                                          (run_id, name))}
        if keep_last:
            keep.update(v[0] for v in vers[-keep_last:])
        if keep_every:
            first_in_window: Dict[int, Tuple[int, int]] = {}
            for vid, step, _ in vers:
                if step is None:
                    continue
                w = step // keep_every
                if w not in first_in_window or step < first_in_window[w][1]:
                    first_in_window[w] = (vid, step)
            keep.update(v for v, _ in first_in_window.values())
        return [v[0] for v in vers if v[0] not in keep]

    def list_files(self, run_id: int) -> List[dict]:
        con = self._r()
        aliases: Dict[int, List[str]] = {}
        for vid, alias in con.execute("SELECT version_id, alias FROM file_aliases WHERE run_id=? ORDER BY alias",
                                      (run_id,)):
            aliases.setdefault(vid, []).append(alias)
        out = []
        for row in con.execute(f"SELECT {self.FILE_COLS} FROM files WHERE run_id=? ORDER BY name, created",
                               (run_id,)):
            d = self._file_row(row)
            d["aliases"] = aliases.get(d["version_id"], [])
            out.append(d)
        return out

    def get_file(self, version_id: int) -> Optional[dict]:
        row = self._r().execute(f"SELECT {self.FILE_COLS} FROM files WHERE version_id=?", (int(version_id),)).fetchone()
        return None if row is None else self._file_row(row)

    def file_manifests(self) -> List[str]:
        return [r[0] for r in self.w.execute("SELECT DISTINCT manifest FROM files")]

    def mark_crashed(self, timeout: float) -> List[int]:
        cutoff = time.time() - timeout
        with self.transaction() as con:
            ids = [r[0] for r in con.execute(
                "SELECT id FROM runs WHERE status='running' AND coalesce(heartbeat, updated) < ?", (cutoff,))]
            if ids:
                con.executemany("UPDATE runs SET status='crashed' WHERE id=?", [(i,) for i in ids])
        return ids

    # -- reads ---------------------------------------------------------
    def projects(self) -> List[dict]:
        rows = self._r().execute(
            "SELECT p.name, count(r.id), max(r.updated) FROM projects p LEFT JOIN runs r ON r.project_id=p.id "
            "GROUP BY p.id ORDER BY max(r.updated) DESC").fetchall()
        return [{"name": n, "runs": c, "updated": u} for n, c, u in rows]

    def run_id_by_uid(self, uid: str) -> Optional[int]:
        row = self._r().execute("SELECT id FROM runs WHERE uid=?", (uid,)).fetchone()
        return row[0] if row else None

    def query_ids(self, project: Optional[str], q: str = "", order: str = "created",
                  desc: bool = True, limit: Optional[int] = None, offset: int = 0) -> List[int]:
        where, params = Q.to_sql(q)
        sql = "SELECT r.id FROM runs r"
        conds = [where]
        if project is not None:
            sql += " JOIN projects p ON p.id=r.project_id"
            conds.append("p.name=?")
            params = params + [project]
        sql += " WHERE " + " AND ".join(conds)
        sql += " ORDER BY " + self._order_sql(order, params) + (" DESC" if desc else " ASC") + ", r.id DESC"
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params += [int(limit), int(offset)]
        return [r[0] for r in self._r().execute(sql, params)]

    def _order_sql(self, order: str, params: list) -> str:
        if order in ("id", "name", "status", "created", "updated", "steps", "heartbeat"):
            return f"r.{order}"
        if order.startswith("last(") and order.endswith(")"):
            params.append(order[5:-1])
            return "(SELECT s.value FROM summary s WHERE s.run_id=r.id AND s.key=?)"
        if order.startswith("config.") and len(order) > 7:  # same rule as the query language
            params += [order, order, order[7:]]
            return ("(SELECT coalesce(c.v_num, c.v_text) FROM config c WHERE c.run_id=r.id AND c.key = CASE WHEN "
                    "EXISTS (SELECT 1 FROM config c2 WHERE c2.run_id=r.id AND c2.key=?) THEN ? ELSE ? END)")
        params.append(order)
        return "(SELECT coalesce(c.v_num, c.v_text) FROM config c WHERE c.run_id=r.id AND c.key=?)"

    def get_runs(self, ids: List[int], include: Iterable[str] = ("config", "summary", "tags")) -> List[dict]:
        """Full run records, in the order of ``ids``."""
        if not ids:
            return []
        con = self._r()
        include = set(include)
        out: Dict[int, dict] = {}
        for chunk in _chunks(ids, 500):
            q = ",".join("?" * len(chunk))
            for row in con.execute(
                    f"SELECT r.id, r.uid, p.name, r.name, r.status, r.created, r.updated, r.heartbeat, "
                    f"r.finished, r.notes, r.steps, r.meta FROM runs r JOIN projects p ON p.id=r.project_id "
                    f"WHERE r.id IN ({q})", chunk):
                out[row[0]] = {
                    "id": row[0], "uid": row[1], "project": row[2], "name": row[3], "status": row[4],
                    "created": row[5], "updated": row[6], "heartbeat": row[7], "finished": row[8],
                    "notes": row[9], "steps": row[10], "meta": json.loads(row[11]),
                    "tags": [], "config": {}, "summary": {}, "code": None,
                }
            if "tags" in include:
                for rid, tag in con.execute(f"SELECT run_id, tag FROM tags WHERE run_id IN ({q}) ORDER BY tag", chunk):
                    out[rid]["tags"].append(tag)
            if "config" in include:
                for rid, k, vt, vtext, vnum in con.execute(
                        f"SELECT run_id, key, vtype, v_text, v_num FROM config WHERE run_id IN ({q}) ORDER BY key", chunk):
                    out[rid]["config"][k] = decode_value(vt, vtext, vnum)
            for rid, mh in con.execute(f"SELECT run_id, manifest FROM run_code WHERE run_id IN ({q})", chunk):
                out[rid]["code"] = mh
            if "summary" in include:
                for rid, k, st, v in con.execute(
                        f"SELECT run_id, key, step, value FROM summary WHERE run_id IN ({q}) ORDER BY key", chunk):
                    out[rid]["summary"][k] = {"step": st, "value": v}
        return [out[i] for i in ids if i in out]

    def get_run(self, run_id: int) -> Optional[dict]:
        r = self.get_runs([run_id])
        return r[0] if r else None

    def config_keys(self, project: Optional[str]) -> List[dict]:
        """All config keys with run count, distinct value count and types."""
        sql = ("SELECT c.key, count(*), count(DISTINCT coalesce(c.v_text, '')), group_concat(DISTINCT c.vtype) "
               "FROM config c")
        params: list = []
        if project is not None:
            sql += " JOIN runs r ON r.id=c.run_id JOIN projects p ON p.id=r.project_id WHERE p.name=?"
            params.append(project)
        sql += " GROUP BY c.key ORDER BY c.key"
        return [{"key": k, "runs": n, "distinct": d, "types": t.split(",")}
                for k, n, d, t in self._r().execute(sql, params)]

    def metric_keys(self, project: Optional[str]) -> List[str]:
        sql = "SELECT DISTINCT s.key FROM summary s"
        params: list = []
        if project is not None:
            sql += " JOIN runs r ON r.id=s.run_id JOIN projects p ON p.id=r.project_id WHERE p.name=?"
            params.append(project)
        return [r[0] for r in self._r().execute(sql + " ORDER BY s.key", params)]

    def tag_values(self, project: Optional[str]) -> List[Tuple[str, int]]:
        sql = "SELECT t.tag, count(*) FROM tags t"
        params: list = []
        if project is not None:
            sql += " JOIN runs r ON r.id=t.run_id JOIN projects p ON p.id=r.project_id WHERE p.name=?"
            params.append(project)
        return list(self._r().execute(sql + " GROUP BY t.tag ORDER BY count(*) DESC", params))

    def config_values(self, project: Optional[str], key: str) -> Optional[List[Tuple[Any, int]]]:
        """Distinct values of a key, or None if the key is high-cardinality."""
        sql = "SELECT c.vtype, c.v_text, c.v_num, count(*) FROM config c"
        params: list = []
        if project is not None:
            sql += " JOIN runs r ON r.id=c.run_id JOIN projects p ON p.id=r.project_id WHERE p.name=? AND c.key=?"
            params += [project, key]
        else:
            sql += " WHERE c.key=?"
            params.append(key)
        sql += " GROUP BY c.vtype, c.v_text ORDER BY count(*) DESC LIMIT ?"
        params.append(MAX_SUGGEST_DISTINCT + 1)
        rows = self._r().execute(sql, params).fetchall()
        if len(rows) > MAX_SUGGEST_DISTINCT:
            return None
        n_float = sum(1 for r in rows if r[0] == "float")
        if n_float > MAX_SUGGEST_FLOAT_DISTINCT:
            return None
        total = sum(r[3] for r in rows)
        # every run has its own value (seeds, ids) -> not useful to suggest
        if len(rows) > 10 and len(rows) >= 0.8 * total:
            return None
        return [(decode_value(vt, vtext, vnum), n) for vt, vtext, vnum, n in rows]

    def suggest(self, project: Optional[str], text: str, limit: int = 30) -> dict:
        """Autocomplete for the query language. ``text`` is the query up to the cursor."""
        ctx = Q.completion_context(text)
        items: List[dict] = []
        pref = ctx.prefix.lower()
        if ctx.kind == "field":
            for f in Q.RUN_FIELDS:
                items.append({"label": f, "insert": f, "kind": "run field"})
            for k in self.config_keys(project):
                if not _plain_ident(k["key"]):
                    ins = f"`{k['key']}`"
                elif k["key"] in Q.RUN_FIELDS:
                    ins = f"config.{k['key']}"  # same name as in tables and the UI
                else:
                    ins = k["key"]
                items.append({"label": k["key"], "insert": ins, "kind": "config",
                              "detail": f"{k['distinct']} values"})
            for m in self.metric_keys(project):
                arg = m if _plain_ident(m) else json.dumps(m)
                items.append({"label": f"last({m})", "insert": f"last({arg})", "kind": "metric"})
            items.append({"label": "not", "insert": "not", "kind": "keyword"})
            items = [i for i in items if pref in i["label"].lower()]
            items.sort(key=lambda i: (not i["label"].lower().startswith(pref), i["kind"] != "config", i["label"]))
        elif ctx.kind == "value" and ctx.field is not None:
            vals: List[Tuple[Any, int]] = []
            f = ctx.field
            if f.kind == "config":
                vals = self.config_values(project, f.name) or []
            elif f.kind == "run" and f.name == "status":
                vals = [(s, 0) for s in STATUSES]
            elif f.kind == "run" and f.name == "tags":
                vals = self.tag_values(project)
            for v, n in vals:
                if isinstance(v, str):
                    ins = json.dumps(v)
                elif v is None:
                    ins = "null"
                elif isinstance(v, bool):
                    ins = "true" if v else "false"
                elif isinstance(v, list):
                    continue
                else:
                    ins = repr(v)
                label = v if isinstance(v, str) else ins
                if pref and pref not in label.lower():
                    continue
                items.append({"label": label, "insert": ins, "kind": "value",
                              "detail": f"{n} runs" if n else ""})
        elif ctx.kind == "operator":
            ops = ["and", "or"] if ctx.field is None else \
                ["=", "!=", "<", "<=", ">", ">=", "in", "not in", "contains", "exists"]
            items = [{"label": o, "insert": o, "kind": "operator"} for o in ops if o.startswith(pref)]
        return {"context": ctx.kind, "start": ctx.start, "items": items[:limit]}


def _plain_ident(s: str) -> bool:
    import re
    return bool(re.fullmatch(r"[A-Za-z_][\w./:-]*", s)) and s.lower() not in Q.KEYWORDS


def _chunks(xs: List[int], n: int):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]
