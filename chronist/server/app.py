"""HTTP layer: REST API, SSE, static UI. Build with ``create_app``."""
from __future__ import annotations

import asyncio
import contextlib
import gzip
import io
import json
import logging
import os
import zlib
from typing import Dict, Optional

import msgpack
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from ..blobs import valid_hash
from ..query import QueryError
from .tokens import TokenStore, from_dict as token_store_from_dict
from ..store import Buckets
from .core import Core, decode_hists, decode_points

log = logging.getLogger("chronist.server")

STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ui", "static")

RAW_PAGE = 250_000  # max raw points per response (bounds response-building memory)
EXPORT_PAGE = 100_000  # points per export page (= parquet row group)

# Endpoints that are reads even though they are POST.
READ_POSTS = {"/api/series", "/api/hist/render"}


# ---------------------------------------------------------------- helpers
class ApiError(Exception):
    def __init__(self, status: int, msg: str):
        super().__init__(msg)
        self.status = status
        self.msg = msg


def _json_default(o):
    import numpy as np
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def respond(request: Request, data, status: int = 200) -> Response:
    if "application/msgpack" in request.headers.get("accept", ""):
        return Response(msgpack.packb(data, default=_json_default), status_code=status,
                        media_type="application/msgpack")
    body = json.dumps(data, default=_json_default, separators=(",", ":"), allow_nan=False)
    return Response(body, status_code=status, media_type="application/json")


async def read_body(request: Request) -> dict:
    raw = await request.body()
    if not raw:
        return {}
    enc = request.headers.get("content-encoding", "")
    try:
        if enc == "gzip":
            raw = gzip.decompress(raw)
        elif enc == "deflate":
            raw = zlib.decompress(raw)
    except Exception:
        raise ApiError(400, "could not decompress body")
    ctype = request.headers.get("content-type", "")
    try:
        if "msgpack" in ctype:
            data = msgpack.unpackb(raw, raw=False, strict_map_key=False)
        else:
            data = json.loads(raw)
    except Exception:
        raise ApiError(400, "could not decode body")
    if not isinstance(data, dict):
        raise ApiError(400, "body must be an object")
    return data


def qint(request: Request, name: str, default=None) -> Optional[int]:
    v = request.query_params.get(name)
    if v is None or v == "":
        return default
    try:
        return int(float(v))
    except ValueError:
        raise ApiError(400, f"{name} must be an integer")


def run_id_param(request: Request) -> int:
    try:
        return int(request.path_params["run_id"])
    except ValueError:
        raise ApiError(400, "bad run id")


# ---------------------------------------------------------------- auth
CODE_HIDDEN = "code not shared"


class AuthMiddleware:
    """Bearer tokens: 'write' tokens may do anything, 'read' tokens only reads
    (and code only if the token carries the code flag). No tokens configured
    -> auth off (local mode). The store is consulted per request, so tokens
    created or revoked at runtime take effect immediately."""

    def __init__(self, app, tokens: TokenStore):
        self.app = app
        self.tokens = tokens

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope["path"].startswith("/api/"):
            return await self.app(scope, receive, send)
        state = scope.setdefault("state", {})
        if not self.tokens.enabled:
            state.update(role="write", code=True, auth=False)
            return await self.app(scope, receive, send)
        req = Request(scope)
        token = None
        auth = req.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            token = auth[7:].strip()
        token = token or req.query_params.get("token")
        entry = self.tokens.lookup(token)
        is_read = scope["method"] in ("GET", "HEAD") or scope["path"] in READ_POSTS
        if entry is None or (entry["role"] == "read" and not is_read):
            status = 401 if entry is None else 403
            resp = JSONResponse({"error": "unauthorized" if status == 401 else "read-only token"}, status)
            return await resp(scope, receive, send)
        state.update(role=entry["role"], code=entry["code"], auth=True)
        return await self.app(scope, receive, send)


def can_see_code(request: Request) -> bool:
    return bool(request.scope.get("state", {}).get("code", True))


def is_writer(request: Request) -> bool:
    return request.scope.get("state", {}).get("role", "write") == "write"


def strip_code(run: dict) -> dict:
    """Remove code-snapshot details from a run record (for no-code tokens)."""
    meta = run.get("meta")
    if isinstance(meta, dict) and "code" in meta:
        run["meta"] = {k: v for k, v in meta.items() if k != "code"}
    if run.get("code"):
        run["code"] = None
    return run


# ---------------------------------------------------------------- app
def create_app(core: Core, tokens=None, crash_check_interval: float = 30.0,
               gc_interval: float = 86400.0) -> Starlette:
    """tokens: a TokenStore, or {token: role | entry} for an in-memory store."""
    if not isinstance(tokens, TokenStore):
        tokens = token_store_from_dict(tokens)

    def code_forbidden():
        return JSONResponse({"error": CODE_HIDDEN}, 403)

    async def submit(fn, *args):
        return await asyncio.wrap_future(core.writer.submit(fn, *args))

    def handle(fn):
        async def endpoint(request: Request):
            try:
                return await fn(request)
            except ApiError as e:
                return JSONResponse({"error": e.msg}, e.status)
            except QueryError as e:
                return JSONResponse({"error": str(e), "pos": e.pos}, 400)
            except KeyError as e:
                return JSONResponse({"error": str(e.args[0]) if e.args else "not found"}, 404)
            except ValueError as e:
                return JSONResponse({"error": str(e)}, 400)
        return endpoint

    def threaded(fn):
        """Run a blocking read handler in the thread pool."""
        async def endpoint(request: Request):
            return await asyncio.get_running_loop().run_in_executor(core.pool, fn, request)
        return endpoint

    # ---- writes ----
    async def create_run(request):
        body = await read_body(request)
        return respond(request, await submit(core.op_create_run, body))

    async def log_points(request):
        rid = run_id_param(request)
        body = await read_body(request)
        try:
            pts = decode_points(body)
            hs = decode_hists(body)
        except (TypeError, AttributeError, KeyError) as e:
            raise ApiError(400, f"bad log payload: {e}")
        return respond(request, await submit(core.op_log, rid, pts, hs))

    async def rewind(request):
        body = await read_body(request)
        step = body.get("step")
        if not isinstance(step, int) or isinstance(step, bool):
            raise ApiError(400, "step must be an integer")
        return respond(request, await submit(core.op_rewind, run_id_param(request), step))

    async def heartbeat(request):
        return respond(request, await submit(core.op_heartbeat, run_id_param(request)))

    async def finish(request):
        body = await read_body(request)
        return respond(request, await submit(core.op_finish, run_id_param(request),
                                             body.get("status", "finished"), body.get("when")))

    async def patch_run(request):
        body = await read_body(request)
        return respond(request, await submit(core.op_patch, run_id_param(request), body))

    async def delete_run(request):
        return respond(request, await submit(core.op_delete, run_id_param(request)))

    # ---- reads ----
    @threaded
    def projects(request):
        return respond(request, {"projects": core.meta.projects()})

    @threaded
    def list_runs(request):
        p = request.query_params
        project = p.get("project") or None
        try:
            ids = core.meta.query_ids(project, p.get("q", ""), order=p.get("order", "created"),
                                      desc=p.get("desc", "1") not in ("0", "false"))
        except QueryError as e:
            return JSONResponse({"error": str(e), "pos": e.pos}, 400)
        total = len(ids)
        limit = qint(request, "limit")
        offset = qint(request, "offset", 0)
        page = ids[offset:offset + limit] if limit is not None else ids[offset:]
        include = [i for i in p.get("include", "config,summary,tags").split(",") if i]
        runs = core.meta.get_runs(page, include)
        if not can_see_code(request):
            runs = [strip_code(r) for r in runs]
        return respond(request, {"total": total, "runs": runs})

    @threaded
    def get_run(request):
        rid = run_id_param(request)
        run = core.meta.get_run(rid)
        if run is None:
            return JSONResponse({"error": f"run {rid} not found"}, 404)
        run["keys"] = core.store.keys(rid)
        if not can_see_code(request):
            strip_code(run)
        return respond(request, run)

    @threaded
    def resolve_uid(request):
        rid = core.meta.run_id_by_uid(request.path_params["uid"])
        if rid is None:
            return JSONResponse({"error": "not found"}, 404)
        return respond(request, {"id": rid})

    @threaded
    def keys(request):
        project = request.query_params.get("project") or None
        return respond(request, {"config": core.meta.config_keys(project),
                                 "metrics": core.meta.metric_keys(project),
                                 "hists": core.meta.hist_keys(project)})

    @threaded
    def suggest(request):
        p = request.query_params
        text = p.get("text", p.get("prefix", ""))
        return respond(request, core.meta.suggest(p.get("project") or None, text))

    @threaded
    def metric(request):
        """Downsampled (max_points) or raw points. Raw reads are paged: at most
        ``limit`` points (default and cap: RAW_PAGE); if more remain, the
        response carries ``next_start`` for the next request."""
        rid = run_id_param(request)
        mp = qint(request, "max_points")
        if mp is not None:
            mp = max(1, min(mp, 20000))
            b: Buckets = core.store.read(rid, request.path_params["key"], qint(request, "start"),
                                         qint(request, "end"), mp)
            return respond(request, {"run_id": rid, "key": request.path_params["key"], **b.to_dict()})
        limit = max(1, min(qint(request, "limit", RAW_PAGE), RAW_PAGE))
        b = core.store.read(rid, request.path_params["key"], qint(request, "start"), qint(request, "end"),
                            None, limit=limit)
        # raw: min = max = mean = last and count = 1, so send just the points
        out = {"run_id": rid, "key": request.path_params["key"], "raw": True,
               "step": b.step.tolist(), "value": b.to_dict()["last"] if len(b) else [],
               "wall_time": b.to_dict().get("wall_time") if len(b) else None, "next_start": None}
        if len(b) == limit:
            out["next_start"] = int(b.step[-1]) + 1
        return respond(request, out)

    @threaded
    def hist_records(request):
        """Histogram records of one run/key: ?start&end&max_steps, or ?at=step
        for the record at (or before) a step."""
        rid = run_id_param(request)
        key = request.path_params["key"]
        at = qint(request, "at")
        if at is not None:
            r = core.hists.at(rid, key, at)
            return respond(request, {"records": [] if r is None else [r.to_json()]})
        ms = qint(request, "max_steps")
        recs = core.hists.read(rid, key, qint(request, "start"), qint(request, "end"),
                               None if ms is None else max(1, min(ms, 100_000)))
        return respond(request, {"records": [r.to_json() for r in recs]})

    async def hist_render(request):
        body = await read_body(request)
        run_ids = body.get("runs")
        key = body.get("key")
        if not isinstance(run_ids, list) or not isinstance(key, str):
            raise ApiError(400, "give runs: [ids] and key")
        yr = body.get("y_range")
        if yr is not None and (not isinstance(yr, list) or len(yr) != 2 or not yr[0] < yr[1]):
            raise ApiError(400, "y_range must be [lo, hi] with lo < hi")

        def work():
            return core.render_hist([int(r) for r in run_ids[:50]], key, body.get("start"), body.get("end"),
                                    body.get("max_steps", 300), body.get("rows", 128),
                                    None if yr is None else (float(yr[0]), float(yr[1])),
                                    body.get("mode", "heatmap"), body.get("step"), bool(body.get("robust", True)))
        return respond(request, await asyncio.get_running_loop().run_in_executor(None, work))

    async def series(request):
        body = await read_body(request)
        run_ids = body.get("runs")
        if run_ids is None:
            if "q" not in body:
                raise ApiError(400, "give runs: [ids] or q + project")
            run_ids = await asyncio.get_running_loop().run_in_executor(
                core.pool, lambda: core.meta.query_ids(body.get("project"), body["q"]))
        keys = body.get("keys") or []
        if not isinstance(keys, list) or not keys:
            raise ApiError(400, "keys must be a non-empty list")
        mp = body.get("max_points", 2000)
        if mp is None:
            raise ApiError(400, "max_points is required here (<= 20000); raw points: GET /api/runs/{id}/metrics/{key}")
        mp = max(1, min(int(mp), 20000))

        def work():
            return core.series([int(r) for r in run_ids], keys, body.get("start"), body.get("end"), mp,
                               body.get("x", "step"), body.get("group_by") or None,
                               body.get("agg", "mean_std"))
        res = await asyncio.get_running_loop().run_in_executor(None, work)
        return respond(request, res)

    async def stream(request):
        project = request.query_params.get("project") or None
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue(maxsize=10000)

        def on_event(ev):
            if project is None or ev.get("project") in (None, project):
                loop.call_soon_threadsafe(_put, ev)

        def _put(ev):
            if not q.full():
                q.put_nowait(ev)

        unsubscribe = core.events.subscribe(on_event)

        async def gen():
            try:
                yield "retry: 3000\n\n"
                while True:
                    try:
                        ev = await asyncio.wait_for(q.get(), timeout=15)
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
                        continue
                    # Coalesce bursts into one message.
                    evs = [ev]
                    while not q.empty() and len(evs) < 500:
                        evs.append(q.get_nowait())
                    yield f"data: {json.dumps(evs, default=_json_default)}\n\n"
            finally:
                unsubscribe()

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @threaded
    def export(request):
        """Streamed export: memory stays bounded by one page of points."""
        import tempfile

        from starlette.background import BackgroundTask

        from ..readback import iter_pages
        p = request.query_params
        project = p.get("project") or None
        ids = core.meta.query_ids(project, p.get("q", ""))
        keys = [k for k in p.get("keys", "").split(",") if k] or None
        fmt = p.get("format", "parquet")
        mp = qint(request, "max_points")
        if mp is not None:
            mp = max(1, min(mp, 20000))
        cols = ["run_id", "run", "key", "step", "value", "wall_time"] + (["min", "max", "count"] if mp else [])
        pages = iter_pages(core.meta, core.store, ids, keys, max_points=mp, page=EXPORT_PAGE)
        if fmt == "csv":
            import csv

            def gen():
                buf = io.StringIO()
                w = csv.writer(buf, lineterminator="\n")
                w.writerow(cols)
                yield buf.getvalue()
                for page in pages:
                    buf.seek(0)
                    buf.truncate()
                    rows = zip(*[page[c].tolist() for c in cols])
                    w.writerows([["" if (isinstance(v, float) and v != v) else v for v in r] for r in rows])
                    yield buf.getvalue()
            return StreamingResponse(gen(), media_type="text/csv",
                                     headers={"Content-Disposition": 'attachment; filename="chronist.csv"'})
        try:
            import pyarrow as pa
        except ImportError:
            return JSONResponse({"error": "pyarrow not installed on the server; use format=csv"}, 400)
        fields = [("run_id", pa.int64()), ("run", pa.string()), ("key", pa.string()), ("step", pa.int64()),
                  ("value", pa.float64()), ("wall_time", pa.float64())]
        if mp:
            fields += [("min", pa.float64()), ("max", pa.float64()), ("count", pa.int64())]
        schema = pa.schema(fields)
        fd, path = tempfile.mkstemp(prefix="chronist-export-", suffix="." + fmt)
        os.close(fd)
        try:
            if fmt == "arrow":
                writer = pa.ipc.new_stream(path, schema)
                media, ext = "application/vnd.apache.arrow.stream", "arrow"
            else:
                import pyarrow.parquet as pq
                writer = pq.ParquetWriter(path, schema, compression="zstd")
                media, ext = "application/vnd.apache.parquet", "parquet"
            with writer:
                for page in pages:
                    writer.write_table(pa.table({c: page[c] for c in cols}, schema=schema))
        except BaseException:
            os.remove(path)
            raise
        return FileResponse(path, media_type=media, filename=f"chronist.{ext}",
                            background=BackgroundTask(os.remove, path))

    # ---- code snapshots ----
    async def blobs_missing(request):
        body = await read_body(request)
        hashes = body.get("hashes") or []
        if not isinstance(hashes, list) or not all(isinstance(h, str) and valid_hash(h) for h in hashes):
            raise ApiError(400, "hashes must be a list of sha256 hex strings")
        return respond(request, {"missing": core.blobs.missing(hashes)})

    async def put_blob(request):
        h = request.path_params["hash"]
        if not valid_hash(h):
            raise ApiError(400, "bad hash")
        data = await request.body()
        if len(data) > 256 * 1024 * 1024:
            raise ApiError(413, "blob too large")
        await submit(core.op_put_blob, h, data)
        return respond(request, {"ok": True})

    @threaded
    def get_blob(request):
        if not can_see_code(request):
            return code_forbidden()
        h = request.path_params["hash"]
        if not valid_hash(h):
            return JSONResponse({"error": "bad hash"}, 400)
        data = core.blobs.get(h)
        if data is None:
            return JSONResponse({"error": "not found"}, 404)
        return Response(data, media_type="application/octet-stream",
                        headers={"Cache-Control": "public, max-age=31536000, immutable"})

    async def set_code(request):
        body = await read_body(request)
        return respond(request, await submit(core.op_set_code, run_id_param(request), str(body.get("manifest", ""))))

    @threaded
    def get_code(request):
        if not can_see_code(request):
            return code_forbidden()
        m = core.manifest(run_id_param(request))
        if m is None:
            return JSONResponse({"error": "no code snapshot for this run"}, 404)
        return respond(request, m)

    @threaded
    def get_code_file(request):
        """Raw content of one file (or env/packages blob) of a run's snapshot."""
        if not can_see_code(request):
            return code_forbidden()
        rid = run_id_param(request)
        m = core.manifest(rid)
        if m is None:
            return JSONResponse({"error": "no code snapshot for this run"}, 404)
        path = request.query_params.get("path", "")
        if path in ("__env__", "__packages__"):
            h = m["env" if path == "__env__" else "packages"]
        else:
            f = next((f for f in m["files"] if f["path"] == path), None)
            if f is None or "sha256" not in f:
                return JSONResponse({"error": "no such file in snapshot"}, 404)
            h = f["sha256"]
        data = core.blobs.read(h)
        return Response(data, media_type="text/plain; charset=utf-8",
                        headers={"Cache-Control": "private, max-age=86400"})

    async def admin_gc(request):
        if not is_writer(request):
            return JSONResponse({"error": "needs a write token"}, 403)
        body = await read_body(request)
        return respond(request, await submit(core.op_gc, bool(body.get("vacuum", False))))

    async def health(request):
        return JSONResponse({"ok": True, "auth": tokens.enabled})

    async def whoami(request):
        st = request.scope.get("state", {})
        return JSONResponse({"auth": tokens.enabled, "role": st.get("role"), "code": st.get("code")})

    # ---- share tokens (write tokens only) ----
    async def list_tokens(request):
        if not is_writer(request):
            return JSONResponse({"error": "needs a write token"}, 403)
        return JSONResponse({"auth": tokens.enabled, "tokens": [t for t in tokens.list() if t["role"] == "read"]})

    async def create_token(request):
        if not tokens.enabled:
            raise ApiError(400, "this server runs without authentication (local mode); start it with "
                                "`chronist serve` and tokens to share views")
        if not is_writer(request):
            return JSONResponse({"error": "needs a write token"}, 403)
        body = await read_body(request)
        label = str(body.get("label") or "")[:100]
        t = tokens.create("read", code=bool(body.get("code", False)), label=label)
        from .tokens import token_id
        return JSONResponse({"token": t, "id": token_id(t), "code": bool(body.get("code", False)), "label": label})

    async def revoke_token(request):
        if not is_writer(request):
            return JSONResponse({"error": "needs a write token"}, 403)
        ident = request.path_params["id"]
        listed = {t["id"]: t for t in tokens.list()}
        if ident not in listed or listed[ident]["role"] != "read":
            return JSONResponse({"error": "no such share token"}, 404)
        tokens.revoke(ident)
        return JSONResponse({"ok": True})

    async def index(request):
        path = os.path.join(STATIC_DIR, "index.html")
        if not os.path.exists(path):
            return Response("UI not built", media_type="text/plain")
        return FileResponse(path, headers={"Cache-Control": "no-cache"})

    routes = [
        Route("/api/health", handle(health)),
        Route("/api/whoami", handle(whoami)),
        Route("/api/admin/gc", handle(admin_gc), methods=["POST"]),
        Route("/api/tokens", handle(list_tokens), methods=["GET"]),
        Route("/api/tokens", handle(create_token), methods=["POST"]),
        Route("/api/tokens/{id}", handle(revoke_token), methods=["DELETE"]),
        Route("/api/projects", handle(projects)),
        Route("/api/runs", handle(create_run), methods=["POST"]),
        Route("/api/runs", handle(list_runs), methods=["GET"]),
        Route("/api/runs/by-uid/{uid}", handle(resolve_uid), methods=["GET"]),
        Route("/api/runs/{run_id}", handle(get_run), methods=["GET"]),
        Route("/api/runs/{run_id}", handle(patch_run), methods=["PATCH"]),
        Route("/api/runs/{run_id}", handle(delete_run), methods=["DELETE"]),
        Route("/api/runs/{run_id}/log", handle(log_points), methods=["POST"]),
        Route("/api/runs/{run_id}/heartbeat", handle(heartbeat), methods=["POST"]),
        Route("/api/runs/{run_id}/rewind", handle(rewind), methods=["POST"]),
        Route("/api/runs/{run_id}/finish", handle(finish), methods=["POST"]),
        Route("/api/runs/{run_id}/metrics/{key:path}", handle(metric), methods=["GET"]),
        Route("/api/blobs/missing", handle(blobs_missing), methods=["POST"]),
        Route("/api/blobs/{hash}", handle(put_blob), methods=["PUT"]),
        Route("/api/blobs/{hash}", handle(get_blob), methods=["GET"]),
        Route("/api/runs/{run_id}/code", handle(set_code), methods=["POST"]),
        Route("/api/runs/{run_id}/code", handle(get_code), methods=["GET"]),
        Route("/api/runs/{run_id}/code/file", handle(get_code_file), methods=["GET"]),
        Route("/api/keys", handle(keys)),
        Route("/api/suggest", handle(suggest)),
        Route("/api/series", handle(series), methods=["POST"]),
        Route("/api/hist/render", handle(hist_render), methods=["POST"]),
        Route("/api/runs/{run_id}/hist/{key:path}", handle(hist_records), methods=["GET"]),
        Route("/api/stream", stream),
        Route("/api/export", handle(export)),
        Route("/", index),
    ]
    if os.path.isdir(STATIC_DIR):
        routes.append(Mount("/", StaticFiles(directory=STATIC_DIR), name="static"))

    async def crash_checker():
        while True:
            await asyncio.sleep(crash_check_interval)
            try:
                await submit(core.op_mark_crashed)
            except Exception:  # pragma: no cover
                log.exception("crash check failed")

    async def daily_gc():
        await asyncio.sleep(3600)
        while True:
            try:
                res = await submit(core.op_gc)
                log.info("gc: %s", res)
            except Exception:  # pragma: no cover
                log.exception("gc failed")
            await asyncio.sleep(gc_interval)

    @contextlib.asynccontextmanager
    async def lifespan(app):
        tasks = [asyncio.create_task(crash_checker()), asyncio.create_task(daily_gc())]
        try:
            yield
        finally:
            for t in tasks:
                t.cancel()

    return Starlette(routes=routes, middleware=[Middleware(AuthMiddleware, tokens=tokens)],
                     lifespan=lifespan)
