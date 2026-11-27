"""HTTP layer: REST API, SSE, static UI. Build with ``create_app``."""
from __future__ import annotations

import asyncio
import contextlib
import gzip
import hmac
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

from ..query import QueryError
from ..store import Buckets
from .core import Core, decode_points

log = logging.getLogger("chronist.server")

STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ui", "static")

# Endpoints that are reads even though they are POST.
READ_POSTS = {"/api/series"}


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
class AuthMiddleware:
    """Bearer tokens: 'write' tokens may do anything, 'read' tokens only reads.
    No tokens configured -> auth off (local mode)."""

    def __init__(self, app, tokens: Dict[str, str]):
        self.app = app
        self.tokens = tokens

    def _role(self, token: Optional[str]) -> Optional[str]:
        if not token:
            return None
        for t, role in self.tokens.items():
            if hmac.compare_digest(t.encode(), token.encode()):
                return role
        return None

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not self.tokens or not scope["path"].startswith("/api/"):
            return await self.app(scope, receive, send)
        req = Request(scope)
        token = None
        auth = req.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            token = auth[7:].strip()
        token = token or req.query_params.get("token")
        role = self._role(token)
        is_read = scope["method"] in ("GET", "HEAD") or scope["path"] in READ_POSTS
        if role is None or (role == "read" and not is_read):
            status = 401 if role is None else 403
            resp = JSONResponse({"error": "unauthorized" if status == 401 else "read-only token"}, status)
            return await resp(scope, receive, send)
        scope.setdefault("state", {})["role"] = role
        return await self.app(scope, receive, send)


# ---------------------------------------------------------------- app
def create_app(core: Core, tokens: Optional[Dict[str, str]] = None,
               crash_check_interval: float = 30.0) -> Starlette:
    tokens = tokens or {}

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
        except (TypeError, AttributeError) as e:
            raise ApiError(400, f"bad points payload: {e}")
        return respond(request, await submit(core.op_log, rid, pts))

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
        return respond(request, {"total": total, "runs": core.meta.get_runs(page, include)})

    @threaded
    def get_run(request):
        rid = run_id_param(request)
        run = core.meta.get_run(rid)
        if run is None:
            return JSONResponse({"error": f"run {rid} not found"}, 404)
        run["keys"] = core.store.keys(rid)
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
                                 "metrics": core.meta.metric_keys(project)})

    @threaded
    def suggest(request):
        p = request.query_params
        text = p.get("text", p.get("prefix", ""))
        return respond(request, core.meta.suggest(p.get("project") or None, text))

    @threaded
    def metric(request):
        rid = run_id_param(request)
        mp = qint(request, "max_points")
        b: Buckets = core.store.read(rid, request.path_params["key"], qint(request, "start"),
                                     qint(request, "end"), mp)
        return respond(request, {"run_id": rid, "key": request.path_params["key"], **b.to_dict()})

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
        if mp is not None:
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
        p = request.query_params
        project = p.get("project") or None
        ids = core.meta.query_ids(project, p.get("q", ""))
        keys = [k for k in p.get("keys", "").split(",") if k]
        fmt = p.get("format", "parquet")
        mp = qint(request, "max_points")
        from ..readback import long_table
        table = long_table(core.meta, core.store, ids, keys or None, max_points=mp)
        if fmt == "csv":
            buf = io.StringIO()
            cols = list(table)
            buf.write(",".join(cols) + "\n")
            for row in zip(*[table[c] for c in cols]):
                buf.write(",".join("" if v is None else str(v) for v in row) + "\n")
            return Response(buf.getvalue(), media_type="text/csv",
                            headers={"Content-Disposition": 'attachment; filename="chronist.csv"'})
        try:
            import pyarrow as pa
        except ImportError:
            return JSONResponse({"error": "pyarrow not installed on the server; use format=csv"}, 400)
        t = pa.table(table)
        sink = pa.BufferOutputStream()
        if fmt == "arrow":
            with pa.ipc.new_stream(sink, t.schema) as w:
                w.write_table(t)
            media, ext = "application/vnd.apache.arrow.stream", "arrow"
        else:
            import pyarrow.parquet as pq
            pq.write_table(t, sink, compression="zstd")
            media, ext = "application/vnd.apache.parquet", "parquet"
        return Response(sink.getvalue().to_pybytes(), media_type=media,
                        headers={"Content-Disposition": f'attachment; filename="chronist.{ext}"'})

    async def health(request):
        return JSONResponse({"ok": True, "auth": bool(tokens)})

    async def index(request):
        path = os.path.join(STATIC_DIR, "index.html")
        if not os.path.exists(path):
            return Response("UI not built", media_type="text/plain")
        return FileResponse(path, headers={"Cache-Control": "no-cache"})

    routes = [
        Route("/api/health", handle(health)),
        Route("/api/projects", handle(projects)),
        Route("/api/runs", handle(create_run), methods=["POST"]),
        Route("/api/runs", handle(list_runs), methods=["GET"]),
        Route("/api/runs/by-uid/{uid}", handle(resolve_uid), methods=["GET"]),
        Route("/api/runs/{run_id}", handle(get_run), methods=["GET"]),
        Route("/api/runs/{run_id}", handle(patch_run), methods=["PATCH"]),
        Route("/api/runs/{run_id}", handle(delete_run), methods=["DELETE"]),
        Route("/api/runs/{run_id}/log", handle(log_points), methods=["POST"]),
        Route("/api/runs/{run_id}/heartbeat", handle(heartbeat), methods=["POST"]),
        Route("/api/runs/{run_id}/finish", handle(finish), methods=["POST"]),
        Route("/api/runs/{run_id}/metrics/{key:path}", handle(metric), methods=["GET"]),
        Route("/api/keys", handle(keys)),
        Route("/api/suggest", handle(suggest)),
        Route("/api/series", handle(series), methods=["POST"]),
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

    @contextlib.asynccontextmanager
    async def lifespan(app):
        task = asyncio.create_task(crash_checker())
        try:
            yield
        finally:
            task.cancel()

    return Starlette(routes=routes, middleware=[Middleware(AuthMiddleware, tokens=tokens)],
                     lifespan=lifespan)
