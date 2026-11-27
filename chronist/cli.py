"""Command line: chronist ui | serve | sync | tokens."""
from __future__ import annotations

import argparse
import json
import logging
import os
import secrets
import sys
import threading
import time
import webbrowser

DEFAULT_PORT = 7070


def _tokens_path(data: str) -> str:
    return os.path.join(os.path.abspath(os.path.expanduser(data)), "tokens.json")


def load_tokens(data: str) -> dict:
    p = _tokens_path(data)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


def save_tokens(data: str, tokens: dict) -> None:
    p = _tokens_path(data)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(tokens, f, indent=1)


def _run_server(core, host: str, port: int, tokens: dict, on_ready=None):
    import uvicorn

    from .server.app import create_app
    app = create_app(core, tokens=tokens)
    config = uvicorn.Config(app, host=host, port=port, log_level="warning", access_log=False,
                            timeout_keep_alive=30)
    server = uvicorn.Server(config)
    if on_ready:
        def waiter():
            while not server.started:
                time.sleep(0.05)
            on_ready()
        threading.Thread(target=waiter, daemon=True).start()
    try:
        server.run()
    finally:
        core.close()


def _watch_local(url: str, root: str, interval: float = 2.0):
    """`chronist ui`: continuously sync offline runs from CHRONIST_DIR into the local server."""
    from .client.sync import find_runs, sync_run
    log = logging.getLogger("chronist")
    while True:
        try:
            for rd in find_runs([root]):
                try:
                    sync_run(rd, url)
                except Exception as e:  # pragma: no cover
                    log.debug("watch: %s: %s", rd.path, e)
        except Exception:  # pragma: no cover
            log.exception("watch failed")
        time.sleep(interval)


def cmd_ui(a):
    from .client.rundir import default_dir
    from .server.core import Core
    core = Core(a.data)
    url = f"http://{a.host}:{a.port}"

    def ready():
        print(f"chronist UI on {url}  (data: {core.data_dir})", flush=True)
        if a.watch:
            root = a.watch_dir or default_dir()
            print(f"watching {root} for offline runs", flush=True)
            threading.Thread(target=_watch_local, args=(url, root), daemon=True).start()
        if a.browser:
            webbrowser.open(url)
    _run_server(core, a.host, a.port, {}, on_ready=ready)


def cmd_serve(a):
    from .server.core import Core
    tokens = load_tokens(a.data)
    for t in a.token or []:
        tokens[t] = "write"
    for t in a.read_token or []:
        tokens[t] = "read"
    if os.environ.get("CHRONIST_SERVER_TOKEN"):
        tokens[os.environ["CHRONIST_SERVER_TOKEN"]] = "write"
    if a.no_auth:
        tokens = {}
    elif not tokens:
        if a.host in ("127.0.0.1", "localhost", "::1"):
            pass  # localhost only: auth off, reach it via ssh -L
        else:
            w, r = secrets.token_urlsafe(24), secrets.token_urlsafe(24)
            tokens = {w: "write", r: "read"}
            save_tokens(a.data, tokens)
            print(f"generated tokens (stored in {_tokens_path(a.data)}):\n  write: {w}\n  read:  {r}")
    core = Core(a.data, heartbeat_timeout=a.heartbeat_timeout)

    def ready():
        auth = f"{sum(1 for v in tokens.values() if v == 'write')} write / " \
               f"{sum(1 for v in tokens.values() if v == 'read')} read tokens" if tokens else "auth OFF"
        print(f"chronist serving on http://{a.host}:{a.port}  ({auth}; data: {core.data_dir})", flush=True)
    _run_server(core, a.host, a.port, tokens, on_ready=ready)


def cmd_sync(a):
    from .client.sync import sync
    res = sync(a.paths or None, url=a.url, token=a.token)
    errors = [r for r in res if "error" in r]
    print(f"synced {len(res) - len(errors)} runs" + (f", {len(errors)} errors" if errors else ""))
    return 1 if errors else 0


def cmd_tokens(a):
    tokens = load_tokens(a.data)
    if a.new:
        t = secrets.token_urlsafe(24)
        tokens[t] = a.new
        save_tokens(a.data, tokens)
        print(f"new {a.new} token: {t}")
        return
    if a.revoke:
        tokens.pop(a.revoke, None)
        save_tokens(a.data, tokens)
        return
    for t, role in tokens.items():
        print(f"{role:6s} {t}")


def _source_args(p):
    p.add_argument("--url", default=None, help="server (default $CHRONIST_URL; else the local data dir)")
    p.add_argument("--token", default=None)
    p.add_argument("--data", default=None, help="local data dir (default ~/.chronist/data)")


def cmd_checkout(a):
    from .readback import run
    r = run(a.run, url=a.url, token=a.token, data_dir=a.data)
    if r.code is None:
        print(f"run {a.run} has no code snapshot", file=sys.stderr)
        return 1
    dest = a.dir or f"chronist-checkout-{r.id}"
    from .client.code import CodeError
    try:
        path = r.code.checkout(dest, force=a.force)
    except CodeError as e:
        print(e, file=sys.stderr)
        return 1
    print(f"restored {len(r.code.files())} files of run {r.id} ({r.name}) into {path}")
    cmd = (r.meta.get("command") or [])
    if cmd:
        print("command:", " ".join(cmd))


def cmd_snapshot(a):
    import os

    from .blobs import FsBlobStore
    from .client import code as C
    from .client.rundir import default_dir
    C._REFERENCE_OVERRIDE = os.getcwd()
    opts = C.parse_code_arg(a.code if a.code.lower() != "none" else None) or C.Code()
    base = os.path.abspath(os.path.expanduser(default_dir()))
    store = FsBlobStore(os.path.join(base, "blobs"))
    snap = C.take_snapshot(opts, store)
    if "manifest" not in snap:
        print(f"snapshot aborted: {snap.get('aborted')}", file=sys.stderr)
        return 1
    print(f"{snap['manifest']}  {snap['files']} files, {snap['bytes'] / 1024:.1f} KB, root {snap['root']} "
          f"({snap['mode']})")
    if a.run:
        import httpx

        from .client.code import manifest_blobs
        url = a.url or os.environ.get("CHRONIST_URL")
        token = a.token or os.environ.get("CHRONIST_TOKEN")
        if not url:
            print("--run needs a server (--url or CHRONIST_URL)", file=sys.stderr)
            return 1
        h = {"Authorization": f"Bearer {token}"} if token else {}
        with httpx.Client(base_url=url.rstrip("/"), headers=h, timeout=60) as http:
            hashes = manifest_blobs(json.loads(store.read(snap["manifest"]))) + [snap["manifest"]]
            r = http.post("/api/blobs/missing", json={"hashes": hashes})
            r.raise_for_status()
            for bh in r.json()["missing"]:
                http.put(f"/api/blobs/{bh}", content=store.get(bh)).raise_for_status()
            r = http.post(f"/api/runs/{a.run}/code", json={"manifest": snap["manifest"]})
            if r.status_code >= 400:
                print(f"attaching failed: {r.text}", file=sys.stderr)
                return 1
            http.patch(f"/api/runs/{a.run}", json={"meta": {"code": snap}}).raise_for_status()
        print(f"attached to run {a.run}")


def cmd_code_diff(a):
    from .readback import code_diff
    d = code_diff(a.a, a.b, url=a.url, token=a.token, data_dir=a.data, unified=not a.stat)
    for k, sign in (("removed", "-"), ("added", "+"), ("changed", "M")):
        for p in d[k]:
            print(f"{sign} {p}")
    c = d["config"]
    for k in c["removed"]:
        print(f"- config {k}")
    for k in c["added"]:
        print(f"+ config {k}")
    for k, (va, vb) in c["changed"].items():
        print(f"M config {k}: {va!r} -> {vb!r}")
    if not a.stat:
        for p, txt in d["diffs"].items():
            print()
            print(txt, end="" if txt.endswith("\n") else "\n")


def main(argv=None):
    from .server.core import default_data_dir
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    ap = argparse.ArgumentParser(prog="chronist", description="lightweight experiment tracker")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ui", help="local server + UI on localhost, no auth")
    p.add_argument("--data", default=default_data_dir(), help="data dir (default: %(default)s)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--no-browser", dest="browser", action="store_false")
    p.add_argument("--no-watch", dest="watch", action="store_false",
                   help="don't auto-sync offline runs from CHRONIST_DIR")
    p.add_argument("--watch-dir", default=None)
    p.set_defaults(fn=cmd_ui)

    p = sub.add_parser("serve", help="server for remote use (put Caddy/nginx in front for TLS)")
    p.add_argument("--data", default=default_data_dir())
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--token", action="append", help="write token (repeatable)")
    p.add_argument("--read-token", action="append", help="read-only token (repeatable)")
    p.add_argument("--no-auth", action="store_true")
    p.add_argument("--heartbeat-timeout", type=float, default=180.0,
                   help="seconds without heartbeat before a running run is marked crashed")
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("sync", help="upload local (offline or backlogged) runs")
    p.add_argument("paths", nargs="*", help="run dirs or CHRONIST_DIR roots (default: $CHRONIST_DIR)")
    p.add_argument("--url", default=None)
    p.add_argument("--token", default=None)
    p.set_defaults(fn=cmd_sync)

    p = sub.add_parser("tokens", help="list / create / revoke server tokens")
    p.add_argument("--data", default=default_data_dir())
    p.add_argument("--new", choices=["write", "read"])
    p.add_argument("--revoke", metavar="TOKEN")
    p.set_defaults(fn=cmd_tokens)

    p = sub.add_parser("checkout", help="restore a run's code snapshot into a directory")
    p.add_argument("run", help="run id or uid")
    p.add_argument("dir", nargs="?", default=None, help="default ./chronist-checkout-<id>")
    p.add_argument("--force", action="store_true", help="write into a non-empty directory")
    _source_args(p)
    p.set_defaults(fn=cmd_checkout)

    p = sub.add_parser("snapshot", help="take a standalone code snapshot (root resolved from the cwd)")
    p.add_argument("--code", default="auto", help="auto | git | <path>")
    p.add_argument("--run", default=None, help="attach to this run id on the server")
    p.add_argument("--url", default=None)
    p.add_argument("--token", default=None)
    p.set_defaults(fn=cmd_snapshot)

    p = sub.add_parser("code", help="code snapshot tools")
    csub = p.add_subparsers(dest="code_cmd", required=True)
    d = csub.add_parser("diff", help="diff code + config of two runs")
    d.add_argument("a")
    d.add_argument("b")
    d.add_argument("--stat", action="store_true", help="only list changed files and config keys")
    _source_args(d)
    d.set_defaults(fn=cmd_code_diff)

    a = ap.parse_args(argv)
    sys.exit(a.fn(a) or 0)


if __name__ == "__main__":
    main()
