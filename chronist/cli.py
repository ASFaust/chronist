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

    a = ap.parse_args(argv)
    sys.exit(a.fn(a) or 0)


if __name__ == "__main__":
    main()
