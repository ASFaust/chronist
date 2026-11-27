"""Client credentials: one token per server, obtained with the server password.

Lookup order for a server URL:
  1. an explicit ``token=`` argument / ``--token``
  2. ``$CHRONIST_TOKEN``
  3. the credentials file (``$CHRONIST_CREDENTIALS``, default
     ``~/.config/chronist/credentials.json``), written by ``chronist login``
  4. if the server needs a login and we're interactive (terminal or Jupyter):
     ask for the password once, log in, save the token, continue.
     Non-interactive (batch jobs): no prompt; run ``chronist login <url>`` once
     on a machine sharing the home directory instead.
"""
from __future__ import annotations

import json
import logging
import os
import socket
import sys
import time
from typing import Optional

log = logging.getLogger("chronist")


class LoginError(RuntimeError):
    pass


def norm_url(url: str) -> str:
    return url.strip().rstrip("/")


def credentials_path() -> str:
    if os.environ.get("CHRONIST_CREDENTIALS"):
        return os.environ["CHRONIST_CREDENTIALS"]
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, "chronist", "credentials.json")


def _read() -> dict:
    try:
        with open(credentials_path()) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _write(d: dict) -> None:
    p = credentials_path()
    os.makedirs(os.path.dirname(p), mode=0o700, exist_ok=True)
    tmp = p + ".tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(d, f, indent=1)
    os.replace(tmp, p)


def stored_token(url: str) -> Optional[str]:
    e = _read().get(norm_url(url))
    return e.get("token") if isinstance(e, dict) else None


def save_token(url: str, token: str) -> None:
    d = _read()
    d[norm_url(url)] = {"token": token, "saved": time.time()}
    _write(d)


def forget_token(url: str) -> bool:
    d = _read()
    hit = d.pop(norm_url(url), None) is not None
    if hit:
        _write(d)
    return hit


def interactive() -> bool:
    if "ipykernel" in sys.modules:
        return True
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except (ValueError, AttributeError):
        return False


def login(url: str, password: Optional[str] = None, label: Optional[str] = None, attempts: int = 3) -> str:
    """Exchange the server password for a token and save it. Prompts when
    ``password`` is None."""
    import getpass

    import httpx
    url = norm_url(url)
    label = label or f"{os.environ.get('USER', 'user')}@{socket.gethostname()}"
    for i in range(attempts if password is None else 1):
        pw = password if password is not None else getpass.getpass(f"chronist password for {url}: ")
        try:
            r = httpx.post(f"{url}/api/login", json={"password": pw, "label": label}, timeout=30)
        except httpx.HTTPError as e:
            raise LoginError(f"cannot reach {url}: {e}")
        if r.status_code == 200:
            tok = r.json()["token"]
            save_token(url, tok)
            return tok
        try:
            msg = r.json().get("error", r.text)
        except ValueError:
            msg = r.text
        if r.status_code == 401 and password is None and i + 1 < attempts:
            print("  wrong password, try again", file=sys.stderr)
            continue
        raise LoginError(f"login to {url} failed: {msg}")
    raise LoginError(f"login to {url} failed")


def server_needs_auth(url: str, token: Optional[str] = None) -> Optional[bool]:
    """True if the server rejects us (no/invalid token), False if it accepts,
    None if it can't be reached right now."""
    import httpx
    try:
        r = httpx.get(f"{norm_url(url)}/api/whoami", timeout=10,
                      headers={"Authorization": f"Bearer {token}"} if token else {})
    except httpx.HTTPError:
        return None
    return r.status_code == 401


def resolve_token(url: Optional[str], token: Optional[str] = None, prompt: bool = True) -> Optional[str]:
    """The token to use for ``url`` (see module docstring)."""
    if not url:
        return token
    if token:
        return token
    if os.environ.get("CHRONIST_TOKEN"):
        return os.environ["CHRONIST_TOKEN"]
    tok = stored_token(url)
    if tok:
        return tok
    need = server_needs_auth(url)
    if not need:  # open server, or unreachable right now (buffer and retry later)
        return None
    if prompt and interactive():
        print(f"chronist: {norm_url(url)} needs a login (once per machine; the token is saved in "
              f"{credentials_path()})", file=sys.stderr)
        return login(url)
    log.warning("chronist: %s needs a login, and there is no terminal to ask for the password. "
                "Run once:  chronist login %s", norm_url(url), norm_url(url))
    return None
