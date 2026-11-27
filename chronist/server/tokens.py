"""Access tokens.

Roles: ``write`` (clients, owner) and ``read`` (shared views). Read tokens
carry a ``code`` flag: without it, code snapshots (manifests, file contents,
blobs, snapshot metadata) are hidden. Write tokens always see code.

Persisted in ``<data>/tokens.json`` (mode 0600) as
``{token: {"role", "code", "label", "created"}}``. The old format
``{token: "write" | "read"}`` is still read (old read tokens get code=False).
Tokens given on the command line are kept in memory only.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from typing import Dict, List, Optional

ROLES = ("write", "read")


def token_id(token: str) -> str:
    """Stable, non-secret identifier for listing/revoking."""
    return hashlib.sha256(token.encode()).hexdigest()[:12]


def _normalize(v) -> dict:
    if isinstance(v, str):
        return {"role": v, "code": v == "write", "label": "", "created": None}
    e = {"role": v.get("role", "read"), "code": bool(v.get("code", False)),
         "label": v.get("label", ""), "created": v.get("created")}
    if e["role"] not in ROLES:
        raise ValueError(f"bad token role {e['role']!r}")
    if e["role"] == "write":
        e["code"] = True
    return e


class TokenStore:
    def __init__(self, path: Optional[str] = None):
        self.path = path
        self._persistent: Dict[str, dict] = {}
        self._ephemeral: Dict[str, dict] = {}
        self._lock = threading.Lock()
        if path and os.path.exists(path):
            with open(path) as f:
                self._persistent = {t: _normalize(v) for t, v in json.load(f).items()}

    # ---- queries ----
    @property
    def enabled(self) -> bool:
        return bool(self._persistent or self._ephemeral)

    def lookup(self, token: Optional[str]) -> Optional[dict]:
        if not token:
            return None
        found = None
        for t, e in list(self._persistent.items()) + list(self._ephemeral.items()):
            if hmac.compare_digest(t.encode(), token.encode()):
                found = e  # no early exit: constant work per request
        return found

    def list(self) -> List[dict]:
        out = []
        for t, e in self._persistent.items():
            out.append({"id": token_id(t), **e, "persistent": True})
        for t, e in self._ephemeral.items():
            out.append({"id": token_id(t), **e, "persistent": False})
        return sorted(out, key=lambda e: e["created"] or 0)

    def counts(self) -> Dict[str, int]:
        c = {"write": 0, "read": 0, "read+code": 0}
        for e in list(self._persistent.values()) + list(self._ephemeral.values()):
            c["read+code" if e["role"] == "read" and e["code"] else e["role"]] += 1
        return c

    # ---- changes ----
    def add(self, token: str, role: str, code: bool = False, label: str = "", persist: bool = False) -> None:
        e = _normalize({"role": role, "code": code, "label": label, "created": time.time()})
        with self._lock:
            (self._persistent if persist else self._ephemeral)[token] = e
            if persist:
                self._save()

    def create(self, role: str, code: bool = False, label: str = "") -> str:
        token = secrets.token_urlsafe(24)
        self.add(token, role, code, label, persist=True)
        return token

    def revoke(self, ident: str) -> bool:
        """Revoke by token or by its id. Returns whether something was removed."""
        with self._lock:
            hit = False
            for d in (self._persistent, self._ephemeral):
                for t in list(d):
                    if t == ident or token_id(t) == ident:
                        del d[t]
                        hit = True
            if hit:
                self._save()
            return hit

    def clear(self) -> None:
        with self._lock:
            self._persistent.clear()
            self._ephemeral.clear()

    def _save(self) -> None:
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(self._persistent, f, indent=1)
        os.replace(tmp, self.path)


def from_dict(tokens: Optional[Dict[str, object]]) -> TokenStore:
    """In-memory store from {token: role | entry} (tests, embedding)."""
    s = TokenStore()
    for t, v in (tokens or {}).items():
        e = _normalize(v)
        s.add(t, e["role"], e["code"], e["label"])
    return s
