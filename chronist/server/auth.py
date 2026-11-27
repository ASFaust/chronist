"""Single-user password login.

The password is stored only as a scrypt hash in ``<data>/auth.json`` (0600).
Logging in issues an ordinary write token (one per device/browser,
individually revocable), so clients never keep the password. The file is
re-read on every login, so ``chronist passwd`` works on a running server.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import threading
import time
from typing import Dict, Optional, Tuple

MIN_PASSWORD_LEN = 8
SCRYPT = {"n": 2 ** 15, "r": 8, "p": 1}


def _hash(password: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    return hashlib.scrypt(password.encode(), salt=salt, n=n, r=r, p=p, maxmem=128 * 1024 * 1024, dklen=32)


class PasswordFile:
    def __init__(self, path: str):
        self.path = path

    def _read(self) -> Optional[dict]:
        try:
            with open(self.path) as f:
                return json.load(f).get("password")
        except (OSError, ValueError):
            return None

    def is_set(self) -> bool:
        return self._read() is not None

    def set(self, password: str) -> None:
        if len(password) < MIN_PASSWORD_LEN:
            raise ValueError(f"password must have at least {MIN_PASSWORD_LEN} characters")
        salt = os.urandom(16)
        rec = {"algo": "scrypt", **SCRYPT, "salt": base64.b64encode(salt).decode(),
               "hash": base64.b64encode(_hash(password, salt, **SCRYPT)).decode(), "set": time.time()}
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        tmp = self.path + ".tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump({"version": 1, "password": rec}, f, indent=1)
        os.replace(tmp, self.path)

    def verify(self, password: str) -> bool:
        rec = self._read()
        if rec is None or rec.get("algo") != "scrypt":
            return False
        got = _hash(password, base64.b64decode(rec["salt"]), rec["n"], rec["r"], rec["p"])
        return hmac.compare_digest(got, base64.b64decode(rec["hash"]))


class LoginThrottle:
    """Slow down guessing: after FREE failed attempts from a client, each
    further attempt must wait 2**k seconds (capped)."""

    FREE = 3
    MAX_WAIT = 300.0

    def __init__(self):
        self._fails: Dict[str, Tuple[int, float]] = {}
        self._lock = threading.Lock()

    def wait_for(self, client: str) -> float:
        with self._lock:
            n, last = self._fails.get(client, (0, 0.0))
        if n < self.FREE:
            return 0.0
        return max(0.0, last + min(2.0 ** (n - self.FREE), self.MAX_WAIT) - time.time())

    def failed(self, client: str) -> None:
        with self._lock:
            n, _ = self._fails.get(client, (0, 0.0))
            self._fails[client] = (n + 1, time.time())

    def succeeded(self, client: str) -> None:
        with self._lock:
            self._fails.pop(client, None)
