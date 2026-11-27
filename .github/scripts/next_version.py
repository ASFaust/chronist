"""Decide the next release version for a push to main.

Usage: next_version.py <last-tag-or-empty> <head-tag-or-empty>
Reads the pushed commit messages from $COMMIT_MESSAGES (a JSON list, passed
via env to avoid shell injection) and an optional $BUMP override
(patch|minor|major). Markers ([major], [minor], [skip release]) count only in
a commit's subject line, so describing them in a body doesn't trigger them.
Prints "version=X.Y.Z" and "release=true|false" lines for $GITHUB_OUTPUT.
"""
import json
import os
import re
import sys


def subjects(raw: str) -> str:
    try:
        msgs = json.loads(raw) if raw.strip() else []
    except ValueError:
        msgs = [raw]
    if isinstance(msgs, str):
        msgs = [msgs]
    return "\n".join((m or "").strip().splitlines()[0] if (m or "").strip() else "" for m in msgs)


def next_version(last: str, msg: str, override: str = "") -> str:
    m = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)", last.strip()) if last.strip() else None
    major, minor, patch = (int(x) for x in m.groups()) if m else (0, 0, 0)
    bump = override or ("major" if "[major]" in msg else "minor" if "[minor]" in msg else "patch")
    if bump == "major":
        return f"{major + 1}.0.0"
    if bump == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def main() -> None:
    last = sys.argv[1] if len(sys.argv) > 1 else ""
    head_tag = sys.argv[2] if len(sys.argv) > 2 else ""
    msg = subjects(os.environ.get("COMMIT_MESSAGES", ""))
    override = os.environ.get("BUMP", "").strip()
    if override not in ("", "patch", "minor", "major"):
        sys.exit(f"invalid BUMP={override!r}")
    if "[skip release]" in msg and not override:
        print("release=false")
        print("reason=commit message says [skip release]")
        return
    if head_tag:
        print("release=false")
        print(f"reason=HEAD is already released as {head_tag}")
        return
    print(f"version={next_version(last, msg, override)}")
    print("release=true")


if __name__ == "__main__":
    main()
