"""Chronist: a lightweight, self-hostable experiment tracker.

    import chronist
    run = chronist.init(project="bptt-free", config={"model": {"arch": "lstm"}})
    for step in range(n):
        run.log({"loss": loss}, step=step)
    run.finish()

    runs = chronist.runs('model.arch = "lstm"', project="bptt-free")
    df = runs.history(["loss"], max_points=2000)
"""
try:  # written by setuptools-scm at build/install time from the git tag
    from ._version import __version__
except ImportError:  # pragma: no cover - plain source checkout without an install
    try:
        from importlib.metadata import version as _v
        __version__ = _v("chronist")
    except Exception:
        __version__ = "0.0.0+unknown"

from .client import Run, active, finish, init, log, sync  # noqa: E402
from .client.code import Code  # noqa: E402
from .hist import Histogram  # noqa: E402
from .readback import RunRef, RunSet, code_diff, run, runs  # noqa: E402

__all__ = ["init", "log", "finish", "active", "Run", "runs", "RunSet", "run", "RunRef", "Code", "Histogram",
           "code_diff", "sync", "__version__"]
