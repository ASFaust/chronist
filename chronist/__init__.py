"""Chronist: a lightweight, self-hostable experiment tracker.

    import chronist
    run = chronist.init(project="bptt-free", config={"model": {"arch": "lstm"}})
    for step in range(n):
        run.log({"loss": loss}, step=step)
    run.finish()

    runs = chronist.runs('model.arch = "lstm"', project="bptt-free")
    df = runs.history(["loss"], max_points=2000)
"""
__version__ = "0.1.0"

from .client import Run, active, finish, init, log, sync  # noqa: E402
from .client.code import Code  # noqa: E402
from .readback import RunRef, RunSet, code_diff, run, runs  # noqa: E402

__all__ = ["init", "log", "finish", "active", "Run", "runs", "RunSet", "run", "RunRef", "Code",
           "code_diff", "sync", "__version__"]
