"""D8 — which lines of the reference a probe executes.

Used only by the `coverage` arm (arm 7, additional-greedy prioritisation)
and by the diversity term that stops the head of a schedule being two probes
that exercise the same branch. It is coverage **of the reference**, so it is a
property of the probe: computed once per (task, probe), cached, and the same for
every round and every arm.

Run in a subprocess, not in-process. Benchmark programs loop hard, recurse to
the C stack, and call `sys.exit`; tracing one inside the driver would take the
driver down with it. The tracer is `sys.settrace` with a line-event callback
filtered on the program's own filename — no third-party dependency, so the
`coverage` arm no longer silently switches itself off when a package is missing
— an optional dependency that silently degrades to a no-op means an arm can
report under a label it was not running.

A failed trace returns None rather than raising: a probe whose coverage cannot
be established still has a kill rate and a cost, and the diversity term degrades
to a plain score sort.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile

from caster.exec.sandbox import _CHILD_GATE, DEFAULT_RECURSION_LIMIT, DEFAULT_TIMEOUT

_DRIVER = r'''
import json, sys

target, out_path = sys.argv[1], sys.argv[2]
sys.setrecursionlimit(int(sys.argv[3]))
hit = set()

def _trace(frame, event, arg):
    if frame.f_code.co_filename == target:
        if event == "line":
            hit.add(frame.f_lineno)
        return _trace
    return None

source = open(target, encoding="utf-8").read()
code = compile(source, target, "exec")
# The program writes to stdout; swallow it so the trace file is the only output
# this process is judged on.
import io
real_stdout = sys.stdout
sys.stdout = io.StringIO()
sys.settrace(_trace)
try:
    exec(code, {"__name__": "__main__", "__file__": target})
except BaseException:
    pass
finally:
    sys.settrace(None)
    sys.stdout = real_stdout
open(out_path, "w", encoding="utf-8").write(json.dumps(sorted(hit)))
'''


def lines_executed(
    source: str,
    stdin_text: str,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
) -> frozenset[int] | None:
    """1-indexed line numbers of `source` executed on `stdin_text`, or None."""
    with tempfile.TemporaryDirectory(prefix="caster-cov-") as tmp:
        d = pathlib.Path(tmp)
        target = d / "program.py"
        driver = d / "_cover_driver.py"
        out = d / "lines.json"
        target.write_text(source, encoding="utf-8")
        driver.write_text(_DRIVER, encoding="utf-8")
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        env.pop("PYTHONSTARTUP", None)
        try:
            # Inside the gate: this is a child of the same process, so its CPU
            # lands in the same process-global RUSAGE_CHILDREN counter a
            # concurrent scan would be reading. `signature()` is public
            # and nothing stops a caller invoking it mid-episode.
            with _CHILD_GATE:
                subprocess.run(
                    [sys.executable, str(driver), str(target), str(out),
                     str(recursion_limit)],
                    input=stdin_text, capture_output=True, text=True,
                    encoding="utf-8", errors="replace",
                    # Tracing costs roughly an order of magnitude; a flat
                    # multiple of the sandbox budget keeps a slow-but-correct
                    # reference from being reported as uncoverable.
                    timeout=max(timeout * 10.0, 30.0),
                    cwd=d, env=env,
                )
        except (subprocess.TimeoutExpired, OSError):
            return None
        if not out.is_file():
            return None
        try:
            return frozenset(json.loads(out.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            return None
