"""Run one program on one input, hard timeout, CPU accounting.

Cost is reported as `getrusage(RUSAGE_CHILDREN)` deltas, never
`time.monotonic()` — Invariant 3. Shards running in parallel make wall clock
noisy, and a schedule that divides by a noisy cost reorders itself with the load
on the machine.

ConDefects programs are whole scripts, not importable functions: they read the
test input from stdin and write their answer to stdout. So the sandbox writes
the source to a file and runs `python3 <file>` with stdin connected to the test
input, exactly as AtCoder judged it. It cannot use the parent's stdin as a
control channel, because the submissions themselves read stdin every way there
is — `input()`, `sys.stdin.readline`, `sys.stdin.buffer.read()`, `open(0).read()`.

The recursion ceiling is installed through a `sitecustomize.py` dropped next to
the program rather than by editing the source, so line numbers in a diff against
the faulty source are not shifted by a prepended preamble.

Two things that are NOT simple ports, each with its reason:

**`getrusage(RUSAGE_CHILDREN)` is process-global**. Deltas around a
`subprocess.run` are correct only if this process has exactly one child in
flight. "Phase 2 parallelises on cores with no shared bottleneck" reads
as permission to use a thread pool, and a thread pool silently attributes a
sibling's CPU to whichever case happened to finish. So phase-2 parallelism is
**process-per-shard, never threads inside one driver** — and this module
*enforces* it rather than documenting it: `_CHILD_GATE` raises
`ConcurrentChildError` the moment a second thread enters `run_program`.

**The timeout stays wall clock, and timeouts are quarantined** (the decision is
recorded in `docs/DECISIONS.md`). A wall-clock kill cannot be made a pure
function of (patch, case): `protocol_sha` covers the interpreter version, the
timeout value, the recursion ceiling and the normalisation rules, but it cannot
cover machine speed or concurrent load, which are the actual inputs, and the
measured base rate of disagreement is small but not zero. Rather than halt a
grid on it, a `timeout` verdict is never reused from the store
(`caster.exec.store`) and never serves a replay-only scan; it is re-executed, a
stable disagreement across repeats on one host halts the run, and a flip is
logged to `jitter.jsonl` and counted in the threats table. The alternative — a
deterministic step cap — would make the verdict pure, but it is not the
instrument the benchmark's own judge used, so the admitted pool (gate G2) would
have to be re-derived under it.
"""
from __future__ import annotations

import dataclasses
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

try:  # pragma: no cover - platform branch
    import resource

    _HAVE_RUSAGE = True
except ImportError:  # pragma: no cover - Windows
    resource = None  # type: ignore[assignment]
    _HAVE_RUSAGE = False

# The instrument's declared capability, not a cost threshold. There is
# deliberately no second, stricter clock anywhere; this one reaches
# `protocol_sha`, so changing it invalidates the verdict store by construction.
#
# **10s does NOT clear the corpus, and that is a deliberate choice — see
# `docs/DECISIONS.md` §16.** Measured directly (CPython 3.13, serial,
# `getrusage` user time, `correctVersion.py` of a real fault):
#
#     abc242_g   32.05s      abc233_g   29.94s      abc257_g   16.96s
#
# `abc233_g` is the clearest case: 2.5 KB of input, an O(N^4) loop nest, and an
# AtCoder-shipped case literally named `killer_T_01.txt`. It produced the
# correct 3-byte answer in 29.94s. At this clock it is a `timeout` instead —
# a REFUTATION of a correct program. On the hardest slice of the corpus (77
# tasks of tier G/H, largest input each) 5% ran past 10s.
#
# What makes that tolerable is that the error is systematic, not random:
# slowness is a property of the TASK, known before any patch runs, and the same
# every time. The fix is `reference_cpu_s` in the gate record and a per-fault
# clock `max(SANDBOX_TIMEOUT_SEC, k * reference_cpu_s)`; until that lands, this
# constant is a floor that mis-judges the heavy tail, and the tasks it
# mis-judges must be reported rather than silently dropped.
DEFAULT_TIMEOUT = float(os.environ.get("SANDBOX_TIMEOUT_SEC", 10.0))
# Contest solutions recurse over the whole input (DFS on 10^5 nodes); CPython's
# default 1000 would turn correct programs into RecursionErrors. Kept well under
# the C stack's real ceiling — a submission that sets its own limit to 10^6, as
# many do, can still crash the interpreter, which surfaces as WorkerCrash below.
DEFAULT_RECURSION_LIMIT = int(os.environ.get("SANDBOX_RECURSION_LIMIT", 10_000))

#: How CPU was obtained. Anything but "getrusage" is a degraded instrument and
#: `caster preflight` refuses to start a reported run on it.
CPU_SOURCE = "getrusage" if _HAVE_RUSAGE else "wallclock-fallback"

_SITECUSTOMIZE = "import sys\nsys.setrecursionlimit({limit})\n"

# Last line of a Python traceback: "ValueError: invalid literal for int()".
_ERROR_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_.]*)\s*(?::\s*(.*))?$", re.DOTALL)


class ConcurrentChildError(RuntimeError):
    """A second child was launched from this process while one was in flight.

    `getrusage(RUSAGE_CHILDREN)` is process-global, so the CPU delta around the
    second launch would include the first child's time. Rather than report a
    silently wrong `cpu_s`, this raises. The fix is always the same: run phase-2
    shards as separate **processes** (`caster evaluate --shards N` forks), never
    as threads inside one driver.
    """


class _ChildGate:
    """One child at a time, per process. See ConcurrentChildError."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._owner: int | None = None

    def __enter__(self) -> "_ChildGate":
        with self._lock:
            if self._owner is not None:
                raise ConcurrentChildError(
                    f"thread {threading.get_ident()} entered the sandbox while "
                    f"thread {self._owner} had a child in flight. "
                    f"getrusage(RUSAGE_CHILDREN) is process-global, so the CPU "
                    f"measurement would be wrong for both. Parallelise phase 2 "
                    f"across processes (--shards), not threads."
                )
            self._owner = threading.get_ident()
        return self

    def __exit__(self, *exc: object) -> None:
        with self._lock:
            self._owner = None


_CHILD_GATE = _ChildGate()


def _children_cpu() -> float:
    """Utime + stime of all reaped children of this process, in seconds."""
    if not _HAVE_RUSAGE:  # pragma: no cover - Windows
        return time.monotonic()
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    return ru.ru_utime + ru.ru_stime


@dataclasses.dataclass(frozen=True)
class Outcome:
    """Result of running one candidate program on one test input.

    `value` is the program's stdout, verbatim — the thing the oracle compares
    against the case's expected output.

    `cpu_s` is `getrusage(RUSAGE_CHILDREN)` delta, never wall clock
    (Invariant 3). `wall_s` is kept beside it because the *timeout* is wall
    clock and a reader needs to see how close a case came to it.
    """

    ok: bool
    value: Any = None
    error_type: str | None = None
    error_message: str | None = None
    traceback: str | None = None
    timed_out: bool = False
    cpu_s: float = 0.0
    wall_s: float = 0.0
    stderr: str = ""
    cpu_source: str = CPU_SOURCE

    @property
    def crashed(self) -> bool:
        """Ran to completion but raised, as opposed to timing out or hanging."""
        return not self.ok and not self.timed_out


def _parse_error(stderr: str) -> tuple[str, str]:
    """(error_type, message) from a traceback."""
    lines = [line for line in stderr.strip().splitlines() if line.strip()]
    if not lines:
        return "NonZeroExit", "program exited non-zero with no traceback"
    if not any(line.startswith("Traceback (most recent call last)") for line in lines):
        # `sys.exit("msg")` prints the bare message and no traceback, and
        # contest solutions bail out that way constantly. Without this branch
        # the message's first word is read as the exception class, giving every
        # distinct bail-out message its own error type.
        return "SystemExit", lines[-1].strip()
    match = _ERROR_LINE.match(lines[-1].strip())
    if not match:
        return "NonZeroExit", lines[-1].strip()
    # "module.ClassName" -> "ClassName", so a type raised from a stdlib module
    # buckets with the same class raised from builtins.
    return match.group(1).split(".")[-1], (match.group(2) or "").strip()


def run_program(
    source: str,
    stdin_text: str,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
) -> Outcome:
    """Execute ``source`` as a script with ``stdin_text`` on stdin.

    Isolation is process-level: a separate ``python3`` interpreter in a
    throwaway directory, killed on ``timeout`` seconds of wall time. An infinite
    loop, a segfault from a self-raised recursion limit, or a bare
    ``sys.exit()`` in candidate code can never take down the caller.
    """
    with _CHILD_GATE:
        with tempfile.TemporaryDirectory(prefix="caster-run-") as tmp:
            tmpdir = pathlib.Path(tmp)
            main_py = tmpdir / "main.py"
            main_py.write_text(source, encoding="utf-8")
            (tmpdir / "sitecustomize.py").write_text(
                _SITECUSTOMIZE.format(limit=recursion_limit), encoding="utf-8"
            )

            env = dict(os.environ)
            env["PYTHONPATH"] = str(tmpdir)  # picks up sitecustomize.py above
            env["PYTHONIOENCODING"] = "utf-8"
            env.pop("PYTHONSTARTUP", None)
            # A candidate that imports a module shadowed by the harness would be
            # a different program. The temp dir holds only main.py and
            # sitecustomize.py, and cwd is that dir, so nothing of ours is on
            # the path.
            env.pop("PYTHONHOME", None)

            cpu0 = _children_cpu()
            wall0 = time.monotonic()
            try:
                proc = subprocess.run(
                    [sys.executable, str(main_py)],
                    input=stdin_text,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=timeout,
                    cwd=tmpdir,
                    env=env,
                )
            except subprocess.TimeoutExpired as exc:
                # subprocess.run kills AND waits on the child before re-raising,
                # so the killed child's CPU is already in RUSAGE_CHILDREN. A
                # timed-out case therefore still reports the CPU it burned,
                # which is what makes `m̂` honest for the expensive cases the
                # scheduler exists to reorder.
                return Outcome(
                    ok=False,
                    timed_out=True,
                    cpu_s=max(0.0, _children_cpu() - cpu0),
                    wall_s=time.monotonic() - wall0,
                    error_type="Timeout",
                    error_message=f"exceeded {timeout}s wall time",
                    stderr=(exc.stderr or "") if isinstance(exc.stderr, str) else "",
                )
            cpu_s = max(0.0, _children_cpu() - cpu0)
            wall_s = time.monotonic() - wall0

    if proc.returncode != 0:
        error_type, message = _parse_error(proc.stderr)
        if proc.returncode < 0:
            # Killed by a signal: a self-raised recursion limit blowing the C
            # stack, or the OOM killer. No traceback exists to parse.
            error_type, message = "WorkerCrash", f"killed by signal {-proc.returncode}"
        return Outcome(
            ok=False,
            value=proc.stdout,
            error_type=error_type,
            error_message=message,
            traceback=proc.stderr,
            cpu_s=cpu_s,
            wall_s=wall_s,
            stderr=proc.stderr,
        )

    return Outcome(ok=True, value=proc.stdout, cpu_s=cpu_s, wall_s=wall_s, stderr=proc.stderr)


def protocol_fields(timeout: float = DEFAULT_TIMEOUT,
                    recursion_limit: int = DEFAULT_RECURSION_LIMIT) -> dict[str, Any]:
    """Everything that can change a verdict without changing the patch.

    Feeds `caster.exec.store.protocol_sha` and the shard `.meta.json`
    (Invariant 13). A value that changes a measured quantity and appears in
    neither a key nor this record is a free variable.
    """
    return {
        "interpreter": sys.version.split()[0],
        "implementation": sys.implementation.name,
        "sandbox_timeout_sec": float(timeout),
        "recursion_limit": int(recursion_limit),
        "cpu_source": CPU_SOURCE,
        # Bumped when `normalize_output` changes. It is part of the verdict.
        "normalisation": "rstrip-lines+drop-trailing-blanks/v1",
    }


if __name__ == "__main__":  # pragma: no cover - operator smoke
    print(run_program("n = int(input())\nprint(n * 2)\n", "21\n"))
    print(run_program("import sys\nprint(sys.stdin.buffer.read().decode().strip()[::-1])\n", "abcde\n"))
    print(run_program("def f(n):\n    return f(n + 1)\nf(0)\n", ""))
    print(run_program("while True:\n    pass\n", "", timeout=1.0))
    print(run_program("import sys\nsys.exit('bye')\n", ""))
