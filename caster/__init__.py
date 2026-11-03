"""CASTER — cost-aware scheduling of test evidence for LLM program repair.

Memory here is keyed by **test case**, not by failure class. A test case is an
*observation*; a class is a *category*, and only an observation can be phrased as
a positive requirement. That is why the scheduler and the steering ship as one
system: the scheduling change is what makes the steering change possible.

Module boundaries that are load-bearing:

  * `caster.bench` is the ONLY module that reads a benchmark. Everything else
    talks to `Task` / `TestCase` / `OracleAdapter`, so a Defects4J port touches
    nothing outside it.
  * `caster.exec.ledger` has no I/O and no benchmark imports. It is pure
    statistics over case names, so it is unit-testable offline.
  * `caster.steer` must not import `caster.exec.verify`. Evidence selection
    reads the ledger; it does not run anything.

`tests/test_invariants.py` asserts all three.
"""

__version__ = "0.1.0"


def _load_dotenv() -> None:
    """Read `.env` beside the package, BEFORE any module reads the environment.

    Every knob in `.env.example` — `MODEL`, `RUN_DIR`, `SANDBOX_TIMEOUT_SEC`,
    `LLM_CONTEXT_TOKENS` — is read at IMPORT time by `paths`, `sandbox` and
    `llm.client`. Loading the file later, in `main()`, would be too late for all
    of them, so it happens here: this package is imported before any of its
    modules are.

    `override=False` deliberately: a variable already in the environment wins,
    so `RUN_DIR=pilot-3b caster evaluate ...` on the command line still beats
    the file. The file is the default, not the authority.

    A missing `python-dotenv` is not fatal — the environment still works, and
    refusing to import over a convenience would be worse than the convenience.
    """
    import os
    import pathlib as _p

    env = _p.Path(os.environ.get("CASTER_ROOT") or
                  _p.Path(__file__).resolve().parent.parent) / ".env"
    if not env.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:          # pragma: no cover - optional at runtime
        return
    load_dotenv(env, override=False)


_load_dotenv()
