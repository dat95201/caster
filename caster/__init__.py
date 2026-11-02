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
