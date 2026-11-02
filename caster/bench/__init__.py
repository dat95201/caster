"""The benchmark adapter — the ONLY module that knows a benchmark.

Everything else in CASTER talks to `Task` / `TestCase` / `OracleAdapter`, so
porting to Defects4J must touch nothing outside this package.
`tests/test_invariants.py` asserts it.
"""

from caster.bench.protocol import (  # noqa: F401
    Authority,
    OracleAdapter,
    ProbeVerdict,
    Program,
    Task,
    TestCase,
)
