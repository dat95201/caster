"""Exceptions that cross module boundaries.

`CacheMiss` is raised on both sides of the phase split — by `llm.client` when
phase 1 never drew a round, and by `exec.verify` when the audit pass never
measured a (patch, probe). Defining it here makes that a decision rather than an
accidental `llm/` → `exec/` import for one class, and
`tests/test_invariants.py` asserts the edge does not exist.

It is deliberately NOT `BackendUnreachable`. That class is caught by the episode
loop, charged as a spent round and fed to the circuit breaker, so five misses
would end a cell as a "backend outage" — exactly the silent degradation a
replay-only run exists to make loud.
"""
from __future__ import annotations


class CacheMiss(RuntimeError):
    """A replay-only run needed something no earlier phase produced.

    Raised loudly rather than silently falling through to a model call or a
    sandbox run: the whole point of phase separation is that a miss is a
    build failure in CI, not an invisible cost.
    """
