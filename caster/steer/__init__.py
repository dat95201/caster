"""Prompt assembly. Slot-based, never string concatenation at call sites.

`caster.steer` must not import `caster.exec.verify`: evidence selection reads
the ledger, it does not run anything. `tests/test_invariants.py` asserts
it.
"""
