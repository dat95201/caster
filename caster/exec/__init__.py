"""Execution: the sandbox, the ledger, one scan, and the two stores.

`caster.exec.ledger` has **no I/O and no benchmark imports**. It is pure
statistics over case names, so the whole scheduler can be exercised offline,
without a benchmark, a sandbox or a model.
"""
