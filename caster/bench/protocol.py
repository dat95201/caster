"""The `OracleAdapter` Protocol — eight dimensions, six functions.

`docs/ORACLE-DIMENSIONS.md` derives these; this module is that derivation as
types. A new benchmark implements six functions and nothing else changes.

| dim | what it decides                                  | where it lands in the code |
|-----|--------------------------------------------------|----------------------------|
| D1  | granularity: what a probe *is*                   | what `CaseLedger.stats` is keyed by |
| D2  | determinism: is `run(p, patch)` a pure function? | whether a verdict may be memoised (`exec.store`) |
| D3  | authority: does "passes" mean "correct"?         | which probes enter the scan; where overfitting is measured |
| D4  | verdict richness: what a refutation returns      | what lines `steer.template` can build |
| D5  | legibility: can the evidence go in a prompt?     | `steer.render`; `ledger.legibility` |
| D6  | cost: what one probe costs                       | **the whole scheduling problem** — `K/m̂` |
| D7  | discriminating power: how much this probe kills  | `ledger.K`, `seed_from` |
| D8  | independence: do two probes test the same thing? | the `coverage` arm's diversity term |

D6 is where the framework actually bears load, and the two benchmarks differ
structurally: **ConDefects pays a fixed cost per probe** (establishing the
reference answer, cacheable), **Defects4J pays it per patch** (compile,
amortised over every probe of that patch). The scheduler must read that from the
adapter rather than hard-code it — hence `fixed_cost_scope`.

D2 for ConDefects is `stable`, **not** `pure`: the timeout is wall clock. That
is why `exec.store` quarantines `timeout` verdicts instead of memoising them.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Literal, Protocol, Sequence, runtime_checkable

#: D3. `judge` is the reference implementation and **never enters a prompt**
#: (Invariant 4). `held_out` is `P_eval` — never shown, never gates acceptance.
Authority = Literal["visible", "held_out", "judge"]

#: D2.
Determinism = Literal["pure", "stable", "flaky"]

#: D5.
Legibility = Literal["verbatim", "shape", "opaque"]

#: D6. Where the fixed cost of a probe lands.
FixedCostScope = Literal["probe", "patch"]

#: The four terminal verdicts a probe can return. `wrong` is a differential
#: disagreement, `crash` a non-zero exit, `timeout` the wall-clock kill.
Status = Literal["pass", "wrong", "crash", "timeout"]


@dataclasses.dataclass(frozen=True)
class TestCase:
    """D1 for ConDefects: one shipped contest case."""

    name: str
    input_text: str
    expected_output: str | None  # None when the benchmark ships no out/ file


@dataclasses.dataclass(frozen=True)
class ProbeVerdict:
    """What `OracleAdapter.run` returns. D2 + D4 + D6 in one object.

    `observation` carries the highest D4 tier the adapter can build:
    tier 0 binary, 1 + localisation, 2 + differential (expected vs actual),
    3 + state, 4 + cause. ConDefects reaches tier 2 on 94.2% of refutations.
    """

    status: Status
    cpu_s: float
    output: str | None = None
    #: True length of `output` before any head-truncation. The observation line
    #: in a steered prompt reports it, so it must survive the trip through the
    #: verdict store or a warm replay renders different bytes from the cold run
    #: that filled it — a different draw, decided by cache warmth, with nothing
    #: in any key to say so.
    output_len: int | None = None
    expected: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    tier: int = 0
    #: Consecutive `timeout` observations on one host behind this verdict.
    #: Set by `caster.exec.verify._measure`'s confirmation loop and written
    #: onto the store entry, which is what makes a confirmed timeout servable.
    timeout_repeats: int = 0
    runs: int = 1          # program executions this verdict cost

    @property
    def refuted(self) -> bool:
        return self.status != "pass"


@dataclasses.dataclass(frozen=True)
class Program:
    """The three sources a fault ships. `reference_source` is D3 `judge`."""

    name: str
    buggy_source: str
    reference_source: str
    fault_lines: tuple[int, ...] = ()


@dataclasses.dataclass(frozen=True)
class Task:
    """One fault. `name` is the identity used everywhere in the system."""

    name: str
    group: str              # the coding task / project a fault belongs to (G4)
    probes: tuple[str, ...]  # usable probe names, in pool order
    meta: dict[str, Any] = dataclasses.field(default_factory=dict)


@runtime_checkable
class OracleAdapter(Protocol):
    """Six functions. Implement them and CASTER runs on your benchmark.

    The attributes below declare the benchmark's position on D2 and D6; they are
    read by `exec.store` (may a verdict be memoised?) and by the scheduler (is
    the fixed cost per probe or per patch?), and they are stamped into the run
    manifest so two adapters can never be pooled silently.
    """

    name: str
    determinism: Determinism
    fixed_cost_scope: FixedCostScope

    def tasks(self) -> dict[str, Task]:
        """Every fault this adapter can serve, keyed by `Task.name`."""
        ...

    def program(self, task_name: str) -> Program:
        """D3. Sources for one fault. `reference_source` is the judge."""
        ...

    def probes(self, task_name: str) -> Sequence[str]:
        """D1. This task's probe universe, in pool order."""
        ...

    def run(self, task_name: str, probe: str, patch: str) -> ProbeVerdict:
        """D2 + D4 + D6. One probe against one patch, cost measured."""
        ...

    def authority(self, task_name: str, probe: str) -> Authority:
        """D3. `held_out` probes never reach the loop."""
        ...

    def render(self, task_name: str, probe: str) -> tuple[Legibility, str]:
        """D5. How this probe's input may be shown. `opaque` ⇒ never shown."""
        ...

    def prior(self, task_name: str, probe: str) -> float | None:
        """D7. An a-priori kill rate if the benchmark supplies one.

        None on ConDefects (learned online). On Defects4J, `defects4j mutation`
        supplies exactly this table — and feeding it is the cross-episode-prior
        experiment, which is **reportable and separate** (`ledger.seed_from`).
        """
        ...

    def signature(self, task_name: str, probe: str) -> frozenset[int] | None:
        """D8. Something to compute overlap over — reference coverage.

        None when it cannot be measured; the diversity term then degrades to a
        plain score sort rather than failing an episode.
        """
        ...
