"""The arms that will actually run. Policy is one injected function.

Every `answers` value below is a heading in `docs/QUESTIONS.md`, and
`tests/test_questions.py` fails the build if one is not. A label with no
statement behind it certifies itself: `fault-record` claimed to be FRTP for a
year while ranking on something FRTP does not rank on, and nothing could catch
it (DECISIONS 26).

A policy is `order(ctx) -> list[str]`: given a ledger and a pool, say what to
run and in what order. Nothing else about an arm is procedural, which is what
makes a new arm cost **zero model calls** and makes the whole of Block A
a replay.

## Block A — replay arms, zero model calls

| # | arm id | ordering function | answers |
|---|---|---|---|
| 0 | `full-suite` | every usable probe, no early stop | calibration — the cost of not stopping; free, `caster audit` runs it anyway |
| 1 | `no-mem` | natural order, **ledger never updated** | calibration + **RQ3's baseline**: this is independent repeated prompting |
| 2 | `natural` | the order the oracle hands probes over (pool order) | RQ1 — *the incumbent*, **not** a neutral control |
| 3 | `random` | seeded shuffle, reseeded per round | RQ1 — the uninformed baseline |
| 4 | `fault-record` | proven refuters by **kill count** descending, then the rest in natural order | RQ1 — FRTP (Qi et al., ICSM 2013), the memory baseline |
| 5 | `cost-only` | m̂ ascending, no K | RQ2 — the cost marginal |
| 6 | `kill-only` | K descending, no cost | RQ2 — the kill-rate marginal |
| 7 | `coverage` | additional-greedy over reference coverage | RQ1 — the traditional prioritiser |
| 8 | `km` | `K/m̂`, cold start `max`, `explore_p = 0.15` | RQ1 — the proposal |
| 9 | `km-cs-median` / `km-cs-q75` | cold start `median` / `q₀.₇₅` | sensitivity |
| 10 | `km-bucket` | proven bucket, then unseen, then survived-only | sensitivity — the structural alternative |
| 11 | `km-explore0` | `km` with `explore_p = 0` | sensitivity, and a precondition of RQ4 |

`no-mem` and `natural` scan in the **same order** and differ only in whether the
ledger is updated. Keeping both is deliberate: it isolates the cost of
*maintaining* the ledger from the benefit of *using* it, and it is free.

**Why `natural` and not `alphabet`.** Sorting by filename is not a scheduling
decision anyone made; it is the order the benchmark's own directory listing
produces, and it is what any implementation gets for free. Calling it "alphabet"
invites the reading that it is an arbitrary control — and it is not: AtCoder's
naming puts the public, small, highly discriminating cases first, and 78.4% of
`natural` rounds start on one. `natural` is the incumbent to beat; `random` is
the control.

**Why `cost-only` is not redundant with `natural`.** RQ2 asks which of the two
terms carries the effect, and `kill-only` supplies only one marginal:

| | no cost term | with cost term |
|---|---|---|
| **no `K`** | `natural` | `cost-only` |
| **with `K`** | `kill-only` | `km` |

Preliminary (7 tasks, full pool, 280 rounds): `fault-record → kill-only` is
−1.5% CPU, `kill-only → km` is −17.7%. If that holds at corpus scale the claim
is *"cost-awareness carries the effect, not counterexample memory"*, which is
sharper and more falsifiable — and it cannot be stated without both marginals.

**Why `coverage` is an arm and not a remark.** The largest prior study of
regression testing in repair found traditional prioritisation beating
repair-specific prioritisation, FRTP included, in most cases. The objection that
*selection* cannot help here — every ConDefects probe runs the whole patched
program, so nothing is excluded — is correct and irrelevant: this is
**prioritisation**. Reference coverage is computed once per probe, cached.

## Block B — steered arms, one model-call grid each

| # | arm id | steering | answers |
|---|---|---|---|
| 12 | `km+steer` | contract, k=2, disclosure `full` | RQ3 |
| 13 | `km+exclusion` | the negative block | RQ3 — the thing being replaced |
| 14 | `km+chat` | full transcript, ChatRepair-style | RQ3 — the published comparator |
| 15 | `km+steer-k1` / `km+steer-k3` | k ∈ {1,3} | RQ4 |
| 16 | `km+steer-sample` | disclosure `sample-only` | RQ4 |
| 17 | `km+steer-noreq` | disclosure `no-required-line` | RQ4 |
| 18 | `km+steer+vs` | contract + verbalized sampling | sensitivity — a different intervention; the paper does not claim it |

Block B arms share round 1 with Block A (empty ledger ⇒ byte-identical prompt ⇒
one cached draw), so each costs `(B−1) × tasks × seeds` model calls, not `B ×`.
That asymmetry — Block A free, Block B one grid each — is the dominant
scheduling constraint on the project and should drive the order of work.

**`--from-cache-only` is meaningful for Block B only on a re-run.** the "every arm differs only in phase 2" holds exactly for the arms whose prompt does
not depend on the ledger. For a steered arm the round-*t* prompt is a function
of the ledger at round *t*, which phase 2 produces: the two phases are **not
separable at all** there, and the dependency is circular, not a cold cache.
"""
from __future__ import annotations

import dataclasses
import hashlib
import random
from typing import Any, Callable

from caster.exec.ledger import CaseLedger
from caster.keys import CellSpec


@dataclasses.dataclass(frozen=True)
class PolicyContext:
    """Everything an ordering function may read. Nothing it may run."""

    ledger: CaseLedger
    pool: tuple[str, ...]                 # natural (pool) order, usable only
    round_index: int
    #: The cell key. Everything seeded in here derives from it, so an order is
    #: reproducible from the record alone and two arms never share a shuffle.
    cell_key: str
    coverage: dict[str, frozenset[int]] = dataclasses.field(default_factory=dict)

    def rank(self) -> dict[str, int]:
        """Natural-order position per probe. Precomputed: an `index()` inside a
        sort key is quadratic, and some pools run to 148 probes."""
        return {n: i for i, n in enumerate(self.pool)}


OrderFn = Callable[[PolicyContext], list[str]]


def _usable(ctx: PolicyContext) -> list[str]:
    """Pool order, minus anything the ledger has learned is unusable."""
    bad = {n for n, s in ctx.ledger.stats.items() if s.unusable}
    return [n for n in ctx.pool if n not in bad]


# ── Block A ordering functions ───────────────────────────────────────────────


def order_natural(ctx: PolicyContext) -> list[str]:
    """The order the benchmark hands probes over. The incumbent."""
    return _usable(ctx)


def order_random(ctx: PolicyContext) -> list[str]:
    """Seeded shuffle, **reseeded per round**. The uninformed control.

    Reseeded per round rather than once per episode: a single shuffle held for a
    whole episode is one draw from the space of orders, and its variance would
    be attributed to the policy. Per round it is an average over orders, which
    is what a control should be.
    """
    names = _usable(ctx)
    rng = random.Random(
        hashlib.sha256(f"random|{ctx.cell_key}|r{ctx.round_index}".encode()).hexdigest()
    )
    rng.shuffle(names)
    return names


def order_fault_record(ctx: PolicyContext) -> list[str]:
    """FRTP — proven refuters by KILL COUNT descending, then the rest in natural
    order.

    Qi, Mao & Lei, *Efficient Automated Program Repair through Fault-Recorded
    Testing Prioritization*, ICSM 2013, pp. 180-189, doi:10.1109/ICSM.2013.29 —
    the method implemented in TrpAutoRepair. It "ranks the test cases based on
    the test case failing count": the count of validated patches this case has
    already falsified. A count, not a rate and not a recency.

    This arm used to rank the proven bucket by `first_kill_round`, which is a
    DIFFERENT method: it reduces the record to one bit and then freezes the
    order at discovery, so a case that goes on to refute fifty patches never
    overtakes one that refuted a single patch earlier. That is not what the
    paper does, and a baseline weaker than the published one inflates the
    margin of whatever is compared against it.

    The discovery-order variant was briefly kept as an arm of its own. It is
    gone: replayed over 68 ceg-mem tasks and 4,670 patches it cost 2,423 s
    against the count's 2,354 s — −2.8%, CI95 [−9.6, +5.4] — so the rung it
    added to RQ2's ladder was one nobody could read, and it was an ablation of
    ours wearing a baseline's clothes.
    """
    names = _usable(ctx)
    stats = ctx.ledger.stats
    rank = ctx.rank()
    proven = [n for n in names if stats[n].proven]
    proven.sort(key=lambda n: (-stats[n].k, rank.get(n, 0)))
    rest = [n for n in names if not stats[n].proven]
    return proven + rest


def order_cost_only(ctx: PolicyContext) -> list[str]:
    """M̂ ascending. The cost marginal: cheapest first, no kill-rate term."""
    names = _usable(ctx)
    m_bar = ctx.ledger.m_bar(names)
    return sorted(names, key=lambda n: (ctx.ledger.m_hat(n, m_bar), ctx.ledger.tie(n)))


def order_kill_only(ctx: PolicyContext) -> list[str]:
    """K descending. The kill-rate marginal: no cost term."""
    names = _usable(ctx)
    return sorted(names, key=lambda n: (-ctx.ledger.K(n), ctx.ledger.tie(n)))


def order_coverage(ctx: PolicyContext) -> list[str]:
    """Additional-greedy over reference coverage. The traditional prioritiser.

    Pick the probe covering the most not-yet-covered reference lines; on a tie,
    the hashed tie-break. When a probe has no coverage signature (the trace
    failed, or the benchmark cannot supply one) it falls to the tail in natural
    order rather than being dropped — a probe with no measurable coverage is
    still a probe.
    """
    names = _usable(ctx)
    with_cov = [n for n in names if ctx.coverage.get(n)]
    without = [n for n in names if not ctx.coverage.get(n)]
    covered: set[int] = set()
    chosen: list[str] = []
    pool = list(with_cov)
    while pool:
        # `min` on (−gain, tie), not `max` on (gain, tie): a tie must fall to the
        # SMALLER tie key, the same direction every other ordering here sorts in.
        # `max` would silently reverse the tie-break for this one arm.
        best = min(pool, key=lambda n: (-len(ctx.coverage[n] - covered), ctx.ledger.tie(n)))
        gain = len(ctx.coverage[best] - covered)
        chosen.append(best)
        pool.remove(best)
        if gain == 0:
            # Nothing left adds coverage: the rest keep natural order, which is
            # what additional-greedy does when the residual is empty.
            chosen.extend(pool)
            break
        covered |= ctx.coverage[best]
    return chosen + without


def order_km(ctx: PolicyContext) -> list[str]:
    """`K/m̂`. The proposal. the SCHEDULE, including `explore_p`."""
    return ctx.ledger.schedule(_usable(ctx))


def order_km_bucket(ctx: PolicyContext) -> list[str]:
    """The structural alternative: proven, then unseen, then survived-only.

    Three buckets rather than one continuous score. Within `proven` the order is
    `K/m̂`; `unseen` keeps natural order (there is nothing to rank on, and
    ranking on a prior is a guess); `survived-only` — run but never refuted — is
    last, because evidence of *not* killing is the one thing the continuous
    score under-weights.

    `explore_p` applies here too: the bucket *structure* is what this
    arm changes, not the presence of exploration.
    """
    names = _usable(ctx)
    stats = ctx.ledger.stats
    m_bar = ctx.ledger.m_bar(names)
    proven = sorted((n for n in names if stats[n].proven),
                    key=lambda n: (-ctx.ledger.score(n, m_bar), ctx.ledger.tie(n)))
    unseen = [n for n in names if stats[n].n == 0]
    survived = [n for n in names if stats[n].n > 0 and not stats[n].proven]
    ordered = proven + unseen + survived
    if ctx.ledger.explore_p > 0 and proven and unseen:
        if ctx.ledger.draw() < ctx.ledger.explore_p:
            promoted = unseen[0]
            ordered = [promoted] + [n for n in ordered if n != promoted]
    return ordered


ORDERINGS: dict[str, OrderFn] = {
    "natural": order_natural,
    "random": order_random,
    "fault-record": order_fault_record,
    "cost-only": order_cost_only,
    "kill-only": order_kill_only,
    "coverage": order_coverage,
    "km": order_km,
    "km-bucket": order_km_bucket,
}


# ── the arm roster ───────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class Arm:
    """One row of the roster. `overrides` are `CellSpec` fields: an arm IS a cell."""

    index: int
    arm_id: str
    ordering: str
    block: str                       # "A" (replay, free) | "B" (one grid each)
    answers: str
    note: str
    overrides: dict[str, Any] = dataclasses.field(default_factory=dict)
    full_scan: bool = False
    update_ledger: bool = True

    def cell(self, task: str, seed: int, **extra: Any) -> CellSpec:
        fields = {"task": task, "seed": seed, "policy": self.arm_id}
        fields.update(self.overrides)
        fields.update(extra)
        return CellSpec(**fields)


_STEER = {"steer_on": True, "k_facts": 2, "disclosure": "full"}

ARMS: dict[str, Arm] = {a.arm_id: a for a in [
    # ── Block A ─────────────────────────────────────────────────────────────
    Arm(0, "full-suite", "natural", "A", "calibration",
        "every usable probe, no early stop — free, the audit pass runs it anyway",
        full_scan=True),
    Arm(1, "no-mem", "natural", "A", "calibration",
        "natural order and the ledger is never updated. Two roles: it isolates "
        "the COST of maintaining the ledger from the BENEFIT of using it, and "
        "— because it redraws every round under a policy-free nonce — it IS "
        "independent repeated prompting, which Chen et al. (arXiv 2503.15050) "
        "measured at 23 bugs against ChatRepair's 18. For RQ3 it is the "
        "baseline to beat, not a floor",
        update_ledger=False),
    Arm(2, "natural", "natural", "A", "RQ1",
        "the incumbent — the order the benchmark's own listing produces"),
    Arm(3, "random", "random", "A", "RQ1",
        "the uninformed control, reseeded per round"),
    Arm(4, "fault-record", "fault-record", "A", "RQ1",
        "FRTP (Qi, Mao & Lei, ICSM 2013, doi:10.1109/ICSM.2013.29) — proven "
        "refuters by kill count descending. The published memory baseline"),
    Arm(5, "cost-only", "cost-only", "A", "RQ2", "the cost marginal"),
    Arm(6, "kill-only", "kill-only", "A", "RQ2", "the kill-rate marginal"),
    Arm(7, "coverage", "coverage", "A", "RQ1", "the traditional prioritiser"),
    Arm(8, "km", "km", "A", "RQ1", "the proposal"),
    Arm(9, "km-cs-median", "km", "A", "sensitivity",
        "cold start = median of observed costs — the one that makes an unseen "
        "probe look cheap and costs a full-pool sweep at round 2",
        overrides={"cold_start": "median"}),
    Arm(9, "km-cs-q75", "km", "A", "sensitivity", "cold start = q₀.₇₅",
        overrides={"cold_start": "q75"}),
    Arm(10, "km-bucket", "km-bucket", "A", "sensitivity", "the structural alternative"),
    Arm(11, "km-explore0", "km", "A", "sensitivity",
        "no exploration — and a PRECONDITION of RQ5: without it a task whose "
        "first probe kills everything reaches the prompt with one proven fact, "
        "k=2 degenerates to k=1, and the k sweep compares k=1 against k=1",
        overrides={"explore_p": 0.0}),

    # ── Block B ─────────────────────────────────────────────────────────────
    Arm(12, "km+steer", "km", "B", "RQ3", "the contract block",
        overrides=dict(_STEER)),
    Arm(12, "km+steer-pass", "km", "B", "RQ3",
        "contract + the regression half: the probes the last candidate already "
        "SATISFIED, as a count. The obvious implementation is dead code — fill "
        "it on accept and the episode ends before it renders — and an oracle "
        "that stops at the first divergence throws away everything a refuted "
        "candidate passed. It is a positive, zero-leak, zero-extra-cost fact, "
        "and it is an arm rather than a default because nothing prices it yet",
        overrides={**_STEER, "passing_block": True}),
    Arm(13, "km+exclusion", "km", "B", "RQ3",
        "the negative block the contract replaces; measured to INVERT at low "
        "semantic pressure (0.052 → 0.117)",
        overrides={**_STEER, "steer_mode": "exclusion"}),
    Arm(14, "km+chat", "km", "B", "RQ3",
        "ChatRepair-style full transcript. IN the grid: measured here at 0.50 "
        "budgeted success against no-memory's 0.63, which independently "
        "reproduces Chen et al. (arXiv 2503.15050) — conversation loses to "
        "redrawing, and 65% of ChatRepair's patches are duplicates. That is a "
        "result, not a reason to drop the arm. It is also the one arm that can "
        "hit the context ceiling, so its context_overflow count is reported "
        "before any of its numbers",
        overrides={**_STEER, "steer_mode": "chat"}),
    Arm(15, "km+steer-k1", "km", "B", "RQ4", "k = 1 fact",
        overrides={**_STEER, "k_facts": 1}),
    Arm(15, "km+steer-k3", "km", "B", "RQ4",
        "k = 3 facts — leakage 16.8% against k=2's 7.0%",
        overrides={**_STEER, "k_facts": 3}),
    Arm(16, "km+steer-sample", "km", "B", "RQ4",
        "disclosure by NAME, not by kill rate — measured exploitability 0.0% (0/273)",
        overrides={**_STEER, "disclosure": "sample-only"}),
    Arm(17, "km+steer-noreq", "km", "B", "RQ4",
        "inputs shown, required outputs withheld",
        overrides={**_STEER, "disclosure": "no-required-line"}),
    Arm(18, "km+steer+vs", "km", "B", "sensitivity",
        "contract + verbalized sampling; rounds 1–2 carry no VS, which is what "
        "keeps the CRN round-1 identity",
        overrides={**_STEER, "k_candidates": 5, "vs_schedule": True}),
]}

#: The default grid. Only `full-suite` is excluded, because the audit pass
#: already produces it. `km+chat` used to be excluded for being measured bad;
#: it is back, because "the published comparator loses" is the finding RQ3
#: rests on and an absent arm cannot report it.
BLOCK_A = tuple(a for a in ARMS.values() if a.block == "A")
BLOCK_B = tuple(a for a in ARMS.values() if a.block == "B")
DEFAULT_GRID = tuple(
    a.arm_id for a in ARMS.values() if a.arm_id != "full-suite"
)

#: Arms whose `success@B` and first-accept round must be IDENTICAL on every
#: cell (Invariant 2). Ordering cannot change acceptance, so a gap is a
#: soundness bug, not a finding. `full-suite` is included: running more probes
#: cannot change whether any refuted.
SOUNDNESS_EQUIVALENT = tuple(a.arm_id for a in ARMS.values() if a.block == "A")


def arm(arm_id: str) -> Arm:
    try:
        return ARMS[arm_id]
    except KeyError:
        raise KeyError(
            f"unknown arm {arm_id!r}. Known: {', '.join(sorted(ARMS))}"
        ) from None


def order_for(arm_id: str) -> OrderFn:
    return ORDERINGS[arm(arm_id).ordering]


def schedule(arm_id: str, ctx: PolicyContext) -> list[str]:
    """The injected function, by name. The only entry point the loop uses."""
    return order_for(arm_id)(ctx)


def describe() -> list[dict[str, Any]]:
    """The roster as data — for `caster report` and for the docs table."""
    return [
        {"index": a.index, "arm": a.arm_id, "block": a.block,
         "ordering": a.ordering, "answers": a.answers, "note": a.note,
         "overrides": a.overrides, "full_scan": a.full_scan,
         "update_ledger": a.update_ledger}
        for a in sorted(ARMS.values(), key=lambda a: (a.block, a.index, a.arm_id))
    ]
