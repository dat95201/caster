"""The case ledger. Pure statistics over case names.

No I/O, no benchmark imports, no subprocess. Everything this module needs about
a case arrives as a number or a string, so the whole scheduler is testable
without a benchmark, a sandbox or a model. `tests/test_invariants.py`
asserts the import graph.

    ledger per (task, episode):  n[i] runs, k[i] refutations, s[i] total CPU

    K(i) = (k[i] + A) / (n[i] + A + B)          Beta(A,B) posterior mean
    M(i) = s[i] / n[i]   if n[i] > 0 else ⊥

    SCHEDULE(ledger, C):
        observed = { M(i) : i ∈ C, n[i] > 0 }
        m̄ = cold_start(observed) if observed else 1.0     # PESSIMISTIC
        m̂[i] = M(i) if n[i] > 0 else m̄
        return C sorted by (−K(i)/max(m̂[i], 1e-6), tie(i))

    UPDATE(i, refuted, cpu, patch):
        n[i]+=1; k[i]+=refuted; s[i]+=cpu                              # O(1)
        if refuted: R[i].add(patch)      # the rejection set, for OVERLAP

    OVERLAP(i, j) = |R[i] ∩ R[j]| / |R[i]|        0 when |R[i]| < min_rej

    REDUNDANT(i, passed) = ∃ a ∈ passed : OVERLAP(i, a) ≥ overlap_thr

A fifth property, added after the pilot:

**`K` is a marginal rate, and under fail-fast the quantity that decides cost is
a conditional one.** Two probes that refute the same patches are
interchangeable; once one of them has PASSED on a candidate the other is very
unlikely to refute it, yet the marginal `K` still ranks it high. `REDUNDANT`
is the correction: a probe whose rejection set is mostly inside one that has
already passed *this round* is moved to the back of **that round's** queue.
It is a deferral, never a drop — an accepting round still runs the whole pool,
so ordering cannot change acceptance (Invariant 2) — and it dies with the
round, because `passed` is round-scoped while `R` is episode-scoped.

A probe may be pushed back **at most once per round**. The walk is finite
either way — the queue shrinks every step — but one deferral each makes the
bound a rule rather than an argument about a loop, and it measures better:
replayed over the pilot, a deferral that lasts the whole round is worth −4.4%
CPU, CI95 [−9.1, +0.2], against −6.6% [−12.6, −1.4] for one deferral each.
Letting a probe be stepped over repeatedly pushes a genuine refuter too far
back, and the cost of that outweighs what the extra suppression saves.

**Pilot v1 measured this clause at exactly zero on episode-local evidence**
(DECISIONS 39). Holding the exploration RNG fixed and toggling only the clause
over 296 episodes and 3,556 rounds, it fires **0 deferrals** and moves CPU by
**0.0%**. Seeded from the audit's kill matrix via `seed_overlap` the same rule
fires 2,971 deferrals and is worth **−6.6% CPU, CI95 [−12.6, −1.4]**. The rule
is cheap; the evidence is not, and that gap is the whole engineering question.
Under fail-fast a round records exactly ONE refuting probe, so a probe that
PASSES this round carries a rejection of its own in only 7.7% of rounds, while
in the audit matrix a median 87% of probe pairs are ≥75% subsumed. Shipping the
rule is therefore a bet on a cheaply-built seeded table, not a measured win on
its own. `report/scripts/coverage_check.py` is the measurement; do NOT quote
the grid pair `km` vs `km-ov0`, whose exploration seeds differ.

Four properties the implementation must preserve, each with what it cost:

**The cold start is asymmetric on purpose.** `K` is optimistic for an unseen
case; `m̄` is pessimistic. Optimistic in both at once makes every unseen case
outrank a proven refuter and empties the pool in one round. A *median* cold
start for `m` makes an unseen case look cheap and lets it outrank a proven
refuter whenever `m_killer > 1.5 × m̄` — a full-pool sweep at round 2. `max` is
the default; `median` and `q75` exist because arm 9 sweeps them, and all three
are pre-registered.

**The prior is a choice, and both choices ship.** `Beta(½,2)` (prior mean 0.2)
is the default: a case that has never been run must not outrank one with a
measured record, and 0.2 sits above this corpus's measured base rate (≈0.03)
without reaching ½. Jeffreys `Beta(½,½)` puts an unseen case at K=0.5, which
outranks a proven refuter whose record has decayed below it — `k=2, n=10` gives
K=0.227. Under homogeneous per-case costs that is a full-pool sweep. The choice
between them is settled by replaying both over the full-scan kill matrix, which
needs no model calls; until that number exists, neither is "the right one".

**The tie-break is the whole tail**, so it is not the case name. See
`caster.keys.schedule_tie_break`.

**`explore_p` is not optional for the steered arms**. Without it a task
whose first case kills everything never learns a second case exists, the
steering contract has one fact for the whole episode, `k=2` silently degenerates
to `k=1`, and the `k ∈ {1,2,3}` sweep compares `k=1` against `k=1`. Measured
saturation is 2.57 distinct refuters per episode — an average, not a floor. So
it is a parameter of **every** `km` arm, in the cell key, default 0.15, and
`km-explore0` (arm 11) is the ablation.

**The ledger lives for one episode.** `seed_from` exists for a separate,
reportable experiment and must not be used to warm a cell from the store: a
prior learned on other episodes of the same task breaks the independence CRN
pairing assumes. Reusing *verdicts* is sound because a verdict is a fact about a
(patch, case); reusing *statistics* leaks the outcome of one cell into the
treatment of another.
"""
from __future__ import annotations

import dataclasses
import math
import random
from typing import Callable, Iterable, Sequence

from caster.keys import COLD_STARTS, DEFAULT_PRIOR, PRIORS, schedule_tie_rank

#: Floor on m̂ in the ratio, so a case measured at 0.000s CPU cannot divide by
#: zero and swamp the order.
MIN_COST = 1e-6

#: Steering only. A case is *legible* when its stdin fits in a prompt; 400 chars
#: is the limit the worked examples already use (`bench.adapter`). Kill rate and
#: showability pull in opposite directions on contest data, and rendering a real
#: prompt is what exposed it: the top kill-rate case for `abc236_b` is a 2.3 MB
#: stress input. As a *schedule* entry it is the right case — cheap to run,
#: kills almost everything. As a *fact* it is worthless: the model is told to
#: produce `66368` on an input it cannot read.
SHOW_INPUT_CHARS = 400
ILLEGIBLE_PENALTY = 0.45

#: The probability that one never-run case is promoted to the front even though
#: proven killers exist. See `explore_p` in the module docstring.
DEFAULT_EXPLORE_P = 0.15

#: Containment at which one probe is treated as a copy of another: defer it
#: when this much of its rejection set is already inside a probe that passed.
#: 0.75 rather than ⅔ on purpose. Swept over the pilot the rule is flat across
#: [0.667, 0.95] and falls off a cliff at 0.50, so the middle of the plateau is
#: the safe place to sit; and small counts make EXACTLY ⅔ a value the data
#: actually takes (2 shared out of 3), where float comparison decides the
#: branch. One pilot task moved 180 s → 740 s on that tie alone.
DEFAULT_OVERLAP_THR = 0.75

#: How many refutations a probe must have on record before its rejection set is
#: allowed to suppress anything. 1 is weak evidence and is the default anyway,
#: because under fail-fast raising it to 2 makes the rule fire zero times on
#: episode-local data. Raise it when `R` is seeded from an audit table.
DEFAULT_OVERLAP_MIN_REJ = 1


@dataclasses.dataclass
class CaseStat:
    """What the ledger knows about one case. n, k, s — nothing else."""

    name: str
    n: int = 0          # candidates this case has been executed against
    k: int = 0          # ... of which it refuted
    s: float = 0.0      # total CPU seconds spent running it
    #: WHICH candidates it refuted, by patch sha. `k` is `len(rejected)` only
    #: when every refutation was a distinct patch, which is not guaranteed —
    #: a proposer that re-emits a patch pays `n` and `k` again while the set is
    #: unchanged. The COUNT is the kill record FRTP ranks on; the SET is what
    #: `overlap` needs, and the two must not be collapsed into one another.
    rejected: set[str] = dataclasses.field(default_factory=set)
    #: When this case first refuted anything. RECORDED, not ranked on: no
    #: policy reads it. `fault-record` ranks on the kill COUNT `k`, as FRTP
    #: does (Qi et al., ICSM 2013). Kept because `snapshot()` carries it and a
    #: discovery timeline is worth having in the record; delete both fields if
    #: that stops being true.
    first_kill_round: int = -1
    last_kill_round: int = -1
    unusable: bool = False

    @property
    def proven(self) -> bool:
        """Has actually refuted something.

        The steering evidence block only ever shows proven cases: a case with a
        prior and no record is a guess, and naming a failure mode the model
        would not otherwise produce *raises* its probability 2.7× (prompt rule 5,
        the priming inversion).
        """
        return self.k > 0


def _quantile(xs: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile. No numpy import — this module stays pure."""
    if not xs:
        raise ValueError("empty sequence")
    ordered = sorted(xs)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


COLD_START_FNS: dict[str, Callable[[Sequence[float]], float]] = {
    "max": max,
    "median": lambda xs: _quantile(xs, 0.5),
    "q75": lambda xs: _quantile(xs, 0.75),
}
assert set(COLD_START_FNS) == set(COLD_STARTS)


class CaseLedger:
    """Per-episode (n, k, s) over a task's usable case pool.

    `case_names` is the **usable** pool — G1/G2 have already run (`bench.gate`),
    so nothing here decides usability from scratch. A case that turns out
    unusable mid-episode (its reference stops being establishable) is marked
    with `mark_unusable` and leaves the schedule without leaving the record.

    `input_chars` is an optional *input*, not something this module computes: a
    name → int map of stdin sizes, for `evidence`'s legibility factor. It
    defaults to empty and `legibility` then treats every probe as showable,
    which is the same order reached more simply.

    Reference coverage is deliberately NOT here. It is a property of the probe,
    the `coverage` arm is the only consumer, and that arm reads it from its own
    `PolicyContext` — a second copy on the ledger was never populated and would
    be one more thing to keep in step.
    """

    def __init__(
        self,
        case_names: Iterable[str],
        *,
        prior: str = DEFAULT_PRIOR,
        cold_start: str = "max",
        explore_p: float = DEFAULT_EXPLORE_P,
        overlap_thr: float = DEFAULT_OVERLAP_THR,
        overlap_min_rej: int = DEFAULT_OVERLAP_MIN_REJ,
        tie_key: Callable[[str], str] | None = None,
        rng: random.Random | None = None,
        input_chars: dict[str, int] | None = None,
    ) -> None:
        if prior not in PRIORS:
            raise ValueError(f"unknown prior {prior!r}; one of {sorted(PRIORS)}")
        if cold_start not in COLD_START_FNS:
            raise ValueError(f"unknown cold_start {cold_start!r}; one of {COLD_STARTS}")
        self.prior_a, self.prior_b = PRIORS[prior]
        self.prior_name = prior
        self.cold_start = cold_start
        self.explore_p = float(explore_p)
        if not 0.0 <= overlap_thr <= 1.0:
            raise ValueError("overlap_thr must be in [0, 1]")
        if overlap_min_rej < 1:
            raise ValueError("overlap_min_rej must be >= 1")
        self.overlap_thr = float(overlap_thr)
        self.overlap_min_rej = int(overlap_min_rej)
        names = list(case_names)
        # Pool rank, not the case name. They agree on ConDefects (pool order IS
        # `sorted(filenames)`) and come apart on any adapter whose listing is
        # not alphabetical — and a default that silently disagrees with
        # `episode.run_episode` is a second place for the rule to drift.
        self._tie = tie_key or schedule_tie_rank(names)
        self._rng = rng or random.Random(0)
        self.input_chars = dict(input_chars or {})
        self.stats: dict[str, CaseStat] = {n: CaseStat(name=n) for n in names}
        #: Program executions attributable to this ledger's scans. Reference
        #: executions and remeasurement runs are counted in their own buckets
        #: or Invariant 9's arithmetic fails on every honest run.
        self.scan_runs = 0

    # ---- the statistics ---------------------------------------------------

    def usable(self) -> list[str]:
        return [n for n, s in self.stats.items() if not s.unusable]

    def tie(self, name: str) -> str:
        """The schedule tie-break for one probe. See `keys.schedule_tie_break`."""
        return self._tie(name)

    def draw(self) -> float:
        """One draw from this ledger's RNG.

        Exposed so a policy can apply `explore_p` without reaching into the
        ledger's internals — and so every draw comes from ONE stream, which is
        what makes a schedule reproducible from the cell key alone.
        """
        return self._rng.random()

    def mark_unusable(self, name: str) -> None:
        st = self.stats.get(name)
        if st is not None:
            st.unusable = True

    def K(self, name: str) -> float:
        """Beta(A,B) posterior mean of this case's kill rate."""
        st = self.stats[name]
        return (st.k + self.prior_a) / (st.n + self.prior_a + self.prior_b)

    def M(self, name: str) -> float | None:
        """Mean CPU seconds per run, or None when the case has never been run."""
        st = self.stats[name]
        return (st.s / st.n) if st.n > 0 else None

    def m_bar(self, over: Sequence[str] | None = None) -> float:
        """The cold-start cost for a never-run case. PESSIMISTIC by default.

        1.0 when nothing has been observed yet, which is the same pessimism by
        another route: on this corpus the mean case run is 0.366 s, so a cold
        m̄ of 1.0 second makes an unseen case look expensive rather than cheap.
        """
        names = list(over) if over is not None else self.usable()
        observed = [m for m in (self.M(n) for n in names) if m is not None]
        if not observed:
            return 1.0
        return float(COLD_START_FNS[self.cold_start](observed))

    def m_hat(self, name: str, m_bar: float | None = None) -> float:
        m = self.M(name)
        if m is not None:
            return m
        return self.m_bar() if m_bar is None else m_bar

    def score(self, name: str, m_bar: float | None = None) -> float:
        """K(i)/m̂(i) — the quantity the proposal arm ranks on."""
        return self.K(name) / max(self.m_hat(name, m_bar), MIN_COST)

    def update(self, name: str, *, refuted: bool, cpu_s: float,
               round_index: int = -1, patch: str | None = None) -> None:
        """UPDATE(i, refuted, cpu, patch). O(1), the only mutator of (n, k, s, R).

        `patch` is the candidate's sha. It is optional so that every existing
        caller and every test keeps working, but a caller that omits it leaves
        `rejected` empty, and `overlap` then reports 0 for that probe forever —
        the rule degrades to plain `K/m̂` rather than to something wrong.
        """
        st = self.stats.get(name)
        if st is None:
            return
        st.n += 1
        st.s += max(0.0, float(cpu_s))
        if refuted:
            if st.k == 0:
                st.first_kill_round = round_index
            st.k += 1
            st.last_kill_round = round_index
            if patch:
                st.rejected.add(patch)

    # ---- overlap: the conditional correction to a marginal K --------------

    def overlap(self, name: str, other: str) -> float:
        """How much of `name`'s rejection set is already inside `other`'s.

        Containment, not Jaccard, and deliberately asymmetric: the question the
        scan asks is "does `name` still have anything to say now that `other`
        has run", which is `|R(name) ∩ R(other)| / |R(name)|`. Jaccard would
        answer a different question — whether the two probes are interchangeable
        in general — and would let a broad probe hide behind a narrow one.

        0.0 when `name` has fewer than `overlap_min_rej` refutations on record,
        which is the no-evidence case, not a low-similarity one.
        """
        a = self.stats.get(name)
        b = self.stats.get(other)
        if a is None or b is None or not b.rejected:
            return 0.0
        if len(a.rejected) < self.overlap_min_rej:
            return 0.0
        return len(a.rejected & b.rejected) / len(a.rejected)

    def redundant(self, name: str, passed: Sequence[str]) -> bool:
        """Is `name` a copy of something that already PASSED this round?

        `passed`, not "already run": a probe that refuted ended the round, and
        a probe that was skipped as unusable proves nothing. The whole argument
        is conditional on a pass — if a near-copy of `name` has just failed to
        refute this candidate, `name` is unlikely to refute it either.
        """
        if self.overlap_thr <= 0.0 or not passed:
            return False
        return any(self.overlap(name, a) >= self.overlap_thr for a in passed)

    def seed_overlap(self, rejected: dict[str, Iterable[str]]) -> None:
        """Install rejection sets measured elsewhere — the audit's kill matrix.

        Separate from `seed_from` and with the same warning: statistics learned
        on other episodes of this task break the independence CRN pairs on, so
        an arm that calls this is reported BESIDE the main grid, never inside
        it. It exists because the measured headroom (−6.6% against −0.9% for
        episode-local evidence) is most of the reason to have the rule at all,
        and because `caster audit` already produces exactly this table.

        Only `rejected` is seeded. `n`, `k` and `s` are left alone: seeding
        those would also move `K` and `m̂`, and then the arm would be measuring
        two changes at once.
        """
        for name, patches in rejected.items():
            st = self.stats.get(name)
            if st is not None:
                st.rejected |= set(patches)

    # ---- the schedule -----------------------------------------------------

    def schedule(self, candidates: Sequence[str] | None = None) -> list[str]:
        """The SCHEDULE, exactly. The `km` ordering; other arms are policies.

        `explore_p` is applied here rather than in the policy, because it is a
        parameter of every `km` arm and a policy that re-implemented it
        would be a second place for it to drift.
        """
        pool = list(candidates) if candidates is not None else self.usable()
        if not pool:
            return []
        m_bar = self.m_bar(pool)
        ordered = sorted(pool, key=lambda n: (-self.score(n, m_bar), self._tie(n)))

        if self.explore_p <= 0.0:
            return ordered
        proven = [n for n in ordered if self.stats[n].proven]
        unseen = [n for n in ordered if self.stats[n].n == 0]
        # Only meaningful when proven killers exist AND something is unseen:
        # otherwise the promotion is either a no-op or already the head.
        if proven and unseen and self._rng.random() < self.explore_p:
            promoted = unseen[0]
            return [promoted] + [n for n in ordered if n != promoted]
        return ordered

    # ---- evidence selection (a DIFFERENT index — see prompt rule 4) ----------

    def legibility(self, name: str) -> float:
        """1.0 when this case's stdin fits in a prompt, else ILLEGIBLE_PENALTY.

        Unknown size counts as legible: `input_chars` is optional, and an
        episode run without it should fall back to pure kill rate rather than
        penalise every case equally (which would be the same order, reached
        more confusingly).
        """
        chars = self.input_chars.get(name)
        if chars is None:
            return 1.0
        return 1.0 if chars <= SHOW_INPUT_CHARS else ILLEGIBLE_PENALTY

    def evidence(self, k: int) -> list[tuple[str, float]]:
        """Top-k proven cases as (name, **raw K**), ranked by K × legibility.

        Different cost currency from the schedule: the scan pays CPU, the prompt
        pays tokens (prompt rule 4). The ranking score is never reported — the
        number that reaches the prompt is a claim about the world ("this input
        an estimated 43% of attempted fixes get this input
        wrong") and must stay true, so the *raw* K is returned.

        Proven cases only (prompt rule 5). A dominant illegible killer still wins
        when nothing else is close, and `steer.render` then shows it as a shape
        rather than as truncated bytes (prompt rule 6).

        Returns AT MOST k, and often fewer: near-duplicate facts are dropped
        (see below) and the pool of proven cases is small. `k_facts_rendered`
        on the record is what actually reached the prompt, and it is the field
        any k-sweep must be read against — not the requested `k`.
        """
        if k <= 0:
            return []
        proven = [n for n, s in self.stats.items() if s.proven and not s.unusable]
        ranked = sorted(
            proven,
            key=lambda n: (-(self.K(n) * self.legibility(n)), self._tie(n)),
        )
        # Two facts that refute the same patches are ONE fact spending two
        # slots. The prompt pays tokens and attention for each, and a second
        # near-copy buys neither a new failure mode nor a new constraint — it
        # just re-states the first in different bytes. Same containment rule as
        # the schedule, symmetric here because neither fact is "already run":
        # the question is whether the PAIR is worth two slots, not whether one
        # follows the other, so a narrow fact inside a broad one is dropped
        # whichever way round it is ranked.
        chosen: list[tuple[str, float]] = []
        for n in ranked:
            if len(chosen) >= k:
                break
            if any(max(self.overlap(n, m), self.overlap(m, n)) >= self.overlap_thr
                   for m, _ in chosen):
                continue
            chosen.append((n, self.K(n)))
        return chosen

    # ---- cross-episode priors: a separate experiment, never the main grid --

    def seed_from(self, prior: dict[str, tuple[int, int, float]]) -> None:
        """Warm-start from (n, k, s) measured elsewhere.

        Unused by the main grid, and it must stay that way — see the module
        docstring. It exists for a cold-start experiment, and for a Defects4J
        port where `defects4j mutation` supplies exactly this table.
        """
        for name, triple in prior.items():
            st = self.stats.get(name)
            if st is None:
                continue
            st.n, st.k, st.s = int(triple[0]), int(triple[1]), float(triple[2])

    def snapshot(self) -> dict[str, dict[str, float | int | bool]]:
        """A JSON-safe view, for the record and for offline replay."""
        return {
            n: {"n": s.n, "k": s.k, "s": round(s.s, 6),
                "first_kill_round": s.first_kill_round,
                "last_kill_round": s.last_kill_round, "unusable": s.unusable,
                # Sorted, so a snapshot is byte-stable; the whole set rather
                # than its size, because `seed_overlap` reads it back and a
                # count cannot be intersected with anything.
                "rejected": sorted(s.rejected)}
            for n, s in self.stats.items()
        }
