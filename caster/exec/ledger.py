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

    UPDATE(i, refuted, cpu):  n[i]+=1; k[i]+=refuted; s[i]+=cpu        # O(1)

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

from caster.keys import COLD_STARTS, DEFAULT_PRIOR, PRIORS

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


@dataclasses.dataclass
class CaseStat:
    """What the ledger knows about one case. n, k, s — nothing else."""

    name: str
    n: int = 0          # candidates this case has been executed against
    k: int = 0          # ... of which it refuted
    s: float = 0.0      # total CPU seconds spent running it
    #: Discovery order, which is what `fault-record` (FRTP) ranks on: the arm
    #: is "proven refuters in the order they were discovered, then the rest in
    #: natural order", and without this the order would silently become kill
    #: rate, i.e. `kill-only` — collapsing the very baseline RQ2 decomposes
    #: against.
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
        self._tie = tie_key or (lambda name: name)
        self._rng = rng or random.Random(0)
        self.input_chars = dict(input_chars or {})
        self.stats: dict[str, CaseStat] = {n: CaseStat(name=n) for n in case_names}
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

    def update(self, name: str, *, refuted: bool, cpu_s: float, round_index: int = -1) -> None:
        """UPDATE(i, refuted, cpu). O(1), and the only mutator of (n, k, s)."""
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
        kills 43% of attempts") and must stay true, so the *raw* K is returned.

        Proven cases only (prompt rule 5). A dominant illegible killer still wins
        when nothing else is close, and `steer.render` then shows it as a shape
        rather than as truncated bytes (prompt rule 6).
        """
        if k <= 0:
            return []
        proven = [n for n, s in self.stats.items() if s.proven and not s.unusable]
        ranked = sorted(
            proven,
            key=lambda n: (-(self.K(n) * self.legibility(n)), self._tie(n)),
        )
        return [(n, self.K(n)) for n in ranked[:k]]

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
                "last_kill_round": s.last_kill_round, "unusable": s.unusable}
            for n, s in self.stats.items()
        }
