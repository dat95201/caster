"""The corpus gate, and the evaluation split. `caster freeze`.

One gate, six conditions, applied before any arm runs and recorded on the frozen
pool. **Nothing is sampled after it**: the corpus *is* the gated pool, in pool
order. A pre-treatment screen on a difficulty proxy puts a covariate upstream of
the thing under study and yields a sample nobody can weight back to the
benchmark, so bands are computed post hoc.

| gate | condition |
|------|-----------|
| G1 | the oracle is sound and consistent for this fault |
| G2 | the test data is intact |
| G3 | `\\|P_loop\\|` ≥ 20 usable probes — counted on the LOOP half, not the pool |
| G4 | one fault per coding task (seeded uniform choice, never "the best one") |
| G5 | contamination cut: drop contests older than `--date-min` |
| G6 | the source fits a prompt (`LINES_MAX`) |

**Not a gate, deliberately: cost.** A fault is unusable only when its reference
cannot be established at the same sandbox clock every arm runs under — never
because it is slower than some second threshold. A separate 10 s filter drops
roughly 9% of candidates — **exactly the expensive-oracle faults the mechanism
helps most** — and biases every cost figure downward. CASTER's entire claim is
about tasks where the oracle is expensive, so the per-case reason keeps the
distinction: `slow` (the reference timed out at the clock) is not `wrong` (the
reference disagrees with `out/`).

**G3 is counted on `P_loop`.** The threshold exists so there is something to
schedule, and the loop only ever schedules `P_loop`. Applying it to the pool and
*then* splitting 30% off into `P_eval` admits tasks the mechanism provably
cannot act on, which is the exact thing the threshold was chosen to exclude.

Why 20, and what it costs — measured over all 1,422 ConDefects problems (median
31 cases, mean 36.4, range 3–148): ≥10 admits 97.5%, ≥15 admits 86.1%, **≥20
admits 76.8%**, ≥25 admits 64.8%, ≥30 admits 54.6%. 20 costs 23.2% of the
corpus. That is affordable, but it is **selection on a variable correlated with
the outcome** and is declared as a scope condition, not buried: at pool size 1
the three policies are identical by construction, so admitting tiny pools would
dilute every corpus-wide average with cells where the mechanism cannot act. The
reported effect is *the effect on tasks with a pool worth scheduling*, and the
paper must say so in exactly those words.

So `freeze` also draws an **unfiltered control arm** — a uniform sample over
the G1-only survivors, pre-registered as "reported whatever it shows". It is the
only thing that makes the scope condition checkable rather than asserted, and it
is what produces the effect size in the excluded 15–19 band.

**The funnel is counted in faults, not problems**, and every fault examined is
published with the first gate it failed, so "N examined → M admitted" reproduces
from the artifact alone.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
import random
from typing import Any, Iterable, Sequence

from caster.bench.adapter import ConDefectsAdapter, outputs_equal
from caster.bench.protocol import TestCase
from caster.exec.sandbox import Outcome, run_program

#: the `P_loop`/`P_eval` ratio, decided and recorded here rather than
#: deferred (see `docs/DECISIONS.md`). 30% held out, stratified within F2P and
#: P2P, seeded from the task name, **drawn once per fault and shared across
#: models**: the split is a property of the task, so sharing it keeps RQ3b
#: comparing curves over one denominator instead of two.
EVAL_RATIO = 0.30

#: G3. See "Why 20" above.
MIN_LOOP_POOL = 20

#: G5. The latest cut that still yields the required number of candidates — a
#: threshold set by pool size rather than by theory, and declared as such.
DATE_MIN = "2022-04-01"

#: G6. Every prompt carries the whole current source (slot S3) and `max_tokens`
#: is sized from it, so a program longer than the output budget produces a
#: truncated response every round on exactly the tasks where repair is hardest.
#: There is deliberately no LOWER bound: a short program is a perfectly good
#: fault, and screening one out would be selection on a difficulty proxy.
LINES_MAX = 120

GATE_ORDER = ("G2", "G1", "G6", "G5", "G3", "G4")


@dataclasses.dataclass(frozen=True)
class GateConfig:
    min_loop_pool: int = MIN_LOOP_POOL
    date_min: str = DATE_MIN
    lines_max: int = LINES_MAX
    eval_ratio: float = EVAL_RATIO
    control_size: int = 0
    #: the frozen pool's ordering. `interleave` is the decision (see
    #: `docs/DECISIONS.md`): ConDefects' own order is contest order, correlated
    #: with date, rating and therefore pool size, so a killed fleet cutting
    #: contiguous ranges from it would leave a *systematically* biased subset
    #: rather than a smaller grid. A seeded interleave makes any contiguous
    #: range a smaller grid. The choice is stamped into `pool_sha256`.
    pool_order: str = "interleave"
    seed: int = 20_260_101


@dataclasses.dataclass
class FrozenTask:
    name: str
    group: str
    admitted: bool
    failed_gate: str | None
    reason: str
    pool: tuple[str, ...] = ()
    unusable: dict[str, str] = dataclasses.field(default_factory=dict)
    f2p: tuple[str, ...] = ()
    p2p: tuple[str, ...] = ()
    p_loop: tuple[str, ...] = ()
    p_eval: tuple[str, ...] = ()
    gates: dict[str, bool] = dataclasses.field(default_factory=dict)
    n_lines: int = 0
    date: str | None = None
    difficulty: int | None = None
    control: bool = False      # in the unfiltered control arm (G1-only survivor)

    @property
    def split_counts(self) -> dict[str, int]:
        """The four counts recorded per task, so a reader can see the split."""
        loop, ev = set(self.p_loop), set(self.p_eval)
        return {
            "f2p_loop": len(set(self.f2p) & loop),
            "f2p_eval": len(set(self.f2p) & ev),
            "p2p_loop": len(set(self.p2p) & loop),
            "p2p_eval": len(set(self.p2p) & ev),
        }

    def as_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["split_counts"] = self.split_counts
        return d


def _stable_reference(adapter: ConDefectsAdapter, task_name: str,
                      case: TestCase) -> tuple[Outcome | None, str | None]:
    """G1 — the reference is SOUND and REPRODUCIBLE for this case.

    Two separate conditions, and the gate is the only place either is checked:

    * **sound** — the reference program agrees with the shipped `out/` file. The
      `out/` file is the authority (it is what AtCoder judged with), so a
      disagreement means this fault's reference is not a reference.
    * **reproducible** — running the reference twice gives the same answer at
      the same sandbox clock every arm will use. Nondeterministic programs
      exist, and one here would make every verdict derived from it a coin flip.

    The second check is why the reference is executed **even when `out/` is
    present**. An earlier version returned the shipped output immediately,
    which made the double run unreachable and the `nondeterministic` reason a
    reason nothing could ever carry — the check existed only in the docstring.

    The cost is two executions per case, **at freeze time only**. It is not the
    per-round cost the store exists to avoid: the reference answer is memoised in the
    adapter and the loop never recomputes it.

    A case that fails either check is marked `unusable` and excluded **from the
    pool, not from the task**.
    """
    program = adapter.program(task_name)
    runs: list[Outcome] = []
    for _ in range(2):
        got = run_program(program.reference_source, case.input_text,
                          timeout=adapter.timeout)
        if got.timed_out:
            # `slow` is NOT `wrong`: a reference that
            # times out at the clock is a cost fact; a reference that disagrees
            # with `out/` is a soundness fact, and conflating them is how a cost
            # filter sneaks back into a gate that must not have one.
            return None, "slow"
        if not got.ok:
            return None, "reference_crashed"
        runs.append(got)
    if not outputs_equal(runs[0].value, runs[1].value):
        return None, "nondeterministic"
    if case.expected_output is not None and not outputs_equal(runs[0].value,
                                                              case.expected_output):
        return None, "reference_disagrees_with_out"
    # The shipped `out/` file is the authority when there is one.
    if case.expected_output is not None:
        return Outcome(ok=True, value=case.expected_output), None
    return runs[0], None


def _split_stratified(f2p: Sequence[str], p2p: Sequence[str], *,
                      ratio: float, seed_text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split WITHIN each of F2P and P2P, seeded from the task name.

    A uniform random split breaks in both directions. If `P_loop` draws no F2P
    case, round 1 accepts the unmodified faulty program and G1's whole purpose
    is undone. If `P_eval` draws no F2P case, it cannot detect a non-repair and
    measures only regressions.
    """
    rng = random.Random(hashlib.sha256(seed_text.encode()).hexdigest())
    loop: list[str] = []
    held: list[str] = []
    for bucket in (list(f2p), list(p2p)):
        shuffled = bucket[:]
        rng.shuffle(shuffled)
        n_eval = int(round(len(shuffled) * ratio))
        # Never hold out a whole stratum: one F2P case must stay in the loop or
        # the fault is invisible to it, and one must stay in P_eval or a
        # non-repair is undetectable. Both are enforced by the caller's G3-ish
        # check below; here we just stop the rounding from zeroing a side.
        n_eval = max(0, min(n_eval, max(0, len(shuffled) - 1)))
        held.extend(shuffled[:n_eval])
        loop.extend(shuffled[n_eval:])
    return tuple(sorted(loop)), tuple(sorted(held))


def gate_one(adapter: ConDefectsAdapter, task_name: str,
             cfg: GateConfig = GateConfig()) -> FrozenTask:
    """Apply G1, G2, G3, G5, G6 to one fault. G4 is cross-fault; `freeze` owns it."""
    task = adapter.task(task_name)
    group = task.group
    gates: dict[str, bool] = {}
    program = adapter.program(task_name)
    n_lines = len(program.buggy_source.splitlines())
    date = adapter.dates().get(group)
    difficulty = adapter.difficulties().get(group)

    # Everything learned before the failing gate is carried onto the record, so
    # a fault cut by a THRESHOLD (G3, G5, G6) still has the pool and the split a
    # driver would need to walk it. That is what makes the unfiltered control
    # arm runnable rather than merely marked — without it, "report whatever it
    # shows" has nothing to report.
    known: dict[str, Any] = {}

    def fail(gate: str, reason: str) -> FrozenTask:
        gates[gate] = False
        return FrozenTask(name=task_name, group=group, admitted=False,
                          failed_gate=gate, reason=reason, gates=dict(gates),
                          n_lines=n_lines, date=date, difficulty=difficulty,
                          **known)

    # ── G2 — the test data is intact ────────────────────────────────────────
    cases = adapter._cases_for_group(group)
    if not cases:
        return fail("G2", f"no test data for {group} (is Test.zip unpacked?)")
    unusable: dict[str, str] = {}
    usable: list[TestCase] = []
    for c in cases:
        if c.expected_output is None:
            # A case with no out/ file is dropped from the pool. The reference
            # could supply one, but then the pool's expected outputs would come
            # from two different authorities and only one of them is AtCoder's.
            unusable[c.name] = "no_expected_output"
            continue
        usable.append(c)
    if not usable:
        return fail("G2", "every case lacks an out/ file")
    gates["G2"] = True

    # ── G1 — the oracle is sound and consistent ─────────────────────────────
    refs: dict[str, Outcome] = {}
    for c in usable:
        ref, why = _stable_reference(adapter, task_name, c)
        if ref is None:
            unusable[c.name] = why or "unusable"
            continue
        refs[c.name] = ref
    # Carried now, not after G1 passes: `freeze` records a per-case reason for
    # every exclusion, and a fault REJECTED by G1 is exactly the one whose
    # per-case reasons a reader wants.
    known["unusable"] = dict(unusable)
    pool_names = [c.name for c in usable if c.name in refs]
    if not pool_names:
        return fail("G1", "no case has an establishable, reproducible reference")

    by_name = {c.name: c for c in usable}
    wrong = [n for n, why in unusable.items() if why == "reference_disagrees_with_out"]
    if wrong:
        # "If it does not pass every case in the pool, it is not a reference and
        # no divergence from it means anything" (G1).
        return fail("G1", f"the reference disagrees with its own shipped out/ on "
                          f"{len(wrong)} case(s): {', '.join(sorted(wrong)[:5])}")
    if not pool_names:
        return fail("G1", "every case was dropped before the faulty version ran")

    f2p: list[str] = []
    p2p: list[str] = []
    for name in pool_names:
        got = run_program(program.buggy_source, by_name[name].input_text,
                          timeout=adapter.timeout)
        failed = got.timed_out or not got.ok or not outputs_equal(got.value, refs[name].value)
        (f2p if failed else p2p).append(name)
    if not f2p:
        return fail("G1", "the faulty version passes the whole pool — the shipped "
                          "pool does not expose this fault, so round 1 would "
                          "accept the unmodified program")
    gates["G1"] = True
    known.update(pool=tuple(sorted(pool_names)), unusable=dict(unusable),
                 f2p=tuple(sorted(f2p)), p2p=tuple(sorted(p2p)))


    # ── G6 — the source fits a prompt ───────────────────────────────────────
    if n_lines > cfg.lines_max:
        return fail("G6", f"{n_lines} lines > LINES_MAX={cfg.lines_max}")
    gates["G6"] = True

    # ── G5 — contamination cut ──────────────────────────────────────────────
    if cfg.date_min and date is not None and date < cfg.date_min:
        return fail("G5", f"contest {date} predates --date-min {cfg.date_min}")
    gates["G5"] = True

    # ── the split, then G3 on its loop half ────────────────────────────
    p_loop, p_eval = _split_stratified(f2p, p2p, ratio=cfg.eval_ratio,
                                       seed_text=f"split|{task_name}")
    known.update(p_loop=p_loop, p_eval=p_eval)
    loop_set, eval_set = set(p_loop), set(p_eval)
    if not (set(f2p) & loop_set):
        return fail("G3", "the split left no F2P case in P_loop — the fault "
                          "would be invisible to the loop")
    if not (set(f2p) & eval_set):
        return fail("G3", "the split left no F2P case in P_eval — a non-repair "
                          "would be undetectable at scoring time")
    if len(p_loop) < cfg.min_loop_pool:
        return fail("G3", f"|P_loop|={len(p_loop)} < {cfg.min_loop_pool} usable probes")
    gates["G3"] = True

    return FrozenTask(
        name=task_name, group=group, admitted=True, failed_gate=None,
        reason="admitted",
        pool=tuple(sorted(pool_names)), unusable=unusable,
        f2p=tuple(sorted(f2p)), p2p=tuple(sorted(p2p)),
        p_loop=p_loop, p_eval=p_eval, gates=dict(gates),
        n_lines=n_lines, date=date, difficulty=difficulty,
    )


def _interleave(names: Sequence[str], seed: int) -> list[str]:
    """Make any contiguous range a smaller grid, not a skewed one.

    ConDefects' own order is contest order, correlated with date (G5), with
    rating, and therefore with pool size — so cutting ranges from it hands one
    machine every hopeless task and another every trivial one. A seeded shuffle
    is the simplest ordering without that property, and because it is seeded it
    is reproducible and goes into `pool_sha256`.
    """
    ordered = sorted(names)
    random.Random(seed).shuffle(ordered)
    return ordered


def control_tasks(doc: dict[str, Any]) -> list[str]:
    """The unfiltered control arm's faults, in a reproducible order."""
    return sorted(n for n, t in doc["tasks"].items()
                  if t.get("control") and t.get("p_loop"))


def freeze(
    adapter: ConDefectsAdapter,
    task_names: Iterable[str] | None = None,
    cfg: GateConfig = GateConfig(),
    *,
    progress: Any = None,
) -> dict[str, Any]:
    """Run the whole gate and return the frozen-corpus document.

    `candidate_order_sha256 == selected_sha256` is asserted at the boundary:
    proof that the freeze step dropped nothing of its own.
    """
    names = sorted(task_names) if task_names is not None else sorted(adapter.tasks())
    candidate_order_sha256 = hashlib.sha256("\n".join(names).encode()).hexdigest()

    results_by_name: dict[str, FrozenTask] = {}
    results: list[FrozenTask] = []
    for i, name in enumerate(names, 1):
        if progress is not None:
            progress(i, len(names), name)
        try:
            results.append(gate_one(adapter, name, cfg))
        except (FileNotFoundError, KeyError) as exc:
            results.append(FrozenTask(name=name, group=name.split("/")[0],
                                      admitted=False, failed_gate="G2",
                                      reason=f"{type(exc).__name__}: {exc}"))
        results_by_name[name] = results[-1]

    # ── G4 — one fault per coding task ──────────────────────────────────────
    # A **seeded uniform choice**, deliberately not "the fault with the largest
    # K-proxy": selecting on the same statistic the analysis reports makes the
    # pool's distribution an artefact of the dedup rule.
    by_group: dict[str, list[FrozenTask]] = {}
    for t in results:
        if t.admitted:
            by_group.setdefault(t.group, []).append(t)
    for group, members in by_group.items():
        if len(members) == 1:
            members[0].gates["G4"] = True
            continue
        rng = random.Random(hashlib.sha256(f"G4|{cfg.seed}|{group}".encode()).hexdigest())
        keep = rng.choice(sorted(members, key=lambda t: t.name))
        for t in members:
            if t is keep:
                t.gates["G4"] = True
            else:
                t.admitted = False
                t.failed_gate = "G4"
                t.reason = (f"G4 dedup: {keep.name} was drawn for coding task "
                            f"{group}. {len(members)} admitted faults share its "
                            f"one test pool, so their cells are correlated and "
                            f"the cluster bootstrap's independence assumption "
                            f"fails. The draw is a SEEDED UNIFORM choice, "
                            f"deliberately not the largest K-proxy: selecting on "
                            f"the statistic the analysis reports would make the "
                            f"pool's distribution an artefact of the dedup rule")
                t.gates["G4"] = False

    admitted = [t for t in results if t.admitted]

    # ── the unfiltered control arm ─────────────────────────────────────
    # Uniform over the G1-only survivors — the faults that pass soundness but
    # were cut by a threshold. Pre-registered as "reported whatever it shows".
    if cfg.control_size > 0:
        # G1-only survivors cut by a THRESHOLD — the faults the scope condition
        # is about. G4's rejects are excluded: they share a test pool with an
        # admitted fault, which is the exact correlation G4 exists to remove,
        # and it would be back inside the arm meant to measure what the pool-
        # size threshold cut.
        g1_only = sorted(
            t.name for t in results
            if t.gates.get("G1") and not t.admitted
            and t.failed_gate in ("G3", "G5", "G6")
        )
        rng = random.Random(hashlib.sha256(f"control|{cfg.seed}".encode()).hexdigest())
        chosen = set(rng.sample(g1_only, min(cfg.control_size, len(g1_only))))
        for t in results:
            if t.name in chosen:
                t.control = True

    admitted_names = [t.name for t in admitted]
    order = (_interleave(admitted_names, cfg.seed) if cfg.pool_order == "interleave"
             else sorted(admitted_names))
    selected_sha256 = hashlib.sha256("\n".join(sorted(order)).encode()).hexdigest()

    # "`candidate_order_sha256 == selected_sha256` is asserted at the
    # boundary: proof that the freeze step dropped nothing of its own."
    #
    # Taken literally the two digests can never be equal: one covers every fault
    # examined and the other only the admitted pool, and the gate drops faults
    # by design. What the sentence is actually asking for is that the freeze
    # step drops nothing **beyond what a gate recorded** — so the assertion is
    # between the selected set and the admitted PROJECTION of the candidate
    # order, and it is a real raise, not a comment.
    projected = [n for n in names if results_by_name[n].admitted]
    projected_sha256 = hashlib.sha256("\n".join(sorted(projected)).encode()).hexdigest()
    if projected_sha256 != selected_sha256:
        raise AssertionError(
            f"freeze dropped {len(set(projected) - set(order))} fault(s) that no "
            f"gate rejected, and added {len(set(order) - set(projected))} that no "
            f"gate admitted. The pool is not the gated pool."
        )

    doc: dict[str, Any] = {
        "adapter": adapter.name,
        "determinism": adapter.determinism,
        "fixed_cost_scope": adapter.fixed_cost_scope,
        "config": dataclasses.asdict(cfg),
        "pool_order": list(order),
        "candidate_order_sha256": candidate_order_sha256,
        "selected_sha256": selected_sha256,
        "funnel": _funnel(results),
        "tasks": {t.name: t.as_dict() for t in results},
    }
    doc["pool_sha256"] = hashlib.sha256(
        json.dumps(
            {"order": doc["pool_order"],
             "tasks": {n: {"p_loop": d["p_loop"], "p_eval": d["p_eval"]}
                       for n, d in doc["tasks"].items() if d["admitted"]},
             "cfg": doc["config"]},
            sort_keys=True, separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return doc


def _funnel(results: Sequence[FrozenTask]) -> dict[str, Any]:
    """"N examined → M admitted", counted in FAULTS, with the first gate failed."""
    counts: dict[str, int] = {g: 0 for g in GATE_ORDER}
    for t in results:
        if not t.admitted and t.failed_gate:
            counts[t.failed_gate] = counts.get(t.failed_gate, 0) + 1
    admitted = sum(1 for t in results if t.admitted)
    sizes = sorted(len(t.p_loop) for t in results if t.admitted)
    return {
        "examined": len(results),
        "admitted": admitted,
        "rejected_by_gate": counts,
        "loop_pool_median": sizes[len(sizes) // 2] if sizes else 0,
        "loop_pool_min": sizes[0] if sizes else 0,
        "loop_pool_max": sizes[-1] if sizes else 0,
    }


def write(doc: dict[str, Any], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=False), encoding="utf-8")
    tmp.replace(path)


def load(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def install(adapter: ConDefectsAdapter, doc: dict[str, Any],
            *, include_control: bool = False) -> None:
    """Point an adapter at a frozen corpus: gated pools in, `P_eval` held out.

    `include_control` additionally installs the unfiltered control arm's faults — the G1-only survivors a threshold cut. They are **never** in
    `pool_order`, so no main-grid analysis can pool them by accident; a driver
    reaches them only by asking for them, and the arm is reported separately,
    "whatever it shows".
    """
    pools: dict[str, list[str]] = {}
    held: dict[str, list[str]] = {}
    for name, t in doc["tasks"].items():
        usable = t.get("admitted") or (include_control and t.get("control")
                                       and t.get("p_loop"))
        if not usable:
            continue
        pools[name] = list(t["p_loop"])
        held[name] = list(t["p_eval"])
    adapter.set_pools(pools)
    adapter.set_held_out(held)


# ── scoring on P_eval ───────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class RegressionResult:
    """One patch, scored on both halves of `P_eval`. No short-circuit anywhere.

    The loop stops at the first counterexample because one is all a refutation
    needs; here the **count** is the point, so a patch that breaks three P2P
    cases comes out as three rather than as "rejected".
    """

    f2p_total: int
    f2p_fixed: int
    p2p_total: int
    p2p_broken: int
    broken_cases: tuple[str, ...]
    still_failing_cases: tuple[str, ...]
    truly_correct: bool
    inconclusive: bool

    @property
    def regression_rate(self) -> float | None:
        """Share of already-working cases this patch broke.

        None when no P2P case is held out — a fault whose footprint is the whole
        pool cannot regress, and 0.0 would claim a measurement never made.
        """
        return (self.p2p_broken / self.p2p_total) if self.p2p_total else None

    @property
    def fix_rate(self) -> float | None:
        return (self.f2p_fixed / self.f2p_total) if self.f2p_total else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "f2p_total": self.f2p_total, "f2p_fixed": self.f2p_fixed,
            "fix_rate": self.fix_rate,
            "p2p_total": self.p2p_total, "p2p_broken": self.p2p_broken,
            "regression_rate": self.regression_rate,
            "regression_broken_cases": list(self.broken_cases),
            "regression_still_failing": list(self.still_failing_cases),
            "regression_inconclusive": self.inconclusive,
        }


def score_on_eval(adapter: ConDefectsAdapter, task_name: str, patch: str,
                  frozen: dict[str, Any]) -> RegressionResult:
    """Score an accepted patch on `P_eval`: fix rate, regression rate, clean or not.

    This is the ONLY place a held-out probe may be executed, and it runs after
    the episode has ended. `verify.scan` raises `HeldOutProbeLeak` if one ever
    reaches the loop.
    """
    t = frozen["tasks"][task_name]
    held = set(t["p_eval"])
    f2p = [n for n in t["f2p"] if n in held]
    p2p = [n for n in t["p2p"] if n in held]
    if not f2p and not p2p:
        return RegressionResult(0, 0, 0, 0, (), (), False, True)

    failed: set[str] = set()
    for name in sorted(held):
        case = adapter.case(task_name, name)
        ref = adapter.reference(task_name, name)
        if case is None or ref is None:
            continue   # became unusable between the two passes; not the patch's fault
        got = run_program(patch, case.input_text, timeout=adapter.timeout)
        if got.timed_out or not got.ok or not outputs_equal(got.value, ref.value):
            failed.add(name)

    broken = tuple(n for n in p2p if n in failed)
    still = tuple(n for n in f2p if n in failed)
    return RegressionResult(
        f2p_total=len(f2p), f2p_fixed=len(f2p) - len(still),
        p2p_total=len(p2p), p2p_broken=len(broken),
        broken_cases=broken, still_failing_cases=still,
        truly_correct=not failed, inconclusive=False,
    )
