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

**Not a gate, deliberately: cost.** No fault is dropped for being expensive,
and there is no second threshold in this file. The sandbox clock is not one
either: it is the instrument's **declared capability** — the same
`SANDBOX_TIMEOUT_SEC` every arm runs under, carried in `protocol_sha` — so a
case whose reference does not complete inside it is a case this instrument
cannot measure, not a case judged too slow to be worth measuring.

The per-case reason keeps the two apart. `exceeds_sandbox_timeout` is a
**capability** fact: the sandbox, at its configured limit, cannot run this
case. `reference_disagrees_with_out` is a **soundness** fact: it ran, and it is
not a reference. Conflating them is how a cost filter sneaks into a gate that
must not have one — and a second, stricter clock added to prune expensive
faults would delete exactly the expensive-oracle population the mechanism helps
most, and bias every cost figure downward.

**G3 does not screen on pool size.** It used to: `|P_loop| ≥ 20`. That
threshold is gone from the gate and lives in the analysis instead. `freeze`
admits every fault whose oracle is sound, the grid runs all of them, and
`analyse` bands the result by `|P_loop|` afterwards (`report/analyse.py`,
`POOL_BANDS`).

Nothing is lost by moving it. Measured over all 1,422 ConDefects problems
(median 31 cases, mean 36.4, range 3–148): ≥10 admits 97.5%, ≥15 admits 86.1%,
**≥20 admits 76.8%**, ≥25 admits 64.8%, ≥30 admits 54.6%. The old pool was the
76.8%, and it was a population nobody could weight back to the benchmark. The
post-hoc `standard` band **is** that pool, reproduced exactly, and the other
23.2% is now reported beside it instead of being deleted. `--min-pool N`
re-imposes the screen for anyone who wants the old behaviour.

It is safe to move because the threshold was never load-bearing for
correctness, only for interpretation: at pool size 1 every policy is the same
schedule by construction, so tiny pools **dilute** a corpus-wide average rather
than corrupt it — and a band separates them, which is exactly what this file's
own rule says to do with a covariate. A screen is the one thing that cannot be
undone after the run; a band can be drawn any number of ways.

**What G3 still rejects, and must.** Two checks, both about the SPLIT rather
than about size:

* `P_loop` with no F2P case — the fault would be invisible to the loop, and
  round 1 would accept the unmodified faulty program.
* `P_eval` with no F2P case — a non-repair would be undetectable at scoring.

Together they put a real floor under the pool (two F2P cases, one per half)
without screening on a difficulty proxy.

**G6, G5 and G4 are off by default too.** Each used to cut the corpus before
any cell existed; each is now either a post-hoc band or a fix somewhere the fix
belongs.

* **G6 (`--lines-max`, default 0 = no limit).** The real constraint is the
  output budget, not a line count: a program longer than `max_tokens` produces
  a truncated reply. That already surfaces where it happens — `context_overflow`
  in the proposal-error table, reported before any scheduler number — so a
  pre-treatment screen on source length was measuring the symptom upstream of
  the thing under study.
* **G5 (`--date-min`, default empty).** Contamination is real, but a cutoff
  *assumes* it away. Running every era and banding by contest date **measures**
  it: `analyse` reports the arms by era, and the pre-cutoff band answers the
  reviewer's question with a number instead of an assertion.
* **G4 (`--dedup-per-group`, default off).** Faults sharing one AtCoder problem
  share a test pool, so their cells are correlated — but the answer to
  correlated observations is to cluster on the shared unit, not to throw data
  away. The cluster bootstrap now resamples **coding tasks**, so every fault can
  stay and the independence assumption holds where it is actually needed.

`freeze` still draws an **unfiltered control arm** — a uniform sample over the
G1-only survivors a threshold cut, pre-registered as "reported whatever it
shows". With every default screen gone it covers only what an operator switches
back on.

**The funnel is counted in faults, not problems**, and every fault examined is
published with the first gate it failed, so "N examined → M admitted" reproduces
from the artifact alone.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pathlib
import random
from typing import Any, Iterable, Sequence

from caster.bench.adapter import ConDefectsAdapter, outputs_equal
from caster.bench.protocol import TestCase
from caster.exec.sandbox import Outcome, run_program

#: **0 by default: the oracle is the WHOLE pool.** RQ1 measures the cost of
#: refuting, and the thing being scheduled is the probe list — holding 30% of
#: it back removes evidence from the question under study and forces G3 to
#: exist. With no holdout, `P_loop` is the pool and `P_eval` is empty.
#:
#: `--holdout 0.3` draws one: stratified within F2P and P2P, seeded from the
#: task name, **drawn once per fault and shared across models**, so the same
#: handicap lands on every arm and cancels in a paired comparison. Switch it on
#: for the one question the full pool cannot answer — whether an accepted patch
#: FIT the disclosed probes rather than fixing the fault. `success@B` without a
#: holdout is a **plausible**-patch rate, not a correctness rate, and must be
#: reported under that name.
HOLDOUT = 0.0

#: G3. The floor, not a screen: a `P_loop` with no probe at all cannot be
#: scheduled by anything. Deliberately **1 rather than 0** — 0 would be a
#: falsy sentinel that switches the check off, and the check should always be
#: on; what changed is the threshold, not whether it is evaluated. Raise it
#: (`--min-pool 20`) to re-impose the old pre-treatment screen; the gate
#: otherwise admits every sound fault and `analyse` bands by `|P_loop|` after
#: the run.
MIN_LOOP_POOL = 1

#: G5. The latest cut that still yields the required number of candidates — a
#: threshold set by pool size rather than by theory, and declared as such.
#: Empty = no cut. Contamination is banded by contest date in `analyse` rather
#: than assumed away by a cutoff — running every era MEASURES the effect, and
#: the pre-2022 band is the number a reviewer actually wants.
#: G5. `None` = no cut. Deliberately not `""`: an empty string is FALSY, so
#: guarding the check with `if cfg.date_min` tests truthiness rather than
#: configuration — the same mistake `min_loop_pool = 0` would have been. `None`
#: means "unset", `is not None` says exactly that, and it serialises to `null`.
DATE_MIN: str | None = None

#: G6. Every prompt carries the whole current source (slot S3) and `max_tokens`
#: is sized from it, so a program longer than the output budget produces a
#: truncated response every round on exactly the tasks where repair is hardest.
#: There is deliberately no LOWER bound: a short program is a perfectly good
#: fault, and screening one out would be selection on a difficulty proxy.
#: 0 = no limit. A program longer than the output budget yields a truncated
#: reply, and that is recorded where it happens (`context_overflow` in the
#: proposal-error table), not screened for upstream.
#: G6. `None` = no ceiling. `0` would be worse than wrong here: it is falsy, so
#: it needs a truthiness guard, AND it is a legitimate line count, so it
#: conflates "no limit" with "reject everything". A ceiling's absence is
#: infinity, not a small number — and `None` expresses that without putting
#: `Infinity` into `tasks.json`, which `pool_sha256` hashes.
LINES_MAX: int | None = None

GATE_ORDER = ("G2", "G1", "G6", "G5", "G3", "G4")


@dataclasses.dataclass(frozen=True)
class GateConfig:
    min_loop_pool: int = MIN_LOOP_POOL
    date_min: str | None = DATE_MIN
    lines_max: int | None = LINES_MAX
    holdout: float = HOLDOUT
    control_size: int = 0
    #: G4. Off: faults sharing a coding task all run, and the cluster bootstrap
    #: resamples coding tasks so their correlation is handled where it matters.
    dedup_per_group: bool = False
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

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FrozenTask":
        """Rebuild one checkpointed fault. `split_counts` is derived, not stored.

        Unknown keys are dropped rather than raising: a checkpoint written by an
        older build should cost a re-gate of that fault at worst, never the
        whole resume.
        """
        fields = {f.name for f in dataclasses.fields(cls)}
        kw = {k: v for k, v in d.items() if k in fields}
        for k in ("pool", "f2p", "p2p", "p_loop", "p_eval"):
            if k in kw and kw[k] is not None:
                kw[k] = tuple(kw[k])
        return cls(**kw)


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

    Both checks work by *running* the reference, so both need the sandbox to be
    able to run it. A case it cannot — the reference exceeds
    `SANDBOX_TIMEOUT_SEC` — falls outside the instrument's declared capability
    and is marked `exceeds_sandbox_timeout`. That is not a judgement that the
    case is too expensive: there is only ever one clock, it is the same one
    every arm runs under, and it is recorded in `protocol_sha`.

    A case that fails any of these is marked `unusable` and excluded **from the
    pool, not from the task**.
    """
    program = adapter.program(task_name)
    runs: list[Outcome] = []
    for _ in range(2):
        got = run_program(program.reference_source, case.input_text,
                          timeout=adapter.timeout)
        if got.timed_out:
            # Outside the sandbox's declared capability — NOT "too expensive".
            # The clock is `SANDBOX_TIMEOUT_SEC`, the same limit every arm runs
            # under and part of `protocol_sha`, so a reference that does not
            # finish inside it cannot be established by this instrument at all.
            # Kept distinct from `reference_disagrees_with_out`, which is a
            # soundness fact, so that no cost filter can hide behind either.
            return None, "exceeds_sandbox_timeout"
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
    if cfg.lines_max is not None and n_lines > cfg.lines_max:
        return fail("G6", f"{n_lines} lines > LINES_MAX={cfg.lines_max}")
    gates["G6"] = True

    # ── G5 — contamination cut ──────────────────────────────────────────────
    if cfg.date_min is not None and date is not None and date < cfg.date_min:
        return fail("G5", f"contest {date} predates --date-min {cfg.date_min}")
    gates["G5"] = True

    # ── the split (only if asked), then G3 on its loop half ────────────
    p_loop, p_eval = _split_stratified(f2p, p2p, ratio=cfg.holdout,
                                       seed_text=f"split|{task_name}")
    known.update(p_loop=p_loop, p_eval=p_eval)
    loop_set, eval_set = set(p_loop), set(p_eval)
    if cfg.holdout > 0:
        # G3 exists ONLY because a holdout exists. Without one, `P_loop` is the
        # whole pool, `f2p` is non-empty by G1, and both checks are satisfied by
        # construction — so asserting them would be theatre.
        if not (set(f2p) & loop_set):
            return fail("G3", "the split left no F2P case in P_loop — the fault "
                              "would be invisible to the loop")
        if not (set(f2p) & eval_set):
            return fail("G3", "the split left no F2P case in P_eval — a "
                              "non-repair would be undetectable at scoring time")
    if len(p_loop) < cfg.min_loop_pool:
        # At the default of 1 this is the floor — a pool no policy can schedule.
        # G1 already implies it (`pool_names` non-empty), so it fires only if
        # that is ever weakened. Above 1 it is the old pre-treatment screen,
        # opt-in: a screen cannot be undone after the run, while the band
        # `analyse` draws from `|P_loop|` can be re-cut any number of ways.
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
    results = gate_many(adapter, names, cfg, progress=progress)
    return assemble(adapter, names, results, cfg)


def gate_one_safe(adapter: ConDefectsAdapter, name: str,
                  cfg: GateConfig) -> FrozenTask:
    """`gate_one`, with a missing or malformed checkout recorded as a G2 reject.

    A fault that raises is a fault whose test data is not intact, which is what
    G2 is. Letting it propagate would end the walk on one bad checkout and lose
    every result before it.
    """
    try:
        return gate_one(adapter, name, cfg)
    except (FileNotFoundError, KeyError) as exc:
        return FrozenTask(name=name, group=name.split("/")[0], admitted=False,
                          failed_gate="G2", reason=f"{type(exc).__name__}: {exc}")


def gate_many(adapter: ConDefectsAdapter, names: Sequence[str],
              cfg: GateConfig = GateConfig(), *, progress: Any = None,
              done_path: pathlib.Path | None = None) -> list[FrozenTask]:
    """Gate each fault in `names`, resumably.

    `done_path` is an append-only JSONL of finished faults, flushed per fault.
    A re-issued command reads it, skips what it holds and continues — the same
    discipline `evaluate` gets from `cells.done`, and the reason a freeze that
    runs for hours is not lost to a closed laptop.

    Per-fault work is INDEPENDENT: G4 and the control arm are cross-fault and
    belong to `assemble`, which is what makes a sharded freeze sound.
    """
    done: dict[str, FrozenTask] = {}
    if done_path is not None and done_path.is_file():
        for line in done_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                done[rec["name"]] = FrozenTask.from_dict(rec)
            except (json.JSONDecodeError, KeyError, TypeError):
                continue   # a line torn by a kill is simply redone
    out: list[FrozenTask] = []
    handle = None
    if done_path is not None:
        done_path.parent.mkdir(parents=True, exist_ok=True)
        handle = done_path.open("a", encoding="utf-8")
    try:
        for i, name in enumerate(names, 1):
            if progress is not None:
                progress(i, len(names), name)
            hit = done.get(name)
            if hit is not None:
                out.append(hit)
                continue
            t = gate_one_safe(adapter, name, cfg)
            out.append(t)
            if handle is not None:
                handle.write(json.dumps(t.as_dict(), sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
    finally:
        if handle is not None:
            handle.close()
    return out


def assemble(adapter: ConDefectsAdapter, names: Sequence[str],
             results: Sequence[FrozenTask],
             cfg: GateConfig = GateConfig()) -> dict[str, Any]:
    """The cross-fault half: G4, the control arm, the order and the digests.

    Separate from `gate_many` because every step here needs the WHOLE corpus —
    G4 dedups across faults of one coding task, the control arm samples over
    all G1-only survivors, and `pool_sha256` covers the complete pool. Running
    any of them on a shard would produce a different corpus per shard.
    """
    names = sorted(names)
    results = list(results)
    candidate_order_sha256 = hashlib.sha256("\n".join(names).encode()).hexdigest()
    results_by_name: dict[str, FrozenTask] = {t.name: t for t in results}

    # ── G4 — one fault per coding task ──────────────────────────────────────
    # A **seeded uniform choice**, deliberately not "the fault with the largest
    # K-proxy": selecting on the same statistic the analysis reports makes the
    # pool's distribution an artefact of the dedup rule.
    by_group: dict[str, list[FrozenTask]] = {}
    for t in results:
        if t.admitted:
            by_group.setdefault(t.group, []).append(t)
    if not cfg.dedup_per_group:
        # Off by default. Correlated cells are a CLUSTERING problem, and
        # `report/analyse.py` resamples coding tasks, so dropping faults to buy
        # independence throws away data for something the bootstrap already
        # handles. `--dedup-per-group` restores the old behaviour.
        for members in by_group.values():
            for t in members:
                t.gates["G4"] = True
        by_group = {}
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
