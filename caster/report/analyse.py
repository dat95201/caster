"""Aggregation, the cluster bootstrap, and the per-arm tables.

**Merge audit before analysis** (Invariant 12). Nothing here re-derives a cell
key by hand: every filter reads `RoundRecord.cell`, which is
`CellSpec.as_dict()` verbatim, so the rule binds consumers by construction. The
two ways an analysis layer breaks it — re-deriving a key from a few pinned
fields, and hand-adding arm-identity fields to a frozen summary — both come from
a hand-maintained list, and there is none here.

Three things reported **before** anything else, because the numbers below them
are unreadable otherwise:

* `proposal_error` counts **per arm and per model**. The steered
  arms carry the longest prompts, so every proposal failure lands
  differentially on exactly the comparison the experiment exists to make. A
  weak proposer above ~20% is a threat to validity, not a rung on a ladder.
* the realised **`k_facts`** distribution per steered arm. A steering
  arm whose realised mean `k_facts` is below the configured `k` is reporting a
  different treatment than its label says — which is what happens when
  `explore_p` is off and a task's first probe kills everything.
* the per-`(task, model)` **repair rate**, computed from the `no-mem` arm and
  written into the frozen artifacts, **never recomputed ad hoc**. Bands
  are computed post hoc from it; it is an outcome, so it may not select the
  corpus.

## What the confidence interval actually measures

With a frozen cost table, every arm's reported CPU is a deterministic function
of that table and of the scan orders, so a bootstrap over cells measures only
the **scan-order** variation and reports an interval that omits the dominant
source of uncertainty. `--remeasure-cost` estimates the within-cell variance of
*one* `cpu_s` sample, which is an **input** to the right calculation, not the
calculation. The right one is available for free because the replay costs
nothing: **resample the cost table from the remeasured draws and re-run the
replay per bootstrap iteration.** `bootstrap_cpu` takes that remeasure sample
and does exactly this; called without one it says so in its result rather than
quietly reporting the narrow interval.

## The cluster, and why it is the coding task

Several faults can share one AtCoder problem and therefore one test pool, so
their cells are correlated. The cluster is the **coding task** — the AtCoder
problem, `abc221_f` in `abc221_f/40898300` — resampled with replacement, which
puts every fault of one problem in or out of a draw together.

This is why the gate no longer drops faults to buy independence (G4, now
`--dedup-per-group`). Dropping data is one answer to correlated observations;
clustering on the shared unit is the better one, because it keeps every fault
and the assumption holds where it is actually needed. A bootstrap that
resampled *rows* would still treat the rounds of one episode as independent,
which is the mistake underneath both.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import pathlib
import random
import statistics
from typing import Any, Callable, Iterable, Sequence

from caster.exec.ledger import CaseLedger
from caster.keys import PRIORS, schedule_tie_break, schedule_tie_rank
from caster.report.records import RoundRecord
from caster.run import policies


# ── per-cell reduction ───────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class Cell:
    """One `(task, seed, policy)` episode, reduced. The analysis unit."""

    task: str
    seed: int
    policy: str
    episode_id: str
    accepted: bool
    first_accept_round: int | None
    rounds_charged: int
    cpu_s: float
    test_calls: int
    sandbox_runs: int
    verdict_hits: int
    proposal_errors: dict[str, int]
    k_facts_mean: float
    distinct_patches: int
    patches: int
    model: str
    prompt_tokens: int
    completion_tokens: int
    usd: float

    @property
    def duplicate_rate(self) -> float:
        """RQ3's PRIMARY metric: the share of draws that were already drawn.

        It was parked under the verbalized-sampling arm as "measurable today".
        It is not a by-product — renormalising the patch family IS half the
        study, and this is the quantity that says whether the family moved.
        Chen et al. (arXiv 2503.15050) measure 65% duplicates in ChatRepair's
        generated patches, which is the number this one is read against.
        """
        if self.patches == 0:
            return 0.0
        return 1.0 - self.distinct_patches / self.patches


def cells(rows: Iterable[RoundRecord], *, budget: int | None = None) -> list[Cell]:
    """Reduce rows to cells, truncating past the first accept when asked.

    `budget` truncates `force_full_budget` runs back out so the floor arm stays
    comparable to the arms it is tabulated against ("draw full, truncate
    at analysis"). Without it, π̂ would be read off a curve of a different B.
    """
    # Keyed on `episode_id` as well as `(task, seed, policy)`, which is the
    # failure `keys.py` names in its own docstring: "re-deriving a cell key by
    # hand from a few pinned fields and keying on `(task, seed)`. Every arm
    # differing only in an unpinned field collapses onto one key". Two of
    # those are live right now — the pre-registered prior pair sweeps `prior`
    # across the same `(task, seed, policy)`, and `model` is in the key but
    # was not in this grouping — so both were being averaged into one cell
    # with the winner decided by sort order. `episode_id` is a hash of the
    # WHOLE key, so it separates exactly what the key separates and nothing
    # else: a cell topped up from B=12 to B=20 keeps one id, because `budget`
    # is deliberately not in the key.
    grouped: dict[tuple[str, int, str, str], list[RoundRecord]] = {}
    for r in rows:
        grouped.setdefault((r.task, r.seed, r.policy, r.episode_id), []).append(r)

    out: list[Cell] = []
    for (task, seed, policy, _episode), rs in sorted(grouped.items()):
        rs.sort(key=lambda r: (r.round_index, r.candidate_index))
        if budget is not None:
            rs = [r for r in rs if r.round_index <= budget]
        if not rs:
            continue
        first = next((r.round_index for r in rs if r.accept), None)
        kept = rs if first is None else [r for r in rs if r.round_index <= first]
        errors: dict[str, int] = {}
        for r in kept:
            if r.proposal_error:
                errors[r.proposal_error] = errors.get(r.proposal_error, 0) + 1
        shas = [r.patch_sha for r in kept if r.patch_sha]
        k_vals = [r.k_facts for r in kept if r.patch_sha]
        out.append(Cell(
            task=task, seed=seed, policy=policy, episode_id=kept[0].episode_id,
            accepted=first is not None, first_accept_round=first,
            rounds_charged=sum(1 for r in kept if r.charged),
            cpu_s=sum(r.cpu_s for r in kept),
            test_calls=sum(r.test_calls for r in kept),
            sandbox_runs=sum(r.sandbox_runs for r in kept),
            verdict_hits=sum(r.verdict_hits for r in kept),
            proposal_errors=errors,
            k_facts_mean=(sum(k_vals) / len(k_vals)) if k_vals else 0.0,
            distinct_patches=len(set(shas)), patches=len(shas),
            model=str((kept[0].cell or {}).get("model", "")),
            # The model half of the bill. Every LLM-repair baseline is priced
            # in tokens (CigaR: 127K/bug against 467K) and a CPU-only table
            # answers the cheaper half of the question.
            prompt_tokens=sum(r.prompt_tokens or 0 for r in kept),
            completion_tokens=sum(r.completion_tokens or 0 for r in kept),
            usd=sum(r.usd or 0.0 for r in kept),
        ))
    return out


# ── the three things reported first ──────────────────────────────────────────


def proposal_error_table(cs: Sequence[Cell]) -> list[dict[str, Any]]:
    """Per arm AND per model, before any scheduler number."""
    agg: dict[tuple[str, str], dict[str, Any]] = {}
    for c in cs:
        slot = agg.setdefault((c.policy, c.model), {
            "policy": c.policy, "model": c.model, "cells": 0, "rounds": 0,
            "backend_unreachable": 0, "context_overflow": 0, "truncated_response": 0,
        })
        slot["cells"] += 1
        slot["rounds"] += c.rounds_charged
        for k, n in c.proposal_errors.items():
            slot[k] = slot.get(k, 0) + n
    rows = []
    for slot in agg.values():
        errs = sum(slot[k] for k in
                   ("backend_unreachable", "context_overflow", "truncated_response"))
        slot["error_rate"] = (errs / slot["rounds"]) if slot["rounds"] else 0.0
        slot["threat"] = slot["error_rate"] > 0.20
        rows.append(slot)
    return sorted(rows, key=lambda r: (r["model"], r["policy"]))


def k_facts_table(cs: Sequence[Cell]) -> list[dict[str, Any]]:
    """Realised `k_facts` per steered arm, against the configured k."""
    agg: dict[str, list[float]] = {}
    for c in cs:
        # A merged log can name an arm the roster has since dropped; that is a
        # reporting problem, not a crash.
        arm = policies.ARMS.get(c.policy)
        if arm is not None and arm.overrides.get("steer_on"):
            agg.setdefault(c.policy, []).append(c.k_facts_mean)
    rows = []
    for policy, vals in sorted(agg.items()):
        configured = policies.ARMS[policy].overrides.get("k_facts", 2)
        realised = statistics.fmean(vals) if vals else 0.0
        rows.append({
            "policy": policy, "configured_k": configured,
            "realised_k_mean": round(realised, 3), "cells": len(vals),
            # "A steering arm whose realised mean k_facts is below the
            # configured k is reporting a different treatment than its label."
            "mislabelled": realised < configured - 0.05,
        })
    return rows


def freeze_repair_rates(rates: dict[str, float], pool_dir: pathlib.Path) -> pathlib.Path:
    """Write the per-`(task, model)` repair rate into the frozen artifacts.

    Written into the frozen artifacts beside `pool_sha256`, and **never
    recomputed ad hoc by an analysis script**. A rate recomputed on
    every call is a rate that changes when the merge changes, and every band
    analysis downstream would move with it without saying so.
    """
    pool_dir.mkdir(parents=True, exist_ok=True)
    path = pool_dir / "repair_rates.json"
    existing = {}
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    existing.update(rates)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(existing, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    return path


def load_repair_rates(pool_dir: pathlib.Path) -> dict[str, float]:
    path = pool_dir / "repair_rates.json"
    if not path.is_file():
        return {}
    try:
        return {k: float(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}


def repair_rates(cs: Sequence[Cell], *, floor_arm: str = "no-mem") -> dict[str, float]:
    """Per `(task, model)` repair rate, from the floor arm over its seeds.

    Written into the frozen artifacts beside `pool_sha256` and read from there
    by every band analysis — **never recomputed ad hoc by an analysis script**. It is an outcome, so it may not select the corpus; and it is **per
    model**, because the same task is a different difficulty for a 3B and for a
    hosted model.
    """
    agg: dict[str, list[bool]] = {}
    for c in cs:
        if c.policy != floor_arm:
            continue
        agg.setdefault(f"{c.task}|{c.model}", []).append(c.accepted)
    return {k: (sum(v) / len(v)) for k, v in sorted(agg.items()) if v}


#: Post-hoc bands over the repair rate (RQ3a). Computed AFTER the
#: run, never used to select the corpus.
BANDS: tuple[tuple[str, float, float], ...] = (
    ("dead", 0.0, 0.0001),
    ("hard", 0.0001, 0.25),
    ("mid", 0.25, 0.60),
    ("easy", 0.60, 0.9999),
    ("trivial", 0.9999, 1.01),
)


def band_of(rate: float) -> str:
    for name, lo, hi in BANDS:
        if lo <= rate < hi:
            return name
    return "trivial"


#: Post-hoc bands over `|P_loop|` — the screen G3 used to apply BEFORE the run.
#: `standard` is exactly the old `--min-pool 20` corpus, so every number that
#: gate would have produced is still computable, and the 23.2% it used to
#: delete is reported beside it instead. Pool size is a property of the frozen
#: pool rather than an outcome, so banding on it is a stratification, not a
#: screen — and unlike a screen it can be re-cut without re-running anything.
POOL_BANDS: tuple[tuple[str, int, int], ...] = (
    ("degenerate", 0, 2),         # |P_loop| = 1: every policy is one schedule
    ("tiny", 2, 10),
    ("small", 10, 20),
    ("standard", 20, 1 << 30),    # what `--min-pool 20` admitted
)


def pool_band_of(n: int) -> str:
    for name, lo, hi in POOL_BANDS:
        if lo <= n < hi:
            return name
    return "standard"


def cluster_of(task: str) -> str:
    """The AtCoder problem a fault belongs to — `abc221_f/40898300` -> `abc221_f`.

    The unit the bootstrap resamples. Faults of one problem share a test pool,
    so their cells are correlated and must move together.
    """
    return task.split("/")[0]


#: The contamination cut G5 used to apply BEFORE the run. A cutoff *assumes*
#: contamination away; running every era and banding by contest date
#: **measures** it, and the pre-cutoff band is the number a reviewer wants.
DATE_CUTOFF = "2022-04-01"


def era_of(date: str | None, cutoff: str = DATE_CUTOFF) -> str:
    if not date:
        return "undated"
    return "pre-cutoff" if date < cutoff else "post-cutoff"


def load_pool_sizes(pool_dir: pathlib.Path) -> dict[str, int]:
    """`|P_loop|` per admitted task, read from the frozen corpus.

    From `tasks.json`, not from the rows: the pool size is a property of the
    frozen pool, and deriving it from whatever cells happened to land in this
    merge would make a band a function of shard completeness.
    """
    path = pool_dir / "tasks.json"
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return {n: len(t.get("p_loop") or ())
                for n, t in doc.get("tasks", {}).items() if t.get("admitted")}
    except (json.JSONDecodeError, TypeError, AttributeError):
        return {}


def load_task_dates(pool_dir: pathlib.Path) -> dict[str, str]:
    """Contest date per admitted task, from the frozen corpus."""
    path = pool_dir / "tasks.json"
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return {n: t["date"] for n, t in doc.get("tasks", {}).items()
                if t.get("admitted") and t.get("date")}
    except (json.JSONDecodeError, TypeError, AttributeError):
        return {}


def _arms_by(cs: Sequence[Cell], key: dict[str, Any],
             band: Any, order: Sequence[str]) -> dict[str, Any]:
    """The arm table split by any pre-run property of the task.

    Every screen the gate used to apply lands here: decided after the run,
    unable to change the corpus, and re-cuttable without re-running anything.
    Cells whose task is not in the frozen pool — the control arm's — are left
    out, since pooling them would restore the correlation that arm measures.
    """
    if not key:
        return {}
    groups: dict[str, list[Cell]] = {}
    for c in cs:
        if c.task in key:
            groups.setdefault(band(key[c.task]), []).append(c)
    out: dict[str, Any] = {}
    for name in order:
        group = groups.get(name) or []
        if group:
            out[name] = {"tasks": len({c.task for c in group}),
                         "cells": len(group), "arms": arm_table(group)}
    return out


def arms_by_era(cs: Sequence[Cell], dates: dict[str, str]) -> dict[str, Any]:
    """RQ1 per contest era — what G5's cutoff used to decide in advance.

    `pre-cutoff` is the corpus `--date-min 2022-04-01` deleted. Reporting it
    beside `post-cutoff` turns "the model may have memorised these" from an
    assumption into a comparison.
    """
    return _arms_by(cs, dict(dates), era_of,
                    ("pre-cutoff", "post-cutoff", "undated"))


def arms_by_pool_band(cs: Sequence[Cell],
                      sizes: dict[str, int]) -> dict[str, Any]:
    """RQ1's arm table, stratified by `|P_loop|`. The screen, moved after the run.

    A pre-treatment screen has to be decided before any cell exists and cannot
    be undone; this is decided afterwards, cannot change the corpus, and can be
    re-cut any number of ways. Cells whose task is not in the frozen pool —
    control-arm faults — are left out, because they are reported separately
    "whatever it shows" and pooling them here would put back the correlation
    the control arm exists to measure.
    """
    if not sizes:
        return {}
    groups: dict[str, list[Cell]] = {}
    for c in cs:
        if c.task in sizes:
            groups.setdefault(pool_band_of(sizes[c.task]), []).append(c)
    out: dict[str, Any] = {}
    for name, _lo, _hi in POOL_BANDS:
        group = groups.get(name) or []
        if not group:
            continue
        tasks = {c.task for c in group}
        out[name] = {
            "tasks": len(tasks),
            "cells": len(group),
            "pool_median": statistics.median(sorted(sizes[t] for t in tasks)),
            "arms": arm_table(group),
        }
    return out


# ── the main tables ──────────────────────────────────────────────────────────


def arm_table(cs: Sequence[Cell]) -> list[dict[str, Any]]:
    """RQ1's primary metric: CPU seconds per episode, plus the usual companions.

    Tokens and USD are here and not in a separate table on purpose: the oracle
    and the model are two halves of one bill, and a reader who sees only one of
    them cannot tell whether a cheaper schedule was bought with more rounds.
    """
    agg: dict[str, list[Cell]] = {}
    for c in cs:
        agg.setdefault(c.policy, []).append(c)
    rows = []
    for policy, group in sorted(agg.items()):
        cpu = [c.cpu_s for c in group]
        first = [c.first_accept_round for c in group if c.first_accept_round]
        rows.append({
            "policy": policy,
            "block": policies.arm(policy).block if policy in policies.ARMS else "?",
            "cells": len(group),
            "cpu_s_mean": round(statistics.fmean(cpu), 4) if cpu else 0.0,
            "cpu_s_median": round(statistics.median(cpu), 4) if cpu else 0.0,
            "test_calls_mean": round(statistics.fmean([c.test_calls for c in group]), 3),
            "sandbox_runs": sum(c.sandbox_runs for c in group),
            "verdict_hits": sum(c.verdict_hits for c in group),
            "success_at_b": round(sum(c.accepted for c in group) / len(group), 4),
            "first_accept_mean": round(statistics.fmean(first), 3) if first else None,
            "duplicate_rate": round(statistics.fmean([c.duplicate_rate for c in group]), 4),
            "prompt_tokens_mean": round(statistics.fmean(
                [c.prompt_tokens for c in group]), 1),
            "completion_tokens_mean": round(statistics.fmean(
                [c.completion_tokens for c in group]), 1),
            "usd_mean": round(statistics.fmean([c.usd for c in group]), 6),
            "usd_total": round(sum(c.usd for c in group), 4),
        })
    return rows


#: RQ2 — one thing per step, so the effect can be attributed. `fault-record →
#: km` changes two (a count becomes a rate, and the rate is divided by cost),
#: so it cannot attribute; this chain can. `fault-record -> kill-only` is the
#: one step that matters in the memory itself: a COUNT of kills becomes a RATE,
#: which is the first quantity that reads `n` — the evidence of NOT killing,
#: which the whole FRTP family discards.
DECOMPOSITION = (
    ("natural", "cost-only"),
    ("natural", "kill-only"),
    ("natural", "fault-record"),
    ("fault-record", "kill-only"),
    ("kill-only", "km"),
    ("cost-only", "km"),
)


def decomposition(cs: Sequence[Cell]) -> list[dict[str, Any]]:
    """The stepwise CPU decomposition, **paired per cell** (CRN).

    Paired, not pooled: under CRN both arms of a step saw the identical patch
    sequence on the identical cell, so the per-cell difference is the estimator
    with the variance removed. Pooling the two means throws that away.
    """
    by_key: dict[tuple[str, int, str], Cell] = {(c.task, c.seed, c.policy): c for c in cs}
    rows = []
    for a, b in DECOMPOSITION:
        deltas = []
        for (task, seed, policy), cell in by_key.items():
            if policy != a:
                continue
            other = by_key.get((task, seed, b))
            if other is None:
                continue
            if cell.cpu_s > 0:
                deltas.append((other.cpu_s - cell.cpu_s) / cell.cpu_s)
        rows.append({
            "step": f"{a} → {b}",
            "paired_cells": len(deltas),
            "cpu_delta_mean_pct": round(100 * statistics.fmean(deltas), 3) if deltas else None,
            "cpu_delta_median_pct": round(100 * statistics.median(deltas), 3) if deltas else None,
        })
    return rows


# ── the cluster bootstrap ────────────────────────────────────────────────────


def bootstrap_cpu(
    cs: Sequence[Cell],
    policy: str,
    *,
    iterations: int = 2000,
    seed: int = 7,
    replay: Callable[[random.Random, Sequence[str]], float] | None = None,
) -> dict[str, Any]:
    """Cluster bootstrap over TASKS of the mean CPU for one arm.

    `replay` is the missing half: a callable `(rng, resampled_tasks) ->
    mean_cpu` that resamples the **cost table** from the remeasured draws and
    re-runs the offline replay over the resampled cluster, returning that
    iteration's mean CPU. It is handed the cluster draw so the interval covers
    BOTH sources, not one in place of the other. Supply it and the interval covers the
    dominant source of uncertainty. Omit it and the result says so in
    `cost_table_resampled: False` — because with a frozen cost table every arm's
    CPU is a deterministic function of that table, so a bootstrap over cells
    alone reports an interval that omits it, and that omission flatters the
    result.
    """
    group = [c for c in cs if c.policy == policy]
    if not group:
        return {"policy": policy, "n": 0}
    # The CLUSTER is the coding task, not the fault: faults of one AtCoder
    # problem share a test pool, so they move in and out of a draw together.
    # With G4 off several faults can land in one cluster, which is exactly what
    # makes dropping them unnecessary.
    by_task: dict[str, list[Cell]] = {}
    for c in group:
        by_task.setdefault(cluster_of(c.task), []).append(c)
    tasks = sorted(by_task)
    rng = random.Random(seed)

    means: list[float] = []
    for _ in range(iterations):
        # The cluster resample runs in BOTH modes. Skipping it when `replay` is
        # supplied would report an interval covering cost-table variation only,
        # under a note claiming it covers both — the same flattering omission in
        # the other direction.
        drawn = [rng.choice(tasks) for _ in tasks]
        if replay is not None:
            means.append(replay(rng, drawn))
            continue
        vals = [c.cpu_s for t in drawn for c in by_task[t]]
        if vals:
            means.append(statistics.fmean(vals))
    means.sort()
    if not means:
        return {"policy": policy, "n": len(group)}
    lo = means[max(0, int(0.025 * len(means)) - 1)]
    hi = means[min(len(means) - 1, int(0.975 * len(means)))]
    return {
        "policy": policy,
        "n": len(group),
        "clusters": len(tasks),
        "mean": round(statistics.fmean([c.cpu_s for c in group]), 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "iterations": iterations,
        "cost_table_resampled": replay is not None,
        "note": (
            "CI covers scan-order variation only; pass `replay` to resample the "
            "cost table from --remeasure-cost draws."
            if replay is None else
            "CI covers cost-table and scan-order variation."
        ),
    }


def km_survival(cs: Sequence[Cell], policy: str, budget: int) -> list[float]:
    """Kaplan–Meier-style `success@b` curve, censored at B. RQ4's primary metric.

    On a repair-rate-capped corpus this is censored almost everywhere: a
    completed pilot of 110 tasks × 550 cells × 33,000 rounds produced **10
    accepted patches (0.03%)** because 97 of its 110 tasks sat at repair rate 0.
    That is why no steered arm may run on a capped corpus.
    """
    group = [c for c in cs if c.policy == policy]
    if not group:
        return []
    out: list[float] = []
    for b in range(1, budget + 1):
        hit = sum(1 for c in group
                  if c.first_accept_round is not None and c.first_accept_round <= b)
        out.append(hit / len(group))
    return out


# ── RQ4: what the prompt showed against what actually refuted ────────────────

#: A verdict that is not `pass` refuted the patch. `timeout` counts: the patch
#: did not produce the required output within the protocol's clock, which is a
#: refutation under this oracle and is what the loop acted on.
REFUTING = ("wrong", "crash", "timeout")


def exploitability(
    rows: Sequence[RoundRecord],
    matrices: dict[str, dict[str, dict[str, str]]],
) -> list[dict[str, Any]]:
    """RQ4's primary metric. **No holdout required.**

    `facts_shown` records, per round, the probes the prompt disclosed — captured
    before the scan runs, so it is what the model actually saw. `caster audit`
    records the full kill vector per (patch, probe). Put them side by side and
    the disclosure axis becomes measurable without holding anything back,
    because the question is not "does this patch generalise" but "is what
    refuted it inside what we told it".

    Two shares, and they are NOT complements — a patch refuted both by a shown
    probe and an unshown one is in neither:

    `fits_disclosed`
        the refuting set is DISJOINT from what was shown. The patch satisfied
        everything it was told and died somewhere else. This is the
        exploitation signature, and the one RQ4 rises or falls on.

    `only_disclosed`
        the refuting set is a SUBSET of what was shown. The patch failed on
        exactly the thing it was handed, so the undisclosed remainder added no
        signal for it. Reported beside the first because it bounds how much the
        measurement could have seen: if it is large, the pool outside the
        disclosure is not discriminating and `fits_disclosed` is measured
        against a thin margin.

    Patches that nothing refutes are excluded from both denominators: a
    plausible patch is evidence about repair, not about disclosure. Rounds
    with no disclosure (every unsteered arm) are skipped rather than counted as
    zero, which would dilute every steered arm it is tabulated against.

    The holdout answers a DIFFERENT question — whether an *accepted* patch
    generalises past the pool that accepted it — and nothing here substitutes
    for it (DECISIONS 23).
    """
    agg: dict[str, dict[str, int]] = {}
    for r in rows:
        if not r.facts_shown or not r.patch_sha:
            continue
        verdicts = matrices.get(r.task, {}).get(r.patch_sha)
        if not verdicts:
            continue                      # the audit never measured this patch
        refuters = {probe for probe, v in verdicts.items() if v in REFUTING}
        if not refuters:
            continue                      # plausible: says nothing about disclosure
        shown = set(r.facts_shown)
        bucket = agg.setdefault(r.policy, {"refuted": 0, "fits": 0, "only": 0,
                                           "k_sum": 0})
        bucket["refuted"] += 1
        bucket["k_sum"] += len(shown)
        if not (refuters & shown):
            bucket["fits"] += 1
        if refuters <= shown:
            bucket["only"] += 1

    out = []
    for policy, b in sorted(agg.items()):
        n = b["refuted"]
        out.append({
            "policy": policy,
            "refuted_patches": n,
            "k_shown_mean": round(b["k_sum"] / n, 3),
            "fits_disclosed": b["fits"],
            "fits_disclosed_pct": round(100 * b["fits"] / n, 2),
            "only_disclosed": b["only"],
            "only_disclosed_pct": round(100 * b["only"] / n, 2),
        })
    return out


# ── offline replay: a new policy costs zero model calls ──────────────────────


def replay_policy(
    kill_matrix: dict[str, dict[str, str]],
    cost_table: dict[str, dict[str, float]],
    patch_sequence: Sequence[str],
    *,
    policy: str = "km",
    cell_key: str = "replay",
    prior: str = "shipped",
    cold_start: str = "max",
    explore_p: float = 0.15,
    tie_break: str = "rank",
    pool: Sequence[str] | None = None,
    coverage: dict[str, frozenset[int]] | None = None,
) -> dict[str, Any]:
    """Score an ordering against an already-measured full-scan matrix.

    This is the whole point of `caster audit`: the full kill vector per (patch,
    probe) lets a counterfactual policy be evaluated with **zero model calls and
    zero sandbox runs**. The 7-task × 40-patch matrix every scheduler comparison
    in this project rests on was built exactly this way.

    It drives the **real** policy and the **real** ledger, so `km`,
    `fault-record`, `cost-only`, `kill-only` and `km-bucket` all replay — not
    just a stateless `order_fn`. Without that, the `Beta(½,½)` vs `Beta(½,2)`
    question could not be settled at all, and it is the one open measurement
    that costs nothing: replay both priors over this matrix and report mean rank
    of the probe that actually fired, and probes run per refuted round.
    """
    probes = tuple(pool) if pool is not None else tuple(sorted(
        {p for v in kill_matrix.values() for p in v}))
    # Mirrors `episode.run_episode` exactly, or a replay scores an order the
    # grid never walks. "rank" is the default: at round 1 every score ties, so
    # the tie-break is the whole order and pool position is the only answer
    # that is not a claim. For "hash", `cell_key` is the TIE SEED — (task,
    # seed) — never a CellSpec key, or the policy and the prior would enter the
    # order and `compare_priors` would compare two different schedules.
    if tie_break == "hash":
        tie = lambda n: schedule_tie_break(cell_key, n)   # noqa: E731
    elif tie_break == "name":
        tie = lambda n: n                                 # noqa: E731
    else:
        tie = schedule_tie_rank(probes)
    ledger = CaseLedger(
        probes, prior=prior, cold_start=cold_start, explore_p=explore_p,
        tie_key=tie,
        rng=random.Random(hashlib.sha256(f"replay|{cell_key}".encode()).hexdigest()),
    )
    arm = policies.arm(policy)
    cpu = 0.0
    calls = 0
    accepted = 0
    ranks: list[int] = []

    for round_index, sha in enumerate(patch_sequence, 1):
        verdicts = kill_matrix.get(sha, {})
        costs = cost_table.get(sha, {})
        ctx = policies.PolicyContext(ledger=ledger, pool=probes,
                                     round_index=round_index, cell_key=cell_key,
                                     coverage=dict(coverage or {}))
        killed = False
        for rank, probe in enumerate(policies.schedule(policy, ctx), 1):
            status = verdicts.get(probe)
            if status is None:
                continue            # the audit never measured this pair
            calls += 1
            cost = float(costs.get(probe, 0.0))
            cpu += cost
            if arm.update_ledger:
                ledger.update(probe, refuted=status != "pass", cpu_s=cost,
                              round_index=round_index)
            if status != "pass":
                if not killed:
                    ranks.append(rank)
                killed = True
                # Arm 0 is "every usable probe, no early stop" — and its whole
                # job is to be the measured upper bound Observation 1 is argued
                # against. Early-stopping it in a replay would report roughly
                # 1/N of its true cost, which is the number the argument turns on.
                if not arm.full_scan:
                    break
        if not killed:
            accepted += 1
    return {
        "policy": policy, "prior": prior, "cold_start": cold_start,
        "patches": len(patch_sequence), "cpu_s": round(cpu, 6), "test_calls": calls,
        "accepted": accepted,
        "mean_rank_of_firing_probe": round(statistics.fmean(ranks), 4) if ranks else None,
        "rank1_share": round(sum(1 for r in ranks if r == 1) / len(ranks), 4) if ranks else None,
        "probes_per_refuted_round": round(calls / len(ranks), 4) if ranks else None,
    }


def compare_priors(
    kill_matrix: dict[str, dict[str, str]],
    cost_table: dict[str, dict[str, float]],
    sequences: dict[str, Sequence[str]],
    *,
    policy: str = "km",
) -> list[dict[str, Any]]:
    """The open question, answered. No model calls, no sandbox runs.

    "Replay both priors over the existing full-scan kill matrix and report mean
    rank of the probe that actually fired, and mean probes run per refuted
    round, for `Beta(½,½)` vs `Beta(½,2)`." Until that number exists, neither
    prior can be described as the right one.
    """
    rows: list[dict[str, Any]] = []
    for prior in sorted(PRIORS):
        per_task = [
            replay_policy(kill_matrix, cost_table, seq, policy=policy,
                          prior=prior, cell_key=task)
            for task, seq in sorted(sequences.items())
        ]
        ranks = [r["mean_rank_of_firing_probe"] for r in per_task
                 if r["mean_rank_of_firing_probe"] is not None]
        per_round = [r["probes_per_refuted_round"] for r in per_task
                     if r["probes_per_refuted_round"] is not None]
        rows.append({
            "prior": prior, "beta": PRIORS[prior], "tasks": len(per_task),
            "mean_rank_of_firing_probe": round(statistics.fmean(ranks), 4) if ranks else None,
            "probes_per_refuted_round": (round(statistics.fmean(per_round), 4)
                                         if per_round else None),
            "cpu_s": round(sum(r["cpu_s"] for r in per_task), 4),
        })
    return rows


def floor_arm_is_complete(cs: Sequence[Cell], universe: Sequence[str],
                          *, floor_arm: str = "no-mem") -> bool:
    """Has the floor arm covered every task, for every seed it ran at all?

    `GAPS` is a non-fatal audit, so `caster merge` succeeds on a partial grid —
    and the very next `caster analyse` would otherwise freeze a repair rate
    computed from one shard. That rate is the denominator of every band in
    RQ3a/RQ3b and, once frozen, it never corrects itself.
    """
    if not universe:
        return False
    floor = [c for c in cs if c.policy == floor_arm]
    if not floor:
        return False
    seeds = {c.seed for c in floor}
    have = {(c.task, c.seed) for c in floor}
    return all((task, seed) in have for task in universe for seed in seeds)


def summarise(rows: Sequence[RoundRecord], *, budget: int,
              pool_dir: pathlib.Path | None = None,
              universe: Sequence[str] = (),
              control: Sequence[str] = (),
              force_freeze: bool = False) -> dict[str, Any]:
    """Everything `caster analyse` writes. Order matters: threats first.

    The repair rate is read from the frozen artifacts when they have it, and
    frozen only when the floor arm is **complete** — every task of the universe,
    at every seed it ran. It is the denominator every band analysis uses,
    it never corrects itself once written, and an incomplete merge is not a
    reason to pick one. `--freeze-rates` forces it when an operator means to.

    `control` is the unfiltered control arm's faults, and **every number below
    excludes them**. The band tables already did, by testing membership of the
    frozen pool — but `arms`, `decomposition`, `survival`, `repair_rates` and
    `floor_arm_complete` did not, so a `caster evaluate --control` pass moved
    the headline arm table. The whole point of that arm is that it is reported
    "separately, whatever it shows": it gets its own key, and the main grid
    stops being a function of whether it ran.
    """
    declared = set(control)
    everything = cells(rows, budget=budget)
    cs = [c for c in everything if c.task not in declared]
    control_cells = [c for c in everything if c.task in declared]
    pool_sizes = load_pool_sizes(pool_dir) if pool_dir else {}
    task_dates = load_task_dates(pool_dir) if pool_dir else {}
    frozen_rates = load_repair_rates(pool_dir) if pool_dir else {}
    computed = repair_rates(cs)
    rates = {**computed, **frozen_rates}       # the frozen value wins
    complete = floor_arm_is_complete(cs, universe)
    source = "frozen" if frozen_rates else "computed-not-frozen"
    if pool_dir is not None and (complete or force_freeze):
        new = {k: v for k, v in computed.items() if k not in frozen_rates}
        if new:
            freeze_repair_rates(new, pool_dir)
            source = "computed-and-frozen"
    return {
        "budget": budget,
        "cells": len(cs),
        # Reported BEFORE any scheduler number — see the module docstring.
        "proposal_errors": proposal_error_table(cs),
        "k_facts": k_facts_table(cs),
        "repair_rates": rates,
        "bands": {k: band_of(v) for k, v in rates.items()},
        "repair_rates_source": source,
        "floor_arm_complete": complete,
        # The pool-size screen G3 used to apply, moved after the run.
        "pool_bands": {t: pool_band_of(n) for t, n in sorted(pool_sizes.items())},
        "arms_by_pool_band": arms_by_pool_band(cs, pool_sizes),
        # The contamination cut G5 used to apply, moved after the run.
        "arms_by_era": arms_by_era(cs, task_dates),
        "date_cutoff": DATE_CUTOFF,
        "arms": arm_table(cs),
        "decomposition": decomposition(cs),
        "survival": {
            p: km_survival(cs, p, budget)
            for p in sorted({c.policy for c in cs})
        },
        # Reported APART, and pre-registered as "whatever it shows". These
        # faults are the ones a pool-size threshold would have cut, so
        # pooling them into `arms` above would put back exactly the
        # correlation this arm exists to measure.
        "control_arm": {
            "tasks": sorted({c.task for c in control_cells}),
            "cells": len(control_cells),
            "arms": arm_table(control_cells),
        } if control_cells else {},
    }


def write(doc: dict[str, Any], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=2, default=_jsonable), encoding="utf-8")
    tmp.replace(path)


def _jsonable(o: Any) -> Any:
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if dataclasses.is_dataclass(o):
        return dataclasses.asdict(o)
    return str(o)
