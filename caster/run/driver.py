"""The driver: walk `(task × seed × policy)`, resumably, writing atomic cells.

**`--resume` is the default, not a flag**. Re-issuing the identical
command after a kill is the documented recovery path and must be safe to do
blindly, so `--no-resume` is what you pass to force a re-walk.

**Resume is per `(task, seed, policy)`, and policies are independent.** Three
consequences this honours:

* adding arm *n+1* later must not re-run arms 1..*n* — the done-index is keyed
  by the full cell key, so a new policy simply has no entries;
* a run may be launched with a **subset** of policies and merged with an earlier
  run that had the others, provided the protocol matches (`shard.merge_audit`);
* a cell finished on another machine is never re-walked, because the done-index
  is merged across shards before the plan is cut.

**Resume reads the done-index, never the row file.** Resuming on `(task, seed)`,
or on the rows themselves, has a boundary defect: a process killed mid-cell
leaves partially flushed rows, and resume then skips the rest of that cell. So a
cell is written atomically and indexed separately, and the index is the only
thing resume consults.

**The universe order makes any contiguous range a smaller grid, not a skewed
one**. It is task-major over the frozen pool's own order — which `freeze`
already seeded-interleaved for exactly this reason — so a range is "these tasks,
every seed, every policy".

**The universe size is computed in exactly one place**: `plan()`. A fleet asks
the driver for a `--dry-run` plan and reads the count out of it. Two scripts
holding two opinions about how big a stage is cut ranges past the end of it.
"""
from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import platform
import sys
import time
from typing import Any, Iterator, Sequence

from caster import paths
from caster.exec.sandbox import DEFAULT_TIMEOUT, protocol_fields
from caster.exec.store import PatchStore, VerdictStore
from caster.keys import KEY_VERSION, CellSpec
from caster.errors import CacheMiss
from caster.llm.client import BackendUnreachable, LLMClient
from caster.report.records import CellDone, CellWriter, load_done, needs_walk
from caster.run import policies
from caster.run.episode import EpisodeConfig, EpisodeResult, run_episode


@dataclasses.dataclass(frozen=True)
class Grid:
    """What to walk. The knobs here are the ones in `CellSpec`, applied to every cell."""

    tasks: tuple[str, ...]
    seeds: tuple[int, ...]
    policies: tuple[str, ...]
    budget: int = 20
    model: str = ""
    temperature: float = 1.0
    reasoning_effort: str = ""
    prior: str = "shipped"
    cold_start: str = "max"
    explore_p: float = 0.15
    overlap_thr: float = 0.75
    overlap_min_rej: int = 1
    serve_stable_timeouts: bool = True
    tie_break: str = "hash"
    #: NOT a cell-key field (Invariant 13 names it as one of the values that
    #: "reach no key at all"), but it changes what a verdict IS — so it must
    #: reach `protocol_sha` and the shard's `.meta.json`, which is what the
    #: runtime-disagreement audit reads. A run at --timeout 5 writing into the
    #: --timeout 10 namespace is two instruments sharing one store.
    sandbox_timeout: float = DEFAULT_TIMEOUT

    #: Grid-wide knobs that an arm's own overrides outrank. `km-explore0` IS its
    #: `explore_p`, and `km+steer-k2` IS its `k_facts`; a grid-wide value must
    #: not silently retune them. Listed once here so adding a knob to `Grid` and
    #: forgetting to forward it is not possible.
    FORWARDED = ("prior", "cold_start", "explore_p", "overlap_thr",
                 "overlap_min_rej", "serve_stable_timeouts", "tie_break")

    def protocol(self) -> dict[str, Any]:
        return protocol_fields(timeout=self.sandbox_timeout)

    def cells(self) -> list[CellSpec]:
        """The universe, in the one order that makes a contiguous range a grid."""
        out: list[CellSpec] = []
        for task in self.tasks:                 # pool order (seeded-interleaved)
            for seed in self.seeds:
                for arm_id in self.policies:
                    arm = policies.arm(arm_id)
                    extra: dict[str, Any] = {
                        "model": self.model,
                        "temperature": self.temperature,
                        "reasoning_effort": self.reasoning_effort,
                    }
                    # An arm's own overrides win over the grid-wide knob: that
                    # is what `km-explore0` and `km-cs-median` ARE.
                    for field in self.FORWARDED:
                        if field not in arm.overrides:
                            extra[field] = getattr(self, field)
                    out.append(arm.cell(task, seed, **extra))
        return out


@dataclasses.dataclass
class Plan:
    """What a `--dry-run` prints. The single authority on universe size."""

    universe: int
    selected: int
    already_done: int
    to_walk: int
    cells: list[CellSpec]
    by_policy: dict[str, int]
    est_model_calls: int
    est_scans: int

    def render(self) -> str:
        lines = [
            f"universe   {self.universe} cells "
            f"(tasks × seeds × policies, task-major)",
            f"selected   {self.selected}  (this shard's range)",
            f"done       {self.already_done}  (from cells.done, merged)",
            f"to walk    {self.to_walk}",
            f"est calls  {self.est_model_calls} model calls, {self.est_scans} scans",
            "",
        ]
        for pol, n in sorted(self.by_policy.items()):
            arm = policies.arm(pol)
            over = ", ".join(f"{k}={v!r}" for k, v in sorted(arm.overrides.items())
                             if k in Grid.FORWARDED)
            tail = f"   [arm overrides {over}]" if over else ""
            lines.append(f"  {pol:18s} block {arm.block}  {n:6d} cells{tail}")
        return "\n".join(lines)


def plan(
    grid: Grid,
    *,
    first: int | None = None,
    last: int | None = None,
    done_paths: Sequence[pathlib.Path] | None = None,
    resume: bool = True,
) -> Plan:
    """The plan. Touches nothing, needs no credentials and no GPU.

    `first`/`last` are **1-based inclusive** indices into the universe, which is
    what the fleet's range arithmetic cuts.
    """
    universe = grid.cells()
    lo = 1 if first is None else max(1, first)
    hi = len(universe) if last is None else min(len(universe), last)
    selected = universe[lo - 1:hi] if universe else []

    done: dict[str, CellDone] = {}
    if resume:
        for p in (done_paths or [paths.CELLS_DONE]):
            done.update(load_done(pathlib.Path(p)))

    to_walk = [c for c in selected if needs_walk(done, c.key(), grid.budget)]
    by_policy: dict[str, int] = {}
    for c in to_walk:
        by_policy[c.policy] = by_policy.get(c.policy, 0) + 1

    # A Block A cell shares its whole prompt family with every other Block A
    # cell on the same (task, seed) — the prompt does not depend on the ledger —
    # so the whole block costs `budget` calls PER (task, seed), not per cell and
    # not once for the grid. A Block B cell shares only round 1 with them, so it
    # costs `budget − 1` of its own.
    block_b = sum(n for p, n in by_policy.items() if policies.arm(p).block == "B")
    families = len({(c.task, c.seed) for c in to_walk
                    if policies.arm(c.policy).block == "A"})
    est_calls = block_b * max(0, grid.budget - 1) + families * grid.budget
    return Plan(
        universe=len(universe), selected=len(selected), already_done=len(selected) - len(to_walk),
        to_walk=len(to_walk), cells=to_walk, by_policy=by_policy,
        est_model_calls=est_calls,
        # A VS round scans k candidates, so arm 18 costs k× the scans of a
        # k=1 arm at the same budget. Counting cells × budget would understate
        # the one arm whose scan cost is not budget-shaped.
        est_scans=sum(c.k_candidates * grid.budget for c in to_walk),
    )


#: How the rows were COLLECTED. Not a key and not a runtime field — it does
#: not change what a cell IS, and two shards legitimately differ on it (phase
#: 1 is live, phase 2 is a replay). It still has to be recorded, because a
#: cell walked against a cold verdict store and a live sandbox is judged by a
#: different oracle state than one replayed after `caster audit`, and when
#: both land in `shards/` the only symptom is Invariant 2 failing on whichever
#: arm was collected first. The audit over it is a WARNING that names the
#: cause.
COLLECTION_FIELDS = ("from_cache_only", "replay_only", "force_full_budget")


def collection_record(cfg: Any) -> dict[str, Any]:
    return {f: bool(getattr(cfg, f, False)) for f in COLLECTION_FIELDS}


def protocol_record(grid: Grid, client: LLMClient, *, shard: str = "0",
                    extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Invariant 13 — every value that changes a measured quantity is in a key
    or in `.meta.json`.

    The cache key covers model, temperature, prompt, `max_tokens` and the nonce.
    The dangerous half is everything else: the **served context window**, the
    sandbox timeout, the interpreter version, the backend and the model digest
    reach no key at all. Each of them moves the numbers for free and leaves no
    trace in the cache: a weaker oracle re-judges the same completions, a
    different sandbox timeout turns a slow correct patch into a wrong one, and
    the served window decides whether the prompt arrived whole.

    So every shard writes one of these, and `shard.merge_audit` refuses to join
    across a disagreement.
    """
    doc = {
        "shard": shard,
        "at": time.time(),
        "protocol": {
            "model": grid.model or client.model,
            "temperature": grid.temperature,
            "reasoning_effort": grid.reasoning_effort or None,
            "budget": grid.budget,
            "policies": list(grid.policies),
            # The seeds this launch ASKED for. The coverage audit used to read
            # the arms and seeds off the rows it was auditing, so an arm or a
            # seed that produced nothing was invisible to the one check whose
            # job is to notice exactly that.
            "seeds": list(grid.seeds),
            # The cell key's own version. It is written here as well as into
            # every key because the ROW does not carry it: `rounds.jsonl` holds
            # the CellSpec fields, which are identical across versions, so v1
            # and v2 rows differ only in `episode_id` and would otherwise pool
            # silently. `shard.merge_audit` refuses to join across a
            # disagreement, which is what makes that a stop rather than a mean.
            "key_version": KEY_VERSION,
        },
        "runtime": {
            **grid.protocol(),
            "context_tokens": client.context_tokens,
            "backend": client.base_url or "default",
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "host": platform.node(),
        },
    }
    if extra:
        doc.update(extra)
    return doc


def write_meta(doc: dict[str, Any], path: pathlib.Path) -> None:
    """The current record, and the HISTORY beside it.

    `meta.json` is the latest launch into this shard directory. It is also
    **overwritten** by each one, and `rounds.jsonl` is not — so a second launch
    erased the only record of the protocol the first launch's rows were
    collected under, and the two audits that exist to catch instrument drift
    had nothing left to compare. `meta.jsonl` is append-only and
    `shard.meta_records` reads both.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    history = path.with_suffix(".jsonl")
    with history.open("a", encoding="utf-8") as f:
        f.write(json.dumps(doc, sort_keys=True, separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())


def walk(
    grid: Grid,
    cells: Sequence[CellSpec],
    *,
    adapter: Any,
    client: LLMClient,
    rows_path: pathlib.Path,
    done_path: pathlib.Path,
    cfg: EpisodeConfig | None = None,
    run_id: str = "local",
    progress: Any = None,
    on_cache_miss: str = "skip",
    misses: list[dict[str, Any]] | None = None,
) -> Iterator[EpisodeResult]:
    """Run each cell and commit it **atomically**. Yields as it goes.

    The commit order is rows → fsync → done-index. A crash between them
    re-walks the cell, which is free and correct; the other order would make
    resume skip a cell that was never written.

    `on_cache_miss` decides what a replay-only run does when phase 1 never
    produced something it needs. `"stop"` is the old behaviour and it is the
    wrong default for a FLEET: a `CacheMiss` left `run_episode` uncaught, the
    driver re-raised it, `cli.main` turned it into exit 1 — and the shard
    process died with its whole remaining RANGE unwalked. One cell whose
    phase-1 draw hit `context_overflow`, or whose probe is stored as a
    quarantined `timeout` (`VerdictStore.get` never serves one), therefore cost
    every later task in that range across EVERY arm. That is the pilot's "49
    task(s) missing" in all twelve arms, and the loudness bought nothing: the
    operator still had to read the merge audit to find out.

    `"skip"` keeps the loudness and drops the collateral: the cell is
    **abandoned whole** — no rows, no done-index entry, so it is re-walked the
    moment the gap is filled — the miss is recorded, the walk continues, and
    the caller exits non-zero with the list. The grid then shows exactly the
    cells phase 1 owes, instead of a cliff at the first one.
    """
    store = VerdictStore(grid.protocol(),
                         serve_stable_timeouts=grid.serve_stable_timeouts)
    patches = PatchStore()
    base = cfg or EpisodeConfig(budget=grid.budget)
    # the refcount needs a producer or it is decoration. A shard registers
    # for the whole walk, not per call: it owns the pool while it is working,
    # including between calls.
    register = getattr(client, "register_session", None)
    unregister = getattr(client, "unregister_session", None)
    if register is not None:
        register()
    try:
        yield from _walk_cells(grid, cells, adapter=adapter, client=client,
                               rows_path=rows_path, done_path=done_path,
                               store=store, patches=patches, base=base,
                               run_id=run_id, progress=progress,
                               on_cache_miss=on_cache_miss, misses=misses)
    finally:
        if unregister is not None:
            unregister()


def _walk_cells(
    grid: Grid,
    cells: Sequence[CellSpec],
    *,
    adapter: Any,
    client: LLMClient,
    rows_path: pathlib.Path,
    done_path: pathlib.Path,
    store: Any,
    patches: Any,
    base: EpisodeConfig,
    run_id: str,
    progress: Any,
    on_cache_miss: str = "skip",
    misses: list[dict[str, Any]] | None = None,
) -> Iterator[EpisodeResult]:
    for i, spec in enumerate(cells, 1):
        writer = CellWriter(rows_path, done_path)
        if progress is not None:
            progress(i, len(cells), spec)
        try:
            result = run_episode(
                spec, adapter=adapter, client=client, store=store, patches=patches,
                cfg=dataclasses.replace(base, budget=grid.budget), run_id=run_id,
            )
        except CacheMiss as exc:
            # The cell is abandoned WHOLE: no rows and no done-index entry, so
            # it re-walks as soon as phase 1 (or `caster audit`) supplies what
            # it needs. Nothing partial is written and nothing is marked done,
            # which is what keeps "skip" from being a silent degradation.
            writer.abandon()
            if on_cache_miss == "stop":
                raise
            record = {"cell_key": spec.key(), "task": spec.task,
                      "seed": spec.seed, "policy": spec.policy,
                      "reason": str(exc)}
            if misses is not None:
                misses.append(record)
            _log_miss(rows_path.parent / "misses.jsonl", record)
            continue
        except BackendUnreachable:
            # the circuit breaker reached the driver. Stop cleanly: every
            # finished cell is already in the done-index and resumable, and this
            # cell is abandoned whole rather than half-written.
            writer.abandon()
            raise
        except Exception:
            writer.abandon()
            raise
        for record in result.records:
            writer.add(record)
        writer.commit(CellDone(
            cell_key=result.cell_key, episode_id=result.episode_id,
            task=spec.task, seed=spec.seed, policy=spec.policy,
            rounds_written=len(result.records), rounds=result.rounds,
            budget=grid.budget,
            accepted=result.accepted, truncated=result.truncated,
        ))
        yield result


def _log_miss(path: pathlib.Path, record: dict[str, Any]) -> None:
    """A skipped cell leaves a row of its own, beside the shard's output.

    The console scrolls and a fleet log is read once; the file is what the
    operator diffs against the next run to see whether phase 1 closed the gap.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")
