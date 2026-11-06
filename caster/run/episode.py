"""The repair loop. Policy is one injected function; the two phases are split.

```
for each round r (charged against the budget, capped at B):
    prompt = build(slots, ledger-at-r)        <- steering acts here
    cands  = draw(prompt, k_candidates)       <- one model call
    for c in order(cands):
        order  = policy(ledger, pool, r)      <- the ONLY thing an arm changes
        result = scan(c, order)               <- triage and oracle, merged
        update ledger, facts, observations
        if result.accept: stop this round
```

## What the budget counts, and why it is rounds

Charging one unit per model call *and* charging it when a guard blocks the
candidate before the oracle runs makes the guard outcome-neutral **by
construction**: blocking a doomed candidate and then burning the round on the
block changes nothing, and the memory arm reproduces the floor to the last
decimal at every budget. That is a measurement that cannot fail.

So **the budget counts rounds, i.e. model calls**, and the reason is RQ7. A
verbalized-sampling round emits *n* candidates from **one** draw; charging each
candidate would let a `k=5` arm spend B in B/5 draws, so "rounds to repair"
would compare arms at different numbers of model calls — exactly the quantity
RQ7 is about. The extra CPU of scanning five candidates is not hidden: it lands
in `cpu_s`, which is the cost unit anyway. Rounds are one reporting axis, not
the unit.

A harness-failure round **is charged**. Leaving it free lets a task whose
proposer keeps truncating draw until a cap, and then `success@B` is a curve of
an unknown B. `charged` is on every row so an analysis can see it either way.

## The two phases, and where the split actually holds

`--from-cache-only` makes the loop fail loudly on a cache miss rather than
silently calling the model. That is exactly separable for **Block A**, whose
prompt does not depend on the ledger. For **Block B** the round-*t* prompt is a
function of the ledger at round *t*, which phase 2 produces: the phases are not
separable at all there, and `--from-cache-only` is meaningful only on a *re-run*
of an arm already collected.

`force_full_budget` keeps drawing after the first accept, because phase 1 has no
oracle and cannot know where to stop — and because the floor arm is an estimator
of π, not just a baseline, so early stopping would bias π̂. The rounds past the
first accept are recorded and truncated back out at analysis.

## Three proposal failures, three policies

`BackendUnreachable` — record, count consecutive, and at 5 raise so the driver
stops cleanly with every finished cell resumable. One failure is weather; a run
of them is climate.

`ContextOverflow` — record and continue. **This one is a deterministic poison
pill if it escapes**: the prompt is a pure function of (task, seed, round,
policy), so an uncaught overflow aborts at the identical round on every re-run,
and under atomic cells the cell is then never written and resume retries it
forever.

`TruncatedResponse` — record and continue, **without touching the ledger**. A
reply the output budget cut short must never become evidence.
"""
from __future__ import annotations

import dataclasses
import hashlib
import random
from typing import Any, Sequence

from caster.bench.protocol import ProbeVerdict
from caster.exec.ledger import CaseLedger
from caster.exec.store import PatchStore, VerdictStore
from caster.exec.verify import ScanResult, scan
from caster.keys import (CellSpec, draw_nonce, schedule_tie_break,
                         schedule_tie_rank, schedule_tie_seed)
from caster.llm.client import (
    BackendUnreachable,
    CallMeta,
    ContextOverflow,
    LLMClient,
    TruncatedResponse,
)
from caster.report.records import RoundRecord
from caster.run import policies, propose
from caster.steer import template

#: "one failure is weather; a run of them is climate". A NON-key flag:
#: raising it extends an episode that hit it, it never changes a draw already
#: made, so it goes on the record instead of into the cell key.
MAX_CONSECUTIVE_BACKEND_FAILURES = 5


@dataclasses.dataclass(frozen=True)
class EpisodeConfig:
    budget: int = 20
    #: the MODEL side. The driver must fail loudly on a miss rather than
    #: silently calling the model.
    from_cache_only: bool = False
    #: The SANDBOX side: after `caster audit` on a task, every policy must
    #: evaluate that task with `sandbox_runs == 0`. Separate from the model flag
    #: on purpose, because they fail for different reasons and at different
    #: times: a model miss means phase 1 never drew this round, a scan miss
    #: means the audit pass never measured this (patch, probe).
    replay_only: bool = False
    force_full_budget: bool = False
    max_consecutive_backend_failures: int = MAX_CONSECUTIVE_BACKEND_FAILURES
    full_scan: bool = False
    concurrency: int | None = None


@dataclasses.dataclass
class EpisodeResult:
    cell_key: str
    episode_id: str
    spec: CellSpec
    records: list[RoundRecord]
    accepted_patch_sha: str | None
    first_accept_round: int | None
    rounds: int
    candidates_seen: int
    cpu_s: float
    sandbox_runs: int
    verdict_hits: int
    reference_runs: int
    truncated: bool
    truncate_reason: str | None

    @property
    def accepted(self) -> bool:
        return self.accepted_patch_sha is not None


def _facts_for(
    adapter: Any,
    task: str,
    ledger: CaseLedger,
    k_facts: int,
    last: dict[str, ProbeVerdict],
) -> tuple[list[template.Fact], tuple[str, ...]]:
    """Turn the ledger's evidence into renderable facts. No execution here.

    `ledger.evidence` ranks by `K × legibility` and returns the posterior K
    (prompt rule 4), which is what reaches the prompt — the number there is a
    claim about the world and must stay true.

    Returns the facts AND the probe names behind them, in the same order. The
    names are what `facts_shown` is cut from: a probe whose case or reference
    is unavailable is dropped here, so the selection and the render can differ
    in length and only the render may be recorded as disclosed.
    """
    out: list[template.Fact] = []
    names: list[str] = []
    for probe, k_rate in ledger.evidence(k_facts):
        case = adapter.case(task, probe)
        ref = adapter.reference(task, probe)
        if case is None or ref is None:
            continue
        seen = last.get(probe)
        out.append(template.Fact(
            input_text=case.input_text,
            required=ref.value,
            kill_rate=k_rate,
            last_status=seen.status if seen else None,
            last_output=seen.output if seen else None,
            last_output_len=seen.output_len if seen else None,
            last_error_type=seen.error_type if seen else None,
            last_error_message=seen.error_message if seen else None,
        ))
        names.append(probe)
    return out, tuple(names)


def _examples_for(adapter: Any, task: str) -> list[template.Example]:
    return [
        template.Example(input_text=c.input_text, output_text=c.expected_output or "")
        for c in adapter.spec_cases(task)
    ]


def run_episode(
    spec: CellSpec,
    *,
    adapter: Any,
    client: LLMClient,
    store: VerdictStore,
    patches: PatchStore,
    cfg: EpisodeConfig = EpisodeConfig(),
    run_id: str = "local",
) -> EpisodeResult:
    """Run one `(task, seed, policy)` cell. Returns its rows; writes nothing.

    Writing is the driver's job, because a cell is written **atomically or not
    at all** — which an episode that appends as it goes cannot do.
    """
    arm = policies.arm(spec.policy)
    cell_key = spec.key()
    episode_id = spec.episode_id
    program = adapter.program(spec.task)
    pool = tuple(adapter.probes(spec.task))
    if not pool:
        raise ValueError(f"{spec.task}: empty probe pool — is the corpus frozen?")

    # At round 1 every score ties, so the tie-break IS the order. The default
    # ranks on pool position, which is `sorted` doing nothing: every arm but
    # `random` walks the natural order until the ledger has something to say.
    # "hash" is the ablation that prices the benchmark's own file order; it is
    # seeded on (task, seed), never on the cell, or the ARM'S NAME would decide
    # round 1. See `keys.schedule_tie_rank`.
    if spec.tie_break == "hash":
        tie_seed = schedule_tie_seed(spec.task, spec.seed)
        tie = lambda n: schedule_tie_break(tie_seed, n)   # noqa: E731
    elif spec.tie_break == "name":
        tie = lambda n: n                                 # noqa: E731
    else:
        tie = schedule_tie_rank(pool)
    ledger = CaseLedger(
        pool,
        prior=spec.prior,
        cold_start=spec.cold_start,
        explore_p=spec.explore_p,
        overlap_thr=spec.overlap_thr,
        overlap_min_rej=spec.overlap_min_rej,
        tie_key=tie,
        rng=random.Random(hashlib.sha256(f"schedule|{cell_key}".encode()).hexdigest()),
        input_chars=adapter.input_chars(spec.task),
    )
    coverage: dict[str, frozenset[int]] = {}
    if arm.ordering == "coverage":
        for probe in pool:
            sig = adapter.signature(spec.task, probe)
            if sig:
                coverage[probe] = sig

    last_obs: dict[str, ProbeVerdict] = {}
    surviving = 0
    transcript: list[str] = []
    history_restarts = 0
    consecutive_failures = 0        # backend outages — the circuit breaker
    consecutive_refutations = 0     # failed repair attempts — the restart

    records: list[RoundRecord] = []
    accepted_sha: str | None = None
    first_accept: int | None = None
    candidates_seen = 0
    cpu_total = 0.0
    runs_total = 0
    hits_total = 0
    truncated = False
    truncate_reason: str | None = None

    ref_runs_before = adapter.reference_runs

    def base_row(round_index: int, **kw: Any) -> RoundRecord:
        return RoundRecord(
            run_id=run_id, episode_id=episode_id, task=spec.task, seed=spec.seed,
            round_index=round_index, policy=spec.policy, cell=spec.as_dict(),
            n_cases=len(ledger.usable()), history_restarts=history_restarts, **kw,
        )

    for round_index in range(1, cfg.budget + 1):
        facts, fact_names = (
            _facts_for(adapter, spec.task, ledger, spec.k_facts, last_obs)
            if spec.steer_on else ([], ())
        )
        prompt = template.build(
            program_label=spec.task,
            buggy_source=program.buggy_source,
            examples=_examples_for(adapter, spec.task),
            facts=facts,
            surviving=surviving,
            round_index=round_index,
            steer_on=spec.steer_on,
            disclosure=spec.disclosure,
            show_kill_rate=spec.show_kill_rate,
            k_candidates=spec.k_candidates,
            vs_schedule=spec.vs_schedule,
            steer_mode=spec.steer_mode,
            transcript=transcript if spec.steer_mode == "chat" else (),
            transcript_cap_tokens=spec.transcript_cap_tokens,
        )

        # What the prompt ACTUALLY SHOWED, captured before the scan runs and
        # updates the ledger. The exploitability bound ("patches refuted only
        # by the disclosed probes") is computed from exactly this field, so:
        #  * reading it back off the post-update ledger would be off by one
        #    round;
        #  * emitting it for an unsteered arm would attribute a disclosure that
        #    never happened;
        #  * and reading the SELECTION rather than the RENDER would do the same
        #    for `disclosure="sample-only"`, where `s5_contract` withholds the
        #    whole block and k facts are selected but none are disclosed.
        # `k_facts_rendered` is the only field that knows what survived to the
        # page, so it is what cuts the list. Null when steering is off.
        shown = fact_names[:prompt.k_facts_rendered] if spec.steer_on else ()

        ref_runs_round = adapter.reference_runs
        meta = CallMeta()
        # Invariant 1, the half that is not about prompt bytes. `max_tokens` is
        # in the cache key, and the output budget is sized from the number of
        # programs the reply must carry. Rounds 1–2 carry no VS, so a VS
        # arm must request ONE program there, or its round-1 cache key differs
        # from every other arm's and the shared draw is silently lost — byte
        # identical prompt, different key, every completion bought twice.
        k_eff = spec.k_candidates if template.vs_active(round_index,
                                                        spec.k_candidates) else 1
        try:
            cands = propose.draw(
                client, prompt.text, program.buggy_source,
                k=k_eff,
                nonce=draw_nonce(spec.task, spec.seed, round_index),
                model=spec.model or None,
                temperature=spec.temperature,
                reasoning_effort=spec.reasoning_effort or None,
                meta=meta,
                from_cache_only=cfg.from_cache_only,
            )
            consecutive_failures = 0
        except (TruncatedResponse, ContextOverflow, BackendUnreachable) as exc:
            kind = (
                "backend_unreachable" if isinstance(exc, BackendUnreachable)
                else "context_overflow" if isinstance(exc, ContextOverflow)
                else "truncated_response"
            )
            records.append(base_row(
                round_index, candidate_index=0, patch_sha=None,
                cache_key=meta.cache_key or None, proposal_error=kind,
                finish_reason=meta.finish_reason, charged=True, accept=False,
                prompt_tokens=meta.prompt_tokens,
                completion_tokens=meta.completion_tokens,
                reasoning_tokens=meta.reasoning_tokens, llm_sec=meta.llm_sec,
                usd=meta.usd, backend=meta.backend,
                facts_shown=tuple(), k_facts=prompt.k_facts_rendered,
            ))
            # The ledger is untouched: a harness failure must never become
            # evidence inside the mechanism under study.
            if isinstance(exc, BackendUnreachable):
                consecutive_failures += 1
                if consecutive_failures >= cfg.max_consecutive_backend_failures:
                    truncated = True
                    truncate_reason = "backend_circuit_breaker"
                    break
            continue

        round_accepted = False
        ordered = propose.order_candidates(list(cands), spec.candidate_policy)
        # The budget counts ROUNDS (model calls), so exactly one row per round
        # carries the charge and the USD. See the module docstring on why a VS
        # round is one unit and not n.
        charged_index = ordered[0].index if ordered else None
        for cand in ordered:
            candidates_seen += 1
            patch_sha = patches.put(cand.source)
            ctx = policies.PolicyContext(
                ledger=ledger, pool=pool, round_index=round_index,
                cell_key=cell_key, coverage=coverage,
            )
            order = policies.schedule(spec.policy, ctx)
            result: ScanResult = scan(
                adapter, spec.task, cand.source, order,
                # `no-mem` never consults OR updates the ledger — that is the
                # whole point of keeping it beside `natural`.
                ledger=ledger if arm.update_ledger else None,
                store=store, round_index=round_index,
                full=arm.full_scan or cfg.full_scan,
                # "After `caster audit` on a task, every policy evaluates that
                # task with sandbox_runs == 0" is checkable only if the scan can
                # be told to refuse to execute. The timeout quarantine then has
                # teeth too: a `timeout` verdict is not servable, so a replay
                # that needs one fails instead of re-executing under a different
                # load than the audit measured.
                from_cache_only=cfg.replay_only,
                source="scan", concurrency=cfg.concurrency,
            )
            cpu_total += result.cpu_s
            runs_total += result.sandbox_runs
            hits_total += result.verdict_hits

            # keep the trace. `scan` stops at the first divergence, but
            # the probes that ran and PASSED are known, and "this input was
            # correct in your previous attempt" is a positive, zero-leak,
            # zero-extra-cost fact — and recording only the killer throws it
            # away for nothing.
            for probe, verdict in result.observations.items():
                last_obs[probe] = verdict
            # The regression half of the contract. `passing_block` is the
            # switch that makes it an arm rather than a silent default (it is
            # in the cell key), because nothing prices it yet.
            surviving = (sum(1 for v in result.verdicts.values() if v == "pass")
                         if spec.passing_block else 0)

            records.append(base_row(
                round_index, candidate_index=cand.index,
                patch_sha=patch_sha, cache_key=meta.cache_key or None,
                proposal_error=None, finish_reason=meta.finish_reason,
                charged=(cand.index == charged_index),
                probability=cand.probability, prob_stated=cand.stated,
                accept=result.accept, test_calls=result.test_calls,
                cpu_s=round(result.cpu_s, 6), killed_by=result.killed_by,
                scan_order=result.scan_order, oracle_error=result.oracle_error,
                sandbox_runs=result.sandbox_runs, verdict_hits=result.verdict_hits,
                timeout_reruns=result.timeout_reruns,
                facts_shown=shown,
                k_facts=prompt.k_facts_rendered,
                reference_runs=adapter.reference_runs - ref_runs_round,
                prompt_tokens=meta.prompt_tokens,
                completion_tokens=meta.completion_tokens,
                reasoning_tokens=meta.reasoning_tokens, llm_sec=meta.llm_sec,
                usd=(meta.usd if cand.index == charged_index else 0.0),
                backend=meta.backend,
            ))

            if spec.steer_mode == "chat":
                transcript.append(_transcript_entry(cand.source, result))
                # ChatRepair restarts after N consecutive FAILED REPAIR
                # ATTEMPTS, not after N backend outages. Keyed on the wrong
                # signal the restart policy is inert, and this is the one arm
                # that can then walk itself into the context ceiling — which
                # would make its `context_overflow` count a property of the
                # harness rather than of the method.
                consecutive_refutations = 0 if result.accept else consecutive_refutations + 1
                history_restarts, transcript = _maybe_restart(
                    history_restarts, transcript, consecutive_refutations,
                    spec.transcript_restart_after)
                if not transcript:
                    consecutive_refutations = 0

            if result.accept and accepted_sha is None:
                accepted_sha = patch_sha
                first_accept = round_index
                round_accepted = True
                break

        if round_accepted and not cfg.force_full_budget:
            break

    rounds = len({r.round_index for r in records})
    if not truncated and accepted_sha is None and rounds < cfg.budget:
        truncated = True
        truncate_reason = truncate_reason or "short_episode"

    return EpisodeResult(
        cell_key=cell_key, episode_id=episode_id, spec=spec, records=records,
        accepted_patch_sha=accepted_sha, first_accept_round=first_accept,
        rounds=rounds, candidates_seen=candidates_seen,
        cpu_s=round(cpu_total, 6), sandbox_runs=runs_total,
        verdict_hits=hits_total,
        reference_runs=adapter.reference_runs - ref_runs_before,
        truncated=truncated, truncate_reason=truncate_reason,
    )


def _transcript_entry(source: str, result: ScanResult) -> str:
    verdict = (
        "it passed every probe it was run on" if result.accept
        else f"it failed on one of the inputs above: {result.killed_by or 'inconclusive'}"
    )
    return f"```python\n{source.rstrip()}\n```\n{verdict}"


def _maybe_restart(restarts: int, transcript: list[str], consecutive: int,
                   restart_after: int) -> tuple[int, list[str]]:
    """ChatRepair's restart policy.

    Without it the chat arm is the one arm that can walk itself into the context
    ceiling, and its `context_overflow` count would then be a property of the
    harness rather than of the method.
    """
    if restart_after and consecutive >= restart_after:
        return restarts + 1, []
    return restarts, transcript


def audit_task(
    adapter: Any,
    task: str,
    patch_shas: Sequence[str],
    *,
    patches: PatchStore,
    store: VerdictStore,
    concurrency: int | None = None,
) -> dict[str, dict[str, str]]:
    """`caster audit` — the dedicated full-scan pass.

    Runs **every usable probe** against every patch, with no early stop and no
    ledger (an audit is not an episode; a ledger it updated would be a
    cross-episode prior, which the store forbids). It produces three things
    nothing else can:

    * the **full kill vector** per (patch, probe), which is the only way to
      compute an honest `K` and the only way to replay a counterfactual policy
      offline **without new model calls**;
    * the **cost table**, written once at a declared concurrency before any arm
      runs, so no arm's own run conditions enter it;
    * the input to the leakage bound — how many patches are refuted *only* by
      the disclosed probes.

    Cost: N runs instead of ~2. Run it on a subset, not the corpus.
    """
    order = list(adapter.probes(task))
    matrix: dict[str, dict[str, str]] = {}
    for sha in patch_shas:
        source = patches.require(sha)
        result = scan(adapter, task, source, order, ledger=None, store=store,
                      full=True, source="audit", concurrency=concurrency)
        matrix[sha] = dict(result.verdicts)
    return matrix
