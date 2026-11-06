"""One scan = triage + oracle, stopped at the first refutation.

The obvious design keeps two mechanisms apart: a guard that replays stored
counterexamples, and an oracle that samples cases at random. Two orderings,
neither informed by the other — so the guard misses refutations the oracle then
pays full price to rediscover, and the same case can be executed twice against
one patch in one round. Here there is **one walk down one schedule**: no bucket
that can be empty, no stopping rule that is unsound, and nothing run twice in a
round.

Soundness is independent of the schedule (Invariant 2): a kill is a real
execution of a real probe against the real reference, so **no ordering can
accept what another rejects**. Reordering can only ever find a true refutation
earlier. `tests/test_exec.py` asserts equality of `accept` and the
first-accept round across every Block A arm on every cell, not on a sample: a
gap there is a soundness bug, not a finding.

**Four terminal states, named.** "`killed_by = None` means accepted" is
ambiguous three ways: a round also has no killer when no patch was built, and
when the scan was inconclusive. So `ScanResult` carries an explicit
`accept: bool` **beside** `oracle_error`, and an inconclusive round is a spent
round with no refutation, never an acceptance.

  accepted         accept=True,  killed_by=None, oracle_error=None
  refuted          accept=False, killed_by=<probe>
  oracle failure   accept=False, killed_by=None, oracle_error set
  proposal failure — not here; the episode loop owns it
"""
from __future__ import annotations

import dataclasses
import hashlib
from typing import Any, Sequence

from caster.bench.protocol import ProbeVerdict
from caster.exec.ledger import CaseLedger
from caster.exec.store import (
    OUTPUT_HEAD_CHARS,
    REPEATS_BEFORE_HALT,
    CacheMiss,
    VerdictStore,
)


class HeldOutProbeLeak(RuntimeError):
    """A held-out (`P_eval`) probe reached the loop. A hard stop, not a warning.

    `P_eval` is never shown and never gates acceptance; a leak makes every
    disclosure figure and every reported success rate meaningless, and it is the
    kind of defect that is invisible in the numbers. So it raises.
    """


@dataclasses.dataclass(frozen=True)
class ScanResult:
    accept: bool
    killed_by: str | None
    scan_order: tuple[str, ...]          # what RAN, in order; never dropped
    cpu_s: float                         # the round's scan cost, CPU seconds
    test_calls: int                      # probes whose verdict was established
    sandbox_runs: int                    # ... of which actually executed
    verdict_hits: int                    # ... of which served from the store
    oracle_error: str | None = None
    verdicts: dict[str, str] = dataclasses.field(default_factory=dict)
    observations: dict[str, ProbeVerdict] = dataclasses.field(default_factory=dict)
    skipped: tuple[str, ...] = ()        # probes with no establishable reference
    timeout_reruns: int = 0              # quarantine repeats; see the assertion

    def __post_init__(self) -> None:
        # Invariant 9, at the only place that can check it per scan.
        #
        # The spec states it as `sandbox_runs + verdict_hits == Σ scan lengths`.
        # the quarantine makes that an inequality unless the repeats are
        # named: a timeout is re-executed up to REPEATS_BEFORE_HALT times to
        # tell a stable disagreement from jitter, and those extra executions are
        # real CPU that must be reported — they are just not extra *verdicts*.
        # So the exact identity is:
        #
        #     (sandbox_runs − timeout_reruns) + verdict_hits == test_calls
        #
        # and `caster merge` re-checks it over the whole run.
        established = self.sandbox_runs - self.timeout_reruns + self.verdict_hits
        if established != self.test_calls:
            raise AssertionError(
                f"Invariant 9: (sandbox_runs {self.sandbox_runs} − timeout_reruns "
                f"{self.timeout_reruns}) + verdict_hits {self.verdict_hits} = "
                f"{established}, but test_calls = {self.test_calls}. Every "
                f"established verdict is either executed exactly once or served "
                f"from the store exactly once."
            )


def _sha(text: str | None) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _measure(
    adapter: Any,
    task_name: str,
    probe: str,
    patch: str,
    store: VerdictStore,
) -> tuple[ProbeVerdict, int, int]:
    """Execute one probe. Returns (verdict, executions, timeout re-executions).

    Timeouts are quarantined. A fresh `timeout` that disagrees with a stored
    non-timeout verdict (or vice versa) is re-run up to `REPEATS_BEFORE_HALT`
    times on one host. **No outcome of that loop halts the run** (DECISIONS 27):
    the conclusive -- non-timeout -- verdict wins, the flip goes to
    `jitter.jsonl` carrying whether it was `stable`, and the threats table
    reports both rates. A disagreement where NEITHER side is a timeout is a
    different thing and still halts unconditionally, in `store.put`.

    In the common case -- no stored entry, or agreement -- this is exactly one
    execution.
    """
    verdict = adapter.run(task_name, probe, patch)
    runs = verdict.runs
    reruns = 0
    stored = store.peek(_sha(patch), probe)
    if stored is None or stored.verdict == verdict.status:
        return verdict, runs, reruns
    if "timeout" not in (stored.verdict, verdict.status):
        # Neither side is a timeout: divergence is fatal. `store.put` is
        # where it raises, so the caller still records the attempt.
        return verdict, runs, reruns

    seen = [verdict.status]
    conclusive = verdict if verdict.status != "timeout" else None
    for _ in range(REPEATS_BEFORE_HALT - 1):
        again = adapter.run(task_name, probe, patch)
        runs += again.runs
        reruns += again.runs
        store.timeout_reexecutions += again.runs
        seen.append(again.status)
        if again.status != "timeout" and conclusive is None:
            conclusive = again
        if again.status == stored.verdict:
            # It reproduced the stored verdict after all — pure jitter.
            store.note_timeout_flip(_sha(patch), probe, seen)
            return again, runs, reruns
    # Every repeat disagreed with the store, and one side is a `timeout`.
    # This does NOT halt the run (DECISIONS 2 and 27). A stored `timeout` is
    # not evidence -- `store.get()` never serves it and `--replay-only` cannot
    # be satisfied by it -- so it must not be the thing that stops a grid.
    #
    # The condition that used to raise here fired when the repeats agreed with
    # each other, reading "stable" as "not jitter, therefore a protocol gap".
    # The repeats run back to back, so under sustained load they sample ONE
    # machine state: on a contended box "stable" measures the load, not the
    # protocol. That is exactly the flip rate at the grid's target concurrency
    # that DECISIONS 2 left open, so it is RECORDED rather than raised --
    # `jitter.jsonl` carries `stable`, and the threats table reports the rate.
    #
    # A disagreement where NEITHER side is a timeout still halts
    # unconditionally: that path returns above, and `store.put` raises.
    store.note_timeout_flip(_sha(patch), probe, seen,
                            stable=all(s == seen[0] for s in seen))
    return conclusive or verdict, runs, reruns


def scan(
    adapter: Any,
    task_name: str,
    patch: str,
    order: Sequence[str],
    *,
    ledger: CaseLedger | None,
    store: VerdictStore,
    round_index: int = 0,
    full: bool = False,
    from_cache_only: bool = False,
    source: str = "scan",
    concurrency: int | None = None,
) -> ScanResult:
    """Walk `order` until something refutes `patch`, or to the end.

    `full=True` is the no-early-stop mode: it produces the **full kill vector**
    per (patch, probe), which is the only way to compute an honest `K` and the
    only way to replay a counterfactual policy offline without new model calls. `caster audit` runs it; arm 0 `full-suite` tabulates it for free.

    `ledger` may be None for an audit pass — the audit is not an episode, and a
    ledger it updated would be a cross-episode prior, which the store forbids.
    """
    patch_sha = _sha(patch)
    ran: list[str] = []
    skipped: list[str] = []
    # The round's queue, in the policy's order. The overlap rule reorders
    # WITHIN it and never removes from it, so nothing is dropped and an
    # accepting round still runs the whole pool — ordering cannot change
    # acceptance (Invariant 2). `scan_order` below records the walk that
    # actually happened, deferrals and all, so a deferral is visible in the
    # record rather than inferred from a cost that moved.
    pending: list[str] = list(order)
    passed: list[str] = []
    #: A probe may be pushed back AT MOST ONCE per round. Without this the
    #: walk is still finite — `pending` shrinks every iteration — but it is
    #: finite by an argument about the loop rather than by a rule, and a probe
    #: could be stepped over many times while `passed` keeps growing. One
    #: deferral each makes the bound a property of the walk: at most one pass
    #: of deferrals, every probe runs, and nothing starves.
    deferred_once: set[str] = set()
    verdicts: dict[str, str] = {}
    observations: dict[str, ProbeVerdict] = {}
    cpu_total = 0.0
    executed = 0
    reruns_total = 0
    hits = 0
    killed_by: str | None = None

    # The rule is off for a full scan (nothing to save when every probe runs
    # anyway, and `full-suite` must stay byte-identical) and off without a
    # ledger (an audit is not an episode and has no rejection sets).
    defer_on = (ledger is not None and not full and ledger.overlap_thr > 0.0)

    while pending:
        probe = None
        if defer_on and passed:
            for i, cand in enumerate(pending):
                if cand in deferred_once:
                    # Already spent its deferral. It is behind the probes that
                    # have not, and it runs as soon as they are gone.
                    continue
                if ledger.redundant(cand, passed):          # type: ignore[union-attr]
                    deferred_once.add(cand)
                    continue
                probe = pending.pop(i)
                break
        if probe is None:
            # Either the rule is off, or nothing is left that has not already
            # been pushed back once. Take the head: among probes that have all
            # spent their deferral the policy's own ranking decides again, and
            # a probe deferred earlier in this round is now AHEAD of whatever
            # was deferred after it. One deferral each, then the walk finishes.
            probe = pending.pop(0)

        # A probe is never run twice against one patch in one round. One walk
        # cannot overlap with itself unless a policy hands back a duplicate, and
        # a policy that does is a bug worth catching here rather than in a CPU
        # total nobody reconciles.
        if probe in verdicts or probe in skipped:
            continue
        if adapter.authority(task_name, probe) != "visible":
            raise HeldOutProbeLeak(
                f"probe {probe!r} of {task_name!r} is held out (P_eval) and "
                f"reached the scan. P_eval is never shown and never gates "
                f"acceptance."
            )

        stored = store.get(patch_sha, probe)
        if stored is not None:
            hits += 1
            # Carry the stored observation, not just the status. Without it a
            # replay against a warm store builds a DIFFERENT steered prompt from
            # the run that filled the store — a different draw, with nothing in
            # any key to say so (`OUTPUT_HEAD_CHARS`).
            verdict = ProbeVerdict(
                status=stored.verdict,  # type: ignore[arg-type]
                cpu_s=stored.cpu_s,
                output=stored.output_head,
                output_len=stored.output_len,
                error_type=stored.error_type,
                error_message=stored.error_message,
                tier=0,
                runs=0,
            )
        else:
            if from_cache_only:
                raise CacheMiss(
                    f"--from-cache-only: (patch {patch_sha[:12]}, probe {probe}, "
                    f"protocol {store.protocol_sha}) is not servable from the "
                    f"verdict store. Run `caster audit` for {task_name} first, "
                    f"or drop the flag. A silent model/sandbox call here is "
                    f"what phase separation exists to prevent."
                )
            verdict, runs, reruns = _measure(adapter, task_name, probe, patch, store)
            executed += runs
            reruns_total += reruns
            store.sandbox_runs += runs
            if verdict.tier < 0:
                # No establishable reference: the probe proves nothing and is
                # never counted against the candidate (G1's runtime backstop).
                skipped.append(probe)
                if ledger is not None:
                    ledger.mark_unusable(probe)
                continue
            store.put(
                patch_sha, probe,
                verdict=verdict.status,
                output_sha256=_sha(verdict.output),
                cpu_s=verdict.cpu_s,
                source=source,
                concurrency=concurrency,
                output_head=(verdict.output or "")[:OUTPUT_HEAD_CHARS] or None,
                output_len=(len(verdict.output) if verdict.output is not None else None),
                error_type=verdict.error_type,
                error_message=verdict.error_message,
                flush=False,
            )

        ran.append(probe)
        verdicts[probe] = verdict.status
        # Truncate the FRESH path to the same head the store keeps, and carry
        # the true length beside it. Otherwise a cold run renders the whole
        # output (or a marker naming its real size) and a warm replay renders
        # the 240-char head — different prompt bytes, different cache key, a
        # different draw, decided by whether the store happened to be warm.
        observations[probe] = dataclasses.replace(
            verdict,
            output=((verdict.output or "")[:OUTPUT_HEAD_CHARS] or None),
            output_len=(verdict.output_len if verdict.output_len is not None
                        else (len(verdict.output) if verdict.output is not None else None)),
        )
        cpu_total += verdict.cpu_s
        if ledger is not None:
            ledger.update(probe, refuted=verdict.refuted, cpu_s=verdict.cpu_s,
                          round_index=round_index, patch=patch_sha)
            ledger.scan_runs += verdict.runs
        if not verdict.refuted:
            passed.append(probe)

        if verdict.refuted and killed_by is None:
            killed_by = probe
            if not full:
                break

    store.flush(patch_sha)

    if not ran:
        return ScanResult(
            accept=False, killed_by=None, scan_order=(), cpu_s=cpu_total,
            test_calls=0, sandbox_runs=executed, verdict_hits=hits,
            oracle_error="caster.exec.verify.NoUsableProbe",
            verdicts={}, observations={}, skipped=tuple(skipped),
            timeout_reruns=reruns_total,
        )
    return ScanResult(
        accept=killed_by is None,
        killed_by=killed_by,
        scan_order=tuple(ran),
        cpu_s=cpu_total,
        test_calls=len(ran),
        sandbox_runs=executed,
        verdict_hits=hits,
        oracle_error=None,
        verdicts=verdicts,
        observations=observations,
        skipped=tuple(skipped),
        timeout_reruns=reruns_total,
    )
