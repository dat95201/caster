"""The sandbox, the verdict store, and one scan. Invariants 2, 3, 9 and 10."""
from __future__ import annotations

import threading

import pytest

from caster.exec import sandbox
from caster.exec.sandbox import ConcurrentChildError, Outcome, run_program
from caster.exec.store import (
    CacheMiss,
    PatchStore,
    VerdictDivergence,
    VerdictStore,
    protocol_sha,
)
from caster.exec.verify import HeldOutProbeLeak, scan


# ── the sandbox ──────────────────────────────────────────────────────────────


def test_cpu_is_getrusage_not_wall_clock():
    """Invariant 3. Wall clock is contaminated by sibling shards and makes m̂ a
    function of load."""
    assert sandbox.CPU_SOURCE == "getrusage"
    busy = run_program("s=0\nfor i in range(400000): s+=i*i\nprint(s)\n", "")
    idle = run_program("import time\ntime.sleep(0.4)\nprint(1)\n", "")
    assert busy.ok and idle.ok
    assert busy.cpu_s > idle.cpu_s          # the sleeper burned wall time, not CPU
    assert idle.wall_s > idle.cpu_s


def test_error_taxonomy_including_the_sys_exit_branch():
    """Without the `sys.exit("msg")` branch every distinct bail-out message
    becomes its own error type."""
    crash = run_program("raise ValueError('bad int')\n", "")
    assert crash.crashed and crash.error_type == "ValueError"

    bail = run_program("import sys\nsys.exit('no solution')\n", "")
    assert bail.error_type == "SystemExit"
    assert bail.error_message == "no solution"

    # Two different bail-out messages must share one error type.
    other = run_program("import sys\nsys.exit('impossible')\n", "")
    assert other.error_type == bail.error_type

    timeout = run_program("while True: pass\n", "", timeout=1.0)
    assert timeout.timed_out and not timeout.crashed

    # A module-qualified exception buckets with the builtin class.
    qualified = run_program("import json\njson.loads('{')\n", "")
    assert qualified.error_type == "JSONDecodeError"


def test_recursion_ceiling_is_installed_without_shifting_line_numbers():
    deep = run_program("import sys\nprint(sys.getrecursionlimit())\n", "")
    assert int(deep.value.strip()) == sandbox.DEFAULT_RECURSION_LIMIT
    # The ceiling arrives via sitecustomize, so line 1 is still line 1.
    lineno = run_program("raise RuntimeError('x')\n", "")
    assert ', line 1,' in (lineno.traceback or "")


def test_a_timed_out_child_still_reports_the_cpu_it_burned():
    """`m̂` must stay honest for the expensive probes the scheduler reorders."""
    out = run_program("s=0\nwhile True: s+=1\n", "", timeout=1.0)
    assert out.timed_out and out.cpu_s > 0.1


def test_a_second_concurrent_child_raises_rather_than_mismeasuring():
    """`getrusage(RUSAGE_CHILDREN)` is process-global.

    "Parallelises on cores with no shared bottleneck" reads as permission to
    use a thread pool, and a thread pool silently attributes a sibling's CPU to
    whichever probe finished. So it raises instead.
    """
    errors: list[BaseException] = []
    started = threading.Event()

    def slow() -> None:
        started.set()
        run_program("import time\ntime.sleep(1.2)\nprint(1)\n", "")

    def fast() -> None:
        started.wait(2)
        try:
            run_program("print(1)\n", "")
        except BaseException as exc:   # noqa: BLE001
            errors.append(exc)

    t1, t2 = threading.Thread(target=slow), threading.Thread(target=fast)
    t1.start(); t2.start(); t1.join(); t2.join()
    assert errors and isinstance(errors[0], ConcurrentChildError)


def test_protocol_fields_cover_everything_that_changes_a_verdict():
    fields = sandbox.protocol_fields()
    assert set(fields) >= {"interpreter", "sandbox_timeout_sec", "recursion_limit",
                           "normalisation", "cpu_source"}
    assert protocol_sha(fields) != protocol_sha({**fields, "sandbox_timeout_sec": 30.0})


# ── the stores ───────────────────────────────────────────────────────────────


def test_patch_store_resolves_a_sha_to_a_patch(run_dir):
    """The response cache holds the model's REPLY, not the extracted program."""
    ps = PatchStore()
    sha = ps.put("print(1)\n")
    assert ps.get(sha) == "print(1)\n"
    assert ps.put("print(1)\n") == sha
    with pytest.raises(CacheMiss):
        ps.require("0" * 64)


def test_a_divergent_verdict_halts(run_dir):
    """Invariant 10 / . Both verdicts are findings; nothing is overwritten."""
    vs = VerdictStore({"p": 1})
    vs.put("sha", "case", verdict="pass", output_sha256="a", cpu_s=0.1)
    with pytest.raises(VerdictDivergence):
        vs.put("sha", "case", verdict="wrong", output_sha256="b", cpu_s=0.1)
    assert vs.peek("sha", "case").verdict == "pass"


def test_a_timeout_verdict_is_stored_but_never_served(run_dir):
    """Timeouts are quarantined. The store is cross-run and cross-HOST, so a
    verdict measured on a fast idle box must not be replayed against a slow
    loaded one."""
    vs = VerdictStore({"p": 1})
    vs.put("sha", "case", verdict="timeout", output_sha256="", cpu_s=9.9)
    assert vs.peek("sha", "case").verdict == "timeout"
    assert vs.get("sha", "case") is None
    assert not vs.has_servable("sha", "case")
    assert vs.verdict_hits == 0
    # ... and it does not halt the run when a fresh execution disagrees.
    vs.put("sha", "case", verdict="wrong", output_sha256="b", cpu_s=0.2)


def test_a_stable_timeout_flip_is_recorded_and_does_not_halt(run_dir):
    """DECISIONS 27. The repeats in `_measure` run back to back, so a `stable`
    disagreement is a statement about machine load, not about the protocol. It
    is written to jitter.jsonl with `stable: true` and the run continues."""
    import json

    from caster import paths
    from caster.exec.store import VerdictStore

    vs = VerdictStore({"p": 1})
    vs.put("sha", "case", verdict="timeout", output_sha256="", cpu_s=9.9)
    # every repeat disagreed with the store -- the old halt condition
    vs.note_timeout_flip("sha", "case", ["pass", "pass", "pass"], stable=True)

    rows = [json.loads(l) for l in
            (paths.DATA_DIR / "jitter.jsonl").read_text().splitlines() if l.strip()]
    assert rows[-1]["stable"] is True
    assert rows[-1]["verdicts"] == ["pass", "pass", "pass"]
    # and the conclusive verdict may overwrite the quarantined timeout
    vs.put("sha", "case", verdict="pass", output_sha256="b", cpu_s=0.2)
    assert vs.peek("sha", "case").verdict == "pass"


def test_an_arm_scan_never_overwrites_an_audited_cost(run_dir):
    """The first arm to touch a (patch, probe) would otherwise write the
    cost every later arm inherits."""
    vs = VerdictStore({"p": 1})
    vs.put("sha", "c", verdict="pass", output_sha256="a", cpu_s=1.0,
           source="audit", concurrency=4)
    vs.put("sha", "c", verdict="pass", output_sha256="a", cpu_s=7.5, source="scan")
    entry = vs.peek("sha", "c")
    assert entry.cpu_s == 1.0 and entry.source == "audit" and entry.concurrency == 4


def test_the_cost_digest_covers_only_audited_entries(run_dir):
    vs = VerdictStore({"p": 1})
    vs.put("a" * 8, "c", verdict="pass", output_sha256="x", cpu_s=1.0, source="audit")
    before = vs.cost_digest()
    vs.put("b" * 8, "c", verdict="pass", output_sha256="x", cpu_s=9.0, source="scan")
    assert vs.cost_digest() == before
    vs.put("b" * 8, "d", verdict="pass", output_sha256="x", cpu_s=2.0, source="audit")
    assert vs.cost_digest() != before


def test_the_store_survives_a_reopen(run_dir):
    vs = VerdictStore({"p": 1})
    vs.put("sha", "c", verdict="pass", output_sha256="a", cpu_s=1.0)
    again = VerdictStore({"p": 1})
    assert again.get("sha", "c").cpu_s == 1.0
    assert VerdictStore({"p": 2}).get("sha", "c") is None   # a protocol change invalidates


# ── one scan ─────────────────────────────────────────────────────────────────


def test_scan_stops_at_the_first_refutation(adapter, frozen, run_dir):
    from caster.exec.sandbox import protocol_fields

    store = VerdictStore(protocol_fields())
    task = "abc001_a/1001"
    order = list(adapter.probes(task))
    result = scan(adapter, task, "n=int(input())\nprint(n*3)\n", order,
                  ledger=None, store=store)
    assert not result.accept and result.killed_by is not None
    assert len(result.scan_order) < len(order)       # it stopped early
    assert result.scan_order[-1] == result.killed_by


def test_full_scan_runs_every_probe(adapter, frozen, run_dir):
    from caster.exec.sandbox import protocol_fields

    store = VerdictStore(protocol_fields())
    task = "abc001_a/1001"
    order = list(adapter.probes(task))
    result = scan(adapter, task, "n=int(input())\nprint(n*3)\n", order,
                  ledger=None, store=store, full=True)
    assert len(result.scan_order) == len(order)
    assert set(result.verdicts) == set(order)


def test_invariant_9_holds_and_the_store_makes_a_rerun_free(adapter, frozen, run_dir):
    """`(sandbox_runs − timeout_reruns) + verdict_hits == test_calls`, and after
    a full scan every policy on that task runs with `sandbox_runs == 0`."""
    from caster.exec.sandbox import protocol_fields

    store = VerdictStore(protocol_fields())
    task = "abc001_a/1001"
    order = list(adapter.probes(task))
    patch = "n=int(input())\nprint(n*3)\n"

    first = scan(adapter, task, patch, order, ledger=None, store=store, full=True)
    assert first.sandbox_runs > 0 and first.verdict_hits == 0

    replay = scan(adapter, task, patch, list(reversed(order)), ledger=None, store=store)
    assert replay.sandbox_runs == 0
    assert replay.verdict_hits == replay.test_calls
    assert replay.cpu_s > 0          # costs are reconstructed from the table


def test_soundness_is_independent_of_the_schedule(adapter, frozen, run_dir):
    """Invariant 2 — no ordering can accept what another rejects."""
    import random

    from caster.exec.sandbox import protocol_fields

    store = VerdictStore(protocol_fields())
    task = "abc001_a/1001"
    order = list(adapter.probes(task))
    for patch, expect in (("n=int(input())\nprint(n*3)\n", False),
                          ("n=int(input())\nprint(n*2)\n", True)):
        verdicts = set()
        for seed in range(6):
            shuffled = order[:]
            random.Random(seed).shuffle(shuffled)
            verdicts.add(scan(adapter, task, patch, shuffled,
                              ledger=None, store=store).accept)
        assert verdicts == {expect}


def test_a_held_out_probe_reaching_the_loop_is_a_hard_stop(adapter, frozen, run_dir):
    """`P_eval` is never shown and never gates acceptance."""
    from caster.exec.sandbox import protocol_fields

    task = "abc001_a/1001"
    held = frozen["tasks"][task]["p_eval"]
    assert held, "the fixture must hold something out for this to mean anything"
    with pytest.raises(HeldOutProbeLeak):
        scan(adapter, task, "print(1)\n", [held[0]],
             ledger=None, store=VerdictStore(protocol_fields()))


def test_from_cache_only_fails_loudly_on_a_miss(adapter, frozen, run_dir):
    from caster.exec.sandbox import protocol_fields

    with pytest.raises(CacheMiss):
        scan(adapter, "abc001_a/1001", "print(1)\n",
             list(adapter.probes("abc001_a/1001"))[:1],
             ledger=None, store=VerdictStore(protocol_fields()),
             from_cache_only=True)


def test_a_duplicate_in_the_order_is_not_run_twice(adapter, frozen, run_dir):
    """A probe run twice against one patch in one round — made impossible."""
    from caster.exec.sandbox import protocol_fields

    task = "abc001_a/1001"
    probe = list(adapter.probes(task))[0]
    result = scan(adapter, task, "n=int(input())\nprint(n*2)\n", [probe, probe, probe],
                  ledger=None, store=VerdictStore(protocol_fields()))
    assert result.test_calls == 1
    assert result.scan_order == (probe,)


def test_the_ledger_learns_from_a_scan(adapter, frozen, run_dir):
    from caster.exec.ledger import CaseLedger
    from caster.exec.sandbox import protocol_fields

    task = "abc001_a/1001"
    order = list(adapter.probes(task))
    ledger = CaseLedger(order, explore_p=0.0)
    scan(adapter, task, "n=int(input())\nprint(n*3)\n", order, ledger=ledger,
         store=VerdictStore(protocol_fields()))
    touched = [n for n, s in ledger.stats.items() if s.n > 0]
    assert touched
    assert any(ledger.stats[n].proven for n in touched)
    assert all(ledger.stats[n].s > 0 for n in touched)


def test_outcome_defaults_are_sane():
    o = Outcome(ok=True, value="x")
    assert not o.crashed and o.cpu_s == 0.0 and o.cpu_source == sandbox.CPU_SOURCE
