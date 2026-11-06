"""The arm roster, the episode loop, atomic cells, resume, and the merge audits."""
from __future__ import annotations

import json
import random

import pytest

from caster.exec.ledger import CaseLedger
from caster.keys import CellSpec
from caster.report.records import (
    CellDone,
    CellWriter,
    RoundRecord,
    collapse,
    load_done,
    needs_walk,
    read_rows,
)
from caster.run import policies, propose, shard
from caster.run.episode import EpisodeConfig, run_episode
from caster.run.policies import PolicyContext

POOL = tuple(f"{i:02d}_probe_{i}" for i in range(12))


def ctx(ledger: CaseLedger, round_index: int = 1, **kw) -> PolicyContext:
    return PolicyContext(ledger=ledger, pool=POOL, round_index=round_index,
                         cell_key="cell", **kw)


def led(**kw) -> CaseLedger:
    kw.setdefault("explore_p", 0.0)
    kw.setdefault("rng", random.Random(0))
    return CaseLedger(POOL, **kw)


# ── the roster ───────────────────────────────────────────────────────────────


def test_every_arm_in_the_spec_exists():
    """The two tables, as a membership assertion."""
    block_a = {"full-suite", "no-mem", "natural", "random", "fault-record",
               "cost-only", "kill-only", "coverage", "km", "km-cs-median",
               "km-cs-q75", "km-bucket", "km-explore0", "km-ov0"}
    block_b = {"km+steer", "km+steer-pass", "km+exclusion", "km+chat",
               "km+steer-k1", "km+steer-k2", "km+steer-sample", "km+steer-noreq",
               "km+steer+vs"}
    assert block_a | block_b == set(policies.ARMS)
    assert {a.arm_id for a in policies.BLOCK_A} == block_a
    assert {a.arm_id for a in policies.BLOCK_B} == block_b


def test_the_default_grid_excludes_only_full_suite():
    """`full-suite` is excluded because `caster audit` already produces it.

    `km+chat` used to be excluded too, for being measured bad here (0.50
    budgeted success against no-memory's 0.63). That was backwards: it is the
    ChatRepair-style comparator RQ3 is argued against, and "the published
    comparator loses to redrawing" is the finding, not a reason to stop
    running it. Chen et al. (arXiv 2503.15050) report the same direction
    independently, so the arm carries a replication.
    """
    assert "full-suite" not in policies.DEFAULT_GRID
    assert "km+chat" in policies.DEFAULT_GRID
    assert set(policies.DEFAULT_GRID) | {"full-suite"} == set(policies.ARMS)


@pytest.mark.parametrize("arm_id", sorted(policies.ARMS))
def test_every_ordering_is_a_permutation_and_is_deterministic(arm_id):
    a, b = led(), led()
    for name in (POOL[3], POOL[7]):
        for x in (a, b):
            x.update(name, refuted=True, cpu_s=0.3, round_index=1)
    for name in (POOL[0], POOL[1]):
        for x in (a, b):
            x.update(name, refuted=False, cpu_s=0.01, round_index=1)
    cov = {n: frozenset({i, i + 1}) for i, n in enumerate(POOL)}
    oa = policies.schedule(arm_id, ctx(a, 4, coverage=cov))
    ob = policies.schedule(arm_id, ctx(b, 4, coverage=cov))
    assert sorted(oa) == sorted(POOL)
    assert oa == ob


def test_no_mem_and_natural_scan_in_the_same_order():
    """They differ ONLY in whether the ledger is updated, which isolates
    the cost of maintaining it from the benefit of using it."""
    a, b = led(), led()
    assert (policies.schedule("no-mem", ctx(a))
            == policies.schedule("natural", ctx(b))
            == list(POOL))
    assert policies.arm("no-mem").update_ledger is False
    assert policies.arm("natural").update_ledger is True


def test_round_one_is_the_natural_order_for_every_arm_but_random():
    """Invariant: an EMPTY ledger means every arm walks the pool order.

    At round 1 nothing has been observed, so every probe carries the identical
    `K` and the identical `m̂`, every score ties, and the tie-break IS the whole
    order. A policy has nothing to say there — so it must say nothing, and
    `sorted` being stable means ranking on pool position is exactly that.

    Any round-1 difference between two arms is therefore a fabricated one. The
    hashed default this replaced put `policy` in the order (`CellSpec.key()`
    writes every field), so the ARM'S NAME picked the first probe: measured on
    the 3B pilot, `km` opened round 1 at mean natural-rank 6.56 while `natural`
    opened at 1.00, and `fault-record`/`km-bucket` — which keep pool order for
    their no-evidence bucket — opened at 1.00 like `natural`. Three arms did
    one thing and three the opposite, and nothing in the roster said so.

    `random` is excluded by construction: being a reshuffle is its definition.
    `coverage` is excluded too — reference coverage is known before round 1, so
    it is the one arm that legitimately has something to rank on.
    """
    same = [a.arm_id for a in policies.BLOCK_A
            if a.ordering not in ("random", "coverage")]
    assert len(same) >= 8
    for arm_id in same:
        assert policies.schedule(arm_id, ctx(led(), 1)) == list(POOL), arm_id


def test_natural_is_pool_order_and_random_reshuffles_per_round():
    x = led()
    assert policies.schedule("natural", ctx(x, 9)) == list(POOL)
    r1 = policies.schedule("random", ctx(x, 1))
    r2 = policies.schedule("random", ctx(x, 2))
    assert r1 != r2 and sorted(r1) == sorted(POOL)
    assert r1 == policies.schedule("random", ctx(led(), 1))   # seeded, reproducible


def test_fault_record_ranks_on_kill_count_as_the_paper_does():
    """FRTP ranks on the COUNT of patches a case has falsified.

    Qi, Mao & Lei, ICSM 2013: "ranks the test cases based on the test case
    failing count". This arm ranked on `first_kill_round` for one revision,
    which freezes the order at discovery and leaves the count unread — a
    weaker baseline than the published one, and weaker in the direction that
    inflates whatever is compared against it. Replayed offline the two differ
    by −2.8%, CI95 [−9.6, +5.4]: small, and the objection is to the claim, not
    to the effect size.
    """
    x = led()
    x.update(POOL[9], refuted=True, cpu_s=0.1, round_index=1)   # discovered first
    for _ in range(6):
        x.update(POOL[2], refuted=True, cpu_s=0.1, round_index=4)  # kills more
    order = policies.schedule("fault-record", ctx(x, 5))
    assert order[0] == POOL[2] and order[1] == POOL[9]
    assert policies.schedule("kill-only", ctx(x, 5))[0] == POOL[2]


def test_the_two_marginals_differ_from_each_other_and_from_km():
    """The 2×2 of : no-K/no-cost, cost-only, kill-only, km."""
    x = led()
    x.update(POOL[1], refuted=True, cpu_s=5.0, round_index=1)    # kills, expensive
    x.update(POOL[2], refuted=False, cpu_s=0.01, round_index=1)  # cheap, no kill
    c = ctx(x, 2)
    assert policies.schedule("kill-only", c)[0] == POOL[1]
    assert policies.schedule("cost-only", c)[0] == POOL[2]
    assert policies.schedule("km", c) != policies.schedule("kill-only", c)
    assert policies.schedule("km", c) != policies.schedule("cost-only", c)


def test_coverage_is_additional_greedy():
    x = led()
    cov = {POOL[0]: frozenset({1, 2, 3}), POOL[1]: frozenset({1, 2, 3}),
           POOL[2]: frozenset({4, 5})}
    order = policies.schedule("coverage", ctx(x, 1, coverage=cov))
    assert order[0] == POOL[0]
    assert order[1] == POOL[2]          # adds new lines; POOL[1] adds nothing
    assert sorted(order) == sorted(POOL)


def test_km_bucket_puts_survived_only_probes_last():
    x = led()
    x.update(POOL[0], refuted=True, cpu_s=0.1, round_index=1)
    x.update(POOL[1], refuted=False, cpu_s=0.1, round_index=1)
    order = policies.schedule("km-bucket", ctx(x, 2))
    assert order[0] == POOL[0]
    assert order[-1] == POOL[1]


def test_km_explore0_is_the_ablation_not_a_different_scheduler():
    assert policies.arm("km-explore0").overrides == {"explore_p": 0.0}
    assert policies.arm("km-explore0").ordering == policies.arm("km").ordering


def test_block_b_arms_are_all_steered():
    for a in policies.BLOCK_B:
        assert a.overrides.get("steer_on") is True, a.arm_id


# ── the draw ─────────────────────────────────────────────────────────────────


def test_parsing_is_forgiving_about_the_vs_format():
    one = propose.parse_candidates("blah\n```python\nprint(1)\n```\ntrailing")
    assert len(one) == 1 and one[0].source == "print(1)\n"
    assert one[0].probability == 1.0 and not one[0].stated

    many = propose.parse_candidates(
        "### Candidate 1 (probability: 0.6)\n```python\nprint(1)\n```\n"
        "### Candidate 2 (prob = 20%)\n```python\nprint(2)\n```\n")
    assert [c.probability for c in many] == [0.6, 0.2]
    assert all(c.stated for c in many)


def test_a_reply_with_no_closed_fence_is_truncated_not_a_patch():
    """Returning the raw reply as though it were a patch produces a candidate
    that always fails — a harness defect entering memory as evidence."""
    from caster.llm.client import TruncatedResponse

    with pytest.raises(TruncatedResponse):
        propose.parse_candidates("I think the bug is on line 2, so")
    with pytest.raises(TruncatedResponse):
        propose.parse_candidates("```python\nprint(1)")


def test_candidate_ordering_policies():
    cands = propose.parse_candidates(
        "### Candidate 1 (probability: 0.1)\n```python\na\n```\n"
        "### Candidate 2 (probability: 0.9)\n```python\nb\n```\n")
    assert [c.index for c in propose.order_candidates(cands, "exploit")] == [2, 1]
    assert [c.index for c in propose.order_candidates(cands, "explore")] == [1, 2]
    assert [c.index for c in propose.order_candidates(cands, "as_given")] == [1, 2]
    with pytest.raises(ValueError):
        propose.order_candidates(cands, "nope")


def test_duplicate_rate_is_measurable_today():
    assert propose.duplicate_rate(["a", "a", "b", "b"]) == 0.5
    assert propose.duplicate_rate([]) == 0.0


# ── atomic cells and resume ───────────────────────────────────────────


def row(episode="e1", r=1, c=0, **kw) -> RoundRecord:
    return RoundRecord(run_id="t", episode_id=episode, task="x", seed=0,
                       round_index=r, policy="km", candidate_index=c, **kw)


def test_a_cell_is_written_whole_or_not_at_all(tmp_path):
    rows, done = tmp_path / "rounds.jsonl", tmp_path / "cells.done"
    w = CellWriter(rows, done)
    w.add(row(r=1)); w.add(row(r=2))
    assert not rows.exists()          # nothing on disk until commit
    w.commit(CellDone(cell_key="k", episode_id="e1", task="x", seed=0, policy="km",
                      rounds_written=2, rounds=2, budget=12))
    assert len(rows.read_text().splitlines()) == 2
    assert list(load_done(done)) == ["k"]

    w2 = CellWriter(rows, done)
    w2.add(row(r=3))
    w2.abandon()
    assert len(rows.read_text().splitlines()) == 2


def test_resume_reads_the_done_index_never_the_row_file(tmp_path):
    done = tmp_path / "cells.done"
    CellWriter(tmp_path / "r.jsonl", done).commit(
        CellDone(cell_key="k", episode_id="e1", task="x", seed=0, policy="km",
                 rounds_written=12, rounds=12, budget=12))
    idx = load_done(done)
    assert not needs_walk(idx, "k", 12)
    assert needs_walk(idx, "other", 12)


def test_a_cell_done_at_a_lower_budget_is_rewalked(tmp_path):
    """`budget` is not in the cell key, so without this check topping a
    grid up from B=12 to B=20 would silently draw nothing."""
    done = tmp_path / "cells.done"
    CellWriter(tmp_path / "r.jsonl", done).commit(
        CellDone(cell_key="k", episode_id="e1", task="x", seed=0, policy="km",
                 rounds_written=12, rounds=12, budget=12))
    idx = load_done(done)
    assert needs_walk(idx, "k", 20)
    assert not needs_walk(idx, "k", 12)
    assert not needs_walk(idx, "k", 8)


def test_a_torn_row_is_dropped_by_the_loader(tmp_path):
    rows = tmp_path / "rounds.jsonl"
    good = json.dumps(row().as_dict())
    rows.write_text(good + "\n" + good[: len(good) // 2], encoding="utf-8")
    assert len(list(read_rows([rows]))) == 1


def test_collapse_is_last_write_wins_so_a_rewalk_rewrites_itself(tmp_path):
    """Collapse on `(episode_id, round_index, candidate_index)`.
    It is what makes topping a cell up mean what it looks like."""
    first = row(r=1, killed_by="a")
    second = row(r=1, killed_by="b")
    out = collapse([first, second])
    assert len(out) == 1 and out[0].killed_by == "b"
    assert len(collapse([row(r=1, c=0), row(r=1, c=1)])) == 2


def test_the_four_terminal_states_are_disjoint_and_exhaustive():
    """`killed_by=None` on its own is ambiguous three ways."""
    assert row(accept=True).terminal == "accepted"
    assert row(killed_by="c").terminal == "refuted"
    assert row(proposal_error="context_overflow").terminal == "proposal_failure"
    assert row(oracle_error="NoUsableProbe").terminal == "oracle_failure"
    # An inconclusive round is a SPENT round with no refutation, never an accept.
    assert row(oracle_error="NoUsableProbe", accept=False).terminal == "oracle_failure"


# ── ranges and the manifest ──────────────────────────────────────────────────


def test_ranges_are_contiguous_and_the_remainder_goes_to_the_first_shards():
    """A fat tail is the shard still running when the others finished."""
    r = shard.plan_ranges(1, 10, 3)
    assert r == [(1, 4), (5, 7), (8, 10)]
    assert r[0][1] - r[0][0] >= r[-1][1] - r[-1][0]
    covered = [i for lo, hi in r for i in range(lo, hi + 1)]
    assert covered == list(range(1, 11))
    assert shard.plan_ranges(5, 5, 1) == [(5, 5)]
    with pytest.raises(ValueError):
        shard.plan_ranges(1, 3, 4)
    with pytest.raises(ValueError):
        shard.plan_ranges(5, 1, 1)


def test_a_second_fleet_refuses_while_one_is_alive(tmp_path):
    import os

    live = shard.ShardEntry(0, 1, 5, os.getpid(), ["caster"], "log")
    shard.write_manifest([live], 5, run_dir=tmp_path)
    with pytest.raises(RuntimeError, match="still running"):
        shard.write_manifest([live], 5, run_dir=tmp_path)

    # A fleet whose processes are all gone is replaceable.
    other = tmp_path / "other"
    dead = shard.ShardEntry(0, 1, 5, 2 ** 30, ["caster"], "log")
    shard.write_manifest([dead], 5, run_dir=other)
    shard.write_manifest([dead], 5, run_dir=other)


# ── the audits ───────────────────────────────────────────────────────────────


def test_runtime_disagreement_is_a_hard_stop(tmp_path):
    """The generalisation of the rule to the values that CANNOT be in a key."""
    for i, ctx_tokens in enumerate((32768, 8192)):
        d = tmp_path / "shards" / f"s{i:02d}"
        d.mkdir(parents=True)
        (d / "meta.json").write_text(json.dumps({
            "protocol": {"model": "m", "temperature": 1.0, "reasoning_effort": None},
            "runtime": {"context_tokens": ctx_tokens, "sandbox_timeout_sec": 10.0,
                        "interpreter": "3.13.0", "recursion_limit": 10000,
                        "normalisation": "v1", "backend": "b", "cpu_source": "getrusage"},
        }), encoding="utf-8")
    audits = shard.audit_protocol(shard.metas(tmp_path))
    runtime = [a for a in audits if a.name == "runtime disagreement"][0]
    assert runtime.fatal and not runtime.ok and "context_tokens" in runtime.detail
    proto = [a for a in audits if a.name == "protocol disagreement"][0]
    assert proto.ok


def test_a_missing_protocol_record_is_reported_not_fatal(tmp_path):
    audits = shard.audit_protocol([])
    assert len(audits) == 1
    assert not audits[0].fatal and not audits[0].ok


def test_foreign_tasks_and_gaps(tmp_path):
    rows = [row(), RoundRecord(run_id="t", episode_id="e2", task="elsewhere",
                               seed=0, round_index=1, policy="km")]
    audits = shard.audit_coverage(rows, universe=["x"], budget=1)
    foreign = [a for a in audits if a.name == "FOREIGN"][0]
    assert foreign.fatal and not foreign.ok and "elsewhere" in foreign.detail


def test_truncated_cells_are_still_flagged_under_atomic_writes():
    """Atomicity solves torn rows, not short episodes."""
    rows = [row(r=1)]
    truncated = [a for a in shard.audit_coverage(rows, ["x"], budget=12)
                 if a.name == "TRUNCATED"][0]
    assert not truncated.ok and not truncated.fatal


def test_disagreement_does_not_pick_a_winner():
    a = row(r=1, killed_by="p1", accept=False)
    b = row(r=1, killed_by="p2", accept=False)
    audit = shard.audit_conflicts([a, b])
    assert audit.fatal and not audit.ok
    assert "do not pick a winner" in audit.detail


def test_invariant_9_over_the_whole_run():
    ok = [row(test_calls=3, sandbox_runs=2, verdict_hits=1)]
    assert shard.audit_invariant9(ok).ok
    bad = [row(test_calls=3, sandbox_runs=2, verdict_hits=0)]
    assert not shard.audit_invariant9(bad).ok


def test_soundness_audit_compares_every_block_a_arm_on_every_cell():
    """Invariant 2: ordering cannot change acceptance, so a gap here is a
    soundness bug, not a finding."""
    same = [RoundRecord(run_id="t", episode_id=p, task="x", seed=0, round_index=2,
                        policy=p, accept=True)
            for p in ("natural", "km", "random")]
    assert shard.audit_soundness(same).ok

    differs = same + [RoundRecord(run_id="t", episode_id="z", task="x", seed=0,
                                  round_index=5, policy="kill-only", accept=True)]
    audit = shard.audit_soundness(differs)
    assert audit.fatal and not audit.ok
    assert audit.items                      # IN THE RESULT, not only on stdout


# ── the loop ─────────────────────────────────────────────────────────────────


def test_an_episode_accepts_when_the_model_eventually_repairs(
        adapter, frozen, run_dir, stores, fake_client):
    store, patches = stores
    client = fake_client(default_source="n=int(input())\nprint(n*3)\n",
                         replies={3: "n=int(input())\nprint(n*2)\n"})
    spec = policies.arm("km").cell("abc001_a/1001", 0)
    r = run_episode(spec, adapter=adapter, client=client, store=store,
                    patches=patches, cfg=EpisodeConfig(budget=6))
    assert r.accepted and r.first_accept_round == 3
    assert r.rounds == 3
    assert [rec.terminal for rec in r.records][-1] == "accepted"
    assert all(rec.cpu_s >= 0 for rec in r.records)
    assert patches.get(r.accepted_patch_sha) == "n=int(input())\nprint(n*2)\n"


def test_force_full_budget_keeps_drawing_past_the_first_accept(
        adapter, frozen, run_dir, stores, fake_client):
    """Phase 1 has no oracle, so it cannot know where to stop; and the
    floor arm is an estimator of π, not just a baseline."""
    store, patches = stores
    client = fake_client(default_source="n=int(input())\nprint(n*3)\n",
                         replies={2: "n=int(input())\nprint(n*2)\n"})
    spec = policies.arm("no-mem").cell("abc001_a/1001", 0)
    r = run_episode(spec, adapter=adapter, client=client, store=store,
                    patches=patches,
                    cfg=EpisodeConfig(budget=5, force_full_budget=True))
    assert r.first_accept_round == 2 and r.rounds == 5


def test_no_mem_never_updates_the_ledger_so_its_order_never_moves(
        adapter, frozen, run_dir, stores, fake_client):
    store, patches = stores
    client = fake_client(default_source="n=int(input())\nprint(n*3)\n")
    r = run_episode(policies.arm("no-mem").cell("abc001_a/1001", 0),
                    adapter=adapter, client=client, store=store, patches=patches,
                    cfg=EpisodeConfig(budget=4))
    orders = {rec.scan_order for rec in r.records}
    assert len(orders) == 1


def test_the_km_arm_learns_and_shortens_its_scan(
        adapter, frozen, run_dir, stores, fake_client):
    store, patches = stores
    client = fake_client(default_source="n=int(input())\nprint(n*3)\n")
    km = run_episode(policies.arm("km").cell("abc001_a/1001", 0, explore_p=0.0),
                     adapter=adapter, client=client, store=store, patches=patches,
                     cfg=EpisodeConfig(budget=5))
    lengths = [len(rec.scan_order) for rec in km.records]
    assert lengths[-1] <= lengths[0]
    assert lengths[-1] == 1              # it has found the cheapest reliable killer


def test_the_three_proposal_failures_are_charged_and_leave_the_ledger_alone(
        adapter, frozen, run_dir, stores, fake_client):
    """And `context_overflow` especially: uncaught it is a deterministic
    poison pill that aborts at the identical round on every re-run."""
    from caster.llm.client import BackendUnreachable, ContextOverflow, TruncatedResponse

    store, patches = stores
    client = fake_client(
        default_source="n=int(input())\nprint(n*3)\n",
        raises={1: ContextOverflow("too long"), 2: TruncatedResponse("no fence")},
    )
    r = run_episode(policies.arm("km").cell("abc001_a/1001", 0),
                    adapter=adapter, client=client, store=store, patches=patches,
                    cfg=EpisodeConfig(budget=4))
    kinds = [rec.proposal_error for rec in r.records if rec.proposal_error]
    assert kinds == ["context_overflow", "truncated_response"]
    assert all(rec.charged for rec in r.records if rec.proposal_error)
    assert all(rec.patch_sha is None for rec in r.records if rec.proposal_error)
    assert r.rounds == 4                 # the failures were CHARGED, not free

    breaker = fake_client(
        default_source="x",
        raises={i: BackendUnreachable("down") for i in range(1, 9)})
    r2 = run_episode(policies.arm("km").cell("abc001_a/1001", 0),
                     adapter=adapter, client=breaker, store=store, patches=patches,
                     cfg=EpisodeConfig(budget=20))
    assert r2.truncated and r2.truncate_reason == "backend_circuit_breaker"
    assert len(r2.records) == 5          # MAX_CONSECUTIVE_BACKEND_FAILURES


def test_a_vs_round_emits_one_row_per_candidate_sharing_one_cache_key(
        adapter, frozen, run_dir, stores, fake_client):
    """VS's schema consequence: `candidate_index`, not one patch per round."""
    store, patches = stores
    client = fake_client(default_source="n=int(input())\nprint(n*3)\n", k_candidates=3)
    spec = policies.arm("km+steer+vs").cell("abc001_a/1001", 0)
    r = run_episode(spec, adapter=adapter, client=client, store=store,
                    patches=patches, cfg=EpisodeConfig(budget=4))
    by_round: dict[int, list] = {}
    for rec in r.records:
        by_round.setdefault(rec.round_index, []).append(rec)
    # Rounds 1–2 carry no VS, so they emit ONE row — that is what keeps the
    # CRN round-1 identity (and the round-1 cache key) intact against the
    # non-VS arms. From round 3 a draw yields k candidates, k rows, one key.
    assert [len(by_round[r]) for r in (1, 2)] == [1, 1]
    assert max(len(v) for v in by_round.values()) == 3
    for r, recs in by_round.items():
        assert len({rec.cache_key for rec in recs}) == 1
        expected = [1] if r < 3 else [1, 2, 3]
        assert sorted(rec.candidate_index for rec in recs) == expected
        # The budget counts ROUNDS, so exactly one row per round is charged.
        assert sum(rec.charged for rec in recs) == 1


def test_a_steered_arm_shows_facts_once_the_ledger_has_proven_probes(
        adapter, frozen, run_dir, stores, fake_client):
    store, patches = stores
    client = fake_client(default_source="n=int(input())\nprint(n*3)\n")
    spec = policies.arm("km+steer").cell("abc001_a/1001", 0)
    r = run_episode(spec, adapter=adapter, client=client, store=store,
                    patches=patches, cfg=EpisodeConfig(budget=4))
    assert r.records[0].k_facts == 0                 # round 1 carries no facts
    assert any(rec.k_facts > 0 for rec in r.records[1:])
    assert "must produce exactly this output" in client.prompts[-1]
    assert "must produce exactly this output" not in client.prompts[0]


def test_round_one_is_one_shared_draw_across_arms(
        adapter, frozen, run_dir, stores, fake_client):
    """Invariant 1, end to end: the steered and unsteered arms must issue the
    SAME nonce and the SAME prompt for round 1."""
    store, patches = stores
    seen = {}
    for arm_id in ("no-mem", "km", "km+steer", "km+steer-k2"):
        client = fake_client(default_source="n=int(input())\nprint(n*3)\n")
        run_episode(policies.arm(arm_id).cell("abc001_a/1001", 0),
                    adapter=adapter, client=client, store=store, patches=patches,
                    cfg=EpisodeConfig(budget=1))
        seen[arm_id] = (client.prompts[0], client.nonces[0])
    assert len(set(seen.values())) == 1


def test_an_episode_records_its_cell_by_construction(
        adapter, frozen, run_dir, stores, fake_client):
    """Every analysis filter reads fields the key constructor produced."""
    store, patches = stores
    client = fake_client(default_source="n=int(input())\nprint(n*3)\n")
    spec = policies.arm("km-cs-median").cell("abc001_a/1001", 1)
    r = run_episode(spec, adapter=adapter, client=client, store=store,
                    patches=patches, cfg=EpisodeConfig(budget=2))
    assert r.records[0].cell == spec.as_dict()
    assert r.records[0].cell["cold_start"] == "median"
    assert CellSpec(**r.records[0].cell).key() == spec.key()
