"""Aggregation, the cluster bootstrap, and the offline replay."""
from __future__ import annotations

import pytest

from caster.report import analyse, charts
from caster.report.records import RoundRecord


def row(task="t1", seed=0, policy="km", r=1, c=0, **kw) -> RoundRecord:
    kw.setdefault("cell", {"model": "qwen:7b"})
    return RoundRecord(run_id="x", episode_id=f"{task}|{seed}|{policy}", task=task,
                       seed=seed, round_index=r, policy=policy, candidate_index=c, **kw)


def test_cells_truncate_at_the_first_accept():
    rows = [row(r=1, killed_by="a", cpu_s=1.0, test_calls=3),
            row(r=2, accept=True, cpu_s=0.5, test_calls=2),
            row(r=3, killed_by="b", cpu_s=9.0, test_calls=9)]   # past the accept
    (c,) = analyse.cells(rows, budget=10)
    assert c.accepted and c.first_accept_round == 2
    assert c.cpu_s == pytest.approx(1.5)        # the 9.0 is truncated out
    assert c.test_calls == 5


def test_the_budget_truncates_a_force_full_budget_run():
    """"draw full, truncate at analysis", so the floor arm stays
    comparable to the arms it is tabulated against."""
    rows = [row(r=i, killed_by="a", cpu_s=1.0) for i in range(1, 9)]
    assert analyse.cells(rows, budget=4)[0].rounds_charged == 4
    assert analyse.cells(rows, budget=8)[0].rounds_charged == 8


def test_proposal_errors_are_split_by_arm_and_by_model():
    """The same task is a different difficulty for a 3B and a hosted
    model, so a pooled error rate hides the threat."""
    rows = [
        row(policy="km+steer", r=1, proposal_error="context_overflow",
            cell={"model": "qwen:3b"}),
        row(policy="km+steer", r=2, killed_by="a", cell={"model": "qwen:3b"}),
        row(policy="km", task="t2", r=1, killed_by="a", cell={"model": "qwen:7b"}),
    ]
    table = analyse.proposal_error_table(analyse.cells(rows, budget=5))
    by = {(t["policy"], t["model"]): t for t in table}
    assert by[("km+steer", "qwen:3b")]["context_overflow"] == 1
    assert by[("km+steer", "qwen:3b")]["error_rate"] == pytest.approx(0.5)
    assert by[("km+steer", "qwen:3b")]["threat"] is True     # above the ~20% line
    assert by[("km", "qwen:7b")]["error_rate"] == 0.0


def test_a_steered_arm_below_its_configured_k_is_flagged():
    """An arm below its configured k is reporting a different treatment."""
    rows = [row(policy="km+steer", r=1, patch_sha="s", k_facts=1),
            row(policy="km+steer", r=2, patch_sha="s", k_facts=1)]
    (entry,) = analyse.k_facts_table(analyse.cells(rows, budget=5))
    assert entry["configured_k"] == 2 and entry["realised_k_mean"] == 1.0
    assert entry["mislabelled"] is True


def test_repair_rate_comes_from_the_floor_arm_and_is_per_model():
    rows = [row(policy="no-mem", seed=0, accept=True),
            row(policy="no-mem", seed=1, killed_by="a"),
            row(policy="km", seed=0, accept=True)]          # must not contribute
    rates = analyse.repair_rates(analyse.cells(rows, budget=5))
    assert rates == {"t1|qwen:7b": 0.5}
    assert analyse.band_of(0.0) == "dead"
    assert analyse.band_of(0.5) == "mid"
    assert analyse.band_of(1.0) == "trivial"


def test_the_decomposition_is_paired_per_cell():
    """Under CRN both arms of a step saw the identical patch sequence on the
    identical cell, so the per-cell difference is the estimator with the
    variance removed. Pooling the two means throws that away."""
    rows = []
    for i, task in enumerate(("t1", "t2")):
        rows.append(row(task=task, policy="kill-only", killed_by="a", cpu_s=10.0 + i))
        rows.append(row(task=task, policy="km", killed_by="a", cpu_s=8.0 + i))
    steps = {d["step"]: d for d in analyse.decomposition(analyse.cells(rows, budget=5))}
    km_step = steps["kill-only → km"]
    assert km_step["paired_cells"] == 2
    assert km_step["cpu_delta_mean_pct"] < 0        # km is cheaper
    assert steps["natural → cost-only"]["paired_cells"] == 0


def test_the_bootstrap_says_when_the_cost_table_was_not_resampled():
    """With a frozen cost table a bootstrap over cells measures only
    scan-order variation, and that omission flatters the result."""
    rows = [row(task=f"t{i}", policy="km", killed_by="a", cpu_s=float(i))
            for i in range(8)]
    cs = analyse.cells(rows, budget=5)
    out = analyse.bootstrap_cpu(cs, "km", iterations=200)
    assert out["clusters"] == 8
    assert out["ci95"][0] <= out["mean"] <= out["ci95"][1]
    assert out["cost_table_resampled"] is False
    assert "resample the cost table" in out["note"]

    import random

    seen_clusters: list[int] = []

    def replay(rng, drawn):
        # The cluster draw is handed in, so the interval covers BOTH sources.
        seen_clusters.append(len(drawn))
        return rng.random()

    resampled = analyse.bootstrap_cpu(cs, "km", iterations=50, replay=replay)
    assert resampled["cost_table_resampled"] is True
    assert seen_clusters and all(n == 8 for n in seen_clusters)


def test_the_bootstrap_clusters_on_the_task_not_the_row():
    """Several faults can share one test pool, and a bootstrap over rows would
    treat the rounds of one episode as independent."""
    rows = [row(task="t1", seed=s, policy="km", killed_by="a", cpu_s=1.0)
            for s in range(5)]
    out = analyse.bootstrap_cpu(analyse.cells(rows, budget=5), "km", iterations=50)
    assert out["clusters"] == 1 and out["n"] == 5


def test_survival_is_censored_at_b():
    rows = [row(task="t1", policy="km", r=3, accept=True),
            row(task="t2", policy="km", r=1, killed_by="a")]
    curve = analyse.km_survival(analyse.cells(rows, budget=5), "km", 5)
    assert curve == [0.0, 0.0, 0.5, 0.5, 0.5]


def test_offline_replay_costs_nothing_and_drives_the_real_ledger():
    """The full kill vector is the only way to replay a counterfactual
    policy without new model calls, and the policy replayed has to be the REAL
    one: `km`, `fault-record` and the rest are ledger-driven, so a stateless
    `order_fn` could not express them at all."""
    kill = {f"p{i}": {"a": "pass", "b": "wrong", "c": "pass"} for i in range(1, 6)}
    cost = {f"p{i}": {"a": 1.0, "b": 0.1, "c": 1.0} for i in range(1, 6)}
    seq = [f"p{i}" for i in range(1, 6)]

    natural = analyse.replay_policy(kill, cost, seq, policy="natural")
    km = analyse.replay_policy(kill, cost, seq, policy="km", explore_p=0.0)

    # `natural` pays for `a` on every patch; `km` learns `b` and stops paying.
    assert natural["test_calls"] == 10
    assert km["test_calls"] < natural["test_calls"]
    assert km["cpu_s"] < natural["cpu_s"]
    assert km["mean_rank_of_firing_probe"] < natural["mean_rank_of_firing_probe"]
    assert km["accepted"] == natural["accepted"] == 0
    assert km["rank1_share"] > 0


def test_replay_marks_a_patch_nothing_refutes_as_accepted():
    kill = {"p1": {"a": "pass", "b": "pass"}}
    out = analyse.replay_policy(kill, {"p1": {"a": 1.0, "b": 1.0}}, ["p1"],
                                policy="natural")
    assert out["accepted"] == 1 and out["mean_rank_of_firing_probe"] is None


def test_both_priors_replay_over_one_matrix_with_no_model_calls():
    """The prior question, settled offline: no model calls, no sandbox runs."""
    kill = {f"p{i}": {f"c{j}": ("wrong" if j == 7 else "pass") for j in range(12)}
            for i in range(8)}
    cost = {f"p{i}": {f"c{j}": 0.1 for j in range(12)} for i in range(8)}
    rows = analyse.compare_priors(kill, cost, {"t1": list(kill)}, policy="km")
    by = {r["prior"]: r for r in rows}
    assert set(by) == {"shipped", "jeffreys"}
    assert by["shipped"]["beta"] == (0.5, 2.0)
    assert by["jeffreys"]["beta"] == (0.5, 0.5)
    for r in rows:
        assert r["mean_rank_of_firing_probe"] is not None
        assert r["probes_per_refuted_round"] is not None
    # Under homogeneous costs, Jeffreys' K=0.5 for an unseen probe outranks a
    # proven refuter whose record has decayed — the full-pool sweep.
    assert (by["jeffreys"]["probes_per_refuted_round"]
            > by["shipped"]["probes_per_refuted_round"])


def test_summarise_puts_the_threats_first():
    rows = [row(policy="no-mem", killed_by="a", cpu_s=1.0),
            row(policy="km", killed_by="a", cpu_s=0.5)]
    doc = analyse.summarise(rows, budget=4)
    assert list(doc)[:5] == ["budget", "cells", "proposal_errors", "k_facts",
                             "repair_rates"]


def test_an_unknown_arm_in_a_merged_log_does_not_crash_the_analysis():
    rows = [row(policy="a-retired-arm", killed_by="a", cpu_s=1.0)]
    doc = analyse.summarise(rows, budget=4)
    assert doc["cells"] == 1
    assert "a-retired-arm" in doc["survival"]


def test_the_repair_rate_is_frozen_then_read_never_recomputed(tmp_path):
    """It is the denominator every band analysis uses, and a denominator
    that moves with the merge is not one."""
    rows = [row(policy="no-mem", seed=0, accept=True),
            row(policy="no-mem", seed=1, killed_by="a")]

    # An INCOMPLETE floor arm does not get to pick the denominator. `GAPS` is a
    # non-fatal audit, so `caster merge` succeeds on a partial grid, and a rate
    # frozen from one shard never corrects itself.
    partial = analyse.summarise(rows, budget=4, pool_dir=tmp_path,
                                universe=["t1", "t2"])
    assert partial["repair_rates_source"] == "computed-not-frozen"
    assert partial["floor_arm_complete"] is False
    assert not (tmp_path / "repair_rates.json").exists()

    first = analyse.summarise(rows, budget=4, pool_dir=tmp_path, universe=["t1"])
    assert first["repair_rates"] == {"t1|qwen:7b": 0.5}
    assert first["repair_rates_source"] == "computed-and-frozen"
    assert (tmp_path / "repair_rates.json").is_file()

    # A later merge that would compute a DIFFERENT rate must not move it.
    more = rows + [row(policy="no-mem", seed=2, accept=True),
                   row(policy="no-mem", seed=3, accept=True)]
    second = analyse.summarise(more, budget=4, pool_dir=tmp_path, universe=["t1"])
    assert second["repair_rates"] == {"t1|qwen:7b": 0.5}
    assert second["repair_rates_source"] == "frozen"


def test_the_report_renders_without_a_network_or_a_build_step(tmp_path):
    rows = [row(policy="km", killed_by="a", cpu_s=0.5, test_calls=2,
                sandbox_runs=2, verdict_hits=0)]
    html = charts.render(analyse.summarise(rows, budget=4))
    assert "<script src=" not in html and "cdn" not in html.lower()
    assert "prefers-color-scheme" in html
    out = charts.write(analyse.summarise(rows, budget=4), tmp_path / "r.html")
    assert out.read_text().startswith("<!doctype html>")


def test_a_duplicate_patch_is_measurable_today():
    """RQ7's measurable half: distinct `patch_sha` per episode, no new arm."""
    rows = [row(r=1, patch_sha="a", killed_by="x"),
            row(r=2, patch_sha="a", killed_by="x"),
            row(r=3, patch_sha="b", killed_by="x")]
    (c,) = analyse.cells(rows, budget=5)
    assert c.patches == 3 and c.distinct_patches == 2
    assert c.duplicate_rate == pytest.approx(1 / 3)


# ── the pool-size screen, moved after the run ────────────────────────────────


def test_the_standard_band_is_exactly_the_old_min_pool_20_corpus():
    """The boundary has to land on 20 or the band does not reproduce the gate."""
    assert analyse.pool_band_of(1) == "degenerate"
    assert analyse.pool_band_of(2) == "tiny"
    assert analyse.pool_band_of(19) == "small"
    assert analyse.pool_band_of(20) == "standard"
    assert analyse.pool_band_of(148) == "standard"


def test_arms_by_pool_band_reproduces_the_screen_without_rerunning(tmp_path):
    """What the gate used to decide up front, computed afterwards from the same
    cells — and the control-arm faults stay out of it, because pooling them here
    would put back the correlation the control arm exists to measure."""
    import json

    sizes = {"big": 40, "small": 12, "one": 1}
    (tmp_path / "tasks.json").write_text(json.dumps({"tasks": {
        "big": {"admitted": True, "p_loop": ["p"] * 40},
        "small": {"admitted": True, "p_loop": ["p"] * 12},
        "one": {"admitted": True, "p_loop": ["p"]},
        "ctl": {"admitted": False, "control": True, "p_loop": ["p"] * 9},
    }}), encoding="utf-8")
    assert analyse.load_pool_sizes(tmp_path) == sizes

    rows = [row(task=t, killed_by="a", cpu_s=1.0)
            for t in ("big", "small", "one", "ctl")]
    cs = analyse.cells(rows, budget=4)
    out = analyse.arms_by_pool_band(cs, sizes)
    assert set(out) == {"degenerate", "small", "standard"}
    assert out["standard"]["tasks"] == 1 and out["standard"]["pool_median"] == 40
    # The control-arm cell is not in the frozen pool and must not be banded.
    assert sum(d["cells"] for d in out.values()) == 3
