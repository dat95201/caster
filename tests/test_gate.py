"""The corpus gate, and the evaluation split."""
from __future__ import annotations

import pytest

from caster.bench.gate import (
    GateConfig,
    _split_stratified,
    freeze,
    gate_one,
    install,
    score_on_eval,
)


def test_a_sound_fault_is_admitted(adapter):
    t = gate_one(adapter, "abc001_a/1001", GateConfig(min_loop_pool=6, date_min="2000-01-01"))
    assert t.admitted, t.reason
    assert t.gates["G1"] and t.gates["G2"] and t.gates["G3"]
    assert t.f2p and t.p2p, "the fixture must produce both halves"
    assert set(t.p_loop) | set(t.p_eval) == set(t.f2p) | set(t.p2p)
    assert not (set(t.p_loop) & set(t.p_eval))


def test_g3_counts_the_LOOP_half_not_the_pool(adapter):
    """Applying the threshold to the pool and THEN splitting 30% off admits
    tasks the mechanism provably cannot act on — the exact thing the threshold
    was chosen to exclude."""
    pool = len(adapter.probes("abc001_a/1001"))
    loose = gate_one(adapter, "abc001_a/1001",
                     GateConfig(min_loop_pool=pool - 2, date_min="2000-01-01"))
    assert not loose.admitted and loose.failed_gate == "G3"
    assert "|P_loop|" in loose.reason


def test_the_split_is_stratified_within_f2p_and_p2p():
    """A uniform split breaks in both directions: no F2P in P_loop and round 1
    accepts the unmodified program; no F2P in P_eval and a non-repair is
    undetectable."""
    f2p = [f"f{i}" for i in range(10)]
    p2p = [f"p{i}" for i in range(10)]
    loop, held = _split_stratified(f2p, p2p, ratio=0.3, seed_text="t")
    assert set(loop) & set(f2p) and set(held) & set(f2p)
    assert set(loop) & set(p2p) and set(held) & set(p2p)
    assert set(loop) | set(held) == set(f2p) | set(p2p)
    # Seeded from the task name: same name, same split, every time.
    assert (loop, held) == _split_stratified(f2p, p2p, ratio=0.3, seed_text="t")
    assert (loop, held) != _split_stratified(f2p, p2p, ratio=0.3, seed_text="u")


def test_a_stratum_is_never_emptied():
    loop, held = _split_stratified(["only"], ["a", "b"], ratio=0.9, seed_text="t")
    assert "only" in loop


def test_g5_drops_a_contaminated_contest(adapter):
    t = gate_one(adapter, "abc001_a/1001",
                 GateConfig(min_loop_pool=6, date_min="2030-01-01"))
    assert not t.admitted and t.failed_gate == "G5"


def test_g6_drops_a_program_too_long_for_a_prompt(adapter):
    t = gate_one(adapter, "abc001_a/1001",
                 GateConfig(min_loop_pool=6, date_min="2000-01-01", lines_max=1))
    assert not t.admitted and t.failed_gate == "G6"


def test_freeze_asserts_it_dropped_nothing_of_its_own(adapter):
    doc = freeze(adapter, cfg=GateConfig(min_loop_pool=6, date_min="2000-01-01"))
    assert doc["funnel"]["examined"] == len(doc["tasks"])
    assert doc["funnel"]["admitted"] == len(doc["pool_order"])
    assert doc["pool_sha256"]
    assert doc["candidate_order_sha256"] and doc["selected_sha256"]


def test_the_pool_order_is_a_seeded_interleave(adapter):
    """ConDefects' own order is contest order, correlated with date,
    rating and therefore pool size. A killed fleet cutting contiguous ranges
    from it would leave a SYSTEMATICALLY biased subset, not a smaller grid."""
    a = freeze(adapter, cfg=GateConfig(min_loop_pool=6, date_min="2000-01-01", seed=1))
    b = freeze(adapter, cfg=GateConfig(min_loop_pool=6, date_min="2000-01-01", seed=2))
    contest = freeze(adapter, cfg=GateConfig(min_loop_pool=6, date_min="2000-01-01",
                                             pool_order="contest"))
    assert sorted(a["pool_order"]) == sorted(contest["pool_order"])
    assert contest["pool_order"] == sorted(contest["pool_order"])
    # The ordering is stamped into the digest, so it cannot change silently.
    assert a["pool_sha256"] != contest["pool_sha256"]
    assert isinstance(b["pool_order"], list)


def test_the_funnel_names_the_first_gate_each_fault_failed(adapter):
    doc = freeze(adapter, cfg=GateConfig(min_loop_pool=1000, date_min="2000-01-01"))
    assert doc["funnel"]["admitted"] == 0
    assert doc["funnel"]["rejected_by_gate"]["G3"] == doc["funnel"]["examined"]
    for t in doc["tasks"].values():
        assert t["failed_gate"] == "G3" and t["reason"]


def test_the_unfiltered_control_arm_is_drawn_over_g1_survivors(adapter):
    """The only thing that makes the scope condition checkable rather than
    asserted."""
    doc = freeze(adapter, cfg=GateConfig(min_loop_pool=1000, date_min="2000-01-01",
                                         control_size=2))
    controls = [n for n, t in doc["tasks"].items() if t["control"]]
    assert controls
    for n in controls:
        assert doc["tasks"][n]["gates"].get("G1") is True
        assert not doc["tasks"][n]["admitted"]


def test_install_narrows_the_pool_and_holds_p_eval_out(adapter):
    doc = freeze(adapter, cfg=GateConfig(min_loop_pool=6, date_min="2000-01-01"))
    install(adapter, doc)
    task = doc["pool_order"][0]
    probes = set(adapter.probes(task))
    assert probes == set(doc["tasks"][task]["p_loop"])
    for p in doc["tasks"][task]["p_eval"]:
        assert adapter.authority(task, p) == "held_out"
    for p in probes:
        assert adapter.authority(task, p) == "visible"


def test_scoring_happens_on_p_eval_only(adapter, frozen):
    """`fix_rate` on F2P, `regression_rate` on P2P, `truly_correct` only when
    nothing scoreable fails."""
    task = "abc001_a/1001"
    good = score_on_eval(adapter, task, "n=int(input())\nprint(n*2)\n", frozen)
    assert good.truly_correct and good.fix_rate == 1.0
    assert good.regression_rate in (0.0, None)

    # A patch that repairs the fault and trades away already-working behaviour.
    traded = score_on_eval(adapter, task,
                           "n=int(input())\nprint(n*2 if n>=3 else n*9)\n", frozen)
    assert not traded.truly_correct
    assert traded.regression_rate and traded.regression_rate > 0


def test_regression_rate_is_none_rather_than_zero_when_nothing_is_scoreable(adapter, frozen):
    """A fault whose footprint is the whole pool cannot regress, and 0.0 would
    claim a measurement never made."""
    from caster.bench.gate import RegressionResult

    r = RegressionResult(3, 3, 0, 0, (), (), True, False)
    assert r.regression_rate is None and r.fix_rate == 1.0


def test_spec_cases_have_no_unsafe_fallback(adapter):
    """Falling back to "the two smallest inputs" discloses a hidden test on 19
    of 1,422 problems. No fallback means safe by construction."""
    cases = adapter.spec_cases("abc001_a/1001")
    assert cases and all("sample" in c.name or "example" in c.name for c in cases)

    class NoSamples(type(adapter)):
        def _cases_for_group(self, task_id):      # noqa: ANN001, ANN202
            return tuple(c for c in super()._cases_for_group(task_id)
                         if "sample" not in c.name and "example" not in c.name)

    stripped = NoSamples(root=adapter.root, timeout=adapter.timeout)
    assert stripped.spec_cases("abc001_a/1001") == ()


def test_g4_keeps_one_fault_per_coding_task_by_a_seeded_uniform_draw(adapter, corpus_root):
    """Several submissions share ONE test pool, so their cells are correlated and
    the cluster bootstrap's independence assumption fails.

    The draw is deliberately NOT "the fault with the largest K-proxy": selecting
    on the same statistic the analysis reports makes the pool's distribution an
    artefact of the dedup rule.
    """
    from conftest import CORRECT, FAULTY, _write
    from caster.bench.adapter import ConDefectsAdapter

    sibling = corpus_root / "Code" / "abc001_a" / "Python" / "1002"
    _write(sibling / "faultyVersion.py",
           FAULTY.replace("n * 3", "n * 4"))
    _write(sibling / "correctVersion.py", CORRECT)
    fresh = ConDefectsAdapter(root=corpus_root, timeout=10.0)
    assert len(fresh.tasks()) == 3

    doc = freeze(fresh, cfg=GateConfig(min_loop_pool=6, date_min="2000-01-01"))
    groups = [doc["tasks"][n]["group"] for n in doc["pool_order"]]
    assert len(groups) == len(set(groups))
    dropped = [t for t in doc["tasks"].values() if t["failed_gate"] == "G4"]
    assert len(dropped) == 1 and "G4 dedup" in dropped[0]["reason"]

    sibling_dir = sibling
    for f in sibling_dir.iterdir():
        f.unlink()
    sibling_dir.rmdir()


def test_a_case_the_sandbox_cannot_run_is_named_as_such(tmp_path):
    """The reason is a CAPABILITY fact, and its name has to say so.

    `exceeds_sandbox_timeout` means the sandbox, at its configured
    `SANDBOX_TIMEOUT_SEC`, cannot run this case — the same limit every arm runs
    under, and part of `protocol_sha`. It is not a verdict that the case is too
    expensive to be worth measuring: there is only ever one clock, and a second
    stricter one would cut exactly the expensive-oracle faults the mechanism is
    for. A reason named after cost would invite that filter back.
    """
    from conftest import _write, build_corpus
    from caster.bench.adapter import ConDefectsAdapter

    root = build_corpus(tmp_path / "beyond")
    code = root / "Code" / "abc003_c" / "Python" / "3003"
    stall = ("import time\n"
             "n = int(input())\n"
             "if n == 99:\n"
             "    time.sleep(5)\n"
             "print(n * 2)\n")
    _write(code / "correctVersion.py", stall)
    _write(code / "faultyVersion.py", stall.replace("n * 2", "n * 3"))
    _write(code / "faultLocation.txt", "5\n")
    test = root / "Test" / "abc003" / "C"
    for i in range(8):
        _write(test / "in" / f"01_random_{i:02d}.txt", f"{i + 1}\n")
        _write(test / "out" / f"01_random_{i:02d}.txt", f"{(i + 1) * 2}\n")
    _write(test / "in" / "02_big_00.txt", "99\n")
    _write(test / "out" / "02_big_00.txt", "198\n")
    _write(root / "date.txt", "abc001_a 2023-05-01\nabc002_b 2023-06-01\n"
                              "abc003_c 2023-07-01\n")

    ad = ConDefectsAdapter(root=root, timeout=0.5)
    t = gate_one(ad, "abc003_c/3003",
                 GateConfig(min_loop_pool=4, date_min="2000-01-01"))
    assert t.admitted, t.reason
    assert t.unusable.get("02_big_00.txt") == "exceeds_sandbox_timeout"
    assert "02_big_00.txt" not in t.pool
    assert "slow" not in set(t.unusable.values()), "a cost-shaped name is the bug"


# ── the pool-size screen moved out of the gate ───────────────────────────────


def test_the_gate_no_longer_screens_on_pool_size(adapter):
    """A pre-treatment screen is the one decision that cannot be undone after
    the run. `|P_loop| >= 20` used to delete 23.2% of the corpus before any cell
    existed; now the grid runs every sound fault and `analyse` bands by pool
    size afterwards, where the cut can be redrawn without re-running anything.
    """
    t = gate_one(adapter, "abc001_a/1001", GateConfig(date_min="2000-01-01"))
    assert t.admitted, t.reason
    assert len(t.p_loop) < 20, "the fixture pool must be small enough to matter"
    assert t.gates["G3"]


def test_the_default_floor_is_one_probe_not_a_disabled_check():
    """1, not 0. Zero would switch the comparison off through a falsy sentinel;
    what changed is the THRESHOLD, not whether it is evaluated. At 1 the check
    states the real floor — an empty `P_loop` is a task no policy can schedule
    — and it stays a live assertion if the two F2P checks are ever weakened."""
    import inspect

    from caster.bench import gate as gate_mod

    assert gate_mod.MIN_LOOP_POOL == 1
    assert GateConfig().min_loop_pool == 1
    src = inspect.getsource(gate_mod.gate_one)
    assert "cfg.min_loop_pool and len(p_loop)" not in src, (
        "the check must not be guarded by truthiness")
    assert "if len(p_loop) < cfg.min_loop_pool:" in src


def test_min_pool_still_re_imposes_the_old_screen(adapter):
    """Reproducibility of the old corpus is the point of keeping the flag."""
    t = gate_one(adapter, "abc001_a/1001",
                 GateConfig(min_loop_pool=20, date_min="2000-01-01"))
    assert not t.admitted and t.failed_gate == "G3"
    assert "|P_loop|" in t.reason


def test_g3_still_puts_a_floor_under_the_pool(adapter, tmp_path):
    """Size is gone; the two SPLIT checks are not, and must not be.

    A `P_loop` with no F2P case makes the fault invisible to the loop, so round
    1 would accept the unmodified faulty program. A `P_eval` with no F2P case
    makes a non-repair undetectable at scoring time. Together they still demand
    two F2P cases, one per half — a floor that is not a difficulty proxy.
    """
    from conftest import _write, build_corpus
    from caster.bench.adapter import ConDefectsAdapter

    root = build_corpus(tmp_path / "onef2p")
    code = root / "Code" / "abc004_d" / "Python" / "4004"
    _write(code / "correctVersion.py", "n = int(input())\nprint(n * 2)\n")
    # Wrong on exactly ONE case (n == 7), so F2P has a single member and the
    # 70/30 split cannot put one in each half.
    _write(code / "faultyVersion.py",
           "n = int(input())\nprint(99 if n == 7 else n * 2)\n")
    _write(code / "faultLocation.txt", "2\n")
    test = root / "Test" / "abc004" / "D"
    for i in range(8):
        _write(test / "in" / f"01_random_{i:02d}.txt", f"{i + 1}\n")
        _write(test / "out" / f"01_random_{i:02d}.txt", f"{(i + 1) * 2}\n")
    _write(root / "date.txt", "abc001_a 2023-05-01\nabc002_b 2023-06-01\n"
                              "abc004_d 2023-07-01\n")

    t = gate_one(ConDefectsAdapter(root=root, timeout=10.0), "abc004_d/4004",
                 GateConfig(date_min="2000-01-01"))
    assert not t.admitted and t.failed_gate == "G3"
    assert "F2P" in t.reason
