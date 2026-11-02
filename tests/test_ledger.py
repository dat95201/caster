"""K, m̂, the two priors, the asymmetric cold start, and `explore_p`."""
from __future__ import annotations

import random

import pytest

from caster.exec.ledger import CaseLedger, ILLEGIBLE_PENALTY, SHOW_INPUT_CHARS
from caster.keys import PRIORS


def make(names=("a", "b", "c", "d"), **kw) -> CaseLedger:
    kw.setdefault("rng", random.Random(0))
    kw.setdefault("explore_p", 0.0)
    return CaseLedger(names, **kw)


def test_shipped_prior_keeps_an_unseen_probe_below_a_proven_one():
    """The whole argument, as an assertion.

    `Beta(½,2)` puts an unseen probe at 0.2, which sits above this corpus's base
    rate (≈0.03) without reaching ½. Jeffreys puts it at 0.5, which outranks a
    proven refuter at k=2,n=10 — under homogeneous costs that is a full-pool
    sweep, the failure names.
    """
    shipped = make(prior="shipped")
    jeffreys = make(prior="jeffreys")
    for led in (shipped, jeffreys):
        for _ in range(6):
            led.update("a", refuted=False, cpu_s=0.1)
        for _ in range(4):
            led.update("a", refuted=True, cpu_s=0.1)   # a solid 4-in-10 refuter

    assert PRIORS["shipped"] == (0.5, 2.0)
    assert PRIORS["jeffreys"] == (0.5, 0.5)
    assert shipped.K("b") == pytest.approx(0.2)        # prior mean, above ≈0.03
    assert jeffreys.K("b") == pytest.approx(0.5)

    assert shipped.K("a") > shipped.K("b")             # the proven probe stays ahead
    assert jeffreys.K("a") < jeffreys.K("b")           # ... and here it does not

    # The spec's own worked example: k=2, n=10 gives Jeffreys K=0.227 against an
    # unseen 0.5. The ratio is what decides whether the pool gets swept.
    for led in (shipped, jeffreys):
        for name in ("c",):
            for _ in range(8):
                led.update(name, refuted=False, cpu_s=0.1)
            for _ in range(2):
                led.update(name, refuted=True, cpu_s=0.1)
    assert jeffreys.K("b") / jeffreys.K("c") > shipped.K("b") / shipped.K("c")


def test_cold_start_is_pessimistic_and_asymmetric():
    """`K` optimistic for an unseen probe, `m̄` pessimistic. Both at once empties
    the pool in one round."""
    led = make(cold_start="max")
    led.update("a", refuted=True, cpu_s=0.01)      # a cheap, proven killer
    led.update("b", refuted=False, cpu_s=2.00)     # an expensive survivor
    assert led.m_bar() == pytest.approx(2.00)      # max, not median
    assert led.m_hat("c") == pytest.approx(2.00)   # the unseen probe looks expensive
    assert led.schedule()[0] == "a"                # so the proven killer stays first


def test_median_cold_start_is_the_v3_defect():
    """The median makes an UNSEEN probe look cheap and lets it outrank a
    proven refuter whenever `m_killer > 1.5 × m̄`. Costs a full-pool sweep.

    The comparison is between the proven refuter `a` and the never-run `c`:
    `b`/`d` are observed and cheap, so they legitimately come first either way —
    running a 0.01 s probe with a 14% chance of refuting IS the right call. What
    must not happen is an unmeasured probe inheriting their cheapness.
    """
    cheap = make(cold_start="median")
    pess = make(cold_start="max")
    for led in (cheap, pess):
        led.update("a", refuted=True, cpu_s=3.0)   # the proven refuter is EXPENSIVE
        led.update("b", refuted=False, cpu_s=0.01)
        led.update("d", refuted=False, cpu_s=0.02)
    assert cheap.schedule().index("c") < cheap.schedule().index("a")
    assert pess.schedule().index("c") > pess.schedule().index("a")


def test_quantile_cold_starts_are_ordered():
    led = make()
    for name, cpu in (("a", 0.1), ("b", 1.0), ("c", 5.0)):
        led.update(name, refuted=False, cpu_s=cpu)
    med = CaseLedger(["a", "b", "c"], cold_start="median")
    q75 = CaseLedger(["a", "b", "c"], cold_start="q75")
    mx = CaseLedger(["a", "b", "c"], cold_start="max")
    for led2 in (med, q75, mx):
        for name, cpu in (("a", 0.1), ("b", 1.0), ("c", 5.0)):
            led2.update(name, refuted=False, cpu_s=cpu)
    assert med.m_bar() <= q75.m_bar() <= mx.m_bar()
    assert mx.m_bar() == pytest.approx(5.0)
    assert med.m_bar() == pytest.approx(1.0)


def test_schedule_divides_by_cost():
    """The gap `docs/ORACLE-DIMENSIONS.md` names: kill rate alone is the wrong
    criterion when per-probe cost varies by 27–196×."""
    led = make()
    for _ in range(4):
        led.update("a", refuted=True, cpu_s=5.0)    # kills often, very expensive
        led.update("b", refuted=True, cpu_s=0.05)   # kills as often, cheap
    assert led.K("a") == pytest.approx(led.K("b"))
    assert led.schedule()[0] == "b"


def test_update_is_o1_and_accumulates():
    led = make()
    led.update("a", refuted=True, cpu_s=0.5, round_index=3)
    led.update("a", refuted=False, cpu_s=1.5, round_index=4)
    st = led.stats["a"]
    assert (st.n, st.k, st.s) == (2, 1, 2.0)
    assert st.first_kill_round == 3 and st.last_kill_round == 3
    assert led.M("a") == pytest.approx(1.0)


def test_first_kill_round_is_discovery_order():
    """`fault-record` ranks on it; without it the arm silently becomes
    `kill-only` and RQ2's decomposition loses a step."""
    led = make()
    led.update("b", refuted=True, cpu_s=0.1, round_index=2)
    led.update("a", refuted=True, cpu_s=0.1, round_index=5)
    for _ in range(5):
        led.update("a", refuted=True, cpu_s=0.1, round_index=6)
    assert led.stats["b"].first_kill_round < led.stats["a"].first_kill_round
    assert led.K("a") > led.K("b")      # ... even though `a` now kills more


def test_explore_p_promotes_exactly_one_unseen_probe():
    """Without it, a task whose first probe kills everything reaches the
    prompt with ONE fact, k=2 degenerates to k=1, and the k sweep compares k=1
    against k=1."""
    always = make(explore_p=1.0, rng=random.Random(1))
    never = make(explore_p=0.0, rng=random.Random(1))
    for led in (always, never):
        led.update("a", refuted=True, cpu_s=0.01)
    assert never.schedule()[0] == "a"
    promoted = always.schedule()
    assert promoted[0] != "a" and promoted[1] == "a"
    assert sorted(promoted) == sorted(never.schedule())   # a permutation, nothing dropped


def test_explore_p_is_a_no_op_without_a_proven_probe():
    led = make(explore_p=1.0)
    assert led.schedule() == sorted(["a", "b", "c", "d"], key=led.tie)


def test_schedule_is_reproducible_and_a_permutation():
    names = [f"p{i}" for i in range(30)]
    a = CaseLedger(names, rng=random.Random(5), explore_p=0.15,
                   tie_key=lambda n: f"h{hash(n) % 97:02d}")
    b = CaseLedger(names, rng=random.Random(5), explore_p=0.15,
                   tie_key=lambda n: f"h{hash(n) % 97:02d}")
    for led in (a, b):
        led.update("p3", refuted=True, cpu_s=0.2)
    for _ in range(5):
        # One call each per iteration: `explore_p` consumes a draw, so calling
        # `a.schedule()` twice would desync the two RNGs and the test would be
        # measuring its own bookkeeping.
        sa, sb = a.schedule(), b.schedule()
        assert sa == sb
        assert sorted(sa) == sorted(names)


def test_evidence_ranks_by_legibility_but_reports_raw_k():
    """Prompt rule 4 — a different cost currency, and the number in the prompt is a
    claim about the world."""
    led = CaseLedger(["big", "small"], explore_p=0.0,
                     input_chars={"big": SHOW_INPUT_CHARS + 1, "small": 10})
    for _ in range(10):
        led.update("big", refuted=True, cpu_s=0.01)
    for _ in range(10):
        led.update("small", refuted=True, cpu_s=0.01)
    for _ in range(8):
        led.update("small", refuted=False, cpu_s=0.01)

    assert led.K("big") > led.K("small")                 # `big` kills more
    assert led.legibility("big") == ILLEGIBLE_PENALTY
    ranked = [n for n, _ in led.evidence(2)]
    assert ranked[0] == "small"                          # but `small` is showable
    reported = dict(led.evidence(2))
    assert reported["big"] == pytest.approx(led.K("big"))   # RAW K, not the score


def test_evidence_shows_only_proven_probes():
    """Prompt rule 5 — a probe with a prior and no record is a guess, and a wrong
    guess primes (2.7×)."""
    led = make()
    assert led.evidence(2) == []
    led.update("c", refuted=True, cpu_s=0.1)
    assert [n for n, _ in led.evidence(2)] == ["c"]


def test_a_dominant_illegible_killer_still_wins():
    led = CaseLedger(["big", "small"], explore_p=0.0,
                     input_chars={"big": 10_000, "small": 10})
    for _ in range(20):
        led.update("big", refuted=True, cpu_s=0.01)
    led.update("small", refuted=True, cpu_s=0.01)
    for _ in range(40):
        led.update("small", refuted=False, cpu_s=0.01)
    assert [n for n, _ in led.evidence(1)] == ["big"]


def test_unusable_probes_leave_the_schedule_not_the_record():
    led = make()
    led.mark_unusable("b")
    assert "b" not in led.schedule()
    assert "b" in led.snapshot()
    assert led.snapshot()["b"]["unusable"] is True


def test_seed_from_is_not_used_by_the_grid():
    """It exists for a separate, reportable experiment: a prior learned on other
    episodes of the same task breaks the independence CRN pairing assumes."""
    led = make()
    led.seed_from({"a": (10, 3, 1.0)})
    assert (led.stats["a"].n, led.stats["a"].k) == (10, 3)
    import pathlib

    src = pathlib.Path(__file__).resolve().parent.parent / "caster"
    callers = [p for p in src.rglob("*.py") if "seed_from(" in p.read_text(encoding="utf-8")]
    assert [p.name for p in callers] == ["ledger.py"]


def test_bad_configuration_is_refused():
    with pytest.raises(ValueError):
        CaseLedger(["a"], prior="nope")
    with pytest.raises(ValueError):
        CaseLedger(["a"], cold_start="nope")
