"""The three identity keys, and the two exclusions that cost money."""
from __future__ import annotations

import dataclasses

import pytest

from caster.keys import (
    CellSpec,
    KEY_FLAGS,
    NONKEY_FLAGS,
    aux_nonce,
    cache_key,
    check_flag_partition,
    draw_nonce,
    schedule_tie_break,
)


def test_as_dict_and_key_read_the_same_fields():
    """The rule binds CONSUMERS, not just producers.

    A frozen summary whose fields are a hand-maintained list loses a whole arm
    into the wrong condition the first time one is missed. Here both the key and
    the summary walk `dataclasses.fields`, so they cannot drift.
    """
    spec = CellSpec(task="t", seed=1, policy="km")
    names = {f.name for f in dataclasses.fields(spec)}
    assert set(spec.as_dict()) == names
    for name in names:
        assert f"{name}=" in spec.key()


def test_budget_is_not_a_field():
    """A B-round episode is the prefix of a longer one.

    Putting `budget` in the key gave one cell two episode ids, so topping a cell
    up from 12 rounds to 20 appended a SECOND episode and every round-averaged
    estimator counted the first 12 draws twice.
    """
    assert "budget" not in {f.name for f in dataclasses.fields(CellSpec)}
    with pytest.raises(TypeError):
        CellSpec(task="t", seed=1, policy="km", budget=12)  # type: ignore[call-arg]


def test_retry_caps_are_not_in_the_key():
    """Raising a giving-up cap extends an episode; it never changes a draw."""
    fields = {f.name for f in dataclasses.fields(CellSpec)}
    assert "max_consecutive_backend_failures" not in fields
    assert "max_consecutive_backend_failures" in NONKEY_FLAGS


def test_every_knob_changes_the_key():
    base = CellSpec(task="t", seed=1, policy="km")
    moved = {
        "prior": "jeffreys", "cold_start": "median", "explore_p": 0.0,
        "tie_break": "name", "steer_on": True, "k_facts": 3,
        "disclosure": "sample-only", "show_kill_rate": False, "k_candidates": 5,
        "vs_schedule": True, "candidate_policy": "explore", "passing_block": True,
        "steer_mode": "exclusion", "transcript_cap_tokens": 4000,
        "transcript_restart_after": 9, "model": "m", "temperature": 0.2,
        "reasoning_effort": "high", "task": "u", "seed": 2, "policy": "natural",
    }
    for field, value in moved.items():
        kw = {field: value}
        if field == "reasoning_effort":
            kw["model"] = "o4-mini"
        assert base.replace(**kw).key() != base.key(), field


def test_episode_id_is_a_prefix_of_the_cell_key_hash():
    spec = CellSpec(task="t", seed=1, policy="km")
    assert len(spec.episode_id) == 12
    assert spec.episode_id == CellSpec(task="t", seed=1, policy="km").episode_id


def test_nonce_is_not_a_function_of_policy():
    """Invariant 1. An engineer who derives the nonce from `cell_key` — which
    DOES contain the policy — destroys CRN silently and buys every completion
    once per arm."""
    assert draw_nonce("abc_a/1", 0, 1) == draw_nonce("abc_a/1", 0, 1)
    assert "km" not in draw_nonce("abc_a/1", 0, 1)
    assert draw_nonce("abc_a/1", 0, 1) != draw_nonce("abc_a/1", 0, 2)
    assert draw_nonce("abc_a/1", 0, 1) != draw_nonce("abc_a/1", 1, 1)


def test_aux_nonce_is_namespaced():
    """No value that SELECTS may be a value that is REPORTED."""
    assert aux_nonce("pi-pilot", "t", 3).startswith("pi-pilot|")
    with pytest.raises(ValueError):
        aux_nonce("", "t")
    with pytest.raises(ValueError):
        aux_nonce("a|b", "t")


def test_cache_key_appends_effort_only_when_set():
    """An always-present empty field would re-key every chat-model response."""
    a = cache_key("p", "gpt-x", 1.0, 100, "n")
    b = cache_key("p", "gpt-x", 1.0, 100, "n", "")
    c = cache_key("p", "o4-mini", 1.0, 100, "n", "high")
    assert a == b
    assert c not in (a, b)


def test_cache_key_moves_with_every_input():
    base = ("p", "m", 1.0, 100, "n")
    assert cache_key(*base) != cache_key("p2", "m", 1.0, 100, "n")
    assert cache_key(*base) != cache_key("p", "m2", 1.0, 100, "n")
    assert cache_key(*base) != cache_key("p", "m", 0.5, 100, "n")
    assert cache_key(*base) != cache_key("p", "m", 1.0, 200, "n")
    assert cache_key(*base) != cache_key("p", "m", 1.0, 100, "n2")


def test_tie_break_is_neutral_with_respect_to_naming():
    """The tie-break is the total order of the unexplored pool.

    Breaking on case name makes that order the filesystem order, and 78.4% of
    `natural` rounds start on a sample/example probe. The hashed tie-break must
    not preserve the alphabetical order of a sample-first pool.
    """
    names = [f"00_sample_{i:02d}.txt" for i in range(4)] + \
            [f"01_random_{i:02d}.txt" for i in range(16)]
    hashed = sorted(names, key=lambda n: schedule_tie_break("cell", n))
    assert hashed != sorted(names)
    samples_in_head = sum(1 for n in hashed[:4] if "sample" in n)
    assert samples_in_head < 4          # the naming artefact is broken up
    # ... and it is still deterministic and cell-specific.
    assert hashed == sorted(names, key=lambda n: schedule_tie_break("cell", n))
    assert hashed != sorted(names, key=lambda n: schedule_tie_break("other", n))


def test_bad_values_are_refused_at_construction():
    for kw in ({"prior": "nope"}, {"cold_start": "nope"}, {"disclosure": "nope"},
               {"tie_break": "nope"}, {"candidate_policy": "nope"},
               {"steer_mode": "nope"}, {"k_candidates": 0}, {"explore_p": 1.5}):
        with pytest.raises(ValueError):
            CellSpec(task="t", seed=1, policy="km", **kw)  # type: ignore[arg-type]


def test_the_two_lists_are_disjoint():
    assert not (KEY_FLAGS & NONKEY_FLAGS)
    assert not check_flag_partition(sorted(KEY_FLAGS | NONKEY_FLAGS))


def test_an_undeclared_flag_fails():
    problems = check_flag_partition(["a_brand_new_knob"])
    assert problems and "NEITHER" in problems[0]
