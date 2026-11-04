"""Every arm names a question that is written down somewhere.

The defect this exists for: `fault-record` carried the label
"RQ1 — FRTP, the memory baseline" through a year of commits while ranking on
`first_kill_round`, which is not what FRTP ranks on. Nothing could catch it,
because the label had nothing to contradict — a research question that lives
only as a string in a dataclass certifies itself. See DECISIONS 26.
"""
from __future__ import annotations

import pathlib
import re

from caster.report import analyse
from caster.report.records import RoundRecord
from caster.run import policies

QUESTIONS = pathlib.Path(__file__).resolve().parents[1] / "docs" / "QUESTIONS.md"


def declared() -> set[str]:
    """The `##` headings of QUESTIONS.md, up to the first em dash."""
    out = set()
    for line in QUESTIONS.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^## (.+)$", line)
        if m:
            out.add(m.group(1).split("—")[0].strip())
    return out


def test_every_arm_answers_a_declared_question():
    names = declared()
    assert {"RQ1", "RQ2", "RQ3", "RQ4", "calibration"} <= names, sorted(names)
    for arm in policies.ARMS.values():
        assert arm.answers in names, f"{arm.arm_id} answers {arm.answers!r}"


def test_there_are_four_questions_and_no_more():
    """Seven labels were in the code and none was ever stated. Four is the
    count the study can defend; a fifth has to earn a heading first."""
    rqs = {n for n in declared() if n.startswith("RQ")}
    assert rqs == {"RQ1", "RQ2", "RQ3", "RQ4"}, sorted(rqs)


def test_every_question_states_what_would_refute_it():
    """A question with no refuting outcome is a name for a group of arms."""
    text = QUESTIONS.read_text(encoding="utf-8")
    for rq in ("RQ1", "RQ2", "RQ3", "RQ4"):
        section = text.split(f"## {rq}")[1].split("\n## ")[0]
        assert "Refuted when" in section, rq


def test_km_chat_is_in_the_default_grid():
    """It is the published comparator RQ3 is argued against. An arm that is
    not run cannot report that the comparator loses."""
    assert "km+chat" in policies.DEFAULT_GRID


def test_no_mem_is_rq3s_baseline_not_merely_a_floor():
    """It redraws every round under a policy-free nonce, which is independent
    repeated prompting — measured elsewhere to BEAT the conversational
    approach. Calling it a floor invites comparing Block B against `natural`,
    which proves nothing."""
    arm = policies.arm("no-mem")
    assert arm.answers == "calibration"
    assert "independent repeated prompting" in arm.note


# ── RQ4's primary metric, which needs no holdout ────────────────────────────


def _row(policy: str, task: str, sha: str, shown: tuple[str, ...]) -> RoundRecord:
    return RoundRecord(
        run_id="r", episode_id="e", task=task, seed=0, policy=policy,
        round_index=1, patch_sha=sha, facts_shown=shown, k_facts=len(shown),
    )


def test_exploitability_separates_fitting_the_shown_from_ignoring_it():
    """Three patches, one disclosure, three different relationships to it."""
    matrices = {
        "t": {
            "fits":   {"p1": "pass",  "p2": "pass", "p3": "wrong"},
            "only":   {"p1": "wrong", "p2": "pass", "p3": "pass"},
            "both":   {"p1": "wrong", "p2": "pass", "p3": "crash"},
            "plaus":  {"p1": "pass",  "p2": "pass", "p3": "pass"},
        }
    }
    shown = ("p1", "p2")
    rows = [_row("km+steer", "t", sha, shown)
            for sha in ("fits", "only", "both", "plaus")]
    [out] = analyse.exploitability(rows, matrices)
    # the plausible patch is in no denominator: it says nothing about disclosure
    assert out["refuted_patches"] == 3
    assert out["fits_disclosed"] == 1          # refuted only outside what was shown
    assert out["only_disclosed"] == 1          # refuted only inside it
    assert out["k_shown_mean"] == 2.0


def test_exploitability_skips_rounds_that_disclosed_nothing():
    """An unsteered arm counted as zero would dilute every steered arm it is
    tabulated beside."""
    matrices = {"t": {"s": {"p1": "wrong"}}}
    rows = [_row("km", "t", "s", ()), _row("km+steer", "t", "s", ("p2",))]
    out = analyse.exploitability(rows, matrices)
    assert [r["policy"] for r in out] == ["km+steer"]


def test_exploitability_ignores_patches_the_audit_never_measured():
    matrices = {"t": {"other": {"p1": "wrong"}}}
    assert analyse.exploitability([_row("km+steer", "t", "s", ("p1",))], matrices) == []
