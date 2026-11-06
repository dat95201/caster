"""The ten verbs, and one pass through the whole pipeline on the tiny corpus."""
from __future__ import annotations

import json
import pathlib
import os

import pytest

from caster import cli
from caster.keys import CellSpec, check_flag_partition


def test_the_count_is_ten():
    """`preflight` and `smoke` are verbs — they are the two things a session
    runs FIRST, and burying them is how they stop being run. `gateway` is a mode
    of `serve`: the thing that starts replicas and the thing that routes to them
    must not be two commands that can disagree."""
    assert len(cli.COMMANDS) == 10
    assert set(cli.COMMANDS) == {
        "freeze", "propose", "evaluate", "audit", "merge", "analyse", "report",
        "serve", "preflight", "smoke",
    }
    ap = cli.build_parser()
    serve_choices = next(a for a in ap._subparsers._group_actions[0].choices["serve"]._actions
                         if a.dest == "subcommand").choices
    assert "gateway" in serve_choices


def test_evaluate_has_no_full_scan_flag():
    """What `audit` does that `evaluate --full-scan` does not.

    The latter does not exist: an arm populating the cost table would make that
    table a function of which arm ran first."""
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["evaluate", "--full-scan"])
    assert cli.build_parser().parse_args(["--help"] and ["audit"]).command == "audit"


def test_every_flag_is_declared():
    assert check_flag_partition(cli.all_dests(cli.build_parser())) == []


def test_no_grid_wide_flag_is_silently_outranked_by_an_arm():
    """A flag that is parsed, keyed and then discarded is the drift bug with
    the arrow reversed: the cell key records the DEFAULT, so the run looks
    internally consistent while being a different treatment than the command
    line says.

    The evidence budget, the disclosure policy and the VS width are set by every
    Block B arm's own overrides, so they are arms (`km+steer-k2`,
    `km+steer-sample`, `km+steer+vs`) and deliberately not flags. The four that
    remain are grid-wide because they are swept ACROSS arms.
    """
    from caster.run import policies
    from caster.run.driver import Grid

    dests = set(cli.all_dests(cli.build_parser()))
    assert {"k_facts", "disclosure", "k_candidates"}.isdisjoint(dests)

    # Whatever IS still grid-wide must actually reach a cell when no arm
    # overrides it, and lose to the arm when one does.
    g = Grid(tasks=("t",), seeds=(0,), policies=("km", "km-explore0"),
             prior="jeffreys", explore_p=0.99)
    by_arm = {c.policy: c for c in g.cells()}
    assert by_arm["km"].explore_p == 0.99 and by_arm["km"].prior == "jeffreys"
    assert by_arm["km-explore0"].explore_p == 0.0        # the arm IS its explore_p
    assert by_arm["km-explore0"].prior == "jeffreys"     # ... but not its prior
    # Every forwarded field must be reachable from the CLI. Two are reachable
    # under a different name and are named here so the alias is deliberate
    # rather than a hole: `prior` is positional-ish, and the CLI exposes the
    # NEGATIVE of `serve_stable_timeouts` as `--quarantine-timeouts`, because
    # the default is to serve and an opt-out reads better than a flag that
    # does nothing. Both are declared in `keys.KEY_FLAGS`.
    aliased = {"prior", "serve_stable_timeouts"}
    assert set(Grid.FORWARDED) <= dests | aliased
    from caster.keys import KEY_FLAGS
    assert "quarantine_timeouts" in dests and "quarantine_timeouts" in KEY_FLAGS
    assert "cost-only" in policies.ORDERINGS


def test_resume_is_the_default_and_no_resume_is_the_flag():
    """Re-issuing the identical command after a kill is the documented
    recovery path and must be safe to do blindly."""
    args = cli.build_parser().parse_args(["evaluate"])
    assert args.no_resume is False
    assert cli.build_parser().parse_args(["evaluate", "--no-resume"]).no_resume


def test_defaults_match_the_spec():
    args = cli.build_parser().parse_args(["evaluate"])
    assert args.seeds == "0,1,2"            # 3 seeds, not 5
    assert args.explore_p == 0.15
    assert args.prior == "shipped"
    assert args.cold_start == "max"
    # Asserted against the SPEC, not against a literal: a flag default and a
    # spec default are one value in two places, and they drifted once already
    # (spec "rank", flag "hash" — every CLI run would have kept the old order).
    assert args.tie_break == CellSpec.tie_break == "rank"
    assert args.budget == 20
    f = cli.build_parser().parse_args(["freeze"])
    # Every pre-treatment screen is OFF by default: the grid runs every sound
    # fault and `analyse` bands by pool size and contest era after the run.
    assert f.min_pool == 1 and f.holdout == 0.0
    # Unset, not a falsy sentinel: `0` is a legitimate line count and `""` is
    # falsy, so either would need a truthiness guard on the check itself.
    assert f.lines_max is None and f.date_min is None
    assert not f.dedup_per_group
    assert f.pool_order == "interleave"


# ── the pipeline ─────────────────────────────────────────────────────────────


@pytest.fixture()
def wired(run_dir, corpus_root, monkeypatch, fake_client):
    """Point the CLI at the miniature corpus and at a model that does not exist."""
    client = fake_client(default_source="n=int(input())\nprint(n*3)\n",
                         replies={2: "n=int(input())\nprint(n*2)\n"})
    monkeypatch.setattr(cli, "_client", lambda args: client)
    real_adapter = cli._adapter

    def patched(args, frozen=None):
        args.condefects_root = str(corpus_root)
        return real_adapter(args, frozen)

    monkeypatch.setattr(cli, "_adapter", patched)
    return client


def _run(*argv: str) -> int:
    return cli.main(list(argv))


def _capture(capsys, *argv: str) -> str:
    """stdout of one command, from its first `{` — `paths.announce` prints first."""
    capsys.readouterr()
    assert _run(*argv) == 0
    out = capsys.readouterr().out
    return out[out.index("{"):]


def test_freeze_then_evaluate_then_merge_then_analyse_then_report(wired, run_dir, capsys):
    assert _run("freeze", "--min-pool", "6", "--date-min", "2000-01-01",
                "--condefects-root", "x") == 0
    out = json.loads(capsys.readouterr().out)
    assert out["funnel"]["admitted"] >= 1
    assert out["pool_sha256"]

    assert _run("evaluate", "--seeds", "0", "--policies", "no-mem,natural,km",
                "--budget", "4") == 0
    capsys.readouterr()

    assert _run("merge", "--budget", "4") == 0
    merged = capsys.readouterr().out
    assert "SOUNDNESS" in merged and "INVARIANT 9" in merged
    assert "[FAIL]" not in merged
    assert run_dir.ROUNDS.is_file()

    assert _run("analyse", "--budget", "4", "--bootstrap", "50") == 0
    capsys.readouterr()
    summary = json.loads((run_dir.DATA_DIR / "summary.json").read_text())
    assert summary["cells"] >= 3
    arms = {a["policy"] for a in summary["arms"]}
    assert arms == {"no-mem", "natural", "km"}
    # Threats are reported, and they come first in the document.
    assert list(summary)[:4] == ["budget", "cells", "proposal_errors", "k_facts"]

    assert _run("report") == 0
    html = (run_dir.DATA_DIR / "report.html").read_text()
    assert "Proposal failures" in html and "getrusage" in html
    assert html.startswith("<!doctype html>")


def test_dry_run_prints_a_plan_and_touches_nothing(wired, run_dir, capsys):
    assert _run("freeze", "--min-pool", "6", "--date-min", "2000-01-01") == 0
    capsys.readouterr()
    assert _run("evaluate", "--dry-run", "--seeds", "0,1",
                "--policies", "km,natural", "--budget", "4") == 0
    text = capsys.readouterr().out
    assert "universe" in text and "est calls" in text
    assert not (run_dir.DATA_DIR / "shards").exists()


def test_resume_re_walks_nothing_and_a_new_arm_costs_no_existing_cell(
        wired, run_dir, capsys):
    """Adding arm n+1 later must not re-run arms 1..n."""
    assert _run("freeze", "--min-pool", "6", "--date-min", "2000-01-01") == 0
    capsys.readouterr()
    assert _run("evaluate", "--seeds", "0", "--policies", "km", "--budget", "3") == 0
    capsys.readouterr()

    assert _run("evaluate", "--seeds", "0", "--policies", "km", "--budget", "3") == 0
    assert "nothing to walk" in capsys.readouterr().out

    # Arm n+1 arrives later: the done-index is keyed by the FULL cell key, so
    # the new policy simply has no entries and the old ones are not re-walked.
    assert _run("evaluate", "--seeds", "0", "--policies", "km,natural",
                "--budget", "3") == 0
    text = capsys.readouterr().out
    plan_lines = [ln for ln in text.splitlines() if "block A" in ln]
    assert any("natural" in ln for ln in plan_lines)
    assert not any("km " in ln and " km " in ln for ln in plan_lines if "natural" not in ln)
    assert "to walk" in text


def test_topping_up_a_budget_re_walks_the_cell(wired, run_dir, capsys):
    """`budget` is not in the cell key, so resume must notice it moved."""
    assert _run("freeze", "--min-pool", "6", "--date-min", "2000-01-01") == 0
    capsys.readouterr()
    assert _run("evaluate", "--seeds", "0", "--policies", "no-mem",
                "--budget", "2", "--force-full-budget") == 0
    capsys.readouterr()
    assert _run("evaluate", "--seeds", "0", "--policies", "no-mem",
                "--budget", "5", "--force-full-budget") == 0
    text = capsys.readouterr().out
    assert "nothing to walk" not in text

    from caster.report.records import collapse, read_rows
    from caster.run import shard

    rows = collapse(read_rows(shard.inputs()))
    per_cell: dict[str, set[int]] = {}
    for r in rows:
        per_cell.setdefault(r.episode_id, set()).add(r.round_index)
    # The rewrite means what it looks like: one episode, five rounds, no
    # duplicated draws (last write wins on (episode, round, candidate)).
    assert all(v == {1, 2, 3, 4, 5} for v in per_cell.values())


def test_audit_builds_the_kill_matrix(wired, run_dir, capsys):
    assert _run("freeze", "--min-pool", "6", "--date-min", "2000-01-01") == 0
    capsys.readouterr()
    assert _run("evaluate", "--seeds", "0", "--policies", "no-mem",
                "--budget", "2") == 0
    capsys.readouterr()
    assert _run("audit", "--concurrency", "1") == 0
    out = json.loads(capsys.readouterr().out)
    assert out["tasks"] >= 1 and out["cost_digest"]
    matrix = json.loads((run_dir.DATA_DIR / "kill_matrix.json").read_text())
    task = next(iter(matrix))
    sha = next(iter(matrix[task]))
    # A full scan: every probe has a verdict, not just the first refuter.
    assert len(matrix[task][sha]) > 1


def test_audit_refuses_before_phase_one(wired, run_dir, capsys):
    assert _run("freeze", "--min-pool", "6", "--date-min", "2000-01-01") == 0
    capsys.readouterr()
    with pytest.raises(SystemExit, match="propose"):
        _run("audit")


def test_preflight_fails_without_a_frozen_corpus(run_dir, capsys):
    assert cli.main(["preflight"]) == 1
    err = capsys.readouterr().err
    assert "frozen corpus" in err


def test_preflight_passes_once_the_corpus_exists(wired, run_dir, capsys, monkeypatch):
    monkeypatch.setenv("MODEL", "qwen2.5-coder:3b")   # preflight requires one
    assert _run("freeze", "--min-pool", "6", "--date-min", "2000-01-01") == 0
    capsys.readouterr()
    assert cli.main(["preflight"]) == 0
    out = capsys.readouterr().out
    assert "pool_sha256" in out and "protocol_sha" in out


def test_smoke_from_cache_only_makes_zero_model_calls(
        run_dir, corpus_root, monkeypatch, capsys, stub_sdk):
    """The CI gate, and the test that the two phases are separable.

    Exercised against the REAL client and its on-disk cache, because the thing
    under test IS the cache: a FakeClient would make the assertion vacuous. The
    first pass fills it; the second must then complete with zero model calls,
    and a call there is a build failure, not a slow test.
    """
    from caster.llm.client import LLMClient

    sdk = stub_sdk("n=int(input())\nprint(n*3)\n")
    client = LLMClient(model="stub-1", cache_dir=run_dir.CACHE_DIR,
                       calls_log=run_dir.CALLS_LOG)
    monkeypatch.setattr(client, "_sdk", lambda: sdk)
    monkeypatch.setattr(cli, "_client", lambda args: client)
    real_adapter = cli._adapter

    def patched(args, frozen=None):
        args.condefects_root = str(corpus_root)
        return real_adapter(args, frozen)

    monkeypatch.setattr(cli, "_adapter", patched)

    assert _run("freeze", "--min-pool", "6", "--date-min", "2000-01-01") == 0
    capsys.readouterr()
    frozen = json.loads((run_dir.POOL_DIR / "tasks.json").read_text())
    universe = run_dir.ROOT / "smoke" / "universe.json"
    universe.parent.mkdir(parents=True, exist_ok=True)
    universe.write_text(json.dumps({
        "tasks": frozen["pool_order"][:1], "seeds": [0],
        "policies": ["no-mem", "km"], "model": "",
    }), encoding="utf-8")

    # Pass 1: live, fills the cache and the ledger.
    assert _run("smoke", "--from-cache-only", "--budget", "2") == 1
    assert "phase separation" in capsys.readouterr().err

    assert _run("smoke", "--live", "--budget", "2") == 0
    capsys.readouterr()
    assert len(sdk.calls) == 1

    # Pass 2: cache-only. Zero calls, and the CRN draw is shared across arms,
    # so one cached round-1 draw serves both policies.
    before = len(sdk.calls)
    assert _run("smoke", "--from-cache-only", "--budget", "1") == 0
    out = json.loads(capsys.readouterr().out)
    assert out["model_calls"] == 0
    assert len(sdk.calls) == before
    assert (run_dir.SMOKE_DIR / "rounds.jsonl").is_file()
    assert (run_dir.ROOT / "smoke" / "expected.json").is_file()


def test_smoke_detects_drift(wired, run_dir, capsys):
    assert _run("freeze", "--min-pool", "6", "--date-min", "2000-01-01") == 0
    capsys.readouterr()
    frozen = json.loads((run_dir.POOL_DIR / "tasks.json").read_text())
    (run_dir.ROOT / "smoke").mkdir(parents=True, exist_ok=True)
    (run_dir.ROOT / "smoke" / "universe.json").write_text(json.dumps({
        "tasks": frozen["pool_order"][:1], "seeds": [0],
        "policies": ["km"], "model": "",
    }), encoding="utf-8")
    (run_dir.ROOT / "smoke" / "expected.json").write_text(json.dumps({
        "cells": 99, "accepted": 0, "rounds": 0, "first_accept": [],
    }), encoding="utf-8")
    assert _run("smoke", "--budget", "2") == 1
    assert "drifted" in capsys.readouterr().err


def test_freeze_shards_then_assembles(wired, run_dir, capsys):
    """Two shards, then one assemble — and the result must equal a single pass.

    A shard writes only its gate records. `tasks.json` appears at `--assemble`,
    because G4, the control arm and `pool_sha256` need the whole corpus.
    """
    import json

    n = json.loads(_capture(capsys, "freeze", "--dry-run"))["universe"]
    assert n >= 2
    half = n // 2
    assert _run("freeze", "--first", "1", "--last", str(half), "--shard", "0",
                "--min-pool", "6", "--date-min", "2000-01-01") == 0
    assert not (run_dir.POOL_DIR / "tasks.json").exists(), "a shard writes no corpus"
    assert _run("freeze", "--first", str(half + 1), "--last", str(n), "--shard", "1",
                "--min-pool", "6", "--date-min", "2000-01-01") == 0
    capsys.readouterr()

    assert _run("freeze", "--assemble", "--min-pool", "6",
                "--date-min", "2000-01-01") == 0
    sharded = json.loads(capsys.readouterr().out)
    assert sharded["shards"] == 2
    doc = json.loads((run_dir.POOL_DIR / "tasks.json").read_text(encoding="utf-8"))

    assert _run("freeze", "--overwrite", "--min-pool", "6",
                "--date-min", "2000-01-01") == 0
    capsys.readouterr()
    whole = json.loads((run_dir.POOL_DIR / "tasks.json").read_text(encoding="utf-8"))
    assert doc["pool_sha256"] == whole["pool_sha256"]
    assert doc["pool_order"] == whole["pool_order"]


def test_assemble_refuses_a_gap(wired, run_dir, capsys):
    """A corpus assembled over a missing shard has a wrong funnel and a
    `pool_sha256` that means nothing. It must fail loudly, not quietly."""
    import json
    import pytest as _pytest

    n = json.loads(_capture(capsys, "freeze", "--dry-run"))["universe"]
    assert _run("freeze", "--first", "1", "--last", "1", "--shard", "0",
                "--min-pool", "6", "--date-min", "2000-01-01") == 0
    capsys.readouterr()
    if n > 1:
        with _pytest.raises(SystemExit) as e:
            _run("freeze", "--assemble", "--min-pool", "6", "--date-min", "2000-01-01")
        assert "no gate record" in str(e.value)


def test_assemble_refuses_a_shard_that_gated_outside_its_range(wired, run_dir, capsys):
    """The silent corruption a sharded `Test/` tree invites.

    Unpack `Test_shard0.zip`, forget `--first/--last`, and the gate walks every
    fault: the ones that shard did not ship have no test data, so they are
    recorded as G2 rejects indistinguishable from real ones. `--assemble` sees
    no gap and writes a corpus whose funnel is a property of which archive was
    unpacked. The range sidecar is what makes that detectable.
    """
    import json

    import pytest as _pytest

    n = json.loads(_capture(capsys, "freeze", "--dry-run"))["universe"]
    if n < 2:
        _pytest.skip("needs at least two candidates")
    # shard 0 claims faults 1..1 but gates the whole universe, as an operator
    # who dropped the range would.
    assert _run("freeze", "--first", "1", "--last", "1", "--shard", "0",
                "--min-pool", "6", "--date-min", "2000-01-01") == 0
    done = run_dir.POOL_DIR / "gate" / "s00.jsonl"
    assert _run("freeze", "--shard", "0", "--last", str(n),
                "--min-pool", "6", "--date-min", "2000-01-01") == 0
    (run_dir.POOL_DIR / "gate" / "s00.range.json").write_text(
        json.dumps({"shard": 0, "first": 1, "last": 1, "universe": n}),
        encoding="utf-8")
    capsys.readouterr()
    assert done.is_file()

    with _pytest.raises(SystemExit) as e:
        _run("freeze", "--assemble", "--min-pool", "6", "--date-min", "2000-01-01")
    assert "does not own" in str(e.value)


def test_assemble_refuses_shards_gated_against_a_different_code_checkout(
        wired, run_dir, capsys):
    """Gate on a runner, assemble at home — the normal way to work, and the way
    it goes silently wrong if the two hold different `Code/` trees.

    `--first 500` then names a different fault on each machine, so every range
    and every index means something else. The candidate digest each shard
    records is what makes that a refusal instead of a corpus.
    """
    import json

    import pytest as _pytest

    assert _run("freeze", "--first", "1", "--last", "1", "--shard", "0",
                "--min-pool", "6", "--date-min", "2000-01-01") == 0
    side = run_dir.POOL_DIR / "gate" / "s00.range.json"
    d = json.loads(side.read_text(encoding="utf-8"))
    assert d["candidate_order_sha256"], "a shard must record what it gated"
    d["candidate_order_sha256"] = "0" * 64        # as another checkout would
    side.write_text(json.dumps(d), encoding="utf-8")
    capsys.readouterr()

    with _pytest.raises(SystemExit) as e:
        _run("freeze", "--assemble", "--min-pool", "6", "--date-min", "2000-01-01")
    assert "different Code/" in str(e.value)


def test_freeze_tasks_makes_those_faults_the_whole_universe(wired, run_dir, capsys):
    """A pilot corpus has to be a corpus, not a slice compared against another.

    `--tasks` sets the candidate list, so `--dry-run`, the ranges and
    `pool_sha256` all refer to these faults. Freezing ten of them and then
    assembling against the full universe would report 2,854 missing.
    """
    import json

    from caster.bench.adapter import ConDefectsAdapter

    names = sorted(ConDefectsAdapter(root=run_dir.ROOT).tasks()) or None
    full = json.loads(_capture(capsys, "freeze", "--dry-run"))["universe"]
    assert full >= 2

    one = json.loads(_capture(capsys, "freeze", "--dry-run",
                              "--tasks", "abc001_a/1001"))
    assert one["universe"] == 1 and one["last"] == 1

    assert _run("freeze", "--tasks", "abc001_a/1001", "--min-pool", "6",
                "--date-min", "2000-01-01") == 0
    capsys.readouterr()
    doc = json.loads((run_dir.POOL_DIR / "tasks.json").read_text(encoding="utf-8"))
    assert list(doc["tasks"]) == ["abc001_a/1001"]
    assert doc["funnel"]["examined"] == 1


def test_freeze_tasks_refuses_a_name_that_is_not_in_the_checkout(wired, run_dir):
    """A typo in a ten-name list is otherwise a corpus one fault short."""
    import pytest as _pytest

    with _pytest.raises(SystemExit) as e:
        _run("freeze", "--tasks", "abc001_a/1001,nope/404")
    assert "not in Code/" in str(e.value)


def test_preflight_fails_on_a_missing_model(wired, run_dir, capsys, monkeypatch):
    """The failure this exists to prevent: `propose` dies on the FIRST draw with
    `no model configured`, after the walk started and wrote rows.

    preflight costs nothing and runs before any of that, so an unset MODEL is a
    preflight failure, not a traceback forty cells in.
    """
    monkeypatch.delenv("MODEL", raising=False)
    assert _run("freeze", "--min-pool", "6", "--date-min", "2000-01-01") == 0
    capsys.readouterr()
    assert cli.main(["preflight"]) == 1
    assert "no MODEL set" in capsys.readouterr().err

    monkeypatch.setenv("MODEL", "qwen2.5-coder:3b")
    capsys.readouterr()
    assert cli.main(["preflight"]) == 0
    assert "model qwen2.5-coder:3b" in capsys.readouterr().out


def test_preflight_checks_the_configured_gateway_without_being_told(
        wired, run_dir, capsys, monkeypatch):
    """Checking the pool only when an operator remembers `--gateway-url` means
    the cheapest check that catches a dead backend is the one that never runs."""
    monkeypatch.setenv("MODEL", "qwen2.5-coder:3b")
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:1/v1")
    assert _run("freeze", "--min-pool", "6", "--date-min", "2000-01-01") == 0
    capsys.readouterr()
    assert cli.main(["preflight"]) == 1
    assert "unreachable" in capsys.readouterr().err


def test_dotenv_is_actually_loaded_and_the_shell_still_wins(tmp_path):
    """`.env.example` says "copy to .env" and `python-dotenv` is a dependency —
    but nothing called it, so every knob in that file was decorative.

    It has to load at PACKAGE import: `paths`, `sandbox` and `llm.client` all
    read the environment at import time, so loading in `main()` would be too
    late for all three. And `override=False`, so `RUN_DIR=pilot-3b caster ...`
    on the command line still beats the file.
    """
    import subprocess
    import sys

    (tmp_path / ".env").write_text(
        "MODEL=from-dotenv\nSANDBOX_TIMEOUT_SEC=17\n", encoding="utf-8")
    repo = str(pathlib.Path(__file__).resolve().parent.parent)
    code = ("import os, caster;"
            "from caster.exec.sandbox import DEFAULT_TIMEOUT;"
            "print(os.environ.get('MODEL'), DEFAULT_TIMEOUT)")

    env = {**os.environ, "CASTER_ROOT": str(tmp_path), "PYTHONPATH": repo}
    env.pop("MODEL", None)
    env.pop("SANDBOX_TIMEOUT_SEC", None)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env=env, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["from-dotenv", "17.0"]

    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, timeout=60,
                         env={**env, "MODEL": "from-shell"})
    assert out.stdout.split()[0] == "from-shell", "the file is a default, not an authority"


def test_phase_one_does_not_mark_phase_two_done(wired, run_dir, capsys):
    """The pilot's soundness failure, end to end.

    `caster propose` and `caster evaluate` both walk `no-mem`. If phase 1's
    cells land in `shards/*/cells.done`, resume calls them done and phase 2
    never evaluates the arm — so the merged `no-mem` rows are phase-1 rows:
    full budget, cold verdict store, live sandbox, collected before `audit`
    corrected anything. Against twelve arms replayed from the warm store that
    is `no-mem=False@None` vs `True@1`, reported as a soundness bug.
    """
    from caster.run import shard

    assert _run("freeze", "--min-pool", "6", "--date-min", "2000-01-01") == 0
    capsys.readouterr()
    assert _run("propose", "--seeds", "0", "--policies", "no-mem",
                "--budget", "2") == 0
    capsys.readouterr()

    assert shard.phase_inputs("propose"), "phase 1 wrote nothing"
    assert not shard.inputs(), "phase 1 wrote into the merge namespace"

    # Phase 2 must still have work to do for the very same arm.
    assert _run("evaluate", "--seeds", "0", "--policies", "no-mem",
                "--budget", "2") == 0
    out = capsys.readouterr().out
    assert "nothing to walk" not in out
    assert shard.inputs(), "phase 2 wrote nothing"


def test_merge_warns_when_one_namespace_holds_two_collection_modes(
        wired, run_dir, capsys):
    """A warning that names the cause, instead of a soundness violation.

    Phase 1 live and phase 2 replayed are different oracle states. When both
    end up in `shards/` anyway — a hand-run driver, an old fleet script — the
    merge has to say so rather than leave the operator reading `no-mem=False`
    against eleven `True`s.
    """
    from caster.run import shard

    assert _run("freeze", "--min-pool", "6", "--date-min", "2000-01-01") == 0
    capsys.readouterr()
    assert _run("evaluate", "--seeds", "0", "--policies", "no-mem",
                "--budget", "2", "--force-full-budget") == 0
    capsys.readouterr()
    assert _run("evaluate", "--seeds", "0", "--policies", "natural",
                "--budget", "2") == 0
    capsys.readouterr()

    audits = shard.audit_protocol(shard.metas(),
                                  records=shard.meta_records())
    mode = next(a for a in audits if a.name == "collection mode")
    assert not mode.ok and not mode.fatal
    assert any("force_full_budget" in item for item in mode.items)


def test_merge_and_analyse_are_told_which_faults_are_the_control_arm(monkeypatch):
    """The wiring, not the arithmetic — that is in test_review_regressions.

    `_control_tasks` reads the flag off the frozen document rather than going
    through `gate.control_tasks`, so a corpus frozen before `--control-size`
    existed (no `control` key on any task) is an empty tuple at merge time
    instead of a KeyError.
    """
    doc = {"tasks": {"t/1": {"control": False}, "ctl/9": {"control": True},
                     "t/2": {}}}
    assert cli._control_tasks(doc) == ("ctl/9",)
    assert cli._control_tasks({}) == ()
    assert cli._control_tasks({"tasks": {"t/1": None}}) == ()
