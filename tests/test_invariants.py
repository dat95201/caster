"""The invariants, each tested where it can actually be broken."""
from __future__ import annotations

import ast
import json
import pathlib

import pytest

from caster import paths
from caster.cli import all_dests, build_parser
from caster.keys import CellSpec, check_flag_partition
from caster.run import policies
from caster.steer import template

ROOT = pathlib.Path(__file__).resolve().parent.parent


# ── Invariant 6 ──────────────────────────────────────────────────────────────


def test_every_cli_flag_is_in_exactly_one_declared_list():
    """A flag in neither list fails the build. That is the whole point.

    A test that only checked the key would pass while a flag went undeclared
    entirely — and the flags that break a grid are usually the ones deliberately
    kept OUT of the key, where nobody has said which half they belong to.
    """
    assert check_flag_partition(all_dests(build_parser())) == []


def test_every_arm_preset_flag_exists_in_the_parser():
    """An arm that passes a flag the parser does not define fails at launch.

    Deleting an arm by slicing a file between two string offsets can swallow a
    flag definition that happened to sit between them: the file still compiles
    and every call site still works.
    """
    fields = {f for f in CellSpec(task="t", seed=0, policy="km").as_dict()}
    for arm in policies.ARMS.values():
        unknown = set(arm.overrides) - fields
        assert not unknown, f"{arm.arm_id} sets {unknown}, which CellSpec has no field for"
        arm.cell("t", 0)          # constructs, so every override is valid


# ── Invariant 1 ──────────────────────────────────────────────────────────────


def test_round1_prompt_is_byte_identical_across_every_arm():
    """Round 1 carries no facts, so the steered prompt must equal the unsteered one.

    Byte-identical, so they share one cache entry and one draw. Anything that
    diverges here buys every completion once per arm.
    """
    prompts = {}
    for arm in policies.ARMS.values():
        spec = arm.cell("abc001_a/1001", 0)
        p = template.build(
            program_label=spec.task,
            buggy_source="n = int(input())\nprint(n * 3)\n",
            examples=[template.Example("1\n", "2\n")],
            facts=[],                    # round 1: the ledger holds nothing
            round_index=1,
            steer_on=spec.steer_on,
            disclosure=spec.disclosure,
            show_kill_rate=spec.show_kill_rate,
            k_candidates=spec.k_candidates,
            vs_schedule=spec.vs_schedule,
            steer_mode=spec.steer_mode,
        )
        prompts[arm.arm_id] = p.text
    distinct = set(prompts.values())
    assert len(distinct) == 1, (
        "round-1 prompts diverge: "
        + ", ".join(sorted(a for a in prompts if prompts[a] != prompts["no-mem"]))
    )


def test_vs_does_not_start_before_round_three():
    """Rounds 1–2 carry no VS — that is what keeps Invariant 1 intact."""
    assert not template.vs_active(1, 5)
    assert not template.vs_active(2, 5)
    assert template.vs_active(3, 5)
    assert not template.vs_active(9, 1)


def test_vs_threshold_schedule_matches_the_template_document():
    """Rounds 1–2 none · 3–5 None · 6–10 0.10 · 11+ 0.01.

    Rounds 1–2 carry no VS at all, which is what keeps the round-1 identity.
    """
    assert template.vs_threshold_for(2) is None
    assert template.vs_threshold_for(3) is None
    assert template.vs_threshold_for(5) is None
    assert template.vs_threshold_for(6) == 0.10
    assert template.vs_threshold_for(10) == 0.10
    assert template.vs_threshold_for(11) == 0.01


# ── Invariant 4 ──────────────────────────────────────────────────────────────


def test_the_reference_source_never_enters_a_prompt():
    """Reference VALUES are allowed; the reference implementation is not."""
    reference = "n = int(input())\nprint(n * 2)   # THE REFERENCE\n"
    facts = [template.Fact(input_text="7\n", required="14\n", kill_rate=0.43,
                           last_status="wrong", last_output="21\n")]
    for mode in ("contract", "exclusion", "chat"):
        for disclosure in template.DISCLOSURES:
            p = template.build(
                program_label="t", buggy_source="n = int(input())\nprint(n * 3)\n",
                examples=[template.Example("1\n", "2\n")], facts=facts,
                round_index=4, steer_on=True, disclosure=disclosure,
                steer_mode=mode, transcript=["```python\nx\n```\nit failed"],
            )
            assert "THE REFERENCE" not in p.text
            assert reference.strip() not in p.text
            # The reference VALUE is allowed, and in `full` it is the point.
            if disclosure == "full" and mode == "contract":
                assert "required:" in p.text


def test_no_probe_name_reaches_a_prompt():
    """Prompt rule 2, satisfied by construction: `render` never writes a name."""
    facts = [template.Fact(input_text="7\n", required="14\n", kill_rate=0.43)]
    p = template.build(program_label="abc001_a/1001",
                       buggy_source="x", facts=facts, round_index=4, steer_on=True)
    assert "01_random" not in p.text
    assert "sample_00" not in p.text


def test_no_steered_prompt_names_a_behaviour_to_avoid():
    """Prompt rule 1. The exclusion arm is the deliberate exception (arm 13)."""
    facts = [template.Fact(input_text="7\n", required="14\n", kill_rate=0.43,
                           last_status="wrong", last_output="21\n")]
    p = template.build(program_label="t", buggy_source="x", facts=facts,
                       round_index=4, steer_on=True, steer_mode="contract")
    lowered = p.text.lower()
    for banned in ("do not", "don't", "avoid", "already-eliminated", "never produce"):
        assert banned not in lowered, banned
    # ... and the exclusion arm DOES name one, which is why it is a comparison.
    x = template.build(program_label="t", buggy_source="x", facts=facts,
                       round_index=4, steer_on=True, steer_mode="exclusion")
    assert "do not" in x.text.lower()


def test_exclusion_block_is_reachable_only_through_arm_13():
    """`s5_exclusion` has exactly one caller, and the roster has exactly one arm
    that reaches it."""
    source = (ROOT / "caster" / "steer" / "template.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "s5_exclusion"]
    assert len(calls) == 1
    arms = [a.arm_id for a in policies.ARMS.values()
            if a.overrides.get("steer_mode") == "exclusion"]
    assert arms == ["km+exclusion"]


# ── Invariant 5 ──────────────────────────────────────────────────────────────


def test_one_path_resolver():
    """No module may compute its own data path. One resolver decides."""
    offenders: list[str] = []
    for py in (ROOT / "caster").rglob("*.py"):
        if py.name == "paths.py":
            continue
        text = py.read_text(encoding="utf-8")
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if (isinstance(func, ast.Attribute) and func.attr == "Path"
                    and isinstance(func.value, ast.Name) and func.value.id == "pathlib"):
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str) \
                            and arg.value.startswith(("data/", "logs/", "cache/")):
                        offenders.append(f"{py.relative_to(ROOT)}: {arg.value}")
    assert not offenders, offenders


def test_bash_and_python_agree_on_the_run_dir(tmp_path, monkeypatch):
    """`scripts/run_dir_paths.sh` and `caster.paths` must agree, and there is a
    test that says so."""
    import importlib
    import subprocess

    script = ROOT / "scripts" / "run_dir_paths.sh"
    assert script.is_file()
    for slug in ("", "campaign-7b"):
        monkeypatch.setenv("RUN_DIR", slug)
        importlib.reload(paths)
        out = subprocess.run(
            ["bash", "-c", f'cd "{ROOT}" && RUN_DIR="{slug}" . {script} && '
                           f'echo "$RUN_DATA|$RUN_LOGS"'],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        data, logs = out.split("|")
        assert data == paths.DATA_REL, (slug, data, paths.DATA_REL)
        assert logs == paths.LOGS_REL, (slug, logs, paths.LOGS_REL)
    monkeypatch.delenv("RUN_DIR", raising=False)
    importlib.reload(paths)


# ── Invariant 14 ─────────────────────────────────────────────────────────────


def test_smoke_output_cannot_be_merged(run_dir):
    """By construction, not by a naming convention an operator must remember."""
    from caster.run import shard

    run_dir.ensure(run_dir.SMOKE_DIR)
    (run_dir.SMOKE_DIR / "rounds.jsonl").write_text("{}\n", encoding="utf-8")
    assert (run_dir.SMOKE_DIR / "rounds.jsonl") not in shard.inputs()


def test_the_shipped_smoke_universe_covers_every_arm():
    """The file claims EVERY arm. Nothing was checking that it still did.

    An arm added after the universe was written is an arm whose first breakage
    is found by a full grid hours in, not by a seconds-long gate — and the gap
    is invisible, because a smoke run over 21 of 22 arms passes exactly like a
    run over all 22.
    """
    universe = json.loads(
        (ROOT / "smoke" / "universe.json").read_text(encoding="utf-8"))
    listed = universe["policies"]
    assert set(listed) == set(policies.ARMS), (
        f"missing from smoke universe: {sorted(set(policies.ARMS) - set(listed))}; "
        f"not an arm: {sorted(set(listed) - set(policies.ARMS))}")
    assert len(listed) == len(set(listed)), "an arm is listed twice"


# ── module boundaries ───────────────────────────────────────────────────


def _imports(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
    return out


def test_the_ledger_has_no_io_and_no_benchmark_imports():
    """It is pure statistics over probe names, so it is unit-testable offline.

    That is what lets the whole scheduler be exercised without a benchmark.
    """
    mods = _imports(ROOT / "caster" / "exec" / "ledger.py")
    assert not any(m.startswith("caster.bench") for m in mods), mods
    for banned in ("subprocess", "pathlib", "json", "os", "caster.exec.sandbox"):
        assert banned not in mods, banned


def test_steer_does_not_import_verify():
    """Evidence selection reads the ledger; it does not run anything."""
    for py in (ROOT / "caster" / "steer").glob("*.py"):
        mods = _imports(py)
        assert "caster.exec.verify" not in mods, py.name
        assert "caster.bench.adapter" not in mods, py.name


def test_only_bench_knows_the_benchmark():
    """A Defects4J port must touch nothing outside `caster/bench/`."""
    for py in (ROOT / "caster").rglob("*.py"):
        if "bench" in py.parts:
            continue
        mods = _imports(py)
        leaked = {m for m in mods if m.startswith("caster.bench.")
                  and m not in ("caster.bench.protocol", "caster.bench.adapter",
                                "caster.bench.gate")}
        assert not leaked, f"{py.relative_to(ROOT)} imports {leaked}"
        # Only the CLI and the gate may name the concrete adapter.
        if py.name not in ("cli.py", "gate.py", "__init__.py"):
            assert "caster.bench.adapter" not in mods, py.relative_to(ROOT)
