"""One test per defect a review found. Each names what it would have cost.

These are not extra coverage. Every one of them is a case where the code did
something defensible-looking and produced a wrong *number* rather than an
exception — which is the only failure mode that matters on a corpus of 1,244
tasks nobody re-reads row by row.
"""
from __future__ import annotations

import json

import pytest

from caster.exec.sandbox import protocol_fields
from caster.exec.store import VerdictStore, protocol_sha
from caster.keys import cache_key, draw_nonce
from caster.run import policies
from caster.run.driver import Grid
from caster.run.episode import EpisodeConfig, run_episode
from caster.steer import render, template


# ── Invariant 1, the half that is not about prompt bytes ─────────────────────


def test_a_vs_arm_shares_round_one_with_every_other_arm(
        adapter, frozen, run_dir, stores, fake_client):
    """`max_tokens` is in the cache key, and the output budget is sized from the
    number of programs the reply must carry.

    Rounds 1–2 carry no VS, so a VS arm must ask for ONE program there.
    Scaling the budget by `k_candidates` unconditionally leaves the prompt
    byte-identical and the cache key different — CRN dies silently and every
    completion is bought once per arm, with nothing in any log to say so.
    """
    store, patches = stores
    seen: dict[str, tuple[str, str, int]] = {}
    for arm_id in ("no-mem", "km+steer", "km+steer+vs"):
        client = fake_client(default_source="n=int(input())\nprint(n*3)\n",
                             k_candidates=5)
        captured: list[int] = []
        real = client.complete

        def spy(prompt, *, nonce, max_tokens=1024, **kw):   # noqa: ANN001
            captured.append(max_tokens)
            return real(prompt, nonce=nonce, max_tokens=max_tokens, **kw)

        client.complete = spy  # type: ignore[method-assign]
        run_episode(policies.arm(arm_id).cell("abc001_a/1001", 0),
                    adapter=adapter, client=client, store=store, patches=patches,
                    cfg=EpisodeConfig(budget=1))
        seen[arm_id] = (client.prompts[0], client.nonces[0], captured[0])

    assert len({v[0] for v in seen.values()}) == 1, "round-1 prompts diverge"
    assert len({v[1] for v in seen.values()}) == 1, "round-1 nonces diverge"
    assert len({v[2] for v in seen.values()}) == 1, "round-1 max_tokens diverge"

    prompt, nonce, mt = seen["km+steer+vs"]
    assert (cache_key(prompt, "m", 1.0, mt, nonce)
            == cache_key(*seen["no-mem"][:1], "m", 1.0, seen["no-mem"][2],
                         seen["no-mem"][1]))


def test_vs_widens_the_budget_once_it_is_actually_on(
        adapter, frozen, run_dir, stores, fake_client):
    """...and the fix must not disable VS, only delay it to round 3."""
    store, patches = stores
    client = fake_client(default_source="n=int(input())\nprint(n*3)\n", k_candidates=5)
    captured: list[int] = []
    real = client.complete

    def spy(prompt, *, nonce, max_tokens=1024, **kw):   # noqa: ANN001
        captured.append(max_tokens)
        return real(prompt, nonce=nonce, max_tokens=max_tokens, **kw)

    client.complete = spy  # type: ignore[method-assign]
    r = run_episode(policies.arm("km+steer+vs").cell("abc001_a/1001", 0),
                    adapter=adapter, client=client, store=store, patches=patches,
                    cfg=EpisodeConfig(budget=4))
    assert captured[0] == captured[1] < captured[2]
    by_round: dict[int, int] = {}
    for rec in r.records:
        by_round[rec.round_index] = by_round.get(rec.round_index, 0) + 1
    assert by_round[1] == 1 and by_round[2] == 1       # no VS in rounds 1–2
    assert by_round[3] > 1                              # ... and VS from round 3


# ── / the disclosure record ──────────────────────────────────────


def test_facts_shown_is_what_the_prompt_showed_and_is_empty_when_unsteered(
        adapter, frozen, run_dir, stores, fake_client):
    """The exploitability bound is computed from exactly this field.

    Read off the post-update ledger it is one round late; emitted for an
    unsteered arm it attributes a disclosure that never happened. Either way the
    leakage figure is wrong and nothing looks broken.
    """
    store, patches = stores
    for arm_id in ("km", "no-mem", "random"):
        client = fake_client(default_source="n=int(input())\nprint(n*3)\n")
        r = run_episode(policies.arm(arm_id).cell("abc001_a/1001", 0),
                        adapter=adapter, client=client, store=store,
                        patches=patches, cfg=EpisodeConfig(budget=4))
        assert all(rec.facts_shown == () for rec in r.records), arm_id
        assert all(rec.k_facts == 0 for rec in r.records), arm_id

    client = fake_client(default_source="n=int(input())\nprint(n*3)\n")
    r = run_episode(policies.arm("km+steer").cell("abc001_a/1001", 0),
                    adapter=adapter, client=client, store=store, patches=patches,
                    cfg=EpisodeConfig(budget=4))
    assert r.records[0].facts_shown == ()        # round 1 shows nothing
    assert r.records[0].k_facts == 0
    later = [rec for rec in r.records[1:] if rec.k_facts > 0]
    assert later
    for rec in later:
        # Exactly the probes the contract block rendered, no more and no fewer.
        assert len(rec.facts_shown) == rec.k_facts


# ── the frozen cost table ─────────────────────────────────────────────


def test_a_changed_verdict_does_not_inherit_the_audited_cost(run_dir):
    """An audited `timeout` whose re-execution comes back `pass` is a probe the
    audit mismeasured.

    Inheriting its 10-second wall-clock kill would put that cost in the frozen
    table for a probe that costs milliseconds — on exactly the probes says
    flip — and every later arm would replay it.
    """
    vs = VerdictStore({"p": 1})
    vs.put("sha", "c", verdict="timeout", output_sha256="", cpu_s=10.0,
           source="audit", concurrency=4)
    before = vs.cost_digest()

    vs.put("sha", "c", verdict="pass", output_sha256="x", cpu_s=0.01, source="scan")
    entry = vs.peek("sha", "c")
    assert entry.verdict == "pass"
    assert entry.cpu_s == pytest.approx(0.01)
    assert entry.source == "scan"
    assert vs.cost_digest() == before      # the stamped artifact did not move

    # The same verdict restated by an arm still inherits — that IS .
    vs.put("sha2", "c", verdict="pass", output_sha256="x", cpu_s=1.0, source="audit")
    vs.put("sha2", "c", verdict="pass", output_sha256="x", cpu_s=9.0, source="scan")
    assert vs.peek("sha2", "c").cpu_s == pytest.approx(1.0)


def test_a_replayed_scan_rebuilds_the_same_steered_prompt(
        adapter, frozen, run_dir, stores, fake_client):
    """A store-served verdict must carry the OBSERVATION, not just the status.

    Without it, "your last attempt printed `…`" disappears on replay, so a
    cold-store and a warm-store run of the same steered arm build different
    prompts — different draws, with nothing in any key to say so.
    """
    store, patches = stores
    spec = policies.arm("km+steer").cell("abc001_a/1001", 0)
    cold = fake_client(default_source="n=int(input())\nprint(n*3)\n")
    run_episode(spec, adapter=adapter, client=cold, store=store, patches=patches,
                cfg=EpisodeConfig(budget=4))

    warm = fake_client(default_source="n=int(input())\nprint(n*3)\n")
    run_episode(spec, adapter=adapter, client=warm, store=store, patches=patches,
                cfg=EpisodeConfig(budget=4))
    assert cold.prompts == warm.prompts
    assert any("your last attempt printed" in p for p in warm.prompts)


# ── prompt rule 6 — the false fact ───────────────────────────────────────────────


def test_a_legible_multiline_input_is_rendered_verbatim():
    """`snippet()` joins lines with " / " and cuts at 240 chars.

    Running a legible input through it and then asserting a required output for
    the result is precisely the false fact rule 6 exists to prevent — and it
    would hit every multi-line stdin at any size.
    """
    text = "5\n1 2 3 4 5\n" + "9 " * 100          # ~212 chars, 3 lines, legible
    assert render.is_legible(text)
    out = render.render_input(text)
    assert "5\n1 2 3 4 5" in out
    assert "/" not in out.split("<<<")[1].split(">>>")[0].replace("9 ", "")
    assert "truncated" not in out

    big = "x" * 5000
    shaped = render.render_input(big)
    # A shape keeps what is informative about a stress case — that it IS one,
    # how big, and its first line (on AtCoder that line carries N) — and drops
    # the bytes. It must be a summary, not a truncation of the data.
    assert "large input" in shaped and "5,000 chars" in shaped
    assert len(shaped) < 300


def test_the_required_output_is_never_truncated_under_an_assertion():
    """A required value the model cannot read is not a requirement."""
    short = render.render_required("15\n42\n")
    assert "15\n42" in short
    huge = render.render_required("y" * 5000)
    assert "withheld rather than truncated" in huge and "yyyy" not in huge


def test_the_contract_block_does_not_open_a_second_code_fence():
    """S7 and S8 ask for the answer in a ```python block; a second fence in the
    prompt is the one thing that can make a model close the wrong one."""
    facts = [template.Fact(input_text="5\n1 2\n", required="3\n", kill_rate=0.4)]
    p = template.build(program_label="t", buggy_source="x", facts=facts,
                       round_index=4, steer_on=True)
    before_answer = p.text[: p.text.index("Return the corrected source")]
    assert "```" not in before_answer.replace("```python\nx\n```", "")


# ── Invariant 13 — the values that reach no key ──────────────────────────────


def test_the_sandbox_timeout_reaches_protocol_sha_and_the_meta(run_dir):
    """A run at `--timeout 5` must not write verdicts into the `--timeout 10`
    namespace. the runtime-disagreement audit reads `sandbox_timeout_sec`, so
    a constant there makes the one audit designed to catch this blind to it."""
    assert protocol_sha(protocol_fields(timeout=5.0)) != protocol_sha(protocol_fields(timeout=10.0))

    g5 = Grid(tasks=("t",), seeds=(0,), policies=("km",), sandbox_timeout=5.0)
    g10 = Grid(tasks=("t",), seeds=(0,), policies=("km",), sandbox_timeout=10.0)
    assert g5.protocol()["sandbox_timeout_sec"] == 5.0
    assert protocol_sha(g5.protocol()) != protocol_sha(g10.protocol())

    from caster.llm.client import LLMClient
    from caster.run.driver import protocol_record

    rec = protocol_record(g5, LLMClient(model="m"), shard="0")
    assert rec["runtime"]["sandbox_timeout_sec"] == 5.0


def test_a_bare_alias_is_reported():
    """"pin the hosted model hard. Exact snapshot id, never a bare
    alias." A hosted model silently re-pointed is a different instrument with
    nothing to say so."""
    from caster.run.shard import looks_like_alias

    assert looks_like_alias("gpt-4o-mini")
    assert not looks_like_alias("gpt-4o-mini-2024-07-18")
    assert not looks_like_alias("qwen2.5-coder:7b")
    assert not looks_like_alias("")


def test_a_shard_with_rows_but_no_meta_is_reported(tmp_path):
    """A MIXED set is the dangerous case: the un-recorded shards are silently
    exempt from the two audits that catch instrument drift."""
    from caster.run import shard

    for i in (0, 1):
        d = tmp_path / "shards" / f"s{i:02d}"
        d.mkdir(parents=True)
        (d / "rounds.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "shards" / "s00" / "meta.json").write_text(json.dumps({
        "protocol": {"model": "m", "temperature": 1.0, "reasoning_effort": None},
        "runtime": {"context_tokens": 1, "sandbox_timeout_sec": 10.0},
    }), encoding="utf-8")

    audits = shard.audit_protocol(shard.metas(tmp_path),
                                  shard_dirs=[p.parent for p in shard.inputs(tmp_path)])
    missing = [a for a in audits if a.name == "no protocol record"][0]
    assert not missing.ok and missing.items == ["s01"]
    assert not missing.fatal


# ── a flag that is parsed, keyed and then discarded ───────────────────


def test_every_grid_knob_reaches_a_cell():
    """The drift bug with the arrow reversed: the cell key records the
    DEFAULT, so the run looks internally consistent while being a different
    treatment than the command line says."""
    g = Grid(tasks=("t",), seeds=(0,), policies=("natural",),
             prior="jeffreys", cold_start="q75", explore_p=0.42, tie_break="name")
    (cell,) = g.cells()
    for field in Grid.FORWARDED:
        assert getattr(cell, field) == getattr(g, field), field


# ── the ledger, not the upstream ──────────────────────────────────────


def test_the_gateway_charges_once_and_the_client_does_not_double_log(tmp_path):
    """Coalescing makes N shards that miss one key produce one upstream CALL.

    the criterion is on the **ledger**: without the write moving to the
    gateway, each of the N clients still appends its own `usd` row and the
    ledger shows N charges for one call.
    """
    from caster.llm.gateway import Backend, Gateway

    log = tmp_path / "calls.jsonl"
    gw = Gateway([Backend(url="http://x/v1", model="m")], calls_log=log,
                 price_in=1_000_000.0, price_out=1_000_000.0, spend_cap=1e9)
    payload = json.dumps({
        "model": "m", "choices": [{"message": {"content": "hi"},
                                   "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4},
    }).encode()

    out = gw._settle("key-1", payload, gw.backends[0])
    blob = json.loads(out)
    assert blob["_caster_gateway"] is True
    assert blob["_caster_backend"] == "local:http://x/v1"
    assert blob["_caster_usd"] == pytest.approx(7.0)
    assert gw.ledger.spent == pytest.approx(7.0)

    rows = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
    assert len(rows) == 1 and rows[0]["via"] == "gateway"
    assert rows[0]["cache_key"] == "key-1"


def test_in_hybrid_the_cloud_is_the_overflow(tmp_path):
    """"admit to a local slot when one is free, else to cloud"."""
    from caster.llm.gateway import Backend, Gateway

    local = Backend(url="http://local/v1", model="m", kind="local", concurrency=1)
    cloud = Backend(url="http://cloud/v1", model="m", kind="cloud", concurrency=8)
    gw = Gateway([local, cloud], calls_log=tmp_path / "c.jsonl")
    assert gw._pick().kind == "local"       # free local slot wins
    assert gw._pick().kind == "cloud"       # local is full: overflow
    gw._release(local, ok=True)
    assert gw._pick().kind == "local"       # ... and back again


def test_the_refcount_holds_the_pool_up(tmp_path):
    """"teardown is driven by a refcount, not by shard exit"."""
    from caster.llm.gateway import Backend, Gateway

    gw = Gateway([Backend(url="http://x/v1", model="m")], calls_log=tmp_path / "c.jsonl",
                 service_timeout=0.5)
    gw.register()
    gw.register()
    assert gw.sessions == 2
    gw.unregister()
    assert gw.sessions == 1
    gw.drain()
    assert gw.sessions == 1                 # drain did not invent an exit


def test_stop_never_touches_a_listener_we_did_not_start(monkeypatch, tmp_path):
    """"not ours → never unload, never stop". `stop()` unloads BEFORE it
    checks ownership, so a bare Replica(owned=False) would drop someone else's
    resident weights."""
    from caster.llm import serve

    monkeypatch.setattr(serve, "OWNERSHIP_FILE", tmp_path / "serve.json")
    foreign = serve.describe(11439, model="m")
    assert foreign.read_only and not foreign.owned
    assert serve.unload(foreign) is False

    serve._save_ownership({"replicas": {"11440": {"pid": 123, "model": "m",
                                                  "context": 4096}}})
    ours = serve.describe(11440)
    assert ours.owned and not ours.read_only and ours.pid == 123


# ── / a miss is a build failure, not an outage ───────────────────


def test_a_cache_only_miss_is_not_absorbed_by_the_circuit_breaker(
        adapter, frozen, run_dir, stores, fake_client):
    """Raised as `BackendUnreachable`, five misses end the cell as a "backend
    outage" — exactly the silent degradation and want to be loud."""
    from caster.exec.store import CacheMiss
    from caster.llm.client import LLMClient

    store, patches = stores
    client = LLMClient(model="fake", cache_dir=run_dir.CACHE_DIR,
                       calls_log=run_dir.CALLS_LOG)
    with pytest.raises(CacheMiss):
        run_episode(policies.arm("km").cell("abc001_a/1001", 0),
                    adapter=adapter, client=client, store=store, patches=patches,
                    cfg=EpisodeConfig(budget=3, from_cache_only=True))


def test_after_an_audit_every_policy_replays_with_zero_sandbox_runs(
        adapter, frozen, run_dir, stores, fake_client):
    """The criterion, and what `--from-cache-only` has to mean on the scan
    side for it to be checkable."""
    from caster.exec.verify import scan
    from caster.run.episode import audit_task

    store, patches = stores
    task = "abc001_a/1001"
    shas = [patches.put(src) for src in
            ("n=int(input())\nprint(n*3)\n", "n=int(input())\nprint(n*2)\n")]
    audit_task(adapter, task, shas, patches=patches, store=store, concurrency=1)

    for arm_id in ("natural", "km", "fault-record", "kill-only"):
        from caster.exec.ledger import CaseLedger
        from caster.run.policies import PolicyContext, schedule

        ledger = CaseLedger(adapter.probes(task), explore_p=0.0)
        ctx = PolicyContext(ledger=ledger, pool=tuple(adapter.probes(task)),
                            round_index=1, cell_key="k")
        order = schedule(arm_id, ctx)
        r = scan(adapter, task, patches.require(shas[0]), order, ledger=ledger,
                 store=store, from_cache_only=True)   # --replay-only
        assert r.sandbox_runs == 0, arm_id
        assert r.verdict_hits == r.test_calls


# ── exactly one call ──────────────────────────────────────────────────


def test_live_smoke_walks_one_round(wired_smoke, run_dir, capsys):
    """"Deliberately one call, so it is cheap enough to run every time." One
    CELL is not one call: a cell at --budget 3 draws three times."""
    from caster import cli

    assert cli.main(["smoke", "--live", "--budget", "9"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["cells"] == 1
    assert out["rounds"] == 1       # one ROUND, whatever --budget said


@pytest.fixture()
def wired_smoke(run_dir, corpus_root, monkeypatch, fake_client, capsys):
    from caster import cli

    client = fake_client(default_source="n=int(input())\nprint(n*3)\n")
    monkeypatch.setattr(cli, "_client", lambda args: client)
    real = cli._adapter

    def patched(args, frozen=None):
        args.condefects_root = str(corpus_root)
        return real(args, frozen)

    monkeypatch.setattr(cli, "_adapter", patched)
    assert cli.main(["freeze", "--min-pool", "6", "--date-min", "2000-01-01"]) == 0
    capsys.readouterr()
    doc = json.loads((run_dir.POOL_DIR / "tasks.json").read_text())
    (run_dir.ROOT / "smoke").mkdir(parents=True, exist_ok=True)
    (run_dir.ROOT / "smoke" / "universe.json").write_text(json.dumps({
        "tasks": doc["pool_order"][:1], "seeds": [0], "policies": ["km"], "model": "",
    }), encoding="utf-8")
    return client


# ── the plan is the thing you check before every fleet launch ─────────


def test_the_dry_run_estimate_counts_prompt_families_not_cells():
    """Block A shares a prompt family per (task, seed); charging the block a
    flat `budget` for the whole grid understates by ~2× at campaign scale."""
    from caster.run.driver import plan

    g = Grid(tasks=tuple(f"t{i}" for i in range(10)), seeds=(0, 1, 2),
             policies=("natural", "km", "random"), budget=20)
    p = plan(g, resume=False)
    assert p.universe == 90
    assert p.est_model_calls == 10 * 3 * 20          # families × budget

    g2 = Grid(tasks=tuple(f"t{i}" for i in range(10)), seeds=(0, 1, 2),
              policies=("natural", "km+steer"), budget=20)
    p2 = plan(g2, resume=False)
    assert p2.est_model_calls == 10 * 3 * 20 + 30 * 19


# ── the unfiltered control arm ──────────────────────────────────────────


def test_the_control_arm_is_runnable_not_merely_marked(adapter):
    """"Reported whatever it shows" needs something to report: a fault cut by a
    THRESHOLD still has to carry the pool and the split a driver would walk."""
    from caster.bench.gate import GateConfig, control_tasks, freeze, install

    doc = freeze(adapter, cfg=GateConfig(min_loop_pool=1000, date_min="2000-01-01",
                                         holdout=0.30, control_size=2))
    names = control_tasks(doc)
    assert names
    for n in names:
        t = doc["tasks"][n]
        assert t["gates"]["G1"] and not t["admitted"]
        assert t["p_loop"] and t["p_eval"]
        assert n not in doc["pool_order"]        # never pooled with the main grid

    install(adapter, doc, include_control=True)
    assert set(adapter.probes(names[0])) == set(doc["tasks"][names[0]]["p_loop"])


# ── , G1 — the double run ──────────────────────────────────────────────────


def test_a_nondeterministic_reference_is_caught(corpus_root, tmp_path):
    """The stability half was unreachable: G2 dropped every case without an
    `out/` file first, so the reference was never run twice and the
    `nondeterministic` reason was one nothing could carry."""
    from conftest import _write
    from caster.bench.adapter import ConDefectsAdapter
    from caster.bench.gate import GateConfig, gate_one

    root = tmp_path / "flaky"
    code = root / "Code" / "abc009_a" / "Python" / "9001"
    _write(code / "correctVersion.py",
           "import random\nn=int(input())\nprint(n*2 + random.randint(0, 1))\n")
    _write(code / "faultyVersion.py", "n=int(input())\nprint(n*3)\n")
    test = root / "Test" / "abc009" / "A"
    for i in range(8):
        _write(test / "in" / f"01_random_{i:02d}.txt", f"{i + 1}\n")
        _write(test / "out" / f"01_random_{i:02d}.txt", f"{(i + 1) * 2}\n")

    ad = ConDefectsAdapter(root=root, timeout=10.0)
    t = gate_one(ad, "abc009_a/9001", GateConfig(min_loop_pool=2, date_min="2000-01-01"))
    assert not t.admitted and t.failed_gate == "G1"
    # The per-case reasons survive the rejection: `freeze` records one for
    # every exclusion, and a G1 rejection is where they matter most.
    assert set(t.unusable.values()) & {"nondeterministic",
                                       "reference_disagrees_with_out"}


# ── a variance that was structurally zero ─────────────────────────────


def test_remeasure_executes_rather_than_replaying_the_store(
        adapter, frozen, run_dir, stores, monkeypatch, capsys):
    """Going through `scan()` is served from the store on every repeat — after
    the audit pass every probe is a hit — so all R draws are byte-identical and
    the reported variance is a structural zero. That number is the input to
    every CPU confidence interval in the project."""
    from caster import cli
    from caster.run.episode import audit_task

    store, patches = stores
    task = "abc001_a/1001"
    shas = [patches.put("n=int(input())\nprint(n*3)\n")]
    matrix = {task: audit_task(adapter, task, shas, patches=patches, store=store,
                               concurrency=1)}

    calls: list[tuple[str, str]] = []
    real_run = adapter.run

    def spy(t, probe, patch):      # noqa: ANN001
        calls.append((t, probe))
        return real_run(t, probe, patch)

    monkeypatch.setattr(adapter, "run", spy)

    args = cli.build_parser().parse_args(
        ["audit", "--remeasure-cost", "100", "--remeasure-repeats", "3",
         "--remeasure-seed", "1"])
    out = cli._remeasure(args, adapter, store, patches, matrix)["remeasure"]

    assert out["keys_sampled"] > 0
    assert len(calls) == out["keys_sampled"] * 3     # it EXECUTED, R times each
    assert out["within_key_cv_mean"] is not None
    assert "flip_rate" in out


def test_remeasure_refuses_fewer_than_three_repeats():
    """One re-execution gives a difference, not a variance."""
    from caster import cli

    args = cli.build_parser().parse_args(["audit", "--remeasure-repeats", "2"])
    with pytest.raises(SystemExit, match="at least 3"):
        cli._remeasure(args, None, None, None, {})


# ── a restart policy wired to the wrong signal is inert ───────────────


def test_the_chat_arm_restarts_on_failed_repairs_not_on_outages(
        adapter, frozen, run_dir, stores, fake_client):
    """ChatRepair restarts after N consecutive FAILED REPAIR ATTEMPTS. Keyed on
    backend outages the policy never fires, and this is the one arm that can
    then walk itself into the context ceiling."""
    store, patches = stores
    client = fake_client(default_source="n=int(input())\nprint(n*3)\n")
    spec = policies.arm("km+chat").cell("abc001_a/1001", 0,
                                        transcript_restart_after=2)
    r = run_episode(spec, adapter=adapter, client=client, store=store,
                    patches=patches, cfg=EpisodeConfig(budget=6))
    assert r.records[-1].history_restarts >= 2
    # ... and the transcript is actually bounded by the restarts.
    assert "Previous attempts" not in client.prompts[-1] or \
        client.prompts[-1].count("### ") <= 2


# ── a key field that changes nothing ─────────────────────────────────


def test_passing_block_actually_changes_the_prompt(
        adapter, frozen, run_dir, stores, fake_client):
    """`passing_block` is in `CellSpec`, hence in the cell key and the episode
    id. A key field read nowhere is the rule inverted.

    The obvious implementation is dead code — fill it on accept, and the
    episode ends before it can render — and an oracle that stops at the first
    divergence throws away the probes a refuted candidate PASSED, even though
    the scan knows them.
    """
    from caster.keys import CellSpec

    store, patches = stores
    prompts: dict[bool, str] = {}
    keys: dict[bool, str] = {}
    for on in (False, True):
        # A patch that is right for small n and wrong for large: under
        # `natural` order the scan then deterministically passes several probes
        # before it reaches a refuter, which is what gives the block anything to
        # say. (A patch refuted by probe 1 leaves nothing surviving, which is
        # also correct and is why calls this a zero-extra-cost fact rather
        # than a free one.)
        client = fake_client(
            default_source="n=int(input())\nprint(n*2 if n < 10 else n*3)\n")
        spec = CellSpec(task="abc001_a/1001", seed=0, policy="natural",
                        steer_on=True, passing_block=on)
        run_episode(spec, adapter=adapter, client=client, store=store,
                    patches=patches, cfg=EpisodeConfig(budget=4))
        prompts[on] = client.prompts[-1]
        keys[on] = spec.key()

    assert keys[True] != keys[False]            # it is a real cell-key field
    assert prompts[True] != prompts[False]
    assert "already produced the right answer" in prompts[True]
    assert "already produced the right answer" not in prompts[False]


# ── second review pass ───────────────────────────────────────────────────────


def test_a_long_output_renders_identically_cold_and_warm(
        adapter, frozen, run_dir, stores, fake_client):
    """The survivor of the first pass.

    The store keeps a 240-char head, and `snippet()` reported the length of what
    it was handed. So a cold run printed "[truncated, 8,890 chars total]" and a
    warm replay printed "[truncated, 240 chars total]" — or, for a long single
    line, no marker at all. Different prompt bytes, different cache key, a
    different draw, decided by whether the store happened to be warm.
    """
    store, patches = stores
    noisy = "n=int(input())\nprint('\\n'.join(str(i) for i in range(2000)))\n"
    spec = policies.arm("km+steer").cell("abc001_a/1001", 0)

    cold = fake_client(default_source=noisy)
    run_episode(spec, adapter=adapter, client=cold, store=store, patches=patches,
                cfg=EpisodeConfig(budget=3))
    warm = fake_client(default_source=noisy)
    run_episode(spec, adapter=adapter, client=warm, store=store, patches=patches,
                cfg=EpisodeConfig(budget=3))

    assert cold.prompts == warm.prompts
    tail = cold.prompts[-1]
    assert "your last attempt printed" in tail
    # ... and the marker still reports the TRUE size, not the head's.
    assert "truncated, 240 chars total" not in tail
    assert "chars total" in tail

    # A long SINGLE line is the nastier case: the head is exactly 240, so a
    # naive length check finds nothing to truncate and drops the marker.
    one_line = "n=int(input())\nprint('9' * 5000)\n"
    spec2 = policies.arm("km+steer").cell("abc002_b/2002", 0)
    a = fake_client(default_source=one_line)
    run_episode(spec2, adapter=adapter, client=a, store=store, patches=patches,
                cfg=EpisodeConfig(budget=3))
    b = fake_client(default_source=one_line)
    run_episode(spec2, adapter=adapter, client=b, store=store, patches=patches,
                cfg=EpisodeConfig(budget=3))
    assert a.prompts == b.prompts


def test_the_regression_signal_is_an_arm_so_it_can_be_run():
    """Making `passing_block` live switched /the line off everywhere:
    no arm set it, so the block rendered in nothing that would run.

    It is speculative — nothing prices it yet — which is why it is an arm
    rather than a default, with a plain contract arm beside it to compare.
    """
    with_pass = [a.arm_id for a in policies.ARMS.values()
                 if a.overrides.get("passing_block")]
    assert with_pass == ["km+steer-pass"]
    assert policies.arm("km+steer").overrides.get("passing_block") is None
    assert "km+steer-pass" in policies.DEFAULT_GRID


def test_an_incomplete_floor_arm_may_not_freeze_the_denominator(tmp_path):
    """`GAPS` is non-fatal, so `caster merge` succeeds on a partial grid — and
    a repair rate frozen from one shard is the denominator of every band in
    RQ3a/RQ3b, forever, because `freeze_repair_rates` only ever adds keys."""
    from caster.report import analyse
    from caster.report.records import RoundRecord

    def r(task, seed, accept):
        return RoundRecord(run_id="x", episode_id=f"{task}{seed}", task=task,
                           seed=seed, round_index=1, policy="no-mem",
                           accept=accept, cell={"model": "m"})

    partial = [r("t1", 0, True)]
    doc = analyse.summarise(partial, budget=4, pool_dir=tmp_path,
                            universe=["t1", "t2", "t3"])
    assert doc["floor_arm_complete"] is False
    assert doc["repair_rates_source"] == "computed-not-frozen"
    assert not (tmp_path / "repair_rates.json").exists()

    # Deliberate is fine; accidental is not.
    forced = analyse.summarise(partial, budget=4, pool_dir=tmp_path,
                               universe=["t1", "t2", "t3"], force_freeze=True)
    assert forced["repair_rates_source"] == "computed-and-frozen"


def test_full_suite_does_not_early_stop_in_a_replay():
    """Arm 0's whole job is to be the measured upper bound Observation 1 is
    argued against. Early-stopping it reports ~1/N of its true cost."""
    from caster.report import analyse

    kill = {"p1": {"a": "wrong", "b": "wrong", "c": "pass"}}
    cost = {"p1": {"a": 1.0, "b": 1.0, "c": 1.0}}
    stop = analyse.replay_policy(kill, cost, ["p1"], policy="natural")
    full = analyse.replay_policy(kill, cost, ["p1"], policy="full-suite")
    assert stop["test_calls"] == 1
    assert full["test_calls"] == 3
    assert full["mean_rank_of_firing_probe"] == 1.0    # still the FIRST refuter


def test_the_client_skips_its_own_ledger_row_when_the_gateway_wrote_one(
        run_dir, stub_sdk, monkeypatch):
    """Fix 9's client half rests on the SDK surfacing unknown top-level fields
    through `model_extra`. Untested, a future SDK that drops them would send
    every shard back to writing its own row — one call, N charges — with
    nothing to catch it."""
    import json as _json

    from caster.llm.client import LLMClient

    stamped = stub_sdk("print(1)\n", extra={"_caster_gateway": True,
                                            "_caster_backend": "local:x",
                                            "_caster_usd": 0.5})
    c = LLMClient(model="m", cache_dir=run_dir.CACHE_DIR, calls_log=run_dir.CALLS_LOG,
                  price_in=1_000_000.0, price_out=1_000_000.0)
    monkeypatch.setattr(c, "_sdk", lambda: stamped)
    c.complete("p", nonce="n1")
    assert not run_dir.CALLS_LOG.exists() or run_dir.CALLS_LOG.read_text().strip() == ""

    plain = stub_sdk("print(1)\n")
    c2 = LLMClient(model="m", cache_dir=run_dir.CACHE_DIR, calls_log=run_dir.CALLS_LOG,
                   price_in=1_000_000.0, price_out=1_000_000.0)
    monkeypatch.setattr(c2, "_sdk", lambda: plain)
    c2.complete("p", nonce="n2")
    rows = [_json.loads(l) for l in run_dir.CALLS_LOG.read_text().splitlines() if l.strip()]
    assert len(rows) == 1 and rows[0]["nonce"] == "n2"
    assert rows[0]["usd"] == pytest.approx(24.0)      # 11 in + 13 out, priced here


def test_the_cache_is_what_makes_a_rerun_free(run_dir, stub_sdk, monkeypatch):
    """Invariant 8, against the real client rather than a stand-in."""
    from caster.llm.client import LLMClient

    sdk = stub_sdk("print(1)\n")
    c = LLMClient(model="m", cache_dir=run_dir.CACHE_DIR, calls_log=run_dir.CALLS_LOG)
    monkeypatch.setattr(c, "_sdk", lambda: sdk)
    first = c.complete("the prompt", nonce="n")
    assert c.live_calls == 1
    second = LLMClient(model="m", cache_dir=run_dir.CACHE_DIR,
                       calls_log=run_dir.CALLS_LOG)
    monkeypatch.setattr(second, "_sdk", lambda: sdk)
    assert second.complete("the prompt", nonce="n") == first
    assert second.live_calls == 0 and len(sdk.calls) == 1


def test_a_shard_registers_with_the_gateway_for_the_whole_walk(
        adapter, frozen, run_dir, fake_client):
    """"teardown is driven by a refcount, not by shard exit". Endpoints
    with no producer are decoration: a shard between calls still owns the pool."""
    from caster.run.driver import Grid, walk

    client = fake_client(default_source="n=int(input())\nprint(n*3)\n")
    g = Grid(tasks=(frozen["pool_order"][0],), seeds=(0,), policies=("km",), budget=1)
    seen: list[int] = []
    for _ in walk(g, g.cells(), adapter=adapter, client=client,
                  rows_path=run_dir.DATA_DIR / "r.jsonl",
                  done_path=run_dir.DATA_DIR / "d.done"):
        seen.append(client.sessions)
    assert seen == [1]                 # registered for the duration
    assert client.sessions == 0        # and released at the end


def test_the_control_arm_excludes_g4s_correlated_siblings(adapter, corpus_root):
    """G4's rejects share a test pool with an ADMITTED fault — the exact
    correlation G4 exists to remove, which would then sit inside the arm meant
    to measure what the pool-size threshold cut."""
    from conftest import CORRECT, FAULTY, _write
    from caster.bench.adapter import ConDefectsAdapter
    from caster.bench.gate import GateConfig, freeze

    sibling = corpus_root / "Code" / "abc001_a" / "Python" / "1003"
    _write(sibling / "faultyVersion.py", FAULTY.replace("n * 3", "n * 7"))
    _write(sibling / "correctVersion.py", CORRECT)
    try:
        fresh = ConDefectsAdapter(root=corpus_root, timeout=10.0)
        doc = freeze(fresh, cfg=GateConfig(min_loop_pool=6, date_min="2000-01-01",
                                           control_size=10))
        for name, t in doc["tasks"].items():
            if t["control"]:
                assert t["failed_gate"] in ("G3", "G5", "G6"), name
    finally:
        for f in sibling.iterdir():
            f.unlink()
        sibling.rmdir()


def test_the_scan_estimate_knows_about_vs_candidates():
    """Arm 18 scans k candidates per round; counting cells × budget understates
    the one arm whose scan cost is not budget-shaped."""
    from caster.run.driver import plan

    g = Grid(tasks=("t1",), seeds=(0,), policies=("km", "km+steer+vs"), budget=10)
    p = plan(g, resume=False)
    assert p.est_scans == 10 + 5 * 10


def test_coalescing_survives_a_proxy_that_normalises_header_case():
    """HTTP header names are case-insensitive. An exact-case lookup silently
    disables coalescing and the bill goes back to N×, with nothing in the log."""
    from caster.llm.gateway import Backend, Gateway

    # A short service timeout: this request is a FOLLOWER by construction, and
    # a follower waits on the leader's generation.
    gw = Gateway([Backend(url="http://x/v1", model="m")], queue_timeout=0.01,
                 service_timeout=0.2)
    leader, _ = gw.coalescer.begin("k")
    assert leader
    status, _, _ = gw.forward(b"{}", {"x-caster-cache-key": "k"})
    assert status == 504                               # it waited, as a follower
    assert gw.stats["coalesced"] == 1

    # ... and the exact-case header still works, so nothing regressed.
    gw.coalescer.begin("k2")
    gw.forward(b"{}", {"X-Caster-Cache-Key": "k2"})
    assert gw.stats["coalesced"] == 2
