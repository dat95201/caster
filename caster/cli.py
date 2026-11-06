"""`caster` — ten verbs.

    freeze propose evaluate audit merge analyse report serve preflight smoke

`preflight` and `smoke` are verbs rather than modes of something else: they are
the two things a session runs *first*, and burying them is how they stop being
run. `gateway` is a **mode of `serve`**, because the gateway *owns* backend
lifecycle — the thing that starts replicas and the thing that routes to them
must not be two commands that can disagree about which replicas exist.

**There is deliberately no `evaluate --full-scan`.**

* `caster audit` is the dedicated full-scan pass: every probe against every
  patch, no early stop, **no ledger**, at a *declared* concurrency, before any
  arm runs. It writes the kill matrix and the cost table, each with its own
  digest. Letting an arm populate the cost table would make that table a
  function of which arm ran first and of the concurrency it ran under.
* the no-early-stop *policy* is `--policies full-suite` (arm 0), which
  **replays** that matrix for free.

Every flag below appears in exactly one of `keys.KEY_FLAGS` /
`keys.NONKEY_FLAGS`, and a flag in neither fails the build — `caster preflight`
and `tests/test_invariants.py` both assert it (Invariant 6).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import time
from typing import Any, Sequence

from caster import paths
from caster.keys import PRIORS, TIE_BREAKS, CellSpec, check_flag_partition
from caster.run import policies


# ── argument parser ──────────────────────────────────────────────────────────


def _add_cell_knobs(p: argparse.ArgumentParser) -> None:
    """KEY flags — they change what the cell IS."""
    p.add_argument("--tasks", default="", help="comma-separated task names, or empty for the frozen pool")
    p.add_argument("--seeds", default="0,1,2",
                   help="comma-separated seeds. 3, not 5: with a corpus this "
                        "size the task-level n is ample and seeds only cut "
                        "within-task noise")
    p.add_argument("--policies", default=",".join(policies.DEFAULT_GRID),
                   help="comma-separated arm ids")
    p.add_argument("--model", default="", help="proposer model id; pin a hosted model by exact snapshot, never an alias")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--reasoning-effort", dest="reasoning_effort", default="")
    p.add_argument("--prior", choices=sorted(PRIORS), default="shipped",
                   help="Beta prior on K. Both ship as a pre-registered pair")
    p.add_argument("--cold-start", dest="cold_start",
                   choices=("max", "median", "q75"), default="max")
    p.add_argument("--explore-p", dest="explore_p", type=float, default=0.15)
    p.add_argument("--overlap-thr", dest="overlap_thr", type=float, default=0.75,
                   help="defer a probe when this much of its rejection set is "
                        "already inside one that passed this round; 0 disables")
    p.add_argument("--overlap-min-rej", dest="overlap_min_rej", type=int, default=1,
                   help="refutations a probe needs before it may suppress anything")
    p.add_argument("--quarantine-timeouts", dest="quarantine_timeouts",
                   action="store_true",
                   help="never serve a stored timeout, even a confirmed one. Restores the blanket quarantine; the cell key records which regime ran")
    # The default is read OFF `CellSpec`, not written here. A flag default and
    # a spec default are two places for one value, and they had already drifted:
    # the spec said "rank" while this said "hash", so every CLI run would have
    # kept the old order while the roster and the tests reported the new one.
    p.add_argument("--tie-break", dest="tie_break", choices=TIE_BREAKS,
                   default=CellSpec.tie_break,
                   help="'hash' is the ABLATION that prices the benchmark's own "
                        "file order; 'rank' (default) leaves it alone")
    # NOT exposed: --k-facts, --disclosure, --k-candidates. Every Block B arm
    # sets all three in its own overrides (the roster), so a grid-wide value
    # would lose to the arm on exactly the arms where it matters — a flag that
    # is parsed, keyed and then discarded, which is the drift bug with the
    # arrow reversed. The evidence budget, the disclosure policy and the VS
    # width ARE arms: `km+steer-k1`, `km+steer-k2`, `km+steer-sample`,
    # `km+steer-noreq`, `km+steer+vs`. Add a row to the roster, not a flag.
    #
    # The four knobs above stay grid-wide because they are swept ACROSS arms:
    # both priors over arms 5–11, and both tie-breaks on the audit subset.


def _add_run_knobs(p: argparse.ArgumentParser) -> None:
    """NON-key flags — they change how the cell is PRODUCED or INSPECTED."""
    p.add_argument("--budget", type=int, default=20,
                   help="rounds per cell. Deliberately NOT in the cell key")
    p.add_argument("--no-resume", dest="no_resume", action="store_true",
                   help="resume is the DEFAULT; this forces a re-walk")
    p.add_argument("--from-cache-only", dest="from_cache_only", action="store_true",
                   help="refuse to CALL THE MODEL: a cache miss is a loud "
                        "failure, not a silent draw")
    p.add_argument("--on-cache-miss", dest="on_cache_miss",
                   choices=("skip", "stop"), default="skip",
                   help="what a replay-only run does when phase 1 never "
                        "produced something it needs. 'skip' abandons THAT "
                        "CELL whole (no rows, no done-index entry, so it "
                        "re-walks later), records it in misses.jsonl and "
                        "exits non-zero at the end; 'stop' is the old "
                        "behaviour and kills the shard, taking its whole "
                        "remaining range with it")
    p.add_argument("--replay-only", dest="replay_only", action="store_true",
                   help="refuse to RUN THE SANDBOX too: every probe must be "
                        "servable from the verdict store. Checks the "
                        "'after `caster audit`, every policy sees "
                        "sandbox_runs == 0'. Needs an audit pass first")
    p.add_argument("--force-full-budget", dest="force_full_budget", action="store_true",
                   help="keep drawing past the first accept so π̂ is unbiased")
    p.add_argument("--control", dest="control", action="store_true",
                   help="walk the UNFILTERED CONTROL ARM instead of the pool: the "
                        "G1-only survivors a threshold cut. Reported "
                        "separately, whatever it shows — never pooled with the "
                        "main grid, because they are not in pool_order")
    p.add_argument("--dry-run", dest="dry_run", action="store_true")
    p.add_argument("--first", type=int, default=None, help="1-based inclusive range start")
    p.add_argument("--last", type=int, default=None, help="1-based inclusive range end")
    p.add_argument("--shard", type=int, default=0,
                   help="this shard's index. Its rows go to shards/sNN/ — a "
                        "non-numeric label would collapse onto shard 0 and "
                        "overwrite a real shard's rows")
    p.add_argument("--run-id", dest="run_id", default="local")
    p.add_argument("--timeout", type=float, default=None, help="sandbox seconds")
    p.add_argument("--concurrency", type=int, default=None,
                   help="declared phase-2 concurrency, stamped into the cost table")
    p.add_argument("--gateway-url", dest="gateway_url", default=None)
    p.add_argument("--context-tokens", dest="context_tokens", type=int, default=None)
    p.add_argument("--spend-cap", dest="spend_cap", type=float, default=None)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="caster", description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", dest="run_dir", default=None,
                    help="overrides RUN_DIR; one path resolver decides where a run writes (Invariant 5)")
    sub = ap.add_subparsers(dest="command", required=True)

    # 1 ── freeze
    p = sub.add_parser("freeze", help="apply G1–G6, split P_loop/P_eval, write pool/tasks.json")
    p.add_argument("--condefects-root", dest="condefects_root", default=None)
    p.add_argument("--test-dir", dest="test_dir", default=None)
    p.add_argument("--min-pool", dest="min_pool", type=int, default=1,
                   help="floor on |P_loop| (default 1: a pool with no probe is "
                        "unschedulable). Raise it to re-impose the old "
                        "pre-treatment screen — `--min-pool 20` reproduces the "
                        "former corpus; otherwise `analyse` bands by pool size "
                        "after the run")
    p.add_argument("--date-min", dest="date_min", default=None,
                   help="G5 contamination cut (unset = no cut; `analyse` bands "
                        "by contest era instead)")
    p.add_argument("--lines-max", dest="lines_max", type=int, default=None,
                   help="G6 source-length screen (unset = no ceiling; "
                        "truncation shows up as context_overflow)")
    p.add_argument("--dedup-per-group", dest="dedup_per_group",
                   action="store_true",
                   help="G4: keep one fault per coding task (default off; the "
                        "cluster bootstrap resamples coding tasks instead)")
    p.add_argument("--holdout", dest="holdout", type=float, default=0.0,
                   help="hold this fraction of each pool out of the loop "
                        "(default 0: the oracle is the WHOLE pool and "
                        "success@B is a plausible-patch rate). 0.3 restores "
                        "the P_loop/P_eval split for measuring overfitting")
    p.add_argument("--control-size", dest="control_size", type=int, default=0,
                   help="the unfiltered control arm over G1-only survivors")
    p.add_argument("--pool-order", dest="pool_order",
                   choices=("interleave", "contest"), default="interleave")
    p.add_argument("--pool-seed", dest="pool_seed", type=int, default=20_260_101)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--tasks", default="",
                   help="comma-separated fault names: freeze ONLY these, as a "
                        "universe of their own. For a pilot — the digests and "
                        "ranges then refer to this corpus and not the full one")
    p.add_argument("--first", type=int, default=None,
                   help="1-based inclusive range start over the candidates")
    p.add_argument("--last", type=int, default=None,
                   help="1-based inclusive range end")
    p.add_argument("--shard", type=int, default=0,
                   help="this shard's index; its gate records go to "
                        "pool/gate/sNN.jsonl")
    p.add_argument("--assemble", action="store_true",
                   help="build tasks.json from every shard's checkpoint. G4, "
                        "the control arm and pool_sha256 need the whole corpus, "
                        "so they happen here and not in a shard")
    p.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="print the candidate count and exit; the one authority "
                        "on how to cut ranges")

    # 2 ── propose  (phase 1)
    p = sub.add_parser("propose", help="phase 1: fill the response cache and the patch store")
    _add_cell_knobs(p)
    _add_run_knobs(p)

    # 3 ── evaluate (phase 2)
    p = sub.add_parser("evaluate", help="phase 2: walk (task × seed × policy)")
    _add_cell_knobs(p)
    _add_run_knobs(p)

    # 4 ── audit
    p = sub.add_parser("audit", help="the dedicated full-scan pass: kill matrix + cost table")
    p.add_argument("--tasks", default="")
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--timeout", type=float, default=None)
    p.add_argument("--remeasure-cost", dest="remeasure_cost", type=float, default=0.0,
                   help="re-run this PERCENT of store keys to estimate within-cell cpu variance")
    p.add_argument("--remeasure-repeats", dest="remeasure_repeats", type=int, default=3,
                   help="R ≥ 3: one re-execution gives a difference, not a variance")
    p.add_argument("--remeasure-seed", dest="remeasure_seed", type=int, default=11,
                   help="an unseeded sample makes the variance estimate itself irreproducible")

    # 5 ── merge
    p = sub.add_parser("merge", help="the six audits, then collapse the shards")
    p.add_argument("--budget", type=int, default=20)
    p.add_argument("--strict", action="store_true", help="exit non-zero on a non-fatal audit too")
    p.add_argument("--out", default=None)

    # 6 ── analyse
    p = sub.add_parser("analyse", help="aggregate; threats first, then the arms")
    p.add_argument("--budget", type=int, default=20)
    p.add_argument("--freeze-rates", dest="freeze_rates", action="store_true",
                   help="freeze the per-(task, model) repair rate even though "
                        "the floor arm is incomplete. It is the denominator of "
                        "every band and it never corrects itself, so this is "
                        "deliberate or it is a mistake")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--out", default=None)

    # 7 ── report
    p = sub.add_parser("report", help="the HTML report")
    p.add_argument("--out", default=None)
    p.add_argument("--title", default="CASTER run report")

    # 8 ── serve  (start | verify | unload | stop | gateway)
    p = sub.add_parser("serve", help="local backend lifecycle, and the gateway")
    p.add_argument("subcommand", choices=("start", "verify", "unload", "stop", "gateway"))
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--model", default="")
    p.add_argument("--context", type=int, default=0)
    p.add_argument("--local-replicas", dest="local_replicas", default="auto")
    p.add_argument("--max-replicas", dest="max_replicas", type=int, default=8)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--keep-serving", dest="keep_serving", action="store_true")
    p.add_argument("--backend", choices=("local", "cloud", "hybrid"), default="local")
    p.add_argument("--queue-timeout", dest="queue_timeout", type=float, default=None,
                   help="willingness to WAIT FOR A SLOT; unbounded by default")
    p.add_argument("--service-timeout", dest="service_timeout", type=float, default=1800.0,
                   help="ONE GENERATION once admitted; size from max_tokens/tok_per_sec×margin")
    p.add_argument("--spend-cap", dest="spend_cap", type=float, default=100.0)
    p.add_argument("--vram", type=int, default=0, help="free VRAM bytes, for --local-replicas auto")
    p.add_argument("--model-bytes", dest="model_bytes", type=int, default=0)
    p.add_argument("--num-parallel", dest="num_parallel", type=int, default=1)
    p.add_argument("--tok-per-sec", dest="tok_per_sec", type=float, default=0.0,
                   help="measured generation speed. With it, --service-timeout is "
                        "derived from the WORK (max_tokens/tok_per_sec × margin), "
                        "which is the number queueing was blowing")
    p.add_argument("--tok-budget", dest="tok_budget", type=int, default=16000,
                   help="the largest output budget any arm will request. It "
                        "SIZES THE SERVICE TIMEOUT and nothing else — the per-"
                        "call max_tokens is derived per program and is in the "
                        "cache key, which this is not")
    p.add_argument("--cloud-url", dest="cloud_url", default="",
                   help="--backend hybrid: the overflow endpoint")

    # 9 ── preflight
    p = sub.add_parser("preflight", help="gateway, windows, corpus, stores, disk, flag lists")
    p.add_argument("--gateway-url", dest="gateway_url", default=None)
    p.add_argument("--strict", action="store_true")

    # 10 ── smoke
    p = sub.add_parser("smoke", help="a real end-to-end run on a pinned tiny universe")
    p.add_argument("--live", action="store_true",
                   help="make exactly ONE real call, to prove the local/cloud path")
    p.add_argument("--from-cache-only", dest="from_cache_only", action="store_true", default=None)
    p.add_argument("--budget", type=int, default=3)
    return ap


def all_dests(ap: argparse.ArgumentParser) -> list[str]:
    """Every `dest` the CLI exposes, including subparsers. Feeds Invariant 6."""
    out: list[str] = []

    def walk(parser: argparse.ArgumentParser) -> None:
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                for child in action.choices.values():
                    walk(child)
                continue
            if action.dest and action.dest != argparse.SUPPRESS:
                out.append(action.dest)

    walk(ap)
    return out


# ── helpers ──────────────────────────────────────────────────────────────────


def _csv(value: str) -> tuple[str, ...]:
    return tuple(v.strip() for v in value.split(",") if v.strip())


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(v) for v in _csv(value))


def _adapter(args: argparse.Namespace, frozen: dict[str, Any] | None = None) -> Any:
    from caster.bench.adapter import ConDefectsAdapter
    from caster.bench.gate import install

    kw: dict[str, Any] = {}
    if getattr(args, "condefects_root", None):
        kw["root"] = args.condefects_root
    if getattr(args, "test_dir", None):
        kw["test_dir"] = args.test_dir
    if getattr(args, "timeout", None):
        kw["timeout"] = args.timeout
    ad = ConDefectsAdapter(**kw)
    if frozen is not None:
        install(ad, frozen, include_control=bool(getattr(args, "control", False)))
    return ad


def _frozen(required: bool = True) -> dict[str, Any] | None:
    path = paths.POOL_DIR / "tasks.json"
    if not path.is_file():
        if required:
            raise SystemExit(
                f"no frozen corpus at {path}. Run `caster freeze` first — nothing "
                f"is sampled after the gate, so the corpus IS the gated pool."
            )
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _model(args: argparse.Namespace) -> str:
    """The proposer model id, resolved ONCE, for the key and the client alike.

    `--model` defaults to the empty string and `LLMClient` falls back to
    `$MODEL`, so the client ran under `qwen2.5-coder:7b` while every CellSpec
    recorded `model=""`. `model` IS a `CellSpec` field — it is in the cell key
    — so the whole grid was keyed under a model that is not the one that drew
    the patches. Three things follow, and the pilot hit all three:

      * two models collide. `(task, seed, policy)` under 7B and under 3B
        produce the IDENTICAL `episode_id`, so `needs_walk` calls the second
        one already done and never walks it, and `collapse` — last write wins
        — drops whichever row file is read first. That is the "a later run
        overwrote the earlier one" data loss, and it is silent;
      * `proposal_error_table` and `repair_rates` are per `(policy, model)`
        and every row reports `model=""`;
      * the merge's protocol audit compares `meta.json`, which records
        `grid.model or client.model` and therefore looked right while the keys
        under it did not.

    Resolving here means the key, the record and the call can no longer
    disagree. `KEY_VERSION` is bumped with it: the meaning of every existing
    key changed, so old and new rows must not pool.
    """
    return (getattr(args, "model", "") or os.environ.get("MODEL", "")).strip()


def _client(args: argparse.Namespace) -> Any:
    from caster.llm.client import LLMClient

    return LLMClient(
        model=_model(args) or None,
        base_url=getattr(args, "gateway_url", None),
        temperature=getattr(args, "temperature", None),
        reasoning_effort=getattr(args, "reasoning_effort", None) or None,
        context_tokens=getattr(args, "context_tokens", None),
        spend_cap=getattr(args, "spend_cap", None),
    )


def _grid(args: argparse.Namespace, frozen: dict[str, Any]) -> Any:
    from caster.bench.gate import control_tasks
    from caster.run.driver import Grid

    if getattr(args, "control", False):
        tasks = _csv(args.tasks) or tuple(control_tasks(frozen))
        if not tasks:
            raise SystemExit(
                "--control: the frozen corpus has no control arm. Re-freeze with "
                "`caster freeze --control-size N` — it is the only thing that "
                "makes the ≥20 scope condition checkable rather than asserted."
            )
    else:
        tasks = _csv(args.tasks) or tuple(frozen["pool_order"])
    from caster.exec.sandbox import DEFAULT_TIMEOUT

    return Grid(
        tasks=tasks, seeds=_ints(args.seeds), policies=_csv(args.policies),
        budget=args.budget, model=_model(args), temperature=args.temperature,
        reasoning_effort=args.reasoning_effort, prior=args.prior,
        cold_start=args.cold_start, explore_p=args.explore_p,
        overlap_thr=args.overlap_thr, overlap_min_rej=args.overlap_min_rej,
        serve_stable_timeouts=not getattr(args, 'quarantine_timeouts', False),
        tie_break=args.tie_break,
        sandbox_timeout=(args.timeout if args.timeout else DEFAULT_TIMEOUT),
    )


# ── the verbs ────────────────────────────────────────────────────────────────


def gate_dir() -> pathlib.Path:
    """Where a freeze checkpoints. One JSONL per shard, appended per fault."""
    return paths.POOL_DIR / "gate"


def _audit_shard_ranges(files: Sequence[pathlib.Path], names: Sequence[str],
                        owner: dict[str, int], here: str = "") -> None:
    """Every record must come from the shard whose range contains it.

    The failure this exists for: a sharded `Test/` tree plus a `caster freeze`
    with no `--first/--last`. The gate walks all of it, finds no test data for
    the faults that shard did not ship, and records them as G2 rejects. They
    look exactly like real ones, `--assemble` sees no gap, and the corpus comes
    out with most of the benchmark marked "no test data" — a number that is a
    property of which archive was unpacked.

    Also checks the shards agree on the candidate list, which catches a shard
    run against a different `Code/` checkout.
    """
    ranges: dict[int, tuple[int, int]] = {}
    digests: dict[int, str] = {}
    for path in files:
        side = path.with_suffix(".range.json")
        if not side.is_file():
            continue
        try:
            d = json.loads(side.read_text(encoding="utf-8"))
            ranges[int(d["shard"])] = (int(d["first"]), int(d["last"]))
            digests[int(d["shard"])] = str(d.get("candidate_order_sha256", ""))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    if not ranges:
        return      # a single unsharded pass records none, and needs none

    distinct = {d for d in digests.values() if d}
    if len(distinct) > 1:
        raise SystemExit(
            f"the shards disagree about the candidate list ({len(distinct)} "
            f"different digests). They were run against different Code/ "
            f"checkouts, so their ranges do not refer to the same faults."
        )
    if here and distinct and here not in distinct:
        # Gating on a runner and assembling somewhere else is the normal way to
        # work; a Code/ that differs between the two is the way it goes wrong.
        # `--first 500` then means a different fault on each machine, and every
        # range check below would be comparing indices into two different lists.
        raise SystemExit(
            f"the shards were gated against a different Code/ than this "
            f"machine has (shards {sorted(distinct)[0][:12]}, here "
            f"{here[:12]}). Assemble where the candidate list matches, or "
            f"re-clone Code/ to the revision the runners used."
        )

    index = {n: i for i, n in enumerate(sorted(names), 1)}
    strays: list[str] = []
    for name, shard in owner.items():
        i = index.get(name)
        if i is None or shard not in ranges:
            continue
        lo, hi = ranges[shard]
        if not (lo <= i <= hi):
            strays.append(f"{name} (#{i}) from shard {shard} [{lo}-{hi}]")
    if strays:
        raise SystemExit(
            f"{len(strays)} fault(s) were gated by a shard that does not own "
            f"them, e.g. {'; '.join(strays[:3])}.\n"
            f"That is what `caster freeze` with no --first/--last looks like on "
            f"a machine holding only one Test_shardN.zip: every other fault is "
            f"recorded as a G2 reject that nothing downstream can tell from a "
            f"real one. Delete pool/gate/ on that machine and re-run it with "
            f"its range."
        )


def cmd_freeze(args: argparse.Namespace) -> int:
    from caster.bench.gate import FrozenTask, GateConfig, assemble, gate_many, write

    paths.announce("freeze")
    adapter = _adapter(args)
    names = sorted(adapter.tasks())
    if args.tasks:
        # A named universe, for a pilot. The candidate list IS these faults, so
        # `--first/--last`, `--assemble` and `pool_sha256` all refer to them and
        # nothing silently compares a ten-fault corpus against the full one.
        wanted = list(_csv(args.tasks))
        unknown = [n for n in wanted if n not in set(names)]
        if unknown:
            raise SystemExit(f"not in Code/: {', '.join(unknown[:5])}")
        names = sorted(wanted)
    if args.limit:
        names = names[: args.limit]
    if not names:
        raise SystemExit(adapter.setup_hint)
    cfg = GateConfig(min_loop_pool=args.min_pool, date_min=args.date_min,
                     dedup_per_group=args.dedup_per_group,
                     lines_max=args.lines_max, holdout=args.holdout,
                     control_size=args.control_size, pool_order=args.pool_order,
                     seed=args.pool_seed)
    out = paths.POOL_DIR / "tasks.json"
    t0 = time.time()

    # The ONE authority on how big a freeze is, so a fleet never cuts ranges
    # past the end of it — the contract `evaluate --dry-run` already has.
    if args.dry_run:
        print(json.dumps({"universe": len(names), "first": 1, "last": len(names)},
                         indent=2))
        return 0

    t_first = time.time()

    def progress(i: int, n: int, name: str) -> None:
        # Every fault on a small universe. A pilot of ten printed nothing at all
        # until it finished, which reads exactly like a hang — and a freeze is
        # the one stage slow enough that the difference matters.
        if n <= 50 or i % 25 == 0 or i == n:
            done = i - 1
            rate = (time.time() - t_first) / done if done else 0.0
            eta = f"  ~{rate * (n - done) / 60:.0f}m left" if done >= 2 else ""
            print(f"  gate {i}/{n}  {name}{eta}", file=sys.stderr, flush=True)

    if not args.assemble:
        lo = max(1, args.first or 1)
        hi = min(len(names), args.last or len(names))
        chunk = names[lo - 1:hi]
        if not chunk:
            raise SystemExit(f"--first {lo} --last {hi} selects nothing of "
                             f"{len(names)} candidates")
        done_path = gate_dir() / f"s{int(args.shard):02d}.jsonl"
        # The range this shard OWNS, recorded beside its records. With a sharded
        # `Test/` tree a missing range is not a slower freeze, it is a silently
        # wrong corpus: the gate would walk every fault, find no test data for
        # the ones this shard did not ship, and write them out as G2 rejects
        # that `--assemble` has no way to tell from real ones.
        done_path.parent.mkdir(parents=True, exist_ok=True)
        done_path.with_suffix(".range.json").write_text(json.dumps({
            "shard": int(args.shard), "first": lo, "last": hi,
            "universe": len(names),
            "candidate_order_sha256": hashlib.sha256(
                "\n".join(names).encode()).hexdigest(),
        }, indent=2), encoding="utf-8")
        gate_many(adapter, chunk, cfg, progress=progress, done_path=done_path)
        if args.first or args.last or args.shard:
            # A shard owns no corpus: G4, the control arm and `pool_sha256` all
            # need every fault, so the document is built once, afterwards.
            print(json.dumps({
                "shard": int(args.shard), "first": lo, "last": hi,
                "gated": len(chunk), "checkpoint": str(done_path),
                "seconds": round(time.time() - t0, 1),
                "next": "caster freeze --assemble (once every shard is done)",
            }, indent=2))
            return 0

    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists; pass --overwrite to replace it. A corpus "
                         f"silently re-frozen is a corpus nobody can reproduce.")

    seen: dict[str, FrozenTask] = {}
    owner: dict[str, int] = {}
    files = sorted(gate_dir().glob("s*.jsonl"))
    for path in files:
        shard = int(path.stem[1:]) if path.stem[1:].isdigit() else -1
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                seen[rec["name"]] = FrozenTask.from_dict(rec)
                owner[rec["name"]] = shard
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    _audit_shard_ranges(files, names, owner,
                        hashlib.sha256("\n".join(names).encode()).hexdigest())
    missing = [n for n in names if n not in seen]
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(names)} candidates have no gate record "
            f"(e.g. {', '.join(missing[:5])}). A corpus assembled over a gap has "
            f"a wrong funnel and a pool_sha256 that means nothing. Re-run the "
            f"shard that covers them."
        )
    doc = assemble(adapter, names, [seen[n] for n in names], cfg)
    if args.control_size > 0 and not any(
            (t or {}).get("control") for t in doc["tasks"].values()):
        # The control arm is drawn from faults a THRESHOLD gate cut (G3, G5,
        # G6). Every one of those is off by default now (decision 22), so
        # `--control-size 40` against default thresholds has nothing to sample
        # and draws zero — silently, until `caster evaluate --control` dies two
        # stages later with "the frozen corpus has no control arm".
        print(f"note: --control-size {args.control_size} drew NOTHING. The "
              f"control arm samples faults rejected by a threshold gate (G3 "
              f"--min-pool, G5 --date-min, G6 --lines-max) and all three are "
              f"off here, so no fault was cut by one. Re-freeze with the "
              f"threshold whose effect you want measured, e.g. `--min-pool 20 "
              f"--control-size {args.control_size}`, or drop the flag.",
              file=sys.stderr)
    write(doc, out)
    print(json.dumps({
        "pool_sha256": doc["pool_sha256"],
        "candidate_order_sha256": doc["candidate_order_sha256"],
        "selected_sha256": doc["selected_sha256"],
        "shards": len(files),
        "funnel": doc["funnel"], "seconds": round(time.time() - t0, 1),
        "out": str(out),
    }, indent=2))
    return 0


def cmd_propose(args: argparse.Namespace) -> int:
    """Phase 1. Block A only — Block B's prompt depends on the ledger."""
    frozen = _frozen()
    grid = _grid(args, frozen)
    block_b = [p for p in grid.policies if policies.arm(p).block == "B"]
    if block_b:
        print(
            f"note: {', '.join(block_b)} are Block B — their round-t prompt is a "
            f"function of the ledger at round t, which phase 2 produces. The two "
            f"phases are NOT separable for them, so they are collected by "
            f"`caster evaluate`. Dropping them from this phase-1 pass.",
            file=sys.stderr,
        )
    grid = type(grid)(**{**grid.__dict__,
                         "policies": tuple(p for p in grid.policies
                                           if policies.arm(p).block == "A")})
    if not grid.policies:
        print("nothing to do: every requested arm is Block B.", file=sys.stderr)
        return 0
    # Phase 1 draws the FULL budget unconditionally: it has no oracle, so it
    # cannot know which round would have accepted, and the floor arm is an
    # estimator of π, not just a baseline.
    args.force_full_budget = True
    return _walk(args, grid, frozen, label="propose")


def cmd_evaluate(args: argparse.Namespace) -> int:
    frozen = _frozen()
    return _walk(args, _grid(args, frozen), frozen, label="evaluate")


def _walk(args: argparse.Namespace, grid: Any, frozen: dict[str, Any],
          *, label: str) -> int:
    from caster.run import shard
    from caster.run.driver import (collection_record, plan, protocol_record,
                                   walk, write_meta)
    from caster.run.episode import EpisodeConfig

    paths.announce(label)
    # Each phase owns its namespace. `evaluate` must not resume from
    # `propose`'s done-index: the two walk the same cell keys (the phase is in
    # no key) but produce different rows, and the skip was silent.
    out_dir = shard.shard_dir(args.shard, phase=label)
    done_paths = shard.done_inputs(phase=label) or [paths.CELLS_DONE]
    p = plan(grid, first=args.first, last=args.last,
             done_paths=[*done_paths, out_dir / "cells.done"],
             resume=not args.no_resume)
    print(p.render())
    if args.dry_run:
        return 0
    if not p.cells:
        print("nothing to walk — every selected cell is already done.")
        return 0

    adapter = _adapter(args, frozen)
    client = _client(args)
    cfg = EpisodeConfig(
        budget=grid.budget, from_cache_only=args.from_cache_only,
        replay_only=args.replay_only,
        force_full_budget=args.force_full_budget, concurrency=args.concurrency,
    )
    meta = protocol_record(grid, client, shard=str(args.shard),
                           extra={"pool_sha256": frozen["pool_sha256"],
                                  "phase": label,
                                  "collection": {
                                      **collection_record(cfg),
                                      "control": bool(getattr(args, "control", False)),
                                  }})
    # Reusing a shard index under a different model, key version, sandbox
    # timeout or served window appends two instruments to one `rounds.jsonl`
    # — and rewriting `meta.json` erases the evidence that they differ, so the
    # merge audit passes. Refuse before the first row rather than after.
    clash = shard.meta_conflict(meta, out_dir)
    if clash:
        raise SystemExit(f"{label}: {clash}")
    write_meta(meta, out_dir / "meta.json")

    accepted = 0
    t0 = time.time()
    misses: list[dict[str, Any]] = []

    def progress(i: int, n: int, spec: Any) -> None:
        print(f"  [{i}/{n}] {spec.policy:16s} {spec.task} seed {spec.seed}",
              file=sys.stderr, flush=True)

    for result in walk(grid, p.cells, adapter=adapter, client=client,
                       rows_path=out_dir / "rounds.jsonl",
                       done_path=out_dir / "cells.done",
                       cfg=cfg, run_id=args.run_id, progress=progress,
                       on_cache_miss=getattr(args, "on_cache_miss", "skip"),
                       misses=misses):
        accepted += int(result.accepted)
    print(json.dumps({"cells": len(p.cells), "accepted": accepted,
                      "skipped_cache_miss": len(misses),
                      "seconds": round(time.time() - t0, 1),
                      "rows": str(out_dir / "rounds.jsonl")}, indent=2))
    if misses:
        # Still a build failure — just not one that takes the rest of the
        # range with it. The cells are unwritten and unindexed, so re-issuing
        # the identical command after phase 1 (or `caster audit`) fills the
        # gap walks exactly these and nothing else.
        tasks = sorted({m["task"] for m in misses})
        print(f"\n{len(misses)} cell(s) SKIPPED on a cache miss across "
              f"{len(tasks)} task(s) — phase 1 never produced what they need. "
              f"They are unwritten and will re-walk. See "
              f"{out_dir / 'misses.jsonl'}.\n"
              f"  e.g. {', '.join(tasks[:5])}\n"
              f"  first reason: {misses[0]['reason']}", file=sys.stderr)
        return 3
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    from caster.exec.sandbox import protocol_fields
    from caster.exec.store import PatchStore, VerdictStore
    from caster.run.episode import audit_task
    from caster.run import shard
    from caster.report.records import read_rows

    paths.announce("audit")
    frozen = _frozen()
    adapter = _adapter(args, frozen)
    # The audit is what MAKES a timeout confirmed, so it must never be
    # satisfied by one: serving a confirmed timeout here would stop the
    # pass that establishes the next confirmation.
    store = VerdictStore(protocol_fields(timeout=adapter.timeout),
                         serve_stable_timeouts=False)
    patches = PatchStore()

    tasks = list(_csv(args.tasks)) or list(frozen["pool_order"])
    if args.limit:
        tasks = tasks[: args.limit]

    # The patches to scan are whatever phase 1 drew for these tasks — and
    # phase 1 now writes one level deeper, where `shard.inputs()` cannot see
    # it by design. The audit runs BETWEEN the two phases, so reading only the
    # evidence namespace would have found nothing at exactly the moment this
    # stage is meant to run. Both, with phase 2's rows second so a patch phase
    # 1 drew is scanned whether or not an arm has replayed it yet.
    by_task: dict[str, list[str]] = {}
    for row in read_rows([*shard.phase_inputs("propose"), *shard.inputs()]):
        if row.patch_sha and row.task in set(tasks):
            by_task.setdefault(row.task, [])
            if row.patch_sha not in by_task[row.task]:
                by_task[row.task].append(row.patch_sha)
    if not by_task:
        raise SystemExit(
            "no patches to audit — run `caster propose` first. The audit pass "
            "scores the patch sequence phase 1 drew; it does not draw its own. "
            f"Looked in {paths.DATA_REL}/shards/propose/*/rounds.jsonl and "
            f"{paths.DATA_REL}/shards/*/rounds.jsonl."
        )

    matrix: dict[str, dict[str, dict[str, str]]] = {}
    for i, task in enumerate(sorted(by_task), 1):
        print(f"  [{i}/{len(by_task)}] {task}  {len(by_task[task])} patches",
              file=sys.stderr, flush=True)
        matrix[task] = audit_task(adapter, task, by_task[task], patches=patches,
                                  store=store, concurrency=args.concurrency)

    out = paths.DATA_DIR / "kill_matrix.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(matrix, separators=(",", ":")), encoding="utf-8")
    # A `timeout` verdict is written but NEVER SERVED (`VerdictStore.get`),
    # so every one of these is a guaranteed cache miss in a `--replay-only`
    # phase 2 — which used to kill the shard and now skips the cell. Counting
    # them here is the difference between "the replay will be short by N
    # cells" and finding out two stages later.
    quarantined = sum(1 for task, by_sha in matrix.items()
                      for sha, per in by_sha.items()
                      for v in per.values() if v == "timeout")
    print(json.dumps({
        "tasks": len(matrix), "cost_digest": store.cost_digest(),
        "concurrency": args.concurrency, **store.stats(),
        "quarantined_timeout_verdicts": quarantined, "out": str(out),
    }, indent=2))
    if quarantined:
        print(f"note: {quarantined} (patch, probe) pair(s) ended in a timeout. "
              f"A timeout is written but never served, so a --replay-only run "
              f"will skip any cell whose schedule reaches one. Raise --timeout "
              f"and re-audit, or expect those cells in GAPS.", file=sys.stderr)
    if args.remeasure_cost > 0:
        print(json.dumps(_remeasure(args, adapter, store, patches, matrix), indent=2))
    return 0


def _remeasure(args: argparse.Namespace, adapter: Any, store: Any,
               patches: Any, matrix: dict[str, Any]) -> dict[str, Any]:
    """The within-key CPU variance. Three unstated decisions, decided.

    `--remeasure-cost PCT --remeasure-repeats R --remeasure-seed S`: PCT is a
    **percentage** (the name says so and the name wins), the sample is
    **seeded** (an unseeded sample makes the variance estimate itself
    irreproducible), and **R ≥ 3**, because one re-execution gives a difference,
    not a variance.

    It **executes**, bypassing the verdict store. Going through `scan()` would
    be served from the store on every repeat — after the audit pass every probe
    is a hit — so all R draws would be byte-identical and the reported variance
    would be a structural zero. That number is the input to every CPU
    confidence interval in the project, so a zero there is worse than no
    number at all.

    It samples **(patch, probe)** keys uniformly, which is the unit the store is
    keyed on and the unit whose variance the cost table inherits — not whole
    tasks.
    """
    import random
    import statistics

    if args.remeasure_repeats < 3:
        raise SystemExit("--remeasure-repeats must be at least 3: one "
                         "re-execution gives a difference, not a variance")
    keys = [(task, sha, probe)
            for task, per in matrix.items()
            for sha, verdicts in per.items()
            for probe in verdicts]
    if not keys:
        return {"remeasure": {"keys_sampled": 0}}
    rng = random.Random(args.remeasure_seed)
    rng.shuffle(keys)
    n = max(1, int(len(keys) * args.remeasure_cost / 100.0))

    cvs: list[float] = []
    flips = 0
    for task, sha, probe in keys[:n]:
        source = patches.require(sha)
        draws: list[float] = []
        verdicts: set[str] = set()
        for _ in range(args.remeasure_repeats):
            v = adapter.run(task, probe, source)
            if v.tier < 0:
                break
            draws.append(v.cpu_s)
            verdicts.add(v.status)
        if len(draws) < args.remeasure_repeats:
            continue
        if len(verdicts) > 1:
            # the other open measurement: the flip rate at this concurrency.
            flips += 1
            store.note_timeout_flip(sha, probe, sorted(verdicts))
        mean = statistics.fmean(draws)
        if mean > 0:
            cvs.append(statistics.pstdev(draws) / mean)
    return {
        "remeasure": {
            "keys_sampled": n, "pct": args.remeasure_cost,
            "repeats": args.remeasure_repeats, "seed": args.remeasure_seed,
            "within_key_cv_mean": round(statistics.fmean(cvs), 5) if cvs else None,
            "within_key_cv_p90": (round(sorted(cvs)[int(0.9 * (len(cvs) - 1))], 5)
                                  if cvs else None),
            "verdict_flips": flips,
            "flip_rate": round(flips / n, 5) if n else None,
            "note": "An input to the CI, not the CI: resample the cost table "
                    "from these draws and re-run the offline replay per "
                    "bootstrap iteration.",
        }
    }


def _control_tasks(frozen: dict[str, Any]) -> tuple[str, ...]:
    """The declared control faults, or () when the corpus drew none.

    Read here rather than through `gate.control_tasks` so a frozen document
    written before `--control-size` existed — which has no `control` flag on
    any task — is an empty tuple instead of a KeyError at merge time.
    """
    tasks = frozen.get("tasks") or {}
    return tuple(sorted(n for n, t in tasks.items() if (t or {}).get("control")))


def cmd_merge(args: argparse.Namespace) -> int:
    from caster.report.records import write_rows
    from caster.run import shard

    paths.announce("merge")
    frozen = _frozen(required=False) or {}
    audits, rows = shard.merge_audit(universe=tuple(frozen.get("pool_order", ())),
                                     budget=args.budget,
                                     control=_control_tasks(frozen))
    for a in audits:
        print(a.render())
    bad = shard.fatal(audits)
    if bad:
        print(f"\n{len(bad)} FATAL audit(s). A merged file is not evidence that the "
              f"grid is complete (Invariant 12). Nothing was written.", file=sys.stderr)
        return 2
    out = pathlib.Path(args.out) if args.out else paths.ROUNDS
    n = write_rows(rows, out)
    print(f"\nmerged {n} rows -> {out}")
    if args.strict and any(not a.ok for a in audits):
        return 1
    return 0


def cmd_analyse(args: argparse.Namespace) -> int:
    from caster.report import analyse
    from caster.report.records import read_rows

    paths.announce("analyse")
    rows = list(read_rows([paths.ROUNDS]))
    if not rows:
        raise SystemExit(f"no rows at {paths.ROUNDS} — run `caster merge` first "
                         f"(Invariant 12: merge audit before analysis).")
    frozen = _frozen(required=False) or {}
    doc = analyse.summarise(rows, budget=args.budget, pool_dir=paths.POOL_DIR,
                            universe=tuple(frozen.get("pool_order", ())),
                            control=_control_tasks(frozen),
                            force_freeze=args.freeze_rates)
    # The bootstrap reads the same main grid the tables do.
    declared = set(_control_tasks(frozen))
    cs = [c for c in analyse.cells(rows, budget=args.budget)
          if c.task not in declared]
    doc["bootstrap"] = [
        analyse.bootstrap_cpu(cs, p, iterations=args.bootstrap)
        for p in sorted({c.policy for c in cs})
    ]
    out = pathlib.Path(args.out) if args.out else paths.DATA_DIR / "summary.json"
    analyse.write(doc, out)
    print(json.dumps({"cells": doc["cells"], "arms": len(doc["arms"]),
                      "out": str(out)}, indent=2))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from caster.report import charts

    path = paths.DATA_DIR / "summary.json"
    if not path.is_file():
        raise SystemExit(f"no {path} — run `caster analyse` first.")
    summary = json.loads(path.read_text(encoding="utf-8"))
    out = pathlib.Path(args.out) if args.out else paths.DATA_DIR / "report.html"
    charts.write(summary, out, title=args.title)
    print(str(out))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from caster.llm import serve as backend
    from caster.llm.gateway import (
        Backend, Gateway, auto_replicas, serve_forever, service_timeout_for,
        DEFAULT_PORT,
    )

    if args.subcommand == "gateway":
        if args.local_replicas == "auto":
            n, bounds = auto_replicas(
                vram_free_bytes=args.vram or 0, model_bytes=args.model_bytes or 1,
                shards=args.shards, max_replicas=args.max_replicas)
            print(f"[serve] replicas={n}  bounds={bounds}  "
                  f"(VRAM-first, shard-count-second — shards do not create capacity)")
        else:
            n, bounds = int(args.local_replicas), {"explicit": int(args.local_replicas)}
        replicas = []
        if args.backend != "cloud":
            ports = backend.plan_ports(n)
            replicas = [backend.start(port, args.model, args.context,
                                      num_parallel=args.num_parallel) for port in ports]

        pool = [Backend(url=r.base_url, model=args.model, kind="local",
                        concurrency=args.num_parallel, context_tokens=args.context,
                        owned=r.owned and not r.adopted)
                for r in replicas]
        if args.backend in ("cloud", "hybrid"):
            # one pool, mixed members. In `hybrid` the cloud is the
            # OVERFLOW: admit to a local slot when one is free, else to cloud if
            # under the spend cap, else queue. `Gateway._pick` implements that
            # preference; the merge audit still refuses to join a mixed-MODEL
            # pool, because a mixed backend is poolable and a mixed model is not.
            if not args.cloud_url:
                raise SystemExit(f"--backend {args.backend} needs --cloud-url")
            pool.append(Backend(url=args.cloud_url.rstrip("/"), model=args.model,
                                kind="cloud", concurrency=8,
                                context_tokens=args.context))
        if not pool:
            raise SystemExit("no backends: --backend local needs a replica, "
                             "--backend cloud needs --cloud-url")

        service_timeout = args.service_timeout
        if args.tok_per_sec > 0:
            service_timeout = service_timeout_for(args.tok_budget, args.tok_per_sec)
            print(f"[serve] service_timeout {service_timeout:.0f}s "
                  f"= {args.tok_budget} tokens / {args.tok_per_sec:g} tok/s × 2 "
                  f"(sized from the WORK, not guessed)")

        def _respawn(b: Any) -> None:
            port = int(b.url.rsplit(":", 1)[1].split("/")[0])
            backend.start(port, args.model, args.context,
                          num_parallel=args.num_parallel)

        gw = Gateway(
            pool,
            queue_timeout=args.queue_timeout, service_timeout=service_timeout,
            spend_cap=args.spend_cap, respawn=_respawn,
        )
        try:
            serve_forever(gw, port=args.port or DEFAULT_PORT)
        finally:
            if not args.keep_serving:
                for r in replicas:
                    # Adopted replicas are unloaded, never killed: on a shared
                    # machine the process is a few MB, the weights gigabytes.
                    backend.stop(r) if (r.owned and not r.adopted) else backend.unload(r)
        return 0

    port = args.port or backend.FIRST_REPLICA_PORT
    # Ownership is read from the record this machine keeps, not assumed. A bare
    # Replica(owned=False) still reaches `unload()` before `stop()`'s ownership
    # check, which would drop the weights of a listener CASTER never started —
    # Not ours → never unload, never stop.
    rep = backend.describe(port, model=args.model, context=args.context)
    if args.subcommand == "start":
        rep = backend.start(port, args.model, args.context,
                            num_parallel=args.num_parallel)
        print(json.dumps({"port": rep.port, "url": rep.base_url, "owned": rep.owned,
                          "adopted": rep.adopted, "read_only": rep.read_only}, indent=2))
    elif args.subcommand == "verify":
        served = backend.verify(rep)
        print(json.dumps({"port": port, "served_context": served}, indent=2))
        if not served:
            # 0 is not a window, it is the absence of one — nothing is listening,
            # or the server would not say. Printing it under exit 0 reads as
            # "verified", which is the one thing this subcommand exists to deny.
            print(f"[FAIL] nothing on port {port} reported a served context "
                  f"window. Start the pool first (`caster serve gateway "
                  f"--backend local --model ... --context ...`); a window that "
                  f"cannot be read cannot be checked against the prompts.",
                  file=sys.stderr)
            return 1
    elif args.subcommand == "unload":
        print(json.dumps({"unloaded": backend.unload(rep)}, indent=2))
    elif args.subcommand == "stop":
        print(json.dumps({"stopped": backend.stop(rep)}, indent=2))
    return 0


def _window_shortfall(window: int, model: str) -> list[tuple[str, int]]:
    """Admitted tasks whose (source in + rewrite out) exceeds `window`.

    Returns `(task, tokens needed)`, largest first. An unset window returns
    nothing: with `LLM_CONTEXT_TOKENS=0` the per-call check is off by
    configuration and there is no number to compare against.
    """
    from caster.steer.template import window_needed

    if not window:
        return []
    frozen = _frozen(required=False) or {}
    tasks = [t for t in frozen.get("pool_order", ())]
    if not tasks:
        return []
    import argparse as _argparse

    adapter = _adapter(_argparse.Namespace())
    out: list[tuple[str, int]] = []
    for name in tasks:
        try:
            source = adapter.program(name).buggy_source
        except (OSError, FileNotFoundError, KeyError):
            continue
        need = window_needed(source, model=model or None)
        if need > window:
            out.append((name, need))
    return sorted(out, key=lambda r: -r[1])


def cmd_preflight(args: argparse.Namespace) -> int:
    """First thing in any session. Budget: zero. Exits non-zero, one line per failure."""
    import urllib.error
    import urllib.request

    from caster.exec.sandbox import CPU_SOURCE, protocol_fields
    from caster.exec.store import VerdictStore

    failures: list[str] = []
    notes: list[str] = []

    problems = check_flag_partition(all_dests(build_parser()))
    failures.extend(problems)

    if CPU_SOURCE != "getrusage":
        failures.append(
            f"cpu_source={CPU_SOURCE}: Invariant 3 requires "
            f"getrusage(RUSAGE_CHILDREN). Wall clock is contaminated by sibling "
            f"shards and makes m̂ a function of load."
        )

    model = (args.gateway_url and "") or os.environ.get("MODEL", "").strip()

    pool = paths.POOL_DIR / "tasks.json"
    if not pool.is_file():
        failures.append(f"no frozen corpus at {pool} — run `caster freeze`")
    else:
        doc = json.loads(pool.read_text(encoding="utf-8"))
        if doc.get("candidate_order_sha256") and doc.get("selected_sha256"):
            notes.append(f"pool_sha256 {doc['pool_sha256'][:12]} "
                         f"({doc['funnel']['admitted']}/{doc['funnel']['examined']} admitted)")
        asked = float((doc.get("config") or {}).get("holdout") or 0.0)
        if asked > 0 and not any(t.get("p_eval")
                                 for t in doc.get("tasks", {}).values()):
            # Only a failure when one was ASKED for. At `--holdout 0` the oracle
            # is the whole pool by design, and `success@B` is a plausible-patch
            # rate rather than a correctness rate.
            failures.append("--holdout was set but P_eval is empty in the "
                            "frozen corpus — it has to exist before the first "
                            "reported number")
        elif asked == 0:
            notes.append("holdout 0 — oracle is the whole pool; success@B is a "
                         "PLAUSIBLE-patch rate, not a correctness rate")

    # ── the output budget against the served window ────────────────────────
    # G6 (`--lines-max`) is off by default, and the README's reason is that
    # "the real constraint is the output budget, not a line count". That is
    # true and it left the constraint UNMEASURED: with no screen, the corpus
    # admits programs whose source plus its own rewrite does not fit the
    # window, and the first anyone hears of it is `context_overflow` on every
    # round of every seed of those tasks — followed, in a replay-only phase 2,
    # by a cache miss on draws that were never made.
    #
    # The window is a config value, so this belongs in the stage whose job is
    # config. One line, before anything long runs, naming the tasks and the
    # window they would need.
    window = int(os.environ.get("LLM_CONTEXT_TOKENS", 0) or 0)
    if pool.is_file():
        try:
            over = _window_shortfall(window, model)
        except Exception as exc:    # noqa: BLE001 - a missing corpus is not fatal here
            notes.append(f"output-budget check skipped: {exc}")
        else:
            if not window:
                notes.append("LLM_CONTEXT_TOKENS unset — the per-call window "
                             "check is OFF, so an over-long prompt is "
                             "TRUNCATED by the server instead of refused")
            elif over:
                need = max(n for _, n in over)
                failures.append(
                    f"{len(over)} admitted task(s) need more than the "
                    f"{window}-token window declared here; the largest needs "
                    f"{need}. Each burns every round of every seed as "
                    f"context_overflow and leaves a cache miss behind it. "
                    f"Raise LLM_CONTEXT_TOKENS and the server's num_ctx "
                    f"together, or screen them out with `freeze --lines-max`: "
                    + ", ".join(f"{t} ({n})" for t, n in over[:4])
                )
            else:
                notes.append(f"output budget fits the {window}-token window "
                             f"for every admitted task")

    for name, p in (("response cache", paths.CACHE_DIR),
                    ("verdict store", paths.VERDICT_DIR),
                    ("patch store", paths.PATCH_DIR)):
        try:
            p.mkdir(parents=True, exist_ok=True)
            probe = p / ".caster-preflight"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError as exc:
            failures.append(f"{name} at {p} is not writable: {exc}")

    import shutil as _shutil

    free = _shutil.disk_usage(paths.ROOT).free
    if free < 2 * 1024**3:
        failures.append(f"only {free / 1024**3:.1f} GiB free under {paths.ROOT}")
    else:
        notes.append(f"disk {free / 1024**3:.1f} GiB free")

    if not model:
        failures.append(
            "no MODEL set. `propose` and every steered arm die on the first "
            "draw with `no model configured` — after the walk has started and "
            "written rows. Put it in .env (an EXACT snapshot id for a hosted "
            "model; `ollama list` gives the tag for a local one)."
        )
    else:
        notes.append(f"model {model}")

    # Default to the configured endpoint. Checking the gateway only when an
    # operator remembers `--gateway-url` means the one check that costs nothing
    # and catches a dead pool is the one that usually does not run.
    url = args.gateway_url or os.environ.get("LLM_BASE_URL", "").strip() or None
    if url:
        try:
            with urllib.request.urlopen(url.rstrip("/").removesuffix("/v1") + "/healthz",
                                        timeout=10) as r:
                blob = json.loads(r.read().decode())
            for b in blob.get("backends", []):
                if b.get("context_tokens") in (0, None):
                    failures.append(f"backend {b.get('url')} has no verified served "
                                    f"context window")
            notes.append(f"gateway ok: {len(blob.get('backends', []))} backend(s), "
                         f"spent ${blob.get('spent_usd', 0):.4f}/{blob.get('cap_usd')}")
        except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError) as exc:
            failures.append(f"gateway at {url} unreachable: {exc}")

    from caster.exec.sandbox import DEFAULT_TIMEOUT

    proto = protocol_fields(timeout=DEFAULT_TIMEOUT)
    notes.append(f"protocol_sha {VerdictStore(proto).protocol_sha} "
                 f"(sandbox_timeout {proto['sandbox_timeout_sec']}s, "
                 f"interpreter {proto['interpreter']})")

    for n in notes:
        print(f"[ok  ] {n}")
    for f in failures:
        print(f"[FAIL] {f}", file=sys.stderr)
    return 1 if failures else 0


# Invariant 14 — the smoke universe is PINNED in the repo, not sampled. A smoke
# test whose corpus moves cannot fail reproducibly. Its rows go under
# `data/<run>/smoke/`, which `caster merge`'s glob cannot match.
#
# Resolved at call time rather than at import, because `--run-dir` and
# `CASTER_ROOT` both move `paths.ROOT` and a module-level constant would capture
# the wrong tree.


def smoke_universe_file() -> pathlib.Path:
    return paths.ROOT / "smoke" / "universe.json"


def smoke_expected_file() -> pathlib.Path:
    return paths.ROOT / "smoke" / "expected.json"


def cmd_smoke(args: argparse.Namespace) -> int:
    """A real end-to-end run on a fixed tiny universe. Two variants, both required.

    `--from-cache-only` (the default) **must complete with zero model calls and
    zero network**. This is the CI gate, and it is also the test that phase 1
    and phase 2 are genuinely separable: a model call here is a build
    failure.

    `--live` makes exactly **one** real call through the gateway, to prove the
    local/cloud path, the window check and the spend ledger all work.
    Deliberately one call, so it is cheap enough to run every time.
    """
    from caster.exec.sandbox import protocol_fields
    from caster.exec.store import PatchStore, VerdictStore
    from caster.keys import CellSpec
    from caster.report.records import CellDone, CellWriter
    from caster.run.episode import EpisodeConfig, run_episode

    universe_file = smoke_universe_file()
    if not universe_file.is_file():
        raise SystemExit(f"no pinned smoke universe at {universe_file}")
    universe = json.loads(universe_file.read_text(encoding="utf-8"))
    from_cache_only = (not args.live) if args.from_cache_only is None else args.from_cache_only
    # "Deliberately one call, so it is cheap enough to run every time".
    # One CELL is not one call: a cell at --budget 3 draws three times. The live
    # variant therefore walks one cell of one round.
    budget = 1 if args.live else args.budget

    frozen = _frozen()
    adapter = _adapter(args, frozen)
    client = _client(args)
    store = VerdictStore(protocol_fields(timeout=adapter.timeout))
    patches = PatchStore()
    paths.ensure(paths.SMOKE_DIR)
    writer = CellWriter(paths.SMOKE_DIR / "rounds.jsonl", paths.SMOKE_DIR / "cells.done")

    # Counted on the CLIENT, not by diffing the call ledger: the gateway owns
    # that ledger now and may be running under a different RUN_DIR, which would
    # make a real call look like a cache hit — on the one check that exists to
    # prove the live path works.
    calls_before = getattr(client, "live_calls", 0)
    results = []
    for task in universe["tasks"]:
        for seed in universe["seeds"]:
            for arm_id in universe["policies"]:
                spec = policies.arm(arm_id).cell(task, seed, model=universe.get("model", ""))
                r = run_episode(spec, adapter=adapter, client=client, store=store,
                                patches=patches,
                                cfg=EpisodeConfig(budget=budget,
                                                  from_cache_only=from_cache_only))
                for rec in r.records:
                    writer.add(rec)
                writer.commit(CellDone(cell_key=r.cell_key, episode_id=r.episode_id,
                                       task=task, seed=seed, policy=arm_id,
                                       rounds_written=len(r.records),
                                       rounds=r.rounds,
                                       budget=budget, accepted=r.accepted,
                                       truncated=r.truncated))
                results.append(r)
                if args.live:
                    break   # exactly one real call
            if args.live:
                break
        if args.live:
            break

    new_calls = getattr(client, "live_calls", 0) - calls_before
    observed = {
        "cells": len(results), "model_calls": new_calls,
        "accepted": sum(r.accepted for r in results),
        "rounds": sum(r.rounds for r in results),
        "first_accept": [r.first_accept_round for r in results],
    }
    print(json.dumps({
        **observed,
        "sandbox_runs": sum(r.sandbox_runs for r in results),
        "verdict_hits": sum(r.verdict_hits for r in results),
        "rows": str(paths.SMOKE_DIR / "rounds.jsonl"),
    }, indent=2))

    # the expected outputs are COMMITTED, so smoke is a regression test on
    # the loop and not merely a liveness check. `--live` skips the comparison: it
    # walks one cell on purpose and is a different measurement.
    expected_file = smoke_expected_file()
    if not args.live:
        if expected_file.is_file():
            expected = json.loads(expected_file.read_text(encoding="utf-8"))
            drift = {k: (expected.get(k), observed.get(k))
                     for k in expected if k != "model_calls" and expected[k] != observed.get(k)}
            if drift:
                print(f"FAIL: smoke output drifted from {expected_file}: "
                      f"{json.dumps(drift)}", file=sys.stderr)
                return 1
        else:
            expected_file.parent.mkdir(parents=True, exist_ok=True)
            expected_file.write_text(json.dumps(observed, indent=2), encoding="utf-8")
            print(f"wrote baseline expectations to {expected_file} — commit it, or "
                  f"the next run has nothing to regress against.",
                  file=sys.stderr)

    if from_cache_only and new_calls:
        print(f"FAIL: --from-cache-only made {new_calls} model call(s). A model call "
              f"here is a build failure: it means phase 1 and phase 2 are not "
              f"separable.", file=sys.stderr)
        return 1
    if args.live:
        if new_calls > 1:
            print(f"FAIL: --live made {new_calls} model calls; one is the "
                  f"contract — it exists to prove the local/cloud path, the "
                  f"window check and the spend ledger, cheaply enough to run "
                  f"every time.", file=sys.stderr)
            return 1
        if new_calls == 0:
            print("note: --live drew from the response cache, so the live path "
                  "was NOT exercised. That is a valid cache hit, not a failure "
                  "— to force a real call, point RUN_DIR at a fresh run or "
                  "remove that one cache key.", file=sys.stderr)
    return 0


COMMANDS = {
    "freeze": cmd_freeze, "propose": cmd_propose, "evaluate": cmd_evaluate,
    "audit": cmd_audit, "merge": cmd_merge, "analyse": cmd_analyse,
    "report": cmd_report, "serve": cmd_serve, "preflight": cmd_preflight,
    "smoke": cmd_smoke,
}
assert len(COMMANDS) == 10, "ten verbs — see this module's docstring"


def main(argv: Sequence[str] | None = None) -> int:
    from caster.errors import CacheMiss

    ap = build_parser()
    args = ap.parse_args(argv)
    if args.run_dir:
        import os

        os.environ["RUN_DIR"] = args.run_dir
        import importlib

        importlib.reload(paths)
    try:
        return COMMANDS[args.command](args)
    except CacheMiss as exc:
        # A replay-only run needed something no earlier phase produced. It is a
        # BUILD FAILURE, so it exits non-zero with the reason on one line —
        # not a traceback, and emphatically not a silently absorbed round
        #.
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
