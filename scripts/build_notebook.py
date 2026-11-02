#!/usr/bin/env python3
"""Build `notebooks/run_colab.ipynb`.

The notebook is generated rather than hand-edited so that the cells stay in one
reviewable file, and so that a change to the pipeline and a change to the
notebook that drives it land in the same diff. Re-run after changing the CLI.
"""
from __future__ import annotations

import json
import pathlib

MD: list[tuple[str, str]] = []
CELLS: list[dict] = []


def md(text: str) -> None:
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)})


def code(stage: str, reads: str, writes: str, skip: str, body: str, *,
         title: str | None = None) -> None:
    rule = "─" * 76
    header = (
        f"# {rule}\n"
        f"# STAGE   {stage}\n"
        f"# READS   {reads}\n"
        f"# WRITES  {writes}\n"
        f"# SKIP?   {skip}\n"
        f"# {rule}\n"
    )
    src = ((f"#@title {title}\n" if title else "") + header + body.strip() + "\n")
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": src.splitlines(True)})


# ── 0 ────────────────────────────────────────────────────────────────────────
md(r"""
# CASTER on Colab

One notebook, the whole pipeline. It is the Colab wrapper around the ten `caster`
verbs; `README.md` and `docs/DECISIONS.md` in the repository are the authority on
what each one does.

**The four rules this notebook enforces, and why each one exists**

1. **Everything worth keeping lives on Drive.** A Colab runtime is reclaimed in
   12–24 hours. The response cache, the verdict store, the patch store and the
   run's rows are symlinked to Drive, so a reclaimed runtime costs you the
   machine and not the work. The three stores are **cross-run by design** — that
   is what makes a re-run free (Invariant 8).

2. **Block A is free; Block B costs one model grid each.** Every unsteered arm
   replays one cached patch sequence, so the thirteen of them together cost
   **zero** model calls after phase 1. The steered arms cannot do that: their
   round-*t* prompt is a function of the ledger at round *t*, which phase 2
   produces, so the dependency is circular, not a cold cache. That asymmetry is
   the dominant scheduling constraint on the project and it is why the cells
   below are in the order they are in.

3. **Nothing long runs before something short has.** `preflight` (budget: zero),
   then `smoke --from-cache-only` (zero model calls, zero network — a model call
   there is a build failure), then the grid.

4. **A merged file is not evidence.** `caster merge` runs six audits, two of
   which exist to catch instrument drift that reaches no cache key at all — a
   shard served a different context window, or cut from a different corpus.
   Analysis refuses to start before they pass (Invariant 12).

**What this costs.** Measured on the completed pilot: **0.366 s per probe run**,
mean pool 42.1 probes. A full `caster audit` over 110 tasks × 100 patches is
~38 CPU-hours. One Colab runtime gives about 5 effective cores, so the work is
split by *measured* cost across N machines and merged — cells B1–B4 below.
""")

# ── 1 config ─────────────────────────────────────────────────────────────────
code(
    "configuration — nothing runs",
    "nothing",
    "python variables and a few environment variables",
    "NO",
    r'''
import os

# ── repository ───────────────────────────────────────────────────────────────
REPO_URL = "REPO_URL"          # e.g. https://github.com/<you>/caster.git
                               # private: https://<TOKEN>@github.com/<you>/caster.git
BRANCH   = "main"
WORKDIR  = "/content/caster"

# ── persistence ──────────────────────────────────────────────────────────────
# Everything that survives a reclaimed runtime hangs off here. Keep Test.zip
# here too: it is 6.4 GB and re-downloading it every session is the single
# largest avoidable cost in the pipeline.
DRIVE_ROOT          = "/content/drive/MyDrive/caster"
TEST_ZIP_DRIVE_PATH = f"{DRIVE_ROOT}/Test.zip"

# ── this run ─────────────────────────────────────────────────────────────────
# RUN_DIR moves every artifact of the run under data/<RUN_DIR>/ and
# logs/<RUN_DIR>/. The three cross-run stores deliberately do NOT follow it.
RUN_DIR = "campaign-7b"

# The proposer. PIN A HOSTED MODEL BY EXACT SNAPSHOT ID, never a bare alias: a
# hosted model silently re-pointed is a different instrument with nothing to say
# so, and the runtime-disagreement audit exists because of exactly that.
MODEL       = "qwen2.5-coder:7b"
TEMPERATURE = 1.0

# The served context window. THE SETTING THAT CAN CORRUPT A RUN SILENTLY: a
# server is free to TRUNCATE an over-long prompt rather than refuse it, and the
# served window reaches neither the cache key nor the record. It lands
# differentially, too — the steered arms carry the longest prompts.
CONTEXT_TOKENS = 32768

# Backend: "local" runs replicas behind the gateway on this runtime; "cloud"
# points at a hosted endpoint. One pool, one code path either way.
# "local" runs replicas behind the gateway on this runtime; "cloud" points at a
# hosted endpoint; "hybrid" is one pool with cloud as the OVERFLOW — admit to a
# local slot when one is free, else cloud if under the spend cap, else queue.
BACKEND = "local"
CLOUD_BASE_URL = ""            # required for "cloud" and "hybrid"

# Measured generation speed, tokens/sec. With it the service timeout is sized
# from the work (max_tokens / tok_per_sec x margin) instead of guessed — which
# is the number queueing was blowing. 0 keeps the flat ceiling.
TOK_PER_SEC = 0.0

# Phase 2 is CPU-bound and parallelises across PROCESSES, never threads:
# getrusage(RUSAGE_CHILDREN) is process-global, so a thread pool would
# attribute a sibling's CPU to whichever probe finished.
SHARDS = 4
BUDGET = 20
SEEDS  = "0,1,2"               # 3, not 5 — with ~1,244 tasks the task-level n is
                               # ample and seeds only cut within-task noise

os.environ.update(
    RUN_DIR=RUN_DIR, MODEL=MODEL, TEMPERATURE=str(TEMPERATURE),
    LLM_CONTEXT_TOKENS=str(CONTEXT_TOKENS),
    CONDEFECTS_ROOT=f"{WORKDIR}/external/ConDefects",
)
print(f"run={RUN_DIR} model={MODEL} backend={BACKEND} shards={SHARDS} budget={BUDGET}")
''',
)

# ── 2 drive ──────────────────────────────────────────────────────────────────
code(
    "mount Drive",
    "your Drive",
    "{DRIVE_ROOT}/{cache,verdicts,patches,data,logs}",
    "NO — without it nothing survives the session",
    r'''
from google.colab import drive
drive.mount("/content/drive")

import os
for sub in ("cache", "verdicts", "patches", "data", "logs"):
    os.makedirs(f"{DRIVE_ROOT}/{sub}", exist_ok=True)
print("persistent root:", DRIVE_ROOT)
!ls -la {DRIVE_ROOT}
''',
)

# ── 3 clone ──────────────────────────────────────────────────────────────────
code(
    "clone, then point the five persistent directories at Drive",
    "GitHub, {DRIVE_ROOT}",
    "/content/caster and five symlinks",
    "no",
    r'''
import os, pathlib, subprocess
os.chdir("/content")
if not pathlib.Path(WORKDIR).exists():
    !git clone --branch {BRANCH} {REPO_URL} {WORKDIR}
os.chdir(WORKDIR)
!git fetch --quiet && git checkout {BRANCH} && git pull --ff-only
!git log --oneline -1

# cache/ verdicts/ patches/ are CROSS-RUN assets (Invariant 8): scoping
# them per run would cold-start every sweep, so they are symlinked whole.
# data/ and logs/ nest RUN_DIR underneath, so they are symlinked whole too.
for sub in ("cache", "verdicts", "patches", "data", "logs"):
    local = pathlib.Path(WORKDIR) / sub
    if local.is_symlink():
        continue
    if local.exists():
        subprocess.run(["cp", "-rn", f"{local}/.", f"{DRIVE_ROOT}/{sub}/"], check=False)
        subprocess.run(["rm", "-rf", str(local)], check=True)
    local.symlink_to(f"{DRIVE_ROOT}/{sub}")
!ls -la | grep -E "cache|verdicts|patches|data|logs"
''',
)

# ── 4 deps ───────────────────────────────────────────────────────────────────
code(
    "python dependencies",
    "pyproject.toml",
    "site-packages",
    "yes on a warm runtime (seconds to confirm)",
    r'''
import os; os.chdir(WORKDIR)
!pip install -q -e ".[dev]"
!python3 -c "import openai, dotenv, numpy, scipy; print('deps ok')"
!caster --help | head -5
''',
)

# ── 5 benchmark ──────────────────────────────────────────────────────────────
code(
    "the benchmark: ConDefects Code (125 MB clone) + Test.zip from Drive",
    "GitHub, {TEST_ZIP_DRIVE_PATH}",
    "external/ConDefects/{Code,Test}",
    "the clone: no. Test.zip: no — a PARTIAL tree silently changes the corpus",
    r'''
import os; os.chdir(WORKDIR)
# The audit at the end of this script is the point. A truncated Test.zip leaves
# some tasks with fewer probes than the contest shipped; G3 then admits or
# rejects them on a pool size that is an artefact of the download, and nothing
# in the run says so. It refuses rather than letting you freeze on it.
!python3 scripts/fetch_condefects.py --test-zip "{TEST_ZIP_DRIVE_PATH}"
''',
)

# ── 6 backend ────────────────────────────────────────────────────────────────
code(
    "the proposer: start the gateway (local) or point at the cloud",
    "nothing",
    "a background gateway on :11534, replicas from :11435",
    "yes when every cell you need is already cached",
    r'''
import os, subprocess, time, urllib.request, json
os.chdir(WORKDIR)

if BACKEND == "local":
    # ollama, then the gateway. The GATEWAY starts the replicas and is the only
    # process that may stop them. If each shard started and stopped the server,
    # the first shard to finish would tear down the backend the others were
    # still using.
    !curl -fsSL https://ollama.com/install.sh | sh > /dev/null 2>&1
    !nohup ollama serve > /content/ollama-root.log 2>&1 &
    time.sleep(5)
    !ollama pull {MODEL}

    # Replicas are sized VRAM-FIRST, shard-count-second — shards do not create
    # capacity. A machine that holds one replica gets one, and the extra shards
    # wait in the gateway's queue, which is correct and no longer fatal.
    gw = subprocess.Popen(
        ["caster", "serve", "gateway",
         "--model", MODEL, "--context", str(CONTEXT_TOKENS),
         "--shards", str(SHARDS), "--num-parallel", "2",
         # Two timeouts, and only one of them is a property of the work.
         # --queue-timeout is left UNBOUNDED: queueing N shards behind one
         # replica is a load signal, not a failure. --service-timeout
         # is derived from the work once --tok-per-sec is measured; until then
         # it falls back to a flat ceiling.
         "--tok-per-sec", str(TOK_PER_SEC), "--tok-budget", "16000",
         "--service-timeout", "1800", "--spend-cap", "25"]
        + (["--backend", "hybrid", "--cloud-url", CLOUD_BASE_URL]
           if BACKEND == "hybrid" else []),
        stdout=open("/content/gateway.log", "a"),
        stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(120):
        try:
            with urllib.request.urlopen("http://127.0.0.1:11534/healthz", timeout=2) as r:
                print(json.dumps(json.loads(r.read()), indent=2)[:600]); break
        except OSError:
            time.sleep(2)
    os.environ["LLM_BASE_URL"] = "http://127.0.0.1:11534/v1"
    os.environ["LLM_API_KEY"] = "unused"
else:
    os.environ["LLM_BASE_URL"] = CLOUD_BASE_URL
    from getpass import getpass
    os.environ["LLM_API_KEY"] = os.environ.get("LLM_API_KEY") or getpass("LLM_API_KEY: ")

print("LLM_BASE_URL =", os.environ["LLM_BASE_URL"])
''',
)

# ── 7 preflight ──────────────────────────────────────────────────────────────
code(
    "preflight — the first thing in any session",
    "the gateway, the corpus, the stores, the disk, the flag lists",
    "nothing",
    "NO. Budget: zero. One line per failure",
    r'''
import os; os.chdir(WORKDIR)
# Checks the served context window per backend, that the corpus is frozen and
# its pool_sha256 matches, that the three stores are writable, that there is
# disk headroom — and that every CLI flag is in exactly one declared list
# (Invariant 6: a flag in neither fails the build).
!caster preflight --gateway-url "$LLM_BASE_URL" || echo "^^ fix these before anything long runs"
''',
)

# ── 8 freeze ─────────────────────────────────────────────────────────────────
md(r"""
---
## Freeze the corpus

One gate, six conditions, applied before any arm runs. **Nothing is sampled
after it**: the corpus *is* the gated pool, in pool order — a pre-treatment
screen on a difficulty proxy puts a covariate upstream of the thing under study
and yields a sample nobody can weight back to the benchmark.

Two things this writes that the rest of the pipeline depends on:

* **`pool_sha256`**, over the probe lists actually found *and* the pool
  ordering. The ordering is a seeded interleave, not contest order: ConDefects'
  own order is correlated with date, rating and therefore pool size, so a killed
  fleet cutting contiguous ranges from it would leave a *systematically* biased
  subset rather than a smaller grid.
* **`P_loop` / `P_eval`**, split 30% within each of F2P and P2P and seeded from
  the task name. `P_eval` is never shown and never gates acceptance; a probe
  from it reaching the loop is a hard stop, not a warning.

`--control-size` draws the unfiltered control arm over the G1-only survivors.
It is the only thing that makes the ≥20 scope condition *checkable* rather than
asserted, and it is pre-registered as "reported whatever it shows".
""")
code(
    "freeze — G1–G6, the split, pool_sha256",
    "external/ConDefects",
    "data/<run>/pool/tasks.json",
    "yes once it exists — re-freezing silently is how a corpus stops reproducing",
    r'''
import os, json, pathlib; os.chdir(WORKDIR)
pool = pathlib.Path(f"data/{RUN_DIR}/pool/tasks.json")
if pool.is_file():
    doc = json.loads(pool.read_text())
    print("already frozen:", doc["pool_sha256"][:16],
          f"{doc['funnel']['admitted']}/{doc['funnel']['examined']} admitted")
else:
    !caster freeze --control-size 40
''',
)
code(
    "the funnel — every fault examined, with the first gate it failed",
    "data/<run>/pool/tasks.json",
    "nothing",
    "no; it is seconds and it is what the paper reports",
    r'''
import json, pathlib, collections, os; os.chdir(WORKDIR)
doc = json.loads(pathlib.Path(f"data/{RUN_DIR}/pool/tasks.json").read_text())
f = doc["funnel"]
print(f"examined {f['examined']} faults -> admitted {f['admitted']}")
for gate, n in sorted(f["rejected_by_gate"].items()):
    if n:
        print(f"   {gate}: {n}")
print(f"|P_loop| median {f['loop_pool_median']} (min {f['loop_pool_min']}, "
      f"max {f['loop_pool_max']})")
print("pool_sha256", doc["pool_sha256"])
# The 20-probe threshold costs 23.2% of the corpus. That is affordable, but it
# is SELECTION ON A VARIABLE CORRELATED WITH THE OUTCOME and must be declared as
# a scope condition: the reported effect is the effect on tasks with a pool
# worth scheduling, and the paper has to say so in exactly those words.
''',
)

# ── 9 smoke ──────────────────────────────────────────────────────────────────
code(
    "smoke — a real end-to-end run on a pinned tiny universe",
    "smoke/universe.json, the caches",
    "data/<run>/smoke/ (which `caster merge` cannot reach — Invariant 14)",
    "NO. Seconds, and it is the CI gate",
    r'''
import os; os.chdir(WORKDIR)
# --from-cache-only MUST complete with zero model calls and zero network. It is
# also the test that phase 1 and phase 2 are genuinely separable: a model
# call here is a build failure, not a slow test.
!caster smoke --budget 3 || echo "^^ smoke failed — do not launch a grid on this"
print()
# Exactly ONE real call, to prove the local/cloud path, the window check and the
# spend ledger all work. Deliberately one, so it is cheap enough to run always.
!caster smoke --live --budget 1
''',
)

# ── 10 phase 1 ───────────────────────────────────────────────────────────────
md(r"""
---
## Phase 1 — PROPOSE (LLM-bound)

Fills the response cache and the patch store. **Draws the full budget
unconditionally**, for two reasons: phase 1 has no oracle, so it cannot know
which round would have accepted and therefore cannot know where to stop; and the
floor arm is an estimator of π, not just a baseline, so early stopping would
bias π̂. The rounds past the first accept are recorded and truncated back out at
analysis.

This is the only cell in the notebook that spends money or GPU time on the model.
""")
code(
    "phase 1 — fill the response cache and the patch store",
    "the frozen pool, the gateway",
    "cache/, patches/, data/<run>/shards/*/rounds.jsonl",
    "yes once the cache is warm — `--dry-run` below says how much is left",
    r'''
import os; os.chdir(WORKDIR)
!caster evaluate --dry-run --policies no-mem --seeds {SEEDS} --budget {BUDGET}
print()
# `--resume` is the DEFAULT, not a flag: re-issuing the identical command after
# a kill is the documented recovery path and must be safe to do blindly.
!bash scripts/fleet.sh launch --shards {SHARDS} -- --policies no-mem --seeds {SEEDS} --budget {BUDGET} --force-full-budget
''',
)
code(
    "watch the fleet",
    "logs/<run>/fleet/manifest.json",
    "nothing",
    "re-run it as often as you like; it returns immediately",
    r'''
import os; os.chdir(WORKDIR)
!bash scripts/fleet.sh status
# `bash scripts/fleet.sh wait` blocks until it drains.
# `bash scripts/fleet.sh stop` SIGTERMs it; a cell is written whole or not at
# all, so there is nothing half-finished to clean up.
''',
)

# ── 11 audit ─────────────────────────────────────────────────────────────────
md(r"""
---
## The audit pass — what makes everything after it free

`caster audit` runs **every** probe against every patch, with no early stop and
no ledger, at a **declared** concurrency, **before any arm runs**. It is the most
expensive thing in the pipeline and it is bought once per model.

It produces three things nothing else can:

* the **full kill vector** per (patch, probe) — the only way to compute an
  honest `K`, and the only way to replay a counterfactual policy offline **with
  zero model calls**. The scheduler comparison this project rests on was built
  exactly this way;
* the **cost table**, written once at one concurrency. If an *arm* populated it,
  the table would be a function of which arm ran first and of the load it ran
  under, and every later arm would inherit that;
* the input to the **leakage bound** — how many patches are refuted *only* by
  the disclosed probes.

After it, every Block A policy evaluates that task with `sandbox_runs == 0`.

`--remeasure-cost` re-runs a seeded sample to estimate within-cell CPU variance.
It is an **input to the confidence interval, not the interval**: with a frozen
cost table, a bootstrap over cells alone measures only scan-order variation and
reports an interval that omits the dominant source of uncertainty, in a
direction that flatters the result.
""")
code(
    "audit — the full-scan pass",
    "patches/, the frozen pool",
    "verdicts/, data/<run>/kill_matrix.json",
    "yes once done for a task; it is cross-run",
    r'''
import os; os.chdir(WORKDIR)
# Concurrency is DECLARED and stamped into the cost table. Keep it the same
# across machines, or the table is a function of each machine's load.
!caster audit --concurrency 4 --remeasure-cost 2 --remeasure-repeats 3 --remeasure-seed 11
''',
)

# ── 12 phase 2 ───────────────────────────────────────────────────────────────
md(r"""
---
## Phase 2 — EVALUATE

**Block A first, because it is free.** Thirteen arms replay one cached patch
sequence; a new policy costs zero model calls and arm comparison is free of
sampling variance by construction, because every arm sees the identical patches.

**Then Block B, one model grid each.** These share round 1 with Block A (empty
ledger ⇒ byte-identical prompt ⇒ one cached draw), so each costs
`(B−1) × tasks × seeds` calls rather than `B ×`.

One sequencing constraint that is not negotiable: **no steered arm may run on a
repair-rate-capped corpus.** A completed pilot of 110 tasks × 550 cells × 33,000
rounds produced **10 accepted patches (0.03%)** because it was cut at
`--max-repair-rate 0.30` and 97 of its 110 tasks sat at repair rate 0. RQ4's
primary metric is rounds-to-first-accept, censored at B; on such a corpus it is
censored almost everywhere, and the same empty denominator defeats RQ5, RQ6 and
RQ7. That cap does not exist in this codebase, and it must not be reintroduced.
""")
code(
    "phase 2, Block A — thirteen arms, zero model calls",
    "cache/, verdicts/, patches/",
    "data/<run>/shards/*/rounds.jsonl",
    "no — but it is CPU only",
    r'''
import os; os.chdir(WORKDIR)
BLOCK_A = ("full-suite,no-mem,natural,random,fault-record,cost-only,kill-only,"
           "coverage,km,km-cs-median,km-cs-q75,km-bucket,km-explore0")
!caster evaluate --dry-run --policies {BLOCK_A} --seeds {SEEDS} --budget {BUDGET}
print()
# --from-cache-only refuses to CALL THE MODEL; --replay-only refuses to RUN
# THE SANDBOX as well. Together they are the "after `caster audit`, every
# policy evaluates that task with sandbox_runs == 0", made checkable instead
# of hoped for: a miss on either side is a loud failure here.
!bash scripts/fleet.sh launch --shards {SHARDS} -- --policies {BLOCK_A} --seeds {SEEDS} --budget {BUDGET} --from-cache-only --replay-only
''',
)
code(
    "phase 2, Block B — the steered arms, one model grid each",
    "the gateway",
    "cache/, data/<run>/shards/*/rounds.jsonl",
    "yes until Block A has run and the bands are known",
    r'''
import os; os.chdir(WORKDIR)
# NOT separable into two phases: the round-t prompt is a function of the ledger
# at round t, which phase 2 produces. The dependency is circular, not a cold
# cache — so these interleave and pay for it (the correction).
BLOCK_B = ("km+steer,km+steer-pass,km+exclusion,km+steer-k1,km+steer-k3,"
           "km+steer-sample,km+steer-noreq,km+steer+vs")
!caster evaluate --dry-run --policies {BLOCK_B} --seeds {SEEDS} --budget {BUDGET}
print()
!bash scripts/fleet.sh launch --shards {SHARDS} -- --policies {BLOCK_B} --seeds {SEEDS} --budget {BUDGET}
''',
)

md(r"""
---
## Two things that cost nothing and settle an open question each

**The prior.** `Beta(½,½)` and `Beta(½,2)` ship as a pre-registered pair, and
the choice between them is settled by replaying both over the full-scan kill matrix
— mean rank of the probe that actually fired, and probes run per refuted round.
That replay needs **no model calls and no sandbox runs**: `caster audit` has
already measured every (patch, probe).

**The scope condition.** The ≥20 `P_loop` threshold costs 23.2% of the corpus.
`caster freeze --control-size 40` drew a uniform sample over the G1-only
survivors, pre-registered as "reported whatever it shows", and `--control`
walks it. It is the only thing that makes the scope condition *checkable*
rather than asserted, and its faults are never in `pool_order`, so no
main-grid analysis can pool them by accident.
""")
code(
    "both priors, replayed over the kill matrix — zero model calls",
    "data/<run>/kill_matrix.json, verdicts/",
    "nothing",
    "no; it is seconds, and the prior question cannot be closed without it",
    r'''
import json, pathlib, os; os.chdir(WORKDIR)
from caster.exec.sandbox import protocol_fields
from caster.exec.store import VerdictStore
from caster.report.analyse import compare_priors

matrix = json.loads(pathlib.Path(f"data/{RUN_DIR}/kill_matrix.json").read_text())
store = VerdictStore(protocol_fields())
cost = {sha: {probe: (store.peek(sha, probe).cpu_s if store.peek(sha, probe) else 0.0)
              for probe in per}
        for task, by_sha in matrix.items() for sha, per in by_sha.items()}
flat = {task: list(by_sha) for task, by_sha in matrix.items()}
kill = {sha: per for by_sha in matrix.values() for sha, per in by_sha.items()}

for row in compare_priors(kill, cost, flat, policy="km"):
    print(f"  {row['prior']:9s} Beta{row['beta']}  "
          f"mean rank {row['mean_rank_of_firing_probe']}  "
          f"probes/refuted round {row['probes_per_refuted_round']}  "
          f"cpu {row['cpu_s']}s")
print()
print("Until these numbers exist, neither prior can be called the right one —")
print("and K=1/2 is NOT 'the exploration': explore_p is.")
''',
)
code(
    "the unfiltered control arm — the 15-19 band the threshold excluded",
    "the frozen corpus's control faults",
    "data/<run>/shards/*/rounds.jsonl",
    "yes until the main grid is done; it is reported separately either way",
    r'''
import os; os.chdir(WORKDIR)
# Never pooled with the main grid: these faults are not in pool_order, so a
# corpus-wide average cannot pick them up by accident.
!caster evaluate --control --policies no-mem,natural,km --seeds 0 --budget {BUDGET}
''',
)

# ── 13 autosave ──────────────────────────────────────────────────────────────
code(
    "autosave — mandatory; a Colab runtime is reclaimed in 12–24 h",
    "data/<run>/",
    "{DRIVE_ROOT} (already symlinked; this flushes and reports)",
    "no",
    r'''
import os, subprocess; os.chdir(WORKDIR)
# data/ cache/ verdicts/ patches/ are already symlinks into Drive, so writes
# land there directly. This loop exists to make the SIZE visible: Drive uploads
# asynchronously and a quiet stall is how a session ends with nothing saved.
subprocess.Popen(["bash", "-c",
  f'while true; do du -sh {DRIVE_ROOT}/data {DRIVE_ROOT}/cache '
  f'{DRIVE_ROOT}/verdicts 2>/dev/null | tr "\\n" " "; echo; sleep 600; done'],
  start_new_session=True)
!du -sh {DRIVE_ROOT}/* 2>/dev/null
print("Lost the runtime? Start a fresh one, run cells 1-4, and re-issue the")
print("identical fleet launch: resume reads cells.done, and a cell is written")
print("whole or not at all, so there is nothing half-finished to clean up.")
''',
)

# ── 14 progress ──────────────────────────────────────────────────────────────
code(
    "progress — what is done, per arm",
    "data/<run>/shards/*/cells.done",
    "nothing",
    "re-run as often as you like; it returns immediately",
    r'''
import os, collections; os.chdir(WORKDIR)
from caster.report.records import load_done
from caster.run import shard

done = {}
for p in shard.done_inputs():
    done.update(load_done(p))
per = collections.Counter(d.policy for d in done.values())
acc = collections.Counter(d.policy for d in done.values() if d.accepted)
trunc = collections.Counter(d.policy for d in done.values() if d.truncated)
print(f"{'arm':20s} {'cells':>7s} {'accepted':>9s} {'truncated':>10s}")
for arm in sorted(per):
    print(f"{arm:20s} {per[arm]:7d} {acc[arm]:9d} {trunc[arm]:10d}")
print(f"\n{sum(per.values())} cells done. TRUNCATED means ended early without")
print("accepting and without reaching the budget — atomicity solves torn rows,")
print("not short episodes.")
''',
)

# ── 15 merge ─────────────────────────────────────────────────────────────────
md(r"""
---
## Merge, analyse, report

**The merge audit runs before analysis, and a merged file is not evidence that
the grid is complete** (Invariant 12). Six checks:

| audit | what it means |
|---|---|
| protocol disagreement | a shard ran under a different model. Rows would never pool. **Hard stop** |
| runtime disagreement | shards differ on served window, sandbox timeout, interpreter, backend. **None of these reaches the cache key**, so those shards re-judged each other's draws against a different instrument. **Hard stop** |
| no protocol record | a hand-run driver. Reported, not fatal |
| GAPS / TRUNCATED | cells missing from an arm; cells that neither accepted nor reached the budget |
| DISAGREEMENT | the same (cell, round) collected twice, differently. **Do not pick a winner** |
| FOREIGN | a task not in the universe — a shard cut from a different corpus |

plus two invariants checked over the whole run: **soundness** (`success@B` and
the first-accept round identical across every Block A arm on *every* cell — a
gap there is a soundness bug, not a finding) and **Invariant 9**
(`(sandbox_runs − timeout_reruns) + verdict_hits == Σ test_calls`).
""")
code(
    "merge — the six audits, then collapse",
    "data/<run>/shards/*/{rounds.jsonl,meta.json}",
    "data/<run>/rounds.jsonl",
    "NO",
    r'''
import os; os.chdir(WORKDIR)
!caster merge --budget {BUDGET}
''',
)
code(
    "analyse and report — threats first, then the arms",
    "data/<run>/rounds.jsonl",
    "data/<run>/summary.json, data/<run>/report.html",
    "no; seconds",
    r'''
import os; os.chdir(WORKDIR)
!caster analyse --budget {BUDGET} --bootstrap 2000
!caster report --title "CASTER · {MODEL} · {RUN_DIR}"

import json, pathlib
s = json.loads(pathlib.Path(f"data/{RUN_DIR}/summary.json").read_text())

# Reported BEFORE any scheduler number: the steered arms carry the longest
# prompts, so every proposal failure lands differentially on exactly the
# comparison the experiment exists to make. Above ~20% a model is a threat to
# validity, not a rung on a ladder.
print("proposal failures, per arm and per model")
for r in s["proposal_errors"]:
    flag = "  <-- THREAT" if r["threat"] else ""
    print(f"  {r['policy']:18s} {r['model']:16s} {r['error_rate']:6.1%}{flag}")

# A steering arm whose realised mean k_facts is below its configured k is
# reporting a different treatment than its label says.
print("\nrealised evidence budget")
for r in s["k_facts"]:
    print(f"  {r['policy']:18s} k={r['configured_k']} realised={r['realised_k_mean']:.2f}"
          f"{'  <-- MISLABELLED' if r['mislabelled'] else ''}")

print("\nCPU seconds per episode")
for r in s["arms"]:
    print(f"  {r['policy']:18s} {r['cpu_s_mean']:9.3f}  probes/ep {r['test_calls_mean']:6.2f}"
          f"  success@B {r['success_at_b']:6.1%}")

print("\nRQ2 — one thing changes per step")
for r in s["decomposition"]:
    d = r["cpu_delta_mean_pct"]
    print(f"  {r['step']:26s} {('%+.2f%%' % d) if d is not None else '—':>10s}"
          f"  ({r['paired_cells']} paired cells)")
''',
)
code(
    "keep the result",
    "data/<run>/report.html",
    "{DRIVE_ROOT}/report-<run>.html",
    "no",
    r'''
import os, shutil; os.chdir(WORKDIR)
src = f"data/{RUN_DIR}/report.html"
dst = f"{DRIVE_ROOT}/report-{RUN_DIR}.html"
if os.path.exists(src):
    shutil.copyfile(src, dst); print("copied ->", dst)
    from IPython.display import IFrame, display
    display(IFrame(src=src, width="100%", height=620))
else:
    print("no report — run `caster analyse` first")
''',
)

# ── part B ───────────────────────────────────────────────────────────────────
md(r"""
---
# Part B — splitting the audit pass across N machines

Part A above is one runtime end to end. Part B is for the one stage that does
not fit in one: `caster audit`, measured at ~38 CPU-hours per 110 tasks × 100
patches. A Colab runtime gives about 5 effective cores, so the work is cut by
**measured cost**, run on N machines, and merged.

Run **B1 on every machine, changing only `SH`.** B2–B3 are per machine; B4 runs
once on the gathering machine.

Two things that bite, both of which cost real time to learn:

* **`nproc` lies.** Colab reports 8 logical threads on 4 physical cores. B2
  measures throughput at several levels and takes the level that actually peaks
  — the right `--concurrency` is the one that reaches that throughput, not a
  number derived from a core count.
* **Drive uploads asynchronously.** A multi-hundred-MB tar can be *partially
  readable* with no error. So the `.md5` beside it is written only after the
  size stops moving, and its existence is the "upload finished" signal.
""")
code(
    "B1 · bootstrap — run on EVERY machine, change only SH",
    "Drive",
    "/content/caster",
    "no",
    r'''
SH       = "01"                                   # 01..06, one per machine
REPO_URL = "REPO_URL"
BUNDLE   = "/content/drive/MyDrive/caster-audit"   # where the gathering machine puts common.tar.gz

import pathlib, subprocess, sys, time
from google.colab import drive; drive.mount("/content/drive")

# Drive uploads ASYNCHRONOUSLY: a few-hundred-MB tar can be partially readable
# with no error at all. The .md5 is written only once the size stops moving, so
# its existence is the "upload finished" signal.
md5 = pathlib.Path(f"{BUNDLE}/common.tar.gz.md5")
while not md5.is_file():
    print("waiting for common.tar.gz.md5 ...", flush=True); time.sleep(30)

!git clone -q --branch main {REPO_URL} /content/caster
import os; os.chdir("/content/caster")
!pip install -q -e ".[dev]"
!python3 scripts/fetch_condefects.py --audit-only || true
!tar xzf {BUNDLE}/common.tar.gz -C /content/caster
!tar xf  {BUNDLE}/shard-{SH}.tar -C /content/caster 2>/dev/null || true
!ls -la data/*/pool/tasks.json external/ConDefects 2>/dev/null | head
''',
    title="B1 · bootstrap",
)
code(
    "B2 · measure the real concurrency, then run this machine's share",
    "data/<run>/pool/tasks.json",
    "verdicts/, data/<run>/kill_matrix.json",
    "no",
    r'''
import os, subprocess, sys, time
os.chdir("/content/caster")

# nproc lies: Colab reports 8 logical threads on 4 physical cores. Sweep, and
# take the level that actually peaks — the right --concurrency is the one that
# reaches that throughput, not a number derived from a core count.
W = "s=0\nfor i in range(3_000_000): s+=i*i\nprint(s)\n"
def throughput(n):
    t0 = time.perf_counter()
    ps = [subprocess.Popen([sys.executable, "-c", W],
                           stdout=subprocess.DEVNULL) for _ in range(n)]
    for p in ps: p.wait()
    return n / (time.perf_counter() - t0)

best, best_n = 0.0, 1
for n in (1, 2, 4, 6, 8, 12):
    tp = throughput(n)
    print(f"  concurrency {n:2d}: {tp:5.2f} jobs/s")
    if tp > best: best, best_n = tp, n
print("using --concurrency", best_n)

# Background, so the cell returns. A reclaimed runtime loses the process, not
# the work: the verdict store is cross-run and on Drive.
subprocess.Popen(
    ["caster", "audit", "--concurrency", str(best_n)],
    stdout=open(f"/content/audit-{SH}.log", "a"),
    stderr=subprocess.STDOUT, start_new_session=True)
print("launched; `tail -f /content/audit-%s.log`" % SH)
''',
    title="B2 · measure, then run",
)
code(
    "B3 · progress — returns immediately, run it as often as you like",
    "/content/audit-<SH>.log, verdicts/",
    "nothing",
    "no",
    r'''
import os, pathlib, json; os.chdir("/content/caster")
log = pathlib.Path(f"/content/audit-{SH}.log")
if log.is_file():
    lines = [l for l in log.read_text(errors="replace").splitlines() if l.strip()]
    print("\n".join(lines[-12:]))
store = pathlib.Path("verdicts")
if store.is_dir():
    n = sum(1 for _ in store.rglob("*.json"))
    print(f"\nverdict store: {n} patches measured")
''',
    title="B3 · progress",
)
code(
    "B4 · merge on ONE machine, then audit the merge",
    "every machine's verdicts/ and shards/",
    "data/<run>/rounds.jsonl",
    "no",
    r'''
import os; os.chdir("/content/caster")
# The verdict store merges by construction: it is content-addressed on
# (patch_sha, probe, protocol_sha), so two machines that measured the same key
# agree or the store refuses. The ROWS still need the six audits.
!caster merge --budget {BUDGET} --strict
!caster analyse --budget {BUDGET}
!caster report
print()
print("If `runtime disagreement` fired, the machines were not one instrument:")
print("they differ on served window, sandbox timeout, interpreter or backend —")
print("none of which reaches the cache key, which is exactly why that audit")
print("exists (Invariant 13). Fix and re-run; do not merge across it.")
''',
    title="B4 · merge",
)

nb = {
    "cells": CELLS,
    "metadata": {
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = pathlib.Path(__file__).resolve().parent.parent / "notebooks" / "run_colab.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"wrote {out}  ({len(CELLS)} cells)")
