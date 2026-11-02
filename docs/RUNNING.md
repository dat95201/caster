# Running CASTER

The README says what the system *is*. This says what to type, in order, and what
each step costs you.

Two rules that save the most time:

* **`caster preflight` first, every session.** It spends nothing and catches the
  four things that waste a run: a corpus that is not there, a gateway that is
  not up, a served context window smaller than the prompts, and a flag in
  neither declared list.
* **Every stage resumes.** Killing a run mid-walk is safe; re-issuing the same
  command picks up where it stopped. Nothing below needs to finish in one go.

---

## 1. Install

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env            # .env is gitignored
```

Python 3.10+.

## 2. Get the benchmark

```bash
python3 scripts/fetch_condefects.py
```

`Code/` (~125 MB) clones. `Test/` **does not** — it is a 6.4 GB archive
distributed out of band, and the script tells you where to get it and refuses to
pretend it is there when it is not.

This asymmetry matters more than it looks. A **partial `Test/` tree is the one
failure that corrupts a corpus quietly**: some tasks come out with fewer probes
than the contest shipped, and nothing in the run says so. Two safeguards:

* `CONDEFECTS_TEST_DIR` — point at a salvaged tree if a download arrived
  incomplete, instead of freezing the truncated one by accident.
* `pool_sha256` — computed over the probe lists actually found. Two freezes over
  two different downloads give two different digests.

## 3. Fill in `.env`

The only lines you must set to run with a model:

```bash
MODEL=                 # EXACT snapshot id, never a bare alias
LLM_BASE_URL=          # point at the GATEWAY, not at a backend
LLM_API_KEY=
```

Two more deserve a look before you freeze anything, because both enter
`protocol_sha` and changing either invalidates the verdict store by
construction:

| variable | default | why you might change it |
|---|---|---|
| `SANDBOX_TIMEOUT_SEC` | `60` | The instrument's declared capability. A reference that cannot finish inside it marks its case `exceeds_sandbox_timeout`. |
| `SANDBOX_RECURSION_LIMIT` | `10000` | Contest solutions recurse over the whole input; CPython's default 1000 turns correct programs into `RecursionError`. |

And one that can corrupt a run silently:

```bash
LLM_CONTEXT_TOKENS=32768     # 0 disables the check; don't
```

A server is free to **truncate** an over-long prompt rather than refuse it, and
the served window reaches neither the cache key nor the record. It lands
differentially — the steered arms carry the longest prompts.

## 4. Check before you spend

```bash
caster preflight      # budget 0: gateway, served window, corpus, stores, disk, flag lists
caster smoke          # a real end-to-end run on a pinned tiny universe, seconds
```

`smoke` is the CI gate. It walks every arm on two pinned tasks with **zero model
calls and zero network** — a model call there is a build failure, because it
means phase 1 and phase 2 stopped being separable.

First run writes `smoke/expected.json`. **Commit it**, or the next run has
nothing to regress against and the gate degrades into a liveness check.

```bash
caster smoke --live   # exactly ONE real call: proves the local/cloud path,
                      # the window check and the spend ledger
```

---

## 5. The pipeline

```bash
caster freeze
caster propose  --policies no-mem --budget 20
caster audit    --concurrency 4
caster evaluate --policies km,natural,random,cost-only,kill-only --budget 20
caster merge && caster analyse && caster report
```

### `freeze` — build the corpus

Walks every fault once, applies the gate, writes `data/pool/tasks.json` and
`pool_sha256`. Reads `Test/`, so it is I/O-heavy; it is also the only stage that
executes the reference program.

**No pre-treatment screen runs by default.** Every fault whose oracle is sound
is admitted, the oracle is the whole pool, and the populations the old gate used
to pre-select are bands that `analyse` draws afterwards.

```bash
# the old narrow corpus, for comparison:
caster freeze --min-pool 20 --date-min 2022-04-01 --lines-max 120 --dedup-per-group
# hold 30% out, for measuring whether patches overfit the disclosed probes:
caster freeze --holdout 0.3
# a few tasks, to try the pipeline:
caster freeze --limit 20
```

`freeze` refuses to overwrite an existing `tasks.json` without `--overwrite`: a
corpus silently re-frozen is a corpus nobody can reproduce.

### `propose` — phase 1, the model calls

Fills the response cache and the patch store. **Block A only** — the 13 replay
arms are unsteered, so their round-*t* prompt is identical whatever the policy,
and one pass fills the cache for all thirteen. That is why `--policies no-mem`
is enough here.

This is where the money goes. `--dry-run` first:

```bash
caster propose --policies no-mem --budget 20 --dry-run
```

**Block B arms cannot use this path.** Their round-*t* prompt is a function of
the ledger at round *t*, which phase 2 produces, so the two phases are not
separable at all — they run their model calls inside `evaluate`.

### `audit` — the full-scan pass

The dedicated pass that builds the **kill matrix** and the **frozen cost table**.
CPU-bound; size `--concurrency` from cores.

```bash
caster audit --concurrency 4
```

There is deliberately no `evaluate --full-scan`: letting an arm populate the
cost table would make that table a function of which arm ran first.

Optional, and worth it once: re-run a sample of store keys to get a
*within-cell* CPU variance instead of assuming there is none.

```bash
caster audit --remeasure-cost 5 --remeasure-repeats 3 --remeasure-seed 1
```

### `evaluate` — phase 2, the walk

Walks `(task × seed × policy)` and writes atomic cells. After `audit`, every
Block A arm should need **zero sandbox runs** — and you can make that an
assertion rather than a hope:

```bash
caster evaluate --policies km --replay-only    # refuses to run the sandbox at all
caster evaluate --policies km --from-cache-only  # refuses to call the model
```

A cache miss under either is a loud failure, not a silent draw.

```bash
caster evaluate --control      # the unfiltered control arm, reported separately
```

### `merge` — six audits, then collapse

**Nothing is evidence before this.** A merged file is not proof the grid is
complete; the audits are what make it so — protocol drift, runtime drift,
missing protocol records, gaps, disagreements, foreign tasks.

```bash
caster merge --strict    # exit non-zero on a non-fatal audit too
```

### `analyse` and `report`

```bash
caster analyse --bootstrap 2000
caster report            # one self-contained HTML file, no network
```

`analyse` writes `data/summary.json`; `report` turns it into
`data/report.html`.

Two things to read there before the arm table:

* **the threats section** — proposal errors and realised `k_facts` per arm. A
  steered arm whose realised k does not match its configured k is mislabelled,
  and nothing downstream will say so.
* **the bands** — by pool size and by contest era. `standard` is the old
  `--min-pool 20` corpus; `pre-cutoff` is what the old date cut deleted.

The repair rate is frozen only when the floor arm is **complete** — every task
of the universe at every seed it ran. `--freeze-rates` forces it when you mean
to.

---

## 6. Serving a model

Point shards at the **gateway**, never at a backend. Shards know nothing about
how many replicas exist, and one global ledger is the only thing that makes a
spend cap a real ceiling — N shards each honouring a per-process cap is a cap of
N× that number.

```bash
caster serve start --model <id> --local-replicas auto --vram <free VRAM bytes>
caster serve gateway --port 8000 --backend local --spend-cap 25 \
                     --tok-per-sec 16 --tok-budget 16000
caster serve verify         # the served context window, measured not assumed
caster serve stop
```

`--backend hybrid --cloud-url ...` overflows to a hosted endpoint when the local
pool is saturated.

Size `--service-timeout` from the work, not from a guess: with `--tok-per-sec`
the gateway derives it as `max_tokens / tok_per_sec × margin`. `--queue-timeout`
is a different thing — willingness to wait for a slot — and conflating the two
is what makes a legitimate call look like a harness failure.

---

## 7. More than one machine

Each stage takes `--first` / `--last` (1-based, inclusive) and `--shard`:

```bash
caster evaluate --dry-run                       # the ONE authority on universe size
caster evaluate --first 1   --last 500  --shard 0
caster evaluate --first 501 --last 1000 --shard 1
```

`--shard` must be **numeric**. A non-numeric label collapses onto shard 0 and
overwrites a real shard's rows.

On one machine, let `fleet.sh` cut the ranges instead of typing them:

```bash
bash scripts/fleet.sh launch --shards 6 -- --policies km,natural --budget 20
bash scripts/fleet.sh status
bash scripts/fleet.sh tail 3
bash scripts/fleet.sh wait      # the only command that blocks
bash scripts/fleet.sh stop
```

It reads the universe size out of `--dry-run`, cuts contiguous non-overlapping
ranges, divides the spend cap and prints the arithmetic, and keeps one log per
shard under `logs/<run>/fleet/` with a manifest of which pid walks which range.

**Phase 2 parallelism must be processes, not threads.** `getrusage` is
process-global, so a thread pool would mis-attribute a sibling's CPU —
`sandbox.py` raises rather than let that happen. A fleet of separate processes
is exactly what it wants. Useful shard count is bounded by cores: throughput was
still climbing at 8 concurrent and flat by 12.

Separate runs on one checkout:

```bash
RUN_DIR=campaign-7b caster evaluate ...     # writes under data/campaign-7b/
```

The cross-run stores — `cache/`, `verdicts/`, `patches/` — are deliberately
**not** under `RUN_DIR`. Scoping them per run cold-starts every sweep, and the
response cache being cross-run is what makes a re-run free.

---

## 8. Colab

`notebooks/run_colab.ipynb` is the same sequence with Drive persistence and a
shard fleet. Regenerate it after changing the pipeline:

```bash
python3 scripts/build_notebook.py
```

---

## 9. When something goes wrong

| symptom | what it means |
|---|---|
| `FAIL: ... not servable from the verdict store` | `--from-cache-only` / `--replay-only` hit a miss. Run `caster audit` for that task first, or drop the flag. The loud failure is the point. |
| `HeldOutProbeLeak` | A `P_eval` probe reached the loop. Only possible under `--holdout`; it halts the run rather than quietly contaminating a score. |
| `VerdictDivergence` | A stored verdict and a fresh one disagree, stably, on one host. **Do not pick a winner** — fix the cause and re-run both. Timeouts are exempt; they are quarantined. |
| merge audit `DISAGREEMENT` | The same (cell, round) collected twice with different results, usually a wall-clock timeout firing on the slower machine. Same rule: fix and re-run both. |
| merge audit `FOREIGN` | A shard was cut from a different corpus. Check `pool_sha256`. |
| `smoke` drifted | The loop's behaviour changed. That is the gate doing its job; read the diff before re-baselining. |
| `caster smoke --live` made 0 calls | A cache hit, not a failure — but the live path was **not** exercised. Point `RUN_DIR` at a fresh run to force one. |
| preflight: *"holdout 0 — success@B is a PLAUSIBLE-patch rate"* | Not an error. A reminder that without a holdout you are measuring plausibility, not correctness, and the write-up must use that word. |

## 10. The one thing to get right before a real run

`caster freeze` is the step everything else inherits. Before the first reported
number, settle:

1. `SANDBOX_TIMEOUT_SEC` — it enters `protocol_sha`; changing it later
   invalidates every verdict.
2. `MODEL` — an exact snapshot id. A hosted model silently re-pointed is a
   different instrument with nothing to say so.
3. Whether you want a `--holdout`. It changes what `success@B` *means*, and it
   cannot be added after the fact without re-running the grid.
4. `smoke/universe.json` — replace the two placeholder task names with faults
   from your own frozen pool, then commit it together with `smoke/expected.json`.
