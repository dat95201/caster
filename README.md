# CASTER

Cost-aware scheduling of test evidence for LLM program repair.

A repair loop proposes a patch and an oracle refutes it. The oracle is the
expensive half — on this corpus a probe run costs 0.366 s and a task's pool
averages 42 probes — and almost all of that cost is spent on probes that were
never going to refute anything. CASTER keys its memory by **test case** rather
than by failure class, learns a per-probe kill rate and a per-probe cost online,
and schedules by `K/m̂`: refute early, and refute cheaply.

Keying on the case rather than the class is also what makes the steering
possible. A test case is an *observation*; a failure class is a *category*, and
only an observation can be phrased as a positive requirement — "produce exactly
this output on this input" rather than "do not do that again". So the scheduler
and the prompt contract ship as one system.

How to actually run it — setup, the pipeline stage by stage, sharding across
machines, and what to do when something fails — is in
**[`docs/RUNNING.md`](docs/RUNNING.md)**. The design decisions, with the reason
for each, are in **[`docs/DECISIONS.md`](docs/DECISIONS.md)**.

---

## Quick start

```bash
pip install -e ".[dev]"
python3 scripts/fetch_condefects.py         # clones Code/; tells you where Test.zip is

caster preflight                            # first thing in any session; budget 0
caster freeze                               # the corpus gate, pool order, pool_sha256
caster propose  --policies no-mem --budget 20
caster audit    --concurrency 4             # the full-scan pass: kill matrix + cost table
caster evaluate --policies km,natural,random,fault-record,cost-only,kill-only
caster merge && caster analyse && caster report
```

On Colab, `notebooks/run_colab.ipynb` is the same sequence with Drive
persistence and a shard fleet.

## The ten verbs

| verb | what it does |
|---|---|
| `freeze` | applies the corpus gate, writes `pool/tasks.json` + `pool_sha256`. **No pre-treatment screen runs by default** — every sound fault is admitted, the oracle is the whole pool, and `analyse` bands by pool size and contest era afterwards. `--holdout 0.3` draws a `P_eval` for measuring overfitting |
| `propose` | **phase 1** — fills the response cache and the patch store. Block A only |
| `evaluate` | **phase 2** — walks `(task × seed × policy)`, resumably, writing atomic cells. `--from-cache-only` refuses to call the model; `--replay-only` refuses to run the sandbox too; `--control` walks the unfiltered control arm |
| `audit` | the dedicated full-scan pass: the kill matrix and the frozen cost table |
| `merge` | the six audits, then collapses the shards. Nothing is evidence before this |
| `analyse` | threats first, then the arms, the decomposition and the bootstrap |
| `report` | one self-contained HTML file, no network |
| `serve` | `start \| verify \| unload \| stop \| gateway` — local backends and routing |
| `preflight` | gateway, served windows, corpus, stores, disk, flag lists. Budget 0 |
| `smoke` | a real end-to-end run on a pinned tiny universe. The CI gate |

There is deliberately no `evaluate --full-scan`: letting an arm populate the
cost table would make that table a function of which arm ran first.

## The module map

```
caster/
  keys.py           the three identity keys, the nonce, the two flag lists
  paths.py          one path resolver
  errors.py         the exceptions that cross a module boundary
  bench/            the ONLY module that knows a benchmark
    protocol.py       OracleAdapter — eight dimensions, six functions
    adapter.py        ConDefects
    gate.py           the corpus gate; the opt-in P_eval split and its scoring
    cover.py          reference coverage, traced in a subprocess
  exec/
    sandbox.py        one program, one input, hard timeout, getrusage CPU
    ledger.py         n, k, s per probe; K, m̂; the schedule. No I/O, no bench
    verify.py         ONE scan = triage + oracle, stopped at the first refutation
    store.py          the verdict store and the patch store
  steer/
    render.py         verbatim ≤400 chars, else a shape. Never a probe name
    template.py       slots S1–S8, disclosure, the k-fact contract
  llm/
    client.py         one OpenAI-compatible path + the cross-run response cache
    gateway.py        admission control, two timeouts, LOR routing, coalescing
    serve.py          local backend lifecycle, owned by the gateway
  run/
    policies.py       the 22 arms. Policy is one injected function
    propose.py        the draw; verbalized-sampling parsing
    episode.py        the repair loop
    driver.py         the grid, resume, atomic cells, the protocol record
    shard.py          ranges, the manifest, the six merge audits
  report/
    records.py        RoundRecord + atomic cell writing
    analyse.py        aggregation, the cluster bootstrap, the offline replay
    charts.py         the HTML report
  cli.py
```

Three boundaries are load-bearing, and `tests/test_invariants.py` asserts all
three:

* `bench/` is the only module that reads a benchmark. Porting to another
  benchmark touches nothing outside it.
* `exec/ledger.py` has no I/O and no benchmark imports. It is pure statistics
  over probe names, so the whole scheduler is testable without a benchmark, a
  sandbox or a model.
* `steer/` does not import `exec/verify.py`. Evidence selection reads the
  ledger; it does not run anything.

## The arms

**Block A — replay arms, zero model calls.** Every arm here differs only in
phase 2, so a new one costs nothing: `full-suite`, `no-mem`, `natural`,
`random`, `fault-record`, `cost-only`, `kill-only`, `coverage`, `km`,
`km-cs-median`, `km-cs-q75`, `km-bucket`, `km-explore0`.

**Block B — steered arms, one model-call grid each.** For these the round-*t*
prompt is a function of the ledger at round *t*, which phase 2 produces, so the
two phases are **not separable at all**: `km+steer`, `km+steer-pass`,
`km+exclusion`, `km+chat`, `km+steer-k1`, `km+steer-k3`, `km+steer-sample`,
`km+steer-noreq`, `km+steer+vs`.

Two framing points govern how the roster is read:

* **`natural` is the incumbent, not a neutral control.** Sorting by filename is
  not a scheduling decision anyone made; it is what any implementation gets for
  free. It is also not arbitrary: AtCoder's naming puts the public, small,
  highly discriminating cases first, and 78.4% of `natural` rounds start on one.
  `random` is the control.
* **An ablation chain changes one thing per step.** `fault-record → km` changes
  two — a membership bit becomes a rate, and the rate is divided by cost — so it
  cannot attribute. `natural → {cost-only, kill-only} → km` can.

## The invariants

Fourteen, each cheap to break by accident and expensive to notice. The ones most
likely to go first:

1. **CRN round-1 identity, and the nonce is what enforces it.** Round 1 carries
   no facts, so a steered prompt must be byte-identical to an unsteered one and
   share its cache entry. That works only if the draw nonce is
   `<task>|seed<S>|r<round>` — deliberately *not* a function of the policy, the
   arm, or any ablation flag. Derive it from the cell key, which does contain
   the policy, and CRN dies silently while every completion is bought once per
   arm. The output budget is part of the same identity: a verbalized-sampling
   arm must request one program in rounds 1–2, because `max_tokens` is in the
   cache key.
2. **Soundness is independent of the schedule.** A patch is accepted only after
   surviving every usable probe, so no ordering can accept what another rejects.
   `success@B` and the first-accept round are therefore equal across every
   Block A arm **on every cell**, not on a sample — a gap there is a soundness
   bug, not a finding.
3. **CPU time, not wall clock.** `getrusage(RUSAGE_CHILDREN)` deltas. Wall clock
   is contaminated by sibling shards and would make `m̂` a function of load. And
   because `getrusage` is process-global, phase-2 parallelism must be
   process-per-shard: `sandbox.py` raises rather than let a thread pool
   mis-attribute a sibling's CPU.
4. **The reference implementation never enters a prompt.** Reference *values*
   for a probe are fine; the reference *source* is not.
5. **One path resolver.** `RUN_DIR` decides where a run writes; no module
   computes its own data path.
6. **Every CLI flag is in exactly one declared list.** "Everything is in the
   key" is the wrong rule: audit flags stay out deliberately, so they can be
   switched on without invalidating finished cells. A flag in neither list fails
   the build.
7. **Atomic cell writes.** A cell is written whole or not at all, and the
   done-index is written after the rows.
8. **The response cache is cross-run** and never scoped per run — it is what
   makes a re-run free.
9. **No `(patch, probe)` is executed twice** under one protocol. Instrumented,
   not asserted in prose: `(sandbox_runs − timeout_reruns) + verdict_hits ==
   Σ test_calls`, per scan and over the run.
10. **A stored verdict and a fresh one never disagree.** Disagreement halts the
    run — except for timeouts, which are quarantined rather than reused.
11. **Re-issuing an identical command after a kill is safe** and resumes.
12. **Merge audit before analysis.** A merged file is not evidence that the grid
    is complete. Six audits, two of which exist to catch instrument drift that
    reaches no key at all.
13. **Every value that changes a measured quantity is either in a key or in
    `.meta.json`.** The cache key covers model, temperature, prompt,
    `max_tokens` and the nonce. The served context window, the sandbox timeout,
    the interpreter version, the backend and the model digest reach no key —
    they travel on the shard's protocol record, and the merge refuses to join
    across a disagreement.
14. **Smoke output is never mergeable**, by construction rather than by a naming
    convention an operator has to remember.

## Tests

```bash
pytest -q
```

`pytest` runs **CASTER's own unit tests** — not the experiment. 238 of them,
seconds, no network and no model. They are unrelated to the ~1,244-fault corpus
that `caster freeze` produces and `caster evaluate` walks.

The benchmark fixture they run against is a real miniature ConDefects tree
(two faults, ~24 probes each, in a temp dir) that is really executed — the point
of the sandbox is process isolation, CPU accounting and an error taxonomy, and
a mock would test none of it.

`tests/test_review_regressions.py` is one test per defect a review pass found,
each named after what it would have cost. They are not extra coverage: every one
is a case where the code did something defensible-looking and produced a wrong
*number* rather than an exception, which is the only failure mode that matters
on a corpus nobody re-reads row by row.
