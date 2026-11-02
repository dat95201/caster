# Design decisions

The choices below are binding on the implementation. Each one has a reason and,
where it could be broken silently, a test named after what breaking it costs.
Reversing a decision means changing this file in the same commit.

Three of them are **schema or key changes, not flags** — they cannot be deferred
past the first reported number without invalidating everything collected before.

---

## 1. The evaluation split is OPT-IN; the oracle is the whole pool

**`--holdout` defaults to 0: `P_loop` is the pool and `P_eval` is empty.**
`caster/bench/gate.py::HOLDOUT`, `_split_stratified`. See decision 23 for why
the default moved; this entry describes the split as it behaves when asked for.

**`--holdout 0.3` holds 30% of each task's pool out as `P_eval`, split within
each of F2P and P2P, seeded from the task name, drawn once per fault and shared
across models.**

* **30%** because the ratio is upstream of the pool-size screen and of every
  leakage figure. Under `--min-pool 20` a 30% holdout means a fault needs ~29
  usable probes to be admitted, which the measured pool-size distribution
  (median 31) supports.
* **Stratified**, because a uniform split breaks in both directions. If `P_loop`
  draws no F2P probe, round 1 accepts the unmodified faulty program and the
  soundness gate's whole purpose is undone. If `P_eval` draws no F2P probe, it
  cannot detect a non-repair and measures only regressions. A task whose split
  empties either side is excluded, with that reason recorded.
* **Shared across models**, because the split is a property of the *task*. The
  replication question compares band curves across models; two models scored on
  two different denominators would not be comparing curves, they would be
  comparing denominators.
* The four counts (`|F2P ∩ P_loop|`, `|F2P ∩ P_eval|`, `|P2P ∩ P_loop|`,
  `|P2P ∩ P_eval|`) are recorded per task in `pool/tasks.json`.

`P_eval` is never read by the loop — `exec/verify.scan` raises
`HeldOutProbeLeak` if a held-out probe reaches it — and is executed only by
`bench/gate.score_on_eval`, after the episode has ended.

## 2. Timeouts are quarantined, not made deterministic

**The sandbox keeps its wall-clock budget, and a `timeout` verdict is never
served from the store.** `caster/exec/store.py`, `caster/exec/verify.py::_measure`.

A deterministic step cap would make the verdict genuinely pure and let
divergence be an unqualified halt — but it is **not the instrument the
benchmark's own judge used**, so the admitted pool (gate G2: "neither version
times out at the sandbox clock") would have to be re-derived under it, and every
admitted/excluded count with it. That is a corpus-scale cost for a small base
rate.

So:

* a `timeout` verdict is written to the store but `get()` never returns it, so
  it never serves a scan and never satisfies `--replay-only`;
* a fresh verdict that disagrees with a stored one, **where either side is a
  timeout**, is re-run up to `REPEATS_BEFORE_HALT = 3` times on one host. A
  *stable* disagreement raises `VerdictDivergence` and halts; a flip is appended
  to `jitter.jsonl`, counted in the threats table, and the conclusive verdict
  wins;
* a disagreement where **neither** side is a timeout halts unconditionally;
* the no-double-execution identity is stated with the repeats named, because
  they are real CPU but not extra verdicts:

      (sandbox_runs − timeout_reruns) + verdict_hits == Σ test_calls

  asserted per scan in `ScanResult.__post_init__` and over the whole run in
  `run/shard.audit_invariant9`.

**The open measurement stands**: the share of (patch, probe) pairs whose verdict
is `timeout` on this corpus, and the flip rate at the grid's target concurrency.
Both fall out of one `caster audit` pass run twice. Until that number exists,
this is the cheap decision, not the demonstrably right one.

## 3. Both priors ship; `Beta(½,2)` is the default

`--prior shipped` (the default) and `--prior jeffreys`.
`caster/keys.py::PRIORS`.

Jeffreys `Beta(½,½)` puts an unseen probe at `K = 0.5`, which outranks a proven
refuter whose record has decayed below it — `k=2, n=10` gives `K = 0.227`. Under
homogeneous per-probe costs that is a full-pool sweep. `Beta(½,2)` has prior
mean 0.2: above this corpus's measured base rate (≈0.03), below ½.

`K = ½` is **not** described anywhere as "the exploration". Exploration is what
`explore_p` is for, and the two are not interchangeable.

**What settles it, and it costs nothing:** replay both priors over the full-scan
kill matrix (`report/analyse.compare_priors`) and report mean rank of the probe
that actually fired, and mean probes run per refuted round. No model calls, no
sandbox runs. Arms 5–11 are replayed under both.

## 4. Verbalized sampling is in scope, so the record carries `candidate_index`

`caster/report/records.py`, `caster/run/propose.py`, arm 18.

A VS round emits *n* candidates from one draw, so the record emits *n* rows
sharing one `cache_key`. Declaring VS out of scope would lose arm 18 and the
exploration question, and would leave the within-episode reuse scope of the
verdict store with nothing to do. It is a schema change, so it is made before
the first row is written.

One consequence: **the budget counts rounds (model calls), not candidates.**
Charging each candidate would let a `k=5` arm spend B in B/5 draws, so "rounds
to repair" would compare arms at different numbers of model calls — exactly the
quantity the exploration question is about. The extra CPU of scanning five
candidates lands in `cpu_s`, which is the cost unit anyway. Exactly one row per
round carries `charged=True` and the USD.

Rounds 1–2 carry no VS, which keeps the round-1 identity intact against the
non-VS arms — and the **output budget** is sized from the number of programs the
round actually requests, because `max_tokens` is in the cache key.

## 5. The pool is ordered by a seeded interleave

`caster/bench/gate.py::_interleave`, `GateConfig.pool_order`.

ConDefects' own order is contest order, correlated with date (the contamination
cut), with rating, and therefore with pool size. Cutting contiguous shard ranges
from it would leave a killed fleet with a *systematically* biased subset rather
than a smaller grid. The seed and the choice both enter `pool_sha256`, so the
ordering cannot change silently. `--pool-order contest` reproduces the unordered
case deliberately.

## 6. Ten verbs, and no `evaluate --full-scan`

`freeze propose evaluate audit merge analyse report serve preflight smoke`.
`caster/cli.py`, asserted by `tests/test_cli.py::test_the_count_is_ten`.

* `preflight` and `smoke` are **verbs**, not modes. They are the two things a
  session runs first, and burying them under another verb is how they stop being
  run.
* `gateway` is a **mode of `serve`**, because the gateway owns backend
  lifecycle: the thing that starts replicas and the thing that routes to them
  must not be two commands that can disagree about which replicas exist.
* `caster audit` is the dedicated full-scan pass: every probe against every
  patch, no early stop, **no ledger** (an audit is not an episode; a ledger it
  updated would be a cross-episode prior), at a **declared** concurrency,
  **before any arm runs**. It writes the kill matrix and the cost table, each
  with its own digest.
* the no-early-stop *policy* is `--policies full-suite` (arm 0), which replays
  that matrix for free.

An arm populating the cost table would make the table a function of which arm
ran first and of the concurrency it ran under, which is why the flag does not
exist.

## 7. Two separate "do not go live" flags

`--from-cache-only` and `--replay-only` are different misses that fail for
different reasons at different times:

* `--from-cache-only` — the **model** side. A miss means phase 1 never drew this
  round. It raises `CacheMiss`, **not** `BackendUnreachable`: that class is
  caught by the loop, charged as a spent round and fed to the circuit breaker,
  so five misses would end the cell as a "backend outage" — exactly the silent
  degradation the flag exists to prevent.
* `--replay-only` — the **sandbox** side. Every probe must be servable from the
  verdict store. This is what makes "after an audit pass, every policy sees
  `sandbox_runs == 0`" checkable rather than hoped for, and it is where the
  timeout quarantine gets teeth: a `timeout` verdict is not servable, so a
  replay that needs one fails instead of silently re-executing under a different
  load than the audit measured.

## 8. The cost table is frozen by the audit pass, and a changed verdict does not inherit from it

`caster/exec/store.py`.

The audit pass writes the cost table once, at one declared concurrency, so no
arm's own run conditions enter it. But an audited `timeout` whose re-execution
comes back `pass` is a probe the audit **mismeasured**: its wall-clock kill is
not the cost of the millisecond run that actually happened. Inheriting it would
poison the frozen table on exactly the probes most likely to flip. So the
inheritance applies only when the verdict is unchanged, and a `timeout` never
enters `cost_digest()` at all — otherwise the stamped artifact would move the
moment a probe was re-run and came back conclusive, which is when it should
least move.

## 9. `--remeasure-cost` executes; it does not replay

`caster/cli.py::_remeasure`.

Running the repeats through `scan()` is served from the verdict store on every
one — after the audit pass every probe is a hit — so all `R` draws are
byte-identical and the reported variance is a **structural zero**. That number
is the input to every CPU confidence interval, so a zero there is worse than no
number at all. It calls the adapter directly, samples `(patch, probe)` keys (the
unit the store is keyed on, not whole tasks), and reports the timeout flip rate
as a by-product.

The flag's three ambiguities are also decided:
`--remeasure-cost PCT --remeasure-repeats R --remeasure-seed S`. PCT is a
**percentage** — the name says so and the name wins. The sample is **seeded**,
because an unseeded one makes the variance estimate itself irreproducible. And
**R ≥ 3**, enforced: one re-execution gives a difference, not a variance.

The number it produces is labelled as what it is — *an input to the CI, not the
CI*. The right calculation is `bootstrap_cpu(..., replay=...)`, which resamples
the cost table from these draws **and** the task clusters, and re-runs the
offline replay per iteration. Called without a `replay`, the result says
`cost_table_resampled: false` rather than quietly reporting the narrow interval.

## 10. A store-served verdict carries the observation, not just the status

`caster/exec/store.py::OUTPUT_HEAD_CHARS`, `ProbeVerdict.output_len`.

The contract block renders "your last attempt printed `…`" from the last
observation. A replay against a warm store that returned only a status would
build a **different prompt** from the run that filled the store — a different
draw, decided by cache warmth, with nothing in any key to say so. So the store
keeps the head of the candidate's stdout (240 chars, the hard quoting cap, so
more could never be rendered anyway), the error class, **and the true length**.

The length matters on its own: truncating on the warm path only would print
"[truncated, 240 chars total]" where the cold run printed "[truncated, 8,890
chars total]", and for a long single line it would drop the marker entirely.
Both paths truncate, and the true size travels with the head.

## 11. A legible input is shown whole, never through `snippet()`

`caster/steer/render.py`.

`snippet()` joins lines with `" / "` and cuts at 240 characters. Running a
legible multi-line input through it and then asserting a required output for the
result is a **false fact** — and it would hit every multi-line stdin at any
size, not just the stress cases. Legible inputs and required outputs are
rendered verbatim between `<<<` / `>>>` markers rather than a ``` fence, because
the output-format slot asks for the answer in a ```python block and a second
fence in the prompt is the one thing that can make a model close the wrong one.

An input too large to show becomes a **shape** — how many lines, how many
characters, the first line verbatim, because on AtCoder that line carries N. An
expected output too large to show is **withheld**, not truncated: a required
value the model cannot read is not a requirement.

## 12. Probe names never reach a prompt

`caster/steer/render.py`.

A filename is a label the model cannot act on, and worse, the naming is *itself*
information: the `sample`/`example`-named probes are the public, small, highly
discriminating ones. Putting the name in the prompt leaks the name class — the
same artefact the hashed schedule tie-break exists to keep out of the ordering.
With no name rendered, "never name a probe whose content is not in the prompt"
holds by construction.

## 13. `facts_shown` is what the prompt showed

`caster/run/episode.py`.

Captured **before** the scan, from the facts the prompt was built from, and
empty when `steer_on` is false. The exploitability bound is computed from
exactly this field: read off the post-update ledger it is one round late, and
emitted for an unsteered arm it attributes a disclosure that never happened.
Either way the leakage figure is wrong and nothing looks broken.

## 14. The evidence budget, the disclosure policy and the VS width are arms, not flags

`caster/run/policies.py`.

Every steered arm sets all three in its own overrides, so a grid-wide flag would
lose to the arm on exactly the arms where it matters — a flag that is parsed,
keyed and then discarded, which puts the *default* in the cell key while the
command line says otherwise. `km+steer-k1`, `km+steer-k3`, `km+steer-sample`,
`km+steer-noreq` and `km+steer+vs` are the mechanism. Add a row to the roster,
not a flag.

The four knobs that stay grid-wide (`--prior`, `--cold-start`, `--explore-p`,
`--tie-break`) do so because they are swept *across* arms, and the plan prints
which arms override them.

## 15. The regression signal is an arm, not a default

`km+steer-pass`, beside `km+steer`.

Showing the probes the last candidate already satisfied is a positive,
zero-leak, zero-extra-cost fact — the scan already knows them. It is also
speculative: nothing prices it yet. So it is an arm, with a plain contract arm
beside it to compare, rather than a default that quietly changes every steered
prompt.

(The obvious implementation is dead code: fill the list on `accept`, and the
episode ends before it can render. And an oracle that stops at the first
divergence throws the information away unless it is kept deliberately.)

## 16. An incomplete floor arm may not freeze the denominator

`caster/report/analyse.py::floor_arm_is_complete`.

`GAPS` is a non-fatal merge audit, so a merge succeeds on a partial grid — and
the next `analyse` would otherwise freeze a repair rate computed from one shard.
That rate is the denominator of every repair-rate band, and the freeze only ever
*adds* keys, so an under-seeded rate never corrects itself.

The freeze is gated on the floor arm covering every task of the universe at
every seed it ran, and `caster analyse --freeze-rates` forces it when an
operator means to. `repair_rates_source` on the summary says which happened.

## 17. The gateway owns one ledger, and the client does not double-log

`caster/llm/gateway.py::SpendLedger`, `_settle`; `caster/llm/client.py`.

Coalescing makes N shards that miss one cache key produce one upstream **call**.
Without the ledger write moving to the gateway, each of the N clients still
appends its own `usd` row — one call, N charges. The gateway prices the reply,
writes the single row, and stamps the body with `_caster_gateway`, which the SDK
surfaces through `model_extra`; the client sees the stamp and skips its own
write. (Response *headers* are not exposed on the SDK's parsed model, which is
why the stamp travels in the body.)

A per-process cap is also not a cap: `BUDGET_USD_CAP=25` across N shards is a
real ceiling of 25N, with the arithmetic falling on the operator.

## 18. The sandbox timeout reaches `protocol_sha` and `.meta.json`

It is correctly a **non-key** flag — it is one of the values that reach no cache
key — but it changes what a verdict **is**, so it must reach the protocol digest
and the shard's runtime record. Otherwise a run at `--timeout 5` writes verdicts
into the `--timeout 10` namespace and the runtime-disagreement audit, the one
check designed to catch exactly this, reads a constant.

## 19. The control arm is runnable, and excludes the dedup rejects

`caster/bench/gate.py::control_tasks`, `caster evaluate --control`.

A threshold that cuts the corpus is selection on a variable correlated with the
outcome, so whatever thresholds remain are declared as scope conditions and
made **checkable**: a seeded uniform sample over the G1-only survivors a
threshold cut, pre-registered as "reported whatever it shows". Their faults are
never in `pool_order`, so no main-grid analysis can pool them by accident.

Since decision 21 removed the pool-size screen, what this arm covers is G5's
date cut, G6's length cut, and an explicitly passed `--min-pool`.

G4's rejects are excluded from it. They share a test pool with an admitted
fault, which is the exact correlation G4 exists to remove, and it would
otherwise sit inside the arm meant to measure what the threshold cut.

## 20. Smaller calls, recorded so they are decisions rather than accidents

* **`CacheMiss` lives in `caster/errors.py`.** It is raised on both sides of the
  phase split, and defining it inside the verdict store would make `llm/` import
  `exec/` for one class.
* **A shard registers with the gateway for the whole walk.** Teardown is driven
  by a refcount, not by shard exit: a shard between calls still owns the pool.
* **`caster smoke --live` counts the client's own calls**, not lines in the call
  ledger — the gateway owns that ledger and may run under a different `RUN_DIR`,
  which would report a real call as a cache hit on the one check that proves the
  live path works.
* **`replay_policy` honours `arm.full_scan`.** Arm 0 early-stopping in a replay
  reports roughly 1/N of its true cost, and being the measured upper bound is
  arm 0's entire job.
* **The `coverage` arm has no optional dependency.** Reference coverage is
  traced with `sys.settrace` in a subprocess, memoised per `(task, probe)` and
  counted into `reference_runs`. A dependency that silently degrades to a no-op
  means an arm can report under a label it was not running.
* **`est_scans` counts VS candidates**, header lookup is case-insensitive (a
  normalising proxy would silently disable coalescing and restore the N× bill),
  connect timeouts are labelled `service_timeout` like read timeouts, and
  discovery lists probe names from the directory rather than reading every
  probe's text.
* **The prompt parser is forgiving, the record is not.** A reply with no closed
  fence raises rather than returning the raw text as a patch: such a candidate
  always fails, so a harness defect would enter memory as evidence inside the
  very mechanism under study.

## 21. The pool-size screen moved out of the gate and into the analysis

`caster/bench/gate.py::MIN_LOOP_POOL` (now 1), `report/analyse.py::POOL_BANDS`.

G3 used to reject a fault with `|P_loop| < 20`. Measured over all 1,422
ConDefects problems (median 31 cases, mean 36.4, range 3–148), that admitted
76.8% and deleted 23.2% before a single cell existed — a population nobody can
weight back to the benchmark.

It is now a **band**, not a screen. `freeze` admits every fault whose oracle is
sound, the grid runs all of them, and `analyse` stratifies by `|P_loop|`
afterwards. The `standard` band is `≥ 20`, so it reproduces the old corpus
exactly and every number the gate would have produced is still computable; the
other 23.2% is reported beside it instead of being deleted.

The asymmetry is the whole argument: **a pre-treatment screen has to be decided
before any cell exists and cannot be undone afterwards, while a band can be
re-cut any number of ways from the same rows.** When both give the same answer,
take the one that is reversible.

Moving it is safe because the threshold was never load-bearing for correctness,
only for interpretation. At `|P_loop|` = 1 every policy is the same schedule by
construction, so tiny pools **dilute** a corpus-wide average rather than
corrupt it — and the `degenerate` band separates them, which is exactly what
the gate's own rule prescribes for a covariate. `--min-pool 20` re-imposes the
screen for anyone who wants the old behaviour.

The default is **1, not 0**. Zero would switch the comparison off through a
falsy sentinel; what changed is the threshold, not whether it is evaluated. At
1 the check states the real floor — an empty `P_loop` is a task no policy can
schedule — and it stays a live assertion if the two F2P checks above are ever
weakened.

What G3 still rejects, and must, are the two checks about the **split** rather
than about size: a `P_loop` with no F2P case (the fault is invisible to the
loop, so round 1 accepts the unmodified faulty program) and a `P_eval` with no
F2P case (a non-repair is undetectable at scoring). Together they still demand
two F2P cases, one per half — a floor that is not a difficulty proxy.

Cells whose task is not in the frozen pool — the control arm's — are excluded
from the bands. Pooling them would put back the exact correlation the control
arm exists to measure.

## 22. Every pre-treatment screen is off by default

`GateConfig` defaults: `min_loop_pool=1`, `lines_max=None`, `date_min=None`,
`dedup_per_group=False`.

**None of the "off" values is falsy.** `0` and `""` would switch a screen off by
being falsy, which forces a truthiness guard onto the check itself — and `0` is
a legitimate line count, so for a ceiling it also conflates "no limit" with
"reject everything". A ceiling's absence is infinity, not a small number, and
`None` expresses that without putting `Infinity` into `tasks.json`, which
`pool_sha256` hashes. `min_loop_pool` is the opposite case and stays an `int`:
there *is* a real floor (1), so the comparison is always meaningful.

Decision 21 moved the pool-size screen out of the gate. The same argument
applies to the three that were left, so they moved too. What remains in the
gate is only what makes the measurement **defined**; everything that merely
shaped the population is now decided after the run, where it can be redrawn.

| was | now | why the gate was the wrong place |
|---|---|---|
| G6 `lines_max=120` | `0` (no limit) | The real constraint is the output budget, not a line count, and a truncated reply already surfaces as `context_overflow` in the proposal-error table. The screen measured a proxy, upstream of the thing under study. |
| G5 `date_min="2022-04-01"` | `""` (no cut) | A cutoff *assumes* contamination away. Running every era and banding by contest date **measures** it: `analyse.arms_by_era` reports `pre-cutoff` beside `post-cutoff`, and the gap is the effect, as a number. |
| G4 dedup per coding task | opt-in `--dedup-per-group` | Correlated cells are a clustering problem. `bootstrap_cpu` now resamples the **coding task** (`cluster_of`), so faults sharing a pool move together and every one of them can stay. Dropping data was the weaker fix. |

G4's removal is the one that required a code change elsewhere, and it is worth
stating plainly: the gate was deleting faults to make an assumption true that
the estimator should have been handling. Clustering on `abc221_f` instead of on
`abc221_f/40898300` keeps every fault **and** keeps the independence assumption
where it belongs.

**What is still rejected, and why none of it is a screen on a task property:**

* G2 — no `out/` file: the expected value would have to come from the reference,
  so the pool would carry two authorities and only one is AtCoder's.
* G1 — the reference disagrees with `out/`, or the faulty program passes the
  whole pool: in the first case nothing derived from that reference means
  anything, in the second round 1 accepts the unmodified program.
* G3 — `P_loop` or `P_eval` with no F2P case. **This one cannot become a band.**
  A diluted average is still a true average of something, so pool size, era and
  length can all be banded afterwards. A missing F2P case in `P_loop` makes the
  recorded accept *false* — `success@B` counts a non-fix as a fix — and no
  post-hoc cut repairs a number that was wrong when it was written.

The rule the whole file now follows: **screen only where the alternative is a
wrong number, never where the alternative is a wider population.**

## 23. The holdout is opt-in, and `success@B` is a plausibility rate

`--holdout`, default `0.0`.

RQ1 measures **the CPU cost of refuting a patch**, and the thing being
scheduled is the probe list. A 30% holdout removes 30% of that list — on a
median pool of 42, thirteen probes — from the exact question under study, and
it is the sole reason G3 exists at all. For the primary claim the split buys
nothing and costs evidence.

What it buys is the secondary claim, and that claim is real: with the loop
accepting on the same probes it was judged by, "accepted" cannot distinguish a
fix from a patch that fit those probes. So the machinery stays, behind a flag.

**The honest naming.** At `--holdout 0`, `success@B` is a **plausible**-patch
rate — the standard APR term for "passes the available tests" — not a
correctness rate. `caster preflight` says so in its notes on every run, and the
paper must use that word. A number that cannot be wrong is not a weaker number
as long as it is not called something it is not.

Two things follow mechanically:

* **G3's two F2P checks run only under a holdout.** Without one, `P_loop` is the
  pool, `f2p` is non-empty by G1, and both are satisfied by construction —
  asserting them would be theatre.
* **`regression_rate` becomes a guarantee rather than a measurement.** The loop
  runs every P2P probe and rejects on failure, so an accepted patch breaks
  nothing in the pool, by construction. `score_on_eval` returns a clean result
  when `P_eval` is empty.

**What is given up, and it is not nothing.** The measured exploitability of the
disclosure axis — `full` 7.0% (19/273) against `sample-only` 0.0% — needs
held-out probes. Steering is half the thesis, so the comparison between
`km+steer`, `km+steer-sample` and `km+steer-noreq` on *overfitting* (as opposed
to on CPU) requires a `--holdout` pass. That pass can be a small subset of
tasks: it answers a different question from RQ1 and does not need the full grid.

## 24. `freeze` checkpoints per fault, and shards

`caster/bench/gate.py::gate_many` / `assemble`, `caster freeze --first/--last/--shard/--assemble`.

Measured on this corpus a fault costs ~9–23 s to gate — one reference run and
one buggy run per case, and nothing in it is cached across faults. At ~2,900
Python faults that is a walk of several hours, single-threaded. Two properties
stop being optional at that length, and `freeze` had neither.

**Checkpoint.** Each `FrozenTask` is appended and fsynced to
`pool/gate/sNN.jsonl` the moment it is gated, and a re-issued command skips
what the file holds. A kill costs the fault in flight, never the walk behind
it. A line torn by that kill fails to parse and is simply re-gated, so the
checkpoint needs no separate integrity record. This is the discipline
`evaluate` already had through `cells.done`; the gate was the one stage
without it.

**Shard.** The per-fault work is independent, so a range can be cut. What is
NOT independent is everything after it: G4 dedups across the faults of one
coding task, the control arm samples over all G1-only survivors, and
`pool_sha256` covers the whole pool. Running any of those in a shard would
give each shard a different corpus. So the split is exactly there —
`gate_many` is per-fault and shardable, `assemble` is cross-fault and runs
once — and `tests/test_gate.py` asserts a two-shard freeze reproduces the
single-pass `pool_sha256`, `pool_order` and funnel.

`--assemble` refuses on a gap. A funnel computed over missing faults is wrong
in a way nothing downstream would notice, and a `pool_sha256` over a partial
corpus is a digest of something nobody can name.

`--dry-run` prints the candidate count and exits. It is the one authority on
how to cut ranges, for the same reason `scripts/fleet.sh` reads the universe
size out of `evaluate --dry-run` rather than writing it down.
