# The questions

One study, one resource. A repair loop spends a counterexample twice and wastes
it both times: once by running the oracle in an order nobody chose, and once by
letting the model redraw from the mode it just drew from. Keying the memory by
**test case** is what lets one record serve both halves — a case is an
*observation*, and only an observation can be phrased as a positive requirement
("produce exactly this output on this input"). A failure *class* can only be
phrased as a prohibition, which is why the exclusion framing is an arm and not
the design.

Four questions. Each one states what would **refute** it, because a question
with no refuting outcome is a name for a group of arms, not a question.

Every arm's `answers` field names one of the headings below, and
`tests/test_questions.py` fails the build if it names anything else. That test
exists because of a real defect: `fault-record` carried the label
"RQ1 — FRTP, the memory baseline" through a year of commits while ranking on
something FRTP does not rank on, and nothing could catch it, because the label
had no statement to contradict (DECISIONS 26).

---

## calibration

Not a research question — the two measurements every number below is read
against.

| | |
|---|---|
| **Statement** | What does the oracle cost when nothing is scheduled, and what does maintaining a ledger cost when nothing reads it? |
| **Metric** | CPU seconds per episode; `test_calls`. |
| **Arms** | `full-suite` (every probe, no early stop) · `no-mem` (pool order, ledger never updated) |
| **Reported as** | The denominator of RQ1 and the baseline of RQ3. Never as a result. |

---

## RQ1 — the cost of refuting a patch

| | |
|---|---|
| **Statement** | Does scheduling probes by `K/m̂` — an online per-case kill rate over an online per-case cost — reduce the CPU spent refuting a patch, against the incumbent, against the uninformed control, and against the published prioritisers? |
| **Metric** | CPU seconds per episode, paired per cell under CRN. `test_calls` alongside, because the published prioritisation literature reports executions rather than time (Lou et al., TOSEM 2024, measure executions explicitly because time is unstable) and the two must be comparable. |
| **Arms** | `natural` (incumbent) · `random` (control) · `fault-record` (FRTP, Qi et al. ICSM 2013) · `coverage` (the additional-greedy family, `RTP_add`) · `km` |
| **What `km` is** | `K/m̂` over a per-episode memory, plus one deferral: a probe whose rejection set is already ≥ `overlap_thr` inside a probe that PASSED this round goes to the back of **that round's** queue, once. The deferral is the conditional correction to a marginal `K`, and it is what makes `km` a member of the `RTP_add` family rather than a `RTP_total` one. |
| **Refuted when** | `km` is not cheaper than `natural` beyond the cluster bootstrap's interval; or it is cheaper than `natural` but not than `fault-record` and `coverage`, in which case the contribution is early-stopping, which is not ours. |
| **Guaranteed, not measured** | `success@B` is identical across every Block A arm (Invariant 2). RQ1 is a *pure cost* claim; an ordering that changed acceptance would be a soundness bug. |

## RQ2 — which term carries it

| | |
|---|---|
| **Statement** | Of the three things `km` adds to the incumbent — a memory, a rate over that memory, and a division by cost — which one produces the effect? |
| **Metric** | The stepwise CPU decomposition, paired per cell. One variable per step. |
| **Arms** | `natural → cost-only` · `natural → kill-only` · `natural → fault-record` · `fault-record → kill-only` (a COUNT of kills becomes a RATE — the first quantity that reads `n`, which is the evidence of *not* killing that the whole FRTP family discards) · `kill-only → km-ov0` · `cost-only → km-ov0` · `km-ov0 → km` (a MARGINAL rate becomes a CONDITIONAL one — the last step, and the only one that reads which patches a probe refuted rather than how many) |
| **Refuted when** | No step clears the interval, so the gain is "having any memory at all" and the score is ornament; or `cost-only` alone carries it, in which case the kill rate is ornament. |
| **What pilot v1 settled about the last step** | **Refuted, and the refutation is the finding** (DECISIONS 39). Holding the exploration RNG fixed and toggling only the clause over the arm's own candidate sequences, `km-ov0 → km` fires **0 deferrals in 3,556 rounds** and moves CPU by **exactly 0.0%**. Seed `R(·)` from the audit's kill matrix and the same rule fires 2,971 deferrals and reaches **−6.6%, CI95 [−12.6, −1.4]**. The rule is not wrong; it is unfed. Under fail-fast a round records exactly ONE refuting probe, so a probe that PASSES this round has a rejection of its own on record in only **7.7% of rounds**, while the audit matrix shows a **median 87%** of probe pairs are ≥75% subsumed. Do not quote the grid pair `km` vs `km-ov0` for this: the ledger RNG is seeded from the cell key, which contains the policy, so the two arms explore differently from round 1 and differ on 1,008 of 3,556 rounds for reasons unrelated to the clause. `report/scripts/coverage_check.py` is the measurement. |

## RQ3 — renormalising the patch family

| | |
|---|---|
| **Statement** | Does handing the model a counterexample *as a positive requirement keyed to a test case* move its proposal distribution off the mode it has already drawn — fewer duplicate patches, more repairs at the same budget — where a conversation transcript and a prohibition do not? |
| **Metric** | **`duplicate_rate`** (1 − distinct/total patches) as the primary; `success@B` and `first_accept_round`; **prompt and completion tokens, and USD**, because the LLM-repair literature prices repairs in tokens (CigaR: 127K/bug against 467K) and a CPU-only table answers the cheaper half. |
| **Arms** | `no-mem` (independent repeated sampling — the strong baseline, see below) · `km` (scheduling, no steering) · `km+steer` (the contract) · `km+steer-pass` · `km+exclusion` (the prohibition) · `km+chat` (ChatRepair-style transcript) |
| **Refuted when** | `duplicate_rate` does not fall against `no-mem`; or it falls and `success@B` does not rise, in which case the distribution moved somewhere no better. |
| **Why `no-mem` is the baseline and not the floor** | It redraws independently every round under a policy-free nonce, which is exactly *independent repeated prompting*. Chen et al. (arXiv 2503.15050) measured that strategy at **23 bugs repaired against ChatRepair's 18** over 24 attempts, and measured **65% of ChatRepair's generated patches to be duplicates**. The obvious iterative approach is already published as *worse* than drawing again. Block B must beat `no-mem`; beating `natural` proves nothing here. |

## RQ4 — how much disclosure before the model fits the tests

| | |
|---|---|
| **Statement** | Showing a counterexample tells the model what to satisfy. At what disclosure does it begin satisfying *the shown probes* rather than repairing the fault? |
| **Metric, primary — needs no holdout** | Of the patches that are refuted at all, the share whose refuting set is **disjoint from what the prompt showed**: the patch passed everything it was told and died elsewhere. Computed from `facts_shown` (on every round record) against the full kill vector from `caster audit`. Reported beside its complement — the share refuted *only* by shown probes, which bounds how much signal the undisclosed remainder still carries. |
| **Metric, secondary — needs `--holdout`** | Of the patches *accepted*, the share that fail on `P_eval`. At `--holdout 0` the loop accepts on exactly the probes that define success, so "accepted" cannot distinguish a repair from a pool-fit and no post-hoc analysis recovers it. One pass on a **subset of tasks** is enough; it answers a different question from RQ1 and does not need the grid (DECISIONS 23). |
| **Arms** | `km+steer-k1` / `km+steer-k2` / `km+steer` (k=3, the default) · `km+steer-sample` (names, no required outputs) · `km+steer-noreq` · `km-explore0` is a **precondition**, not an arm here: without exploration a task whose first probe refutes everything reaches the prompt with one fact, every k degenerates to k=1, and the sweep compares k=1 against k=1 |
| **Read against `k_facts_rendered`, never `k_facts`** | The selector deduplicates near-identical facts and the pool of proven refuters is thin: on the pilot, **55% of episodes ended with fewer than two distinct proven refuters**, so `k` is a ceiling far more often than a count. A sweep read against the REQUESTED k reports a comparison it did not run. This is the same failure `km-explore0` is kept to prevent, arriving by a different road. |
| **The default moved on an argument, not a measurement** | `k_facts` went 2 → 3 because deduplication frees a slot, not because k=3 was measured better. It was measured **worse** on the metric above: 16.8% leakage against k=2's 7.0%. Whether dropping near-duplicates pulls that back is what `km+steer-k2` vs `km+steer` now has to show. If it does not, the default is wrong and reverts. |
| **Refuted when** | The share is flat across k and across disclosure, in which case there is no dose and the disclosure axis does not exist. |

---

## sensitivity

Not a research question either — the arms that ask whether the answers above
survive our own design choices. They belong in threats to validity. Keeping
them is free (Block A costs no model calls); calling them a research question
would pad the count from four to seven, which is what the roster used to do.

| | |
|---|---|
| **Statement** | Does RQ1's answer depend on the cold start for `m̄`, on a continuous score rather than buckets, or on `explore_p`? Does RQ3's depend on how the draw itself is sampled? |
| **Arms** | `km-cs-median` · `km-cs-q75` · `km-bucket` · `km-explore0` · `km+steer+vs` |
| **Reported as** | A sensitivity table. `km-explore0` is additionally a **precondition** of RQ4, and `km+steer+vs` is a different intervention — it diversifies the *draw*, not the *evidence* — so the arm stays and the paper does not claim it. |

---

## What was dropped

Seven labels used to exist in the code and none was ever written down. Three
are gone as questions:

* `RQ3` (cold start, bucket, exploration) → **sensitivity**, above.
* `RQ5` (how many facts) and `RQ6` (how much disclosure) → merged into **RQ4**.
  They are one dose-response curve asked twice, and splitting them invited
  reporting a k-sweep and a disclosure-sweep as two findings.
* `RQ7` (verbalized sampling) → **sensitivity**. Its one measurable quantity,
  `duplicate_rate`, moved to where it belongs: it is RQ3's primary metric, not
  this arm's by-product.
