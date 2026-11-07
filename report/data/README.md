# data/ — generated artefacts

Every pilot number printed in the report is read from one of these files, and
`scripts/verify_pilot_numbers.py` fails the build if any has drifted.

| File | Produced by | Contents |
|---|---|---|
| `pilot_v1.json` | `scripts/pilot_v1.py <run-dir>` | Everything Chapter 6 reports about Block A: the corpus summary, per-arm totals, the seven paired contrasts with cluster-bootstrap CIs and Holm-corrected Wilcoxon p-values, the ablation ladder, the per-round curve, the per-task totals, and the coverage-check evidence counts |
| `coverage_check.json` | `PYTHONPATH=../.. python3 scripts/coverage_check.py <run-dir>` | The overlap clause priced three ways — off, fed by the session, fed by the audit table — over the arm's own candidate sequences with the exploration RNG held fixed |

Regenerate both, then the figures:

```bash
make analysis     # -> data/*.json
make figures      # -> figures/fig-*.pdf
python3 scripts/verify_pilot_numbers.py
```

`RUN=../../data/pilot-v2 make analysis` points them at another merged run.

## What the inputs are

`<run-dir>` is a finished `caster merge` output and needs three files:
`rounds.jsonl` (one row per round of every cell), `kill_matrix.json` (the audit
pass's complete (patch × test) verdict matrix) and `pool/tasks.json` (the
frozen corpus).

## One reconstruction, and where it does and does not matter

The per-test cost table is not shipped in the run directory, so `pilot_v1.py`
recovers it by least squares: in a replayed run a round's CPU is the sum of the
frozen costs of the tests it ran, which gives one linear equation per distinct
(scan order, CPU) pair. This is exact for a test that appears in enough rounds
and noisy for one that appears in few.

It feeds the descriptive statistics (Table 6.2), the cost-spread split
(Table 6.5) and the coverage-check replay. It does **not** feed any
head-to-head total: those are summed directly from `rounds.jsonl`.

## The older files

`arena_stats.json`, `measured_stats.json`, `head_verify.json` and `iso.json`
come from the earlier CEGMem study (68 tasks, patches drawn by `gpt-4o-mini`)
that preceded this pilot. Nothing in the current report reads them. They are
kept because `CITATIONS.md` traces claims to them.
