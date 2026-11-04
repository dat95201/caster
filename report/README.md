# report/

LaTeX source for the Internship 2 proposal,
**Cost-Aware Scheduling of Test Evidence for LLM-Based Program Repair**.

```bash
make          # -> main.pdf
make clean    # drop aux files
```

Needs `latexmk` + `pdflatex` and the usual TeX Live packages (`booktabs`,
`tabularx`, `siunitx`, `natbib`, `pgfgantt`, `fancyhdr`, `titlesec`).

## Before you build

Edit **`metadata.tex`** — it holds the student name, student ID, instructor and
month, and nothing else reads them. The student ID is a placeholder.

A BK logo is optional: drop `bk-logo.pdf` into `figures/` and it appears on the
title page and in the running header automatically. Without it both degrade
gracefully.

## Layout

```
main.tex              document setup; chapter order
metadata.tex          cover-page details  <-- EDIT THIS
refs.bib              17 references, every one checked against the publisher
                      record or the paper (see data/CITATIONS.md)
frontmatter/          title page, signature page, acknowledgements, abstract
chapters/             01 intro · 02 related · 03 background
                      04 framework + the pilot · 05 evaluation plan · 06 summary
figures/              the four pilot figures, as PDF
data/                 the generated artefacts every pilot number is read from
scripts/              the analysis that produced data/ and figures/
```

## What is evidence and what is plan

Section 4.3 is the only part of the document that reports measurements. It is a
pilot on the **frozen CEGMem data**, not on this repository: a scheduler can
only be compared counterfactually when the verdict of every case on every patch
is known, and only the CEGMem run left a complete patch × case verdict matrix
behind. Everything derived from the CASTER repository itself — the 22-arm grid,
Block B, the duplicate-rate and token measurements — is stated as Phase 2 work
throughout, and Chapter 5 contains no results, only targets.

## Reproducing the pilot numbers

```bash
python3 scripts/verify_numbers.py     # 75 assertions; exit 0 iff the report
                                      # matches data/*.json
```

That gate is the cheap check and runs with no inputs beyond `data/`. To
regenerate `data/` from scratch you need three raw files from the CEGMem run
directory:

| File | Source | What it is |
|---|---|---|
| `verdicts_matrix.json` | `ceg-mem/data/sched3/` | per task: the case pool, and for every stored patch the indices of the cases that kill it |
| `cost_all.json` | `ceg-mem/data/sched3/` | per (task, case): CPU of the **reference** program, measured with `getrusage` |
| `cov_all.json` | `ceg-mem/data/sched3/` | per (task, case): the line set the reference program walks |
| `rounds.csv` | `ceg-mem/data/v4nomem/` | the **measured** round log — 33,000 rows, 3 policies × 11,000 rounds |

Then, with those four beside `scripts/`:

```bash
cd scripts
python3 verify_head.py      # replay head-to-head, 11 policies, 20 seeds
python3 run_iso.py          # cost-exponent sweep + coverage isolation
python3 measured_stats.py   # paired stats over the measured round log
python3 make_figs.py        # the four figures
```

`arena.py` builds the 68-task arena as the intersection of the three
measurement passes; `sim.py` is the replay simulator and carries the rules it
reproduces in its docstring.

## Provenance of the CEGMem premise figures

Table 3.1 and the surrounding numbers in §3.4 come from
`ceg-mem/paper/common/numbers.json`, the frozen provenance file of that study
(309 tasks × 5 seeds, budget 20, `gpt-4o-mini`). They are quoted, not recomputed
here.
