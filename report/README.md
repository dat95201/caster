# report/

LaTeX source for the Internship 2 report,
**Cost-Aware Validation Scheduling and Selected Hints for LLM-Based Program
Repair**.

```bash
make          # -> main.pdf
make clean    # drop aux files
```

Needs `latexmk` + `pdflatex` and the usual TeX Live packages (`booktabs`,
`tabularx`, `natbib`, `tikz`, `fancyhdr`, `titlesec`).

## Before you build

Edit **`metadata.tex`** — student name, student ID, instructor and month.
Nothing else reads them.

A BK logo is optional: drop `bk-logo.pdf` into `figures/` and it appears on the
title page and in the running header automatically. Without it both degrade
gracefully.

## Layout

```
main.tex              document setup; chapter order; the \pend placeholders
metadata.tex          cover-page details  <-- EDIT THIS
refs.bib              the outline's 17 references, plus 7 extras used by the
                      Background and Experiments chapters
frontmatter/          title page, signature page, acknowledgements, abstract
chapters/             01 introduction   02 related works   03 background
                      04 methodology    05 experiments     06 results
                      07 discussion     08 threats         09 conclusion
figures/              figures, as PDF
data/                 generated artefacts
scripts/              the analysis that produces data/ and figures/
_old_v1/              the superseded proposal draft, kept for reference
```

## Placeholders

The 100-task ConDefects pilot is still running, so **Chapter 6 is a skeleton**:
every table and figure has its final shape, its units and its reading, and
every value yet to be measured is marked with one of three macros defined in
`main.tex`.

| macro | renders as | use |
|---|---|---|
| `\pend` | a short red rule | one missing number inside a table |
| `\pendlong` | `[pending]` in red | a missing value in running text |
| `\pendnote{...}` | a red-bordered box | a paragraph-level notice about what is still to come |
| `\figplaceholder{...}` | a red-bordered box | a figure that has not been generated yet |

Grep for `\pend` before submitting:

```bash
grep -rn 'pend\|figplaceholder' chapters/ | grep -v '^main.tex'
```

## Document conventions

* **CASTER** is written `\caster`, **FRTP** `\frtp`, inline code `\code{...}`.
* The scheduling index is `\km` (`K/m̂`) and `\kmc{c}` (`K/m̂^c`).
* CPU seconds are the primary cost unit; test executions are reported
  alongside, never instead (Section 7.2 says why).
* Research questions: RQ1 scheduling, RQ2 hints, RQ3 the two combined,
  RQ4 ablation and sensitivity. Each is stated with the outcome that would
  refute it (Section 5.1).

## Build health

The build is expected to be clean: zero undefined references, zero undefined
citations, zero overfull boxes.

```bash
latexmk -pdf -interaction=nonstopmode main.tex
grep -cE 'LaTeX Warning: (Reference|Citation)' main.log   # -> 0
grep -c 'Overfull \\hbox' main.log                        # -> 0
```

## Reference verification

Every entry in `refs.bib` was checked against arXiv abstract pages and
OpenAlex/Crossref records in November 2026. Corrections applied at that pass:

| entry | what was wrong |
|---|---|
| `lin2026torun` | five given names wrong — correct: Zhihao Lin, Junhua Zhu, Mingyi Zhou, Xin Wang, Zhensu Sun, Renyu Yang, David Lo, Li Li |
| `le2026semantic` | last author is **Tien N. Nguyen**, not "Thanh Nam Nguyen" |
| `shinn2023reflexion` | **Edward Berman** was missing (3rd author) |
| `chen2025sweexp` | author list wrong: order, one misspelling (Longfei not Longhui), one author who is not on the paper, two missing |
| `xiao2024expressapr` | first author is **Yuan-An Xiao**, not "Yu-An Xiao" |
| `yin2024thinkrepair` | pages supplied: 1274–1286 |
| `xia2025agentless` | pages supplied: 801–824. Published title has **no** "Agentless:" prefix |
| `parasaram2025factselection` | Earl **T.** Barr |
| `kang2025autosd` | article number could not be confirmed; dropped in favour of vol. 30, no. 2 |
| `lou2024aprregression` | "Article 180" could not be confirmed; dropped |

Two claim wordings in the text were also corrected to match the sources: FRTP
performs "at least as well as" GenProg (not "matched"), ExpressAPR identifies
five "classes" (not "families"), Parasaram et al. repair 88/157 "above the best
configuration" (not "best fixed configuration"), and the Lou et al. sentence
about reporting executions is now a paraphrase rather than a verbatim quote,
because the quoted wording could not be located in the accessible text.

## Band protocol (added after the pilot was frozen)

The corpus is frozen at `--min-pool 1` on purpose. Pre-run screens were moved
into the analysis as bands (DECISIONS 21-22), so the three conditions under
which the method is expected to pay off are reported as strata, not baked into
the corpus:

| axis | implemented? | where |
|---|---|---|
| pool size | yes — `POOL_BANDS`, `arms_by_pool_band` | `report/analyse.py` |
| task difficulty | partly — `BANDS` + `band_of` label per task, but **no `arms_by_band` aggregation** | `report/analyse.py` |
| within-task cost spread | **not implemented** — needs a tercile band over the frozen cost table | — |

Two small additions are therefore outstanding before Chapter 6's
`tab:res-bands` can be filled:

1. `arms_by_difficulty_band(cs, rates)` — the same shape as
   `arms_by_pool_band`, keyed by `band_of(rate)`.
2. `COST_BANDS` + `cost_band_of(cv)` over the within-task coefficient of
   variation from the frozen cost table, plus `arms_by_cost_band`.

Both are stratifications over a frozen population, so neither requires
re-running anything.
