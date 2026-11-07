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

Block A of the pilot is measured and Chapter 6 reports it. **Block B — the
configurations that change the prompt — has not been run**, so the three tables
in Section 6.5 are laid out with their final shape and marked with the
placeholder macros defined in `main.tex`.

| macro | renders as | use |
|---|---|---|
| `\pend` | a short red rule | one missing number inside a table |
| `\pendlong` | **[pending]** | a missing value in running text |
| `\pendnote{...}` | a boxed red notice | a whole section not yet measured |

Searching for `\pend` finds everything still outstanding.

## Reproducing the numbers

```bash
make analysis   # data/pilot_v1.json + data/coverage_check.json
make figures    # figures/fig-*.pdf
make            # main.pdf
python3 scripts/verify_pilot_numbers.py
```

`coverage_check.py` prices the scheduler's marginal-rate correction on its own.
Its result is a repo finding (DECISIONS 39), not report material: the clause
never fires on session-local evidence, so the report describes the rule and
does not report a measurement of it.

`make analysis` reads `../data/pilot-v1`; override with `RUN=...`. See
`data/README.md` for what each artefact contains and for the one quantity that
is reconstructed rather than read.

## Page budget

The report is held near 60 pages. Three settings in `main.tex` do most of that
and are the first things to revisit if the faculty template disagrees:
`\setstretch{1.15}`, `\setcounter{tocdepth}{1}` (no subsections in the
contents, and no List of Figures or Tables), and the float fractions just
above `\usepackage{enumitem}`. Reverting all three adds roughly twelve
pages.
