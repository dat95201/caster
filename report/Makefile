MAIN := main

.PHONY: all clean distclean view figures analysis

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex metadata.tex refs.bib \
             $(wildcard chapters/*.tex) $(wildcard frontmatter/*.tex) \
             $(wildcard figures/*.pdf)
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(MAIN).tex

# Re-derive every pilot number and figure from the merged run in
# ../data/pilot-v1. `analysis` writes data/*.json; `figures` draws from them.
RUN ?= ../../data/pilot-v1

analysis:
	cd scripts && python3 pilot_v1.py $(RUN)
	cd scripts && PYTHONPATH=../.. python3 coverage_check.py $(RUN)

figures: analysis
	cd scripts && python3 pilot_figs.py

clean:
	latexmk -c $(MAIN).tex
	rm -f $(MAIN).bbl

distclean: clean
	rm -f $(MAIN).pdf
