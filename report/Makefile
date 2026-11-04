MAIN := main

.PHONY: all clean distclean view figures

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex metadata.tex refs.bib \
             $(wildcard chapters/*.tex) $(wildcard frontmatter/*.tex) \
             $(wildcard figures/*.pdf)
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(MAIN).tex

# Regenerate the pilot figures from the analysis outputs in data/.
# Needs the three raw inputs (see data/README.md) next to scripts/.
figures:
	cd scripts && python3 make_figs.py && cp figs/*.pdf ../figures/

clean:
	latexmk -c $(MAIN).tex
	rm -f $(MAIN).bbl

distclean: clean
	rm -f $(MAIN).pdf
