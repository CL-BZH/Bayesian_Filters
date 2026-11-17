.PHONY: book figures media test notebooks clean

book:
	cd book && latexmk -pdf -interaction=nonstopmode main.tex

figures:
	python3 book/figures/make_figures.py
	python3 book/figures/make_bayesnet.py

media:
	python3 scripts/make_media.py

test:
	python3 -m pytest -q

notebooks:
	cd notebooks && for nb in *.ipynb; do \
	  python3 -m nbconvert --to notebook --execute --inplace $$nb; done

clean:
	cd book && latexmk -c
