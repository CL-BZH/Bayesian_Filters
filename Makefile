.PHONY: book figures media test notebooks notebooks-refresh clean

book:
	cd book && latexmk -pdf -interaction=nonstopmode main.tex

figures:
	python3 book/figures/make_figures.py
	python3 book/figures/make_bayesnet.py

media:
	python3 scripts/make_media.py

test:
	python3 -m pytest -q

# Run every notebook without touching the repository: the executed copies
# go to build/notebooks/ (ignored by git), and no __pycache__ is written.
notebooks:
	mkdir -p build/notebooks
	cd notebooks && for nb in *.ipynb; do \
	  PYTHONDONTWRITEBYTECODE=1 python3 -m nbconvert --to notebook --execute \
	    --ExecutePreprocessor.record_timing=False \
	    --output-dir ../build/notebooks $$nb || exit 1; done

# Re-run the notebooks in place, to update the outputs stored in the
# repository (after a change to the code), without run-dependent metadata.
notebooks-refresh:
	cd notebooks && for nb in *.ipynb; do \
	  PYTHONDONTWRITEBYTECODE=1 python3 -m nbconvert --to notebook --execute \
	    --ExecutePreprocessor.record_timing=False --inplace $$nb || exit 1; done
	python3 scripts/clean_notebooks.py notebooks/*.ipynb

clean:
	cd book && latexmk -c
