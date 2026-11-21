"""Remove run-dependent metadata from notebooks, so that re-running them
does not show up as a change in git.

Removed: per-cell execution timings ("execution") and jupytext's
"lines_to_next_cell". Outputs and execution counts are kept, so the
notebooks still display their results on GitHub.

Usage: python3 scripts/clean_notebooks.py notebooks/*.ipynb
"""
import json
import sys

VOLATILE_CELL_KEYS = ("execution", "lines_to_next_cell")


def clean(path):
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)
    changed = False
    for cell in nb["cells"]:
        meta = cell.get("metadata", {})
        for key in VOLATILE_CELL_KEYS:
            if key in meta:
                del meta[key]
                changed = True
    if changed:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write("\n")
    return changed


if __name__ == "__main__":
    for p in sys.argv[1:]:
        if clean(p):
            print("cleaned", p)
