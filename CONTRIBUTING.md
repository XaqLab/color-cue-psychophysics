# Contributing to `color-cue-psychophysics`

## Layout

- `src/color_cue/`
  - package source of truth
- `tests/`
  - package-local tests
- `notebooks/`
  - demo and launch notebooks that import from `color_cue`

## Local setup

From the repo root:

```bash
pip install -e .
```

For notebook-based human data collection:

```bash
pip install -e ".[collect]"
```

For development extras:

```bash
pip install -e ".[dev]"
```

## Tests

Run:

```bash
pytest tests
```

GUI interaction in the notebooks is not fully automated, so if you touch the
interactive task code you should also do a manual smoke test of at least one
fixed-level notebook and one staircase notebook.

## Notebook updates

When changing public APIs:

1. Update the package code in `src/color_cue/`.
2. Update the notebooks in `notebooks/`.

## Generated outputs

Do not commit local subject/session data or notebook-generated result CSVs.
Relevant output directories are ignored in the repo-level `.gitignore`.
