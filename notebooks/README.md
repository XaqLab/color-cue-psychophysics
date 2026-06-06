# `color-cue-psychophysics` Notebooks

These notebooks are the package-local demos and launch notebooks for
`color_cue`. They are designed to be thin wrappers over the installable package
in `src/color_cue/`.

If you are collecting psychophysics data, start here:

```text
interactive-calibration-sessions.ipynb
```

That notebook is the canonical workflow. The older interactive notebook names
are retained only as legacy pointers.

## Setup For Data Collection

From the repository root:

```bash
pip install -e ".[collect]"
jupyter lab notebooks
```

If you use conda for GUI dependencies, this alternative is fine:

```bash
pip install -e .
conda install pyqt
pip install jupyterlab ipykernel
jupyter lab notebooks
```

In the notebook, uncomment:

```python
# %matplotlib qt
```

so it becomes:

```python
%matplotlib qt
```

The interactive task uses a separate Matplotlib window. Subjects answer with
the left/right arrow keys and can press `q` to quit early.

## Recommended Human Workflow

Use `interactive-calibration-sessions.ipynb` from top to bottom.

1. **Set IDs and data roots.** Set `SUBJECT_ID`, `BASELINE_SESSION_ID`,
   `CALIBRATED_SESSION_ID`, and optional notes. Use unique session IDs for each
   run.
2. **Run Experiment 1 baseline staircase.** This estimates the subject's
   baseline perceptual threshold with no injected external theta noise.
3. **Save Experiment 1.** After the task finishes, uncomment and run the
   `save_staircase_session(...)` block.
4. **Load baseline sessions.** Load one or more saved baseline sessions for the
   subject.
5. **Estimate `sigma_internal`.** The notebook averages late staircase reversal
   thresholds and converts the threshold to per-stimulus internal noise.
6. **Choose `SIGMA_TARGETS`.** The notebook computes subject-specific
   `sigma_ext = sqrt(sigma_target^2 - sigma_internal^2)` and excludes infeasible
   targets.
7. **Run Experiment 2 calibrated task.** This is a fixed-level 2AFC task using
   the calibrated external noise values.
8. **Save Experiment 2.** After the task finishes, annotate and save the
   calibrated session.
9. **Analyze pooled sessions.** Load saved calibrated sessions and fit Gaussian
   2AFC curves by subject, context, and `sigma_target`.

The save cells are intentionally commented by default so the notebook does not
accidentally write data until the operator has confirmed the subject/session
IDs.

## Session Files

By default, saved data go under:

```text
notebooks/session_data/
```

Baseline staircase sessions contain:

```text
results.csv
staircase_summary.csv
metadata.json
```

Calibrated fixed-level sessions contain:

```text
results.csv
metadata.json
```

The `session_data/` directory is ignored by git. If a collaborator needs to
share data, they should send the relevant subject/session folders directly, not
commit them to the repository.

## Notebook Guide

- `interactive-calibration-sessions.ipynb`: canonical human data-collection and
  analysis workflow.
- `noise-calibration-demo.ipynb`: simulation-only explanation of baseline
  psychometric curves, external-noise calibration, and matched-noise curves.
- `stimulus-definitions.ipynb`: mathematical definitions and derivations.
- `stimulus-definitions.tex`: LaTeX version suitable for Overleaf.
- `interactive-task.ipynb`, `interactive-task-sessions.ipynb`, and
  `interactive-staircase-sessions.ipynb`: legacy pointer notebooks.

## Practical Checks Before Running A Subject

- Confirm the task window opens after `%matplotlib qt`.
- Confirm left/right arrow keys register after clicking inside the task window.
- Run a short pilot with `N_REPEATS` or staircase trial limits reduced.
- Confirm `save_staircase_session(...)` or `save_task_session(...)` creates the
  expected `results.csv` and `metadata.json`.
- Confirm the load/analysis cells can reload the saved session before collecting
  a long batch of subjects.
