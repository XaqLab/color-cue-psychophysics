# color_cue

`color_cue` is an installable Python package for the color-cue psychophysics
workflow. It contains the reusable code for generating stimuli, running
notebook-launched human 2AFC tasks, saving sessions, and analyzing baseline and
calibrated-noise behavior.

Colleagues who need to collect or inspect color-cue psychophysics data can
install this package directly and use the notebooks in `notebooks/`.

## What This Package Does

The current experiment is a two-stage noise-calibration design:

1. **Experiment 1: baseline calibration.** Run a 2AFC staircase with no injected
   external theta noise. This estimates the subject's baseline perceptual
   threshold and converts it to a per-stimulus internal-noise estimate,
   `sigma_internal`.
2. **Experiment 2: matched total noise.** Choose one or more experimenter-defined
   total per-stimulus noise levels, `sigma_target`, then compute the
   subject-specific external noise:

```text
sigma_ext = sqrt(sigma_target^2 - sigma_internal^2)
```

3. **Analysis.** Load saved sessions, fit Gaussian 2AFC psychometric curves, and
   check whether subjects with different baseline thresholds produce matched
   curves after calibrated external noise is added.

All `sigma_internal`, `sigma_ext`, and `sigma_target` values are
**per-stimulus** standard deviations. Under the default independent left/right
noise model, the comparison-level standard deviation is `sqrt(2)` times the
per-stimulus value.

## Repository Layout

```text
color-cue/
├── pyproject.toml
├── README.md
├── CONTRIBUTING.md
├── notebooks/
│   ├── README.md
│   ├── interactive-calibration-sessions.ipynb
│   ├── noise-calibration-demo.ipynb
│   └── stimulus-definitions.ipynb
├── src/color_cue/
│   ├── stimulus.py
│   ├── psychophysics.py
│   ├── interactive.py
│   ├── staircase.py
│   └── data/cmap.csv
└── tests/
```

The notebooks are demos and launch surfaces for carrying out experiments. The package source in
`src/color_cue/` contains all the coding logic.

## Install

From the repo root, install the package in editable mode:

```bash
pip install -e .
```

For collaborators collecting human data in notebooks, install the collection
extra:

```bash
pip install -e ".[collect]"
```

The `collect` extra installs JupyterLab, an IPython kernel, and PyQt6 for the
Matplotlib Qt task window. If you prefer conda-managed Qt packages, this is also
fine:

```bash
pip install -e .
conda install pyqt
pip install jupyterlab ipykernel
```

For development and tests:

```bash
pip install -e ".[dev]"
pytest tests
```

## Quick Import Check

After installation, verify that the package and colormap data are available:

```bash
python - <<'PY'
import color_cue
from color_cue import get_cmap

print(color_cue.__file__)
print(get_cmap())
PY
```

## Running The Human Data-Collection Notebook

Open the canonical workflow notebook:

```bash
jupyter lab notebooks/interactive-calibration-sessions.ipynb
```

Then work through the notebook from top to bottom:

1. **Setup.** Set `SUBJECT_ID`, data roots, stimulus constants, and session IDs.
   Use a unique `SUBJECT_ID` and a unique session ID for every collection run.
2. **GUI backend.** Uncomment `%matplotlib qt` in the setup cell before running
   interactive tasks. The task uses a Matplotlib window and keyboard events.
3. **Experiment 1.** Configure and run the baseline staircase. Subjects respond
   with the left/right arrow keys and can press `q` to quit early.
4. **Save baseline session.** After the task window closes, uncomment and run the
   `save_staircase_session(...)` block. Saved baseline sessions contain
   `results.csv`, `staircase_summary.csv`, and `metadata.json`.
5. **Estimate `sigma_internal`.** Load one or more baseline sessions. The
   notebook summarizes late staircase reversals and converts the threshold to
   per-stimulus `sigma_internal`.
6. **Experiment 2.** Choose `SIGMA_TARGETS`. The notebook computes feasible
   `sigma_ext` values and builds the calibrated fixed-level 2AFC task.
7. **Save calibrated session.** After running Experiment 2, annotate the results
   with `sigma_internal`, `sigma_target`, and `sigma_ext`, then save the session.
8. **Analysis.** Load saved calibrated sessions and fit Gaussian 2AFC curves
   grouped by subject, context, and `sigma_target`.

Saved sessions are written under the notebook's `session_data/` folder by
default. Those folders are ignored by git so local subject/session data is not
accidentally committed.

## Notebook Guide

- `interactive-calibration-sessions.ipynb`: recommended human workflow. This is
  the notebook collaborators should use for session-based data collection.
- `noise-calibration-demo.ipynb`: simulation-only conceptual demo showing the
  baseline curves before external noise and the matched curves after
  calibration.
- `stimulus-definitions.ipynb` and `stimulus-definitions.tex`: math/reference
  documents for the old stimulus, the noise-calibration model, and the
  per-stimulus versus comparison-level noise distinction.

## Public API Sketch

```python
from color_cue.stimulus import get_theta_array, theta_to_cue, cue_to_theta
from color_cue.psychophysics import (
    fit_all_gaussian_psychometrics,
    make_noise_calibration_table,
    stimulus_sigma_from_jnd,
)
from color_cue.interactive import (
    InteractiveTaskConfig,
    InteractiveColorCueTask,
    StaircaseTaskConfig,
    StaircaseColorCueTask,
    save_task_session,
    save_staircase_session,
    load_multiple_sessions,
    load_multiple_staircase_sessions,
)
```

## Troubleshooting

- If no task window appears, make sure `%matplotlib qt` is uncommented and that
  a Qt binding is installed (`pyqt6` from pip or `pyqt` from conda).
- If the window appears but keypresses do not register, click once inside the
  stimulus window before pressing the left/right arrow keys.
- If `sigma_target` is marked infeasible, the selected target is smaller than
  the estimated `sigma_internal`; adding external noise cannot make the subject
  more precise.
- If a save call raises `FileExistsError`, change the session ID or set
  `overwrite=True` only when you intentionally want to replace that session.

## Contributing

For GitHub issue and pull-request guidance, see
`CONTRIBUTING.md`.
