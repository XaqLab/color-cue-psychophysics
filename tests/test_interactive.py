import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from color_cue.interactive import (
    InteractiveColorCueTask,
    InteractiveTaskConfig,
    StaircaseColorCueTask,
    StaircaseTaskConfig,
    list_saved_sessions,
    list_saved_staircase_sessions,
    load_multiple_sessions,
    load_multiple_staircase_sessions,
    quickstart_calibrated_task,
    save_staircase_session,
    save_task_session,
    summarize_loaded_staircase_sessions,
)


def test_fixed_session_roundtrip(tmp_path):
    task = InteractiveColorCueTask(
        InteractiveTaskConfig(
            delta_thetas=(-0.1, 0.1),
            sigma_ext_levels=(0.0,),
            n_repeats=1,
            contexts=("redder",),
        )
    )
    rows = task.trials.head(2).copy()
    rows["response_key"] = ["left", "right"]
    rows["choose_right"] = [0, 1]
    rows["is_correct"] = [1, 0]
    rows["trial_index"] = [0, 1]
    task._responses = rows.to_dict(orient="records")

    save_task_session(task, tmp_path, "s1", "sess1")
    listing = list_saved_sessions(tmp_path, "s1")
    assert len(listing) == 1
    combined = load_multiple_sessions(tmp_path, "s1")
    assert len(combined) == 2


def test_quickstart_calibrated_task_uses_target_total_noise():
    task = quickstart_calibrated_task(
        sigma_internal=0.05,
        sigma_targets=(0.08, 0.10),
        n_repeats=1,
        delta_thetas=(-0.1, 0.1),
    )
    expected = ((0.08**2 - 0.05**2) ** 0.5, (0.10**2 - 0.05**2) ** 0.5)
    assert np.allclose(task.config.sigma_ext_levels, expected)
    assert np.allclose(sorted(task.trials["sigma_ext"].unique()), sorted(expected))


def test_staircase_summary_and_roundtrip(tmp_path):
    task = StaircaseColorCueTask(
        StaircaseTaskConfig(
            sigma_ext_levels=(0.0, 0.1),
            delta_grid=(0.03, 0.05, 0.08),
            start_index=2,
            max_trials_per_staircase=10,
            max_reversals_per_staircase=3,
        )
    )
    rows = []
    for i, sigma in enumerate(task.config.sigma_ext_levels):
        sid = f"sigma_{i}"
        rows.append(
            {
                "context": task.config.context,
                "shared_noise": task.config.shared_noise,
                "sigma_ext": float(sigma),
                "theta0": task.config.theta0,
                "delta_theta": 0.08,
                "abs_delta_theta": 0.08,
                "theta_left_target": 0.0,
                "theta_right_target": 0.0,
                "eps_left": 0.0,
                "eps_right": 0.0,
                "theta_left": 0.0,
                "theta_right": 0.0,
                "effective_delta": 0.08,
                "correct_right": True,
                "staircase_id": sid,
                "step_index_before": 2,
                "reversal_count_before": 0,
                "global_trial_index": i,
                "response_key": "right",
                "choose_right": 1,
                "is_correct": 1,
                "step_direction": -1,
                "reversal": True,
                "step_index_after": 1,
                "reversal_count_after": 1,
                "staircase_trial_index": 1,
                "staircase_active_after": True,
            }
        )
        task.staircases[sid]["n_trials"] = 1
        task.staircases[sid]["reversal_count"] = 1
        task.staircases[sid]["reversal_abs_deltas"] = [0.08]
    task._responses = rows

    summary = task.staircase_summary()
    assert len(summary) == 2
    assert (summary["threshold_estimate"] == 0.08).all()

    save_staircase_session(task, tmp_path, "s1", "stair1")
    listing = list_saved_staircase_sessions(tmp_path, "s1")
    assert len(listing) == 1
    combined = load_multiple_staircase_sessions(tmp_path, "s1")
    loaded_summary = summarize_loaded_staircase_sessions(
        combined, threshold_reversal_count=1
    )
    assert len(loaded_summary) == 2
    assert (loaded_summary["threshold_estimate"] == 0.08).all()
