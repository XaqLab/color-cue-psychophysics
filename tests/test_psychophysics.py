import numpy as np
import pandas as pd
from color_cue.psychophysics import (
    choose_right_probability,
    comparison_sd_from_stimulus_sigma,
    compute_external_noise,
    fit_all_gaussian_psychometrics,
    fit_all_psychometrics,
    jnd_from_stimulus_sigma,
    make_noise_calibration_table,
    make_trial_table,
    sample_bounded_gaussian_noise,
    simulate_matched_noise_observers,
    simulate_observer,
    stimulus_sigma_from_jnd,
)


def test_make_trial_table_columns_and_size():
    trials = make_trial_table(
        theta0=-np.pi / 4,
        delta_thetas=(-0.1, 0.1),
        sigma_ext_levels=(0.0, 0.1),
        n_repeats=3,
        contexts=("redder",),
        rng=0,
    )
    assert len(trials) == 2 * 2 * 3
    assert {"effective_delta", "correct_right", "theta_left", "theta_right"} <= set(
        trials.columns
    )


def test_bounded_gaussian_noise_stays_within_theta_bounds():
    rng = np.random.default_rng(0)
    target = -0.01
    samples = np.array(
        [
            sample_bounded_gaussian_noise(
                target,
                sigma=0.5,
                rng=rng,
                theta_min=-1.0,
                theta_max=0.0,
            )
            for _ in range(200)
        ]
    )
    observed = target + samples
    assert np.all(observed >= -1.0)
    assert np.all(observed <= 0.0)


def test_trial_table_uses_truncated_noise_without_clipping():
    trials = make_trial_table(
        theta0=-0.05,
        delta_thetas=(0.08,),
        sigma_ext_levels=(0.5,),
        n_repeats=200,
        contexts=("redder",),
        rng=0,
        theta_min=-1.0,
        theta_max=0.0,
    )
    assert np.all(trials["theta_left"].between(-1.0, 0.0))
    assert np.all(trials["theta_right"].between(-1.0, 0.0))
    assert not trials["left_clipped"].any()
    assert not trials["right_clipped"].any()


def test_shared_truncated_noise_is_valid_for_both_sides():
    trials = make_trial_table(
        theta0=-0.05,
        delta_thetas=(0.08,),
        sigma_ext_levels=(0.5,),
        n_repeats=50,
        contexts=("redder",),
        rng=1,
        theta_min=-1.0,
        theta_max=0.0,
        shared_noise=True,
    )
    assert np.allclose(trials["eps_left"], trials["eps_right"])
    assert np.all(trials["theta_left"].between(-1.0, 0.0))
    assert np.all(trials["theta_right"].between(-1.0, 0.0))


def test_simulated_observer_and_fit():
    trials = make_trial_table(
        theta0=-np.pi / 4,
        delta_thetas=(-0.2, -0.1, 0.1, 0.2),
        sigma_ext_levels=(0.0, 0.1),
        n_repeats=50,
        contexts=("redder", "bluer"),
        rng=1,
    )
    observed = simulate_observer(trials, sigma_int=0.05, rng=2)
    fits = fit_all_psychometrics(observed)
    assert len(fits) == 4
    assert fits["success"].all()
    assert (fits["threshold_75"] > 0).all()


def test_jnd_and_stimulus_sigma_round_trip():
    sigma_int = 0.07
    jnd = jnd_from_stimulus_sigma(sigma_int, p_correct=0.75)
    recovered = stimulus_sigma_from_jnd(jnd, p_correct=0.75)
    assert np.isclose(recovered, sigma_int)
    assert np.isclose(
        comparison_sd_from_stimulus_sigma(sigma_int),
        np.sqrt(2) * sigma_int,
    )


def test_correlated_internal_noise_changes_comparison_sd():
    sigma_int = 0.1
    rho_int = 0.4
    expected = np.sqrt(2 * (1 - rho_int)) * sigma_int
    assert np.isclose(
        comparison_sd_from_stimulus_sigma(sigma_int, rho=rho_int),
        expected,
    )


def test_external_noise_calibration_table():
    subjects = pd.DataFrame(
        {
            "subject_id": ["low_threshold", "high_threshold"],
            "sigma_internal": [0.05, 0.10],
        }
    )
    table = make_noise_calibration_table(subjects, sigma_targets=(0.12,))
    assert len(table) == 2
    assert table["is_feasible"].all()
    total = np.sqrt(table["sigma_internal"] ** 2 + table["sigma_ext"] ** 2)
    assert np.allclose(total, table["sigma_target"])
    assert np.isnan(compute_external_noise(0.12, 0.10))


def test_gaussian_probability_uses_per_stimulus_sigma():
    p = choose_right_probability(0.0, sigma_total=0.08, context="redder")
    assert np.isclose(p, 0.5)
    stronger = choose_right_probability(0.2, sigma_total=0.08, context="redder")
    assert stronger > 0.9


def test_matched_noise_simulation_recovers_target_noise():
    subjects = pd.DataFrame(
        {
            "subject_id": ["precise", "noisy"],
            "sigma_internal": [0.04, 0.09],
        }
    )
    trials = simulate_matched_noise_observers(
        subjects=subjects,
        sigma_targets=(0.11,),
        theta0=-np.pi / 4,
        delta_thetas=(-0.24, -0.16, -0.08, 0.08, 0.16, 0.24),
        n_repeats=120,
        contexts=("redder",),
        rng=4,
    )
    fits = fit_all_gaussian_psychometrics(
        trials,
        group_cols=("subject_id", "context", "sigma_target"),
    )
    assert len(fits) == 2
    assert fits["success"].all()
    assert np.allclose(fits["sigma_stimulus"], 0.11, atol=0.035)
