from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit, minimize
from scipy.special import expit, logit, ndtri, ndtr
from scipy.stats import truncnorm

from .stimulus import get_cmap, get_theta_array


def psychometric(
    x: np.ndarray | float,
    bias: float,
    scale: float,
    lapse_low: float,
    lapse_high: float,
) -> np.ndarray:
    """Evaluate a four-parameter psychometric function.

    Args:
        x: Signed stimulus strength or effective comparison value.
        bias: Horizontal shift of the curve.
        scale: Slope parameter of the logistic transition.
        lapse_low: Lower asymptotic lapse rate.
        lapse_high: Upper asymptotic lapse rate.

    Returns:
        The predicted probability of choosing the right stimulus at ``x``.
    """
    x = np.asarray(x, dtype=float)
    return lapse_low + (1.0 - lapse_low - lapse_high) * expit((x - bias) / scale)


def comparison_sd_from_stimulus_sigma(
    sigma_stimulus: np.ndarray | float,
    rho: float = 0.0,
) -> np.ndarray:
    """Convert per-stimulus noise to comparison-level noise.

    Args:
        sigma_stimulus: Per-stimulus noise standard deviation.
        rho: Correlation between the left and right per-stimulus noise terms.
            ``rho=0`` gives the standard independent 2AFC model.

    Returns:
        The standard deviation of the right-minus-left comparison noise.

    Raises:
        ValueError: If ``rho`` is outside ``[-1, 1)``.
    """
    if not -1.0 <= rho < 1.0:
        raise ValueError("rho must be in [-1, 1).")
    sigma_stimulus = np.asarray(sigma_stimulus, dtype=float)
    return np.sqrt(2.0 * (1.0 - rho)) * sigma_stimulus


def gaussian_psychometric(
    x: np.ndarray | float,
    bias: float,
    sigma_stimulus: float,
    lapse_low: float = 0.0,
    lapse_high: float = 0.0,
    rho: float = 0.0,
) -> np.ndarray:
    """Evaluate the Gaussian 2AFC psychometric function.

    Args:
        x: Effective signed hue difference. Positive values favor a rightward
            response after context sign-flipping.
        bias: Horizontal shift of the psychometric curve.
        sigma_stimulus: Per-stimulus total noise standard deviation.
        lapse_low: Lower asymptotic lapse rate.
        lapse_high: Upper asymptotic lapse rate.
        rho: Correlation between left and right per-stimulus noise terms.

    Returns:
        Predicted probability of choosing the right stimulus.
    """
    x = np.asarray(x, dtype=float)
    comparison_sd = comparison_sd_from_stimulus_sigma(sigma_stimulus, rho=rho)
    z = (x - bias) / np.maximum(comparison_sd, 1e-12)
    return lapse_low + (1.0 - lapse_low - lapse_high) * ndtr(z)


def stimulus_sigma_from_jnd(
    jnd: np.ndarray | float,
    p_correct: float = 0.75,
    rho: float = 0.0,
) -> np.ndarray:
    """Convert a 2AFC criterion threshold into per-stimulus noise.

    Args:
        jnd: Absolute effective-delta threshold at criterion ``p_correct``.
        p_correct: Psychometric criterion used to define the threshold.
        rho: Correlation between left and right per-stimulus noise terms.

    Returns:
        Per-stimulus noise standard deviation implied by the JND.

    Raises:
        ValueError: If ``p_correct`` is not in ``(0.5, 1)``.
    """
    if not 0.5 < p_correct < 1.0:
        raise ValueError("p_correct must be in (0.5, 1).")
    jnd = np.asarray(jnd, dtype=float)
    denom = comparison_sd_from_stimulus_sigma(1.0, rho=rho) * ndtri(p_correct)
    return jnd / denom


def jnd_from_stimulus_sigma(
    sigma_stimulus: np.ndarray | float,
    p_correct: float = 0.75,
    rho: float = 0.0,
) -> np.ndarray:
    """Convert per-stimulus noise into a 2AFC criterion threshold.

    Args:
        sigma_stimulus: Per-stimulus noise standard deviation.
        p_correct: Psychometric criterion used to define the threshold.
        rho: Correlation between left and right per-stimulus noise terms.

    Returns:
        Absolute effective-delta threshold at criterion ``p_correct``.
    """
    if not 0.5 < p_correct < 1.0:
        raise ValueError("p_correct must be in (0.5, 1).")
    return comparison_sd_from_stimulus_sigma(sigma_stimulus, rho=rho) * ndtri(
        p_correct
    )


def threshold_from_fit(
    fit: dict,
    p_correct: float = 0.75,
) -> float:
    """Compute a threshold from fitted psychometric parameters.

    Args:
        fit: Dictionary containing at least ``scale``, ``lapse_low``, and
            ``lapse_high``.
        p_correct: Performance level at which to define threshold.

    Returns:
        The absolute effective-delta value required to reach ``p_correct`` on
        the fitted psychometric curve.
    """
    denom = 1.0 - fit["lapse_low"] - fit["lapse_high"]
    q = (p_correct - fit["lapse_low"]) / denom
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return float(abs(fit["scale"] * logit(q)))


def threshold_from_gaussian_fit(
    fit: dict,
    p_correct: float = 0.75,
) -> float:
    """Compute a criterion threshold from a Gaussian 2AFC fit.

    Args:
        fit: Dictionary containing at least ``sigma_stimulus``,
            ``lapse_low``, and ``lapse_high``.
        p_correct: Performance level at which to define threshold.

    Returns:
        The absolute effective-delta value required to reach ``p_correct``
        relative to the fitted bias.
    """
    denom = 1.0 - fit["lapse_low"] - fit["lapse_high"]
    q = (p_correct - fit["lapse_low"]) / denom
    q = np.clip(q, 1e-6, 1 - 1e-6)
    rho = float(fit.get("rho", 0.0))
    return float(jnd_from_stimulus_sigma(fit["sigma_stimulus"], q, rho=rho))


def choose_right_probability(
    delta_theta: np.ndarray | float,
    sigma_total: float,
    context: str,
    rho: float = 0.0,
) -> np.ndarray:
    """Return the ideal-observer right-choice probability.

    Args:
        delta_theta: Signed hue differences between right and left targets.
        sigma_total: Per-stimulus total noise standard deviation.
        context: Task prompt, either ``"redder"`` or ``"bluer"``.
        rho: Correlation between left and right per-stimulus noise terms.

    Returns:
        The probability of choosing the right-hand stimulus under the idealized
        Gaussian comparison model.

    Raises:
        ValueError: If ``context`` is not one of the supported task prompts.
    """
    delta_theta = np.asarray(delta_theta, dtype=float)
    if context not in {"redder", "bluer"}:
        raise ValueError(f"Unknown context: {context}")
    direction = 1.0 if context == "redder" else -1.0
    comparison_sd = comparison_sd_from_stimulus_sigma(sigma_total, rho=rho)
    return ndtr(direction * delta_theta / np.maximum(comparison_sd, 1e-12))


def sample_bounded_gaussian_noise(
    target_theta: np.ndarray | Iterable[float] | float,
    sigma: float,
    rng: np.random.Generator | int | None = None,
    theta_min: float = -np.pi / 2,
    theta_max: float = 0,
) -> float:
    """Sample external Gaussian noise conditional on valid rendered theta.

    The returned noise value is drawn from ``N(0, sigma^2)`` truncated so that
    ``target_theta + noise`` lies inside ``[theta_min, theta_max]``. If
    ``target_theta`` contains multiple targets, the same noise draw is valid for
    every target; this supports shared-noise trials.
    """
    if theta_min > theta_max:
        raise ValueError("theta_min must be less than or equal to theta_max.")
    sigma = float(sigma)
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")

    targets = np.asarray(target_theta, dtype=float)
    lower = float(np.max(theta_min - targets))
    upper = float(np.min(theta_max - targets))
    if lower > upper:
        raise ValueError("No valid bounded-noise interval for the target theta values.")

    if sigma == 0:
        if lower <= 0.0 <= upper:
            return 0.0
        raise ValueError("Zero external noise cannot place target theta inside bounds.")
    if np.isclose(lower, upper):
        return lower

    rng = np.random.default_rng(rng)
    return float(
        truncnorm.rvs(lower / sigma, upper / sigma, scale=sigma, random_state=rng)
    )


def make_trial_table(
    theta0: float,
    delta_thetas: Iterable[float],
    sigma_ext_levels: Iterable[float],
    n_repeats: int,
    contexts: Iterable[str] = ("redder", "bluer"),
    rng: np.random.Generator | int | None = None,
    theta_min: float = -np.pi / 2,
    theta_max: float = 0,
    shared_noise: bool = False,
) -> pd.DataFrame:
    """Construct the fixed-level condition table for the 2AFC experiment.

    Each row corresponds to one trial. The left and right latent hue angles are
    defined symmetrically around ``theta0`` and then perturbed by either
    independent or shared bounded external Gaussian noise.

    Args:
        theta0: Midpoint latent angle around which both stimuli are built.
        delta_thetas: Signed target differences between right and left.
        sigma_ext_levels: External theta-noise standard deviations to include.
        n_repeats: Number of repeats per unique condition combination.
        contexts: Iterable of task prompts, typically ``("redder", "bluer")``.
        rng: Optional NumPy random generator or seed.
        theta_min: Lower bound for rendered latent angles.
        theta_max: Upper bound for rendered latent angles.
        shared_noise: If ``True``, use one common noise draw for both sides on a
            trial. If ``False``, draw left and right noise independently.

    Returns:
        A DataFrame with one row per trial, including target angles, noisy
        angles, task context, effective signed difference, and correctness of a
        rightward choice.
    """
    rng = np.random.default_rng(rng)
    rows = []
    for context in contexts:
        for sigma_ext in sigma_ext_levels:
            sigma_ext = float(sigma_ext)
            for delta_theta in delta_thetas:
                delta_theta = float(delta_theta)
                left_target = theta0 - delta_theta / 2.0
                right_target = theta0 + delta_theta / 2.0
                for repeat in range(int(n_repeats)):
                    if shared_noise:
                        shared = sample_bounded_gaussian_noise(
                            (left_target, right_target),
                            sigma_ext,
                            rng=rng,
                            theta_min=theta_min,
                            theta_max=theta_max,
                        )
                        eps_left = shared
                        eps_right = shared
                    else:
                        eps_left = sample_bounded_gaussian_noise(
                            left_target,
                            sigma_ext,
                            rng=rng,
                            theta_min=theta_min,
                            theta_max=theta_max,
                        )
                        eps_right = sample_bounded_gaussian_noise(
                            right_target,
                            sigma_ext,
                            rng=rng,
                            theta_min=theta_min,
                            theta_max=theta_max,
                        )
                    theta_left = left_target + eps_left
                    theta_right = right_target + eps_right
                    rows.append(
                        {
                            "context": context,
                            "sigma_ext": sigma_ext,
                            "delta_theta": delta_theta,
                            "theta0": theta0,
                            "theta_left_target": left_target,
                            "theta_right_target": right_target,
                            "eps_left": eps_left,
                            "eps_right": eps_right,
                            "theta_left": theta_left,
                            "theta_right": theta_right,
                            "repeat": repeat,
                            "shared_noise": shared_noise,
                            "left_clipped": False,
                            "right_clipped": False,
                        }
                    )
    trials = pd.DataFrame(rows)
    context_sign = trials["context"].map({"redder": 1.0, "bluer": -1.0}).astype(float)
    trials["effective_delta"] = context_sign * trials["delta_theta"]
    trials["correct_right"] = trials["effective_delta"] > 0
    return trials


def simulate_observer(
    trials: pd.DataFrame,
    sigma_int: float,
    rho_int: float = 0.0,
    rng: np.random.Generator | int | None = None,
) -> pd.DataFrame:
    """Simulate a latent-theta observer on a trial table.

    The observer is assumed to form noisy internal estimates of the displayed
    left and right latent angles, then compare those estimates according to the
    task prompt.

    Args:
        trials: Trial table produced by ``make_trial_table``.
        sigma_int: Standard deviation of internal observer noise on each side.
        rho_int: Correlation between left and right internal noise. The default
            ``0`` matches the independent-noise calibration model.
        rng: Optional NumPy random generator or seed.

    Returns:
        A copy of ``trials`` augmented with internal estimates, binary
        right-choice responses, and correctness labels.
    """
    if not -1.0 <= rho_int < 1.0:
        raise ValueError("rho_int must be in [-1, 1).")
    rng = np.random.default_rng(rng)
    trials = trials.copy()
    internal_noise = rng.multivariate_normal(
        mean=[0.0, 0.0],
        cov=[
            [sigma_int**2, rho_int * sigma_int**2],
            [rho_int * sigma_int**2, sigma_int**2],
        ],
        size=len(trials),
    )
    trials["eps_left_internal"] = internal_noise[:, 0]
    trials["eps_right_internal"] = internal_noise[:, 1]
    trials["rho_int"] = rho_int
    trials["theta_left_est"] = trials["theta_left"] + trials["eps_left_internal"]
    trials["theta_right_est"] = trials["theta_right"] + trials["eps_right_internal"]
    is_redder = trials["context"] == "redder"
    choose_right = np.where(
        is_redder,
        trials["theta_right_est"] > trials["theta_left_est"],
        trials["theta_right_est"] < trials["theta_left_est"],
    )
    trials["choose_right"] = choose_right.astype(int)
    trials["is_correct"] = (trials["choose_right"] == trials["correct_right"]).astype(
        int
    )
    return trials


def _fit_start(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    bias0 = 0.0
    scale0 = max(np.std(x), 1e-3)
    p = np.clip(y.mean(), 1e-3, 1 - 1e-3)
    if np.isfinite(logit(p)):
        bias0 = np.median(x)
    return np.array([bias0, scale0, 0.02, 0.02], dtype=float)


def fit_psychometric(trials: pd.DataFrame) -> dict:
    """Fit a psychometric curve to individual binary responses.

    Args:
        trials: Trial DataFrame containing ``effective_delta`` and
            ``choose_right`` columns.

    Returns:
        A dictionary containing the fitted parameters, optimization status, and
        the derived 75%-correct threshold.

        The returned dictionary summarizes one psychometric fit over many trials
        from a single condition. In the typical workflow, that condition is one
        pair ``(context, sigma_ext)`` and contains all stimulus-difference
        levels for that prompt and external-noise level.

        The ``success`` field is an optimizer-status flag, not a behavioral
        performance metric. ``success=True`` means the numerical optimization
        used to fit the sigmoid reported convergence. ``success=False`` means
        the fit may be unreliable even if a parameter vector was returned.
    """
    x = trials["effective_delta"].to_numpy(dtype=float)
    y = trials["choose_right"].to_numpy(dtype=float)

    def nll(params: np.ndarray) -> float:
        bias, scale, lapse_low, lapse_high = params
        p = psychometric(x, bias, scale, lapse_low, lapse_high)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).sum())

    start = _fit_start(x, y)
    bounds = [
        (x.min(), x.max()),
        (1e-4, max(np.ptp(x), 1e-3) * 3.0),
        (0.0, 0.15),
        (0.0, 0.15),
    ]
    result = minimize(nll, x0=start, bounds=bounds, method="L-BFGS-B")
    bias, scale, lapse_low, lapse_high = result.x
    fit = {
        "bias": float(bias),
        "scale": float(scale),
        "lapse_low": float(lapse_low),
        "lapse_high": float(lapse_high),
        "success": bool(result.success),
        "nll": float(result.fun),
        "n_trials": int(len(trials)),
    }
    fit["threshold_75"] = threshold_from_fit(fit)
    return fit


def fit_gaussian_psychometric(trials: pd.DataFrame, rho: float = 0.0) -> dict:
    """Fit the Gaussian 2AFC calibration model to binary responses.

    The fitted ``sigma_stimulus`` parameter is a per-stimulus total noise
    standard deviation. In a baseline condition with no external theta noise,
    this is interpreted as the subject's per-stimulus internal noise estimate
    under the independent-noise model.

    Args:
        trials: Trial DataFrame containing ``effective_delta`` and
            ``choose_right`` columns.
        rho: Assumed correlation between left and right per-stimulus noise
            terms. The default ``rho=0`` is the standard independent 2AFC
            model.

    Returns:
        A dictionary containing fitted parameters, optimization status, and
        the derived 75%-correct threshold.
    """
    x = trials["effective_delta"].to_numpy(dtype=float)
    y = trials["choose_right"].to_numpy(dtype=float)

    def nll(params: np.ndarray) -> float:
        bias, sigma_stimulus, lapse_low, lapse_high = params
        p = gaussian_psychometric(
            x,
            bias=bias,
            sigma_stimulus=sigma_stimulus,
            lapse_low=lapse_low,
            lapse_high=lapse_high,
            rho=rho,
        )
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).sum())

    start = np.array([np.median(x), max(np.std(x) / 2.0, 1e-3), 0.02, 0.02])
    bounds = [
        (x.min(), x.max()),
        (1e-5, max(np.ptp(x), 1e-3) * 3.0),
        (0.0, 0.15),
        (0.0, 0.15),
    ]
    result = minimize(nll, x0=start, bounds=bounds, method="L-BFGS-B")
    bias, sigma_stimulus, lapse_low, lapse_high = result.x
    fit = {
        "bias": float(bias),
        "sigma_stimulus": float(sigma_stimulus),
        "rho": float(rho),
        "lapse_low": float(lapse_low),
        "lapse_high": float(lapse_high),
        "success": bool(result.success),
        "nll": float(result.fun),
        "n_trials": int(len(trials)),
    }
    fit["threshold_75"] = threshold_from_gaussian_fit(fit)
    return fit


def fit_all_gaussian_psychometrics(
    trials: pd.DataFrame,
    rho: float = 0.0,
    group_cols: Iterable[str] = ("context", "sigma_ext"),
) -> pd.DataFrame:
    """Fit Gaussian 2AFC curves for every requested condition.

    Args:
        trials: Trial DataFrame containing simulated or empirical choices.
        rho: Assumed left/right noise correlation for the Gaussian model.
        group_cols: Columns that define independent psychometric fits.

    Returns:
        One row per condition with fitted Gaussian parameters and recovered
        per-stimulus total noise estimates.
    """
    group_cols = list(group_cols)
    rows = []
    for keys, group in trials.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        fit = fit_gaussian_psychometric(group, rho=rho)
        fit.update(dict(zip(group_cols, keys)))
        rows.append(fit)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def fit_all_psychometrics(trials: pd.DataFrame) -> pd.DataFrame:
    """Fit psychometric curves for every context and noise-level combination.

    Args:
        trials: Trial DataFrame containing all simulated or empirical choices.

    Returns:
        A DataFrame with one row per ``(context, sigma_ext)`` condition and
        columns for fitted parameters and recovered threshold.

        Each row is a condition-level summary, not a single trial. For example,
        if the experiment includes two task prompts (``redder`` and ``bluer``)
        and six external-noise levels, this function returns twelve rows: one
        fit for each prompt/noise combination. Each fit pools across all
        ``delta_theta`` levels within that condition.

        Important columns:
            - ``context``: Task prompt for the fit.
            - ``sigma_ext``: External theta-noise level for the fit.
            - ``bias``: Horizontal shift of the psychometric curve.
            - ``scale``: Slope-related width parameter of the psychometric
              curve.
            - ``threshold_75``: Effective-delta magnitude needed to reach 75%
              correct on the fitted curve.
            - ``lapse_low`` and ``lapse_high``: Lower and upper lapse-rate
              parameters.
            - ``success``: Whether the optimizer reported convergence for that
              row's fit.
    """
    rows = []
    for (context, sigma_ext), group in trials.groupby(
        ["context", "sigma_ext"], sort=True
    ):
        fit = fit_psychometric(group)
        fit["context"] = context
        fit["sigma_ext"] = float(sigma_ext)
        rows.append(fit)
    return (
        pd.DataFrame(rows).sort_values(["context", "sigma_ext"]).reset_index(drop=True)
    )


def fit_variance_model(summary: pd.DataFrame) -> dict:
    """Fit the legacy additive-variance threshold growth model.

    The fitted form is ``JND(sigma_ext)^2 = JND0^2 + a * sigma_ext^2``.
    This helper is retained for older notebooks. The preferred calibration
    workflow is now ``fit_gaussian_psychometric`` followed by
    ``compute_external_noise`` or ``make_noise_calibration_table``.

    Args:
        summary: DataFrame containing at least ``sigma_ext`` and
            ``threshold_75`` columns.

    Returns:
        A dictionary containing the fitted baseline threshold ``jnd0``, coupling
        parameter ``a``, covariance estimate, and the implied swamping point
        ``sigma_swamp``.
    """
    sigma = summary["sigma_ext"].to_numpy(dtype=float)
    jnd = summary["threshold_75"].to_numpy(dtype=float)
    y = jnd**2

    def model(x, jnd0, a):
        return jnd0**2 + a * x**2

    params, cov = curve_fit(
        model,
        sigma,
        y,
        p0=[max(jnd.min(), 1e-3), 1.0],
        bounds=([1e-6, 0.0], [np.inf, np.inf]),
    )
    jnd0, a = params
    sigma_swamp = np.nan if a <= 0 else float(jnd0 / np.sqrt(a))
    return {
        "jnd0": float(jnd0),
        "a": float(a),
        "cov": cov,
        "sigma_swamp": sigma_swamp,
    }


def compute_external_noise(
    sigma_internal: np.ndarray | float,
    sigma_target: np.ndarray | float,
    on_infeasible: str = "nan",
) -> np.ndarray:
    """Compute external noise needed to reach target total per-stimulus noise.

    Args:
        sigma_internal: Per-stimulus internal noise estimate.
        sigma_target: Desired per-stimulus total noise level.
        on_infeasible: Behavior when ``sigma_target < sigma_internal``. Use
            ``"nan"`` to return ``NaN`` or ``"raise"`` to raise a
            ``ValueError``.

    Returns:
        Per-stimulus external noise standard deviation.

    Raises:
        ValueError: If a target is infeasible and ``on_infeasible="raise"``.
    """
    if on_infeasible not in {"nan", "raise"}:
        raise ValueError("on_infeasible must be 'nan' or 'raise'.")
    sigma_internal = np.asarray(sigma_internal, dtype=float)
    sigma_target = np.asarray(sigma_target, dtype=float)
    variance = sigma_target**2 - sigma_internal**2
    infeasible = variance < -1e-12
    if np.any(infeasible) and on_infeasible == "raise":
        raise ValueError("sigma_target must be >= sigma_internal.")
    variance = np.where(variance < 0, np.nan, variance)
    return np.sqrt(variance)


def make_noise_calibration_table(
    subjects: pd.DataFrame,
    sigma_targets: Iterable[float],
    subject_col: str = "subject_id",
    sigma_internal_col: str = "sigma_internal",
    on_infeasible: str = "nan",
) -> pd.DataFrame:
    """Build subject-specific external-noise settings.

    Args:
        subjects: DataFrame containing one row per subject.
        sigma_targets: Experimenter-chosen per-stimulus total noise levels.
        subject_col: Column containing subject identifiers.
        sigma_internal_col: Column containing per-stimulus internal noise
            estimates.
        on_infeasible: Passed to ``compute_external_noise``.

    Returns:
        A table with one row per subject and target total-noise level.
    """
    rows = []
    for _, subject in subjects.iterrows():
        sigma_internal = float(subject[sigma_internal_col])
        for sigma_target in sigma_targets:
            sigma_target = float(sigma_target)
            sigma_ext = compute_external_noise(
                sigma_internal,
                sigma_target,
                on_infeasible=on_infeasible,
            )
            sigma_ext = float(np.asarray(sigma_ext))
            rows.append(
                {
                    subject_col: subject[subject_col],
                    "sigma_internal": sigma_internal,
                    "sigma_target": sigma_target,
                    "sigma_ext": sigma_ext,
                    "is_feasible": bool(np.isfinite(sigma_ext)),
                }
            )
    return pd.DataFrame(rows)


def matched_snr(
    delta_theta: np.ndarray | float,
    sigma_target: np.ndarray | float,
) -> np.ndarray:
    """Compute comparison-level SNR for matched per-stimulus total noise.

    Args:
        delta_theta: Hue separation in theta units.
        sigma_target: Per-stimulus total noise standard deviation.

    Returns:
        Comparison-level signal-to-noise ratio.
    """
    delta_theta = np.asarray(delta_theta, dtype=float)
    sigma_target = np.asarray(sigma_target, dtype=float)
    return np.abs(delta_theta) / comparison_sd_from_stimulus_sigma(sigma_target)


def simulate_matched_noise_observers(
    subjects: pd.DataFrame,
    sigma_targets: Iterable[float],
    theta0: float,
    delta_thetas: Iterable[float],
    n_repeats: int,
    contexts: Iterable[str] = ("redder", "bluer"),
    rho_int: float = 0.0,
    rng: np.random.Generator | int | None = None,
    theta_min: float = -np.pi / 2,
    theta_max: float = 0,
) -> pd.DataFrame:
    """Simulate observers after subject-specific external-noise matching.

    Args:
        subjects: DataFrame with ``subject_id`` and ``sigma_internal`` columns.
        sigma_targets: Desired per-stimulus total noise levels.
        theta0: Midpoint latent angle around which stimuli are built.
        delta_thetas: Signed target differences between right and left.
        n_repeats: Repeats per unique condition.
        contexts: Task prompts to simulate.
        rho_int: Correlation between left and right internal noise.
        rng: Optional NumPy random generator or seed.
        theta_min: Lower clipping bound for rendered latent angles.
        theta_max: Upper clipping bound for rendered latent angles.

    Returns:
        Trial-level simulated responses. Each row includes subject-specific
        internal noise, target total noise, and computed external noise.
    """
    rng = np.random.default_rng(rng)
    calibration = make_noise_calibration_table(subjects, sigma_targets)
    frames = []
    for _, condition in calibration[calibration["is_feasible"]].iterrows():
        trial_seed = rng.integers(0, np.iinfo(np.uint32).max)
        observer_seed = rng.integers(0, np.iinfo(np.uint32).max)
        trials = make_trial_table(
            theta0=theta0,
            delta_thetas=delta_thetas,
            sigma_ext_levels=(condition["sigma_ext"],),
            n_repeats=n_repeats,
            contexts=contexts,
            rng=int(trial_seed),
            theta_min=theta_min,
            theta_max=theta_max,
            shared_noise=False,
        )
        observed = simulate_observer(
            trials,
            sigma_int=float(condition["sigma_internal"]),
            rho_int=rho_int,
            rng=int(observer_seed),
        )
        observed["subject_id"] = condition["subject_id"]
        observed["sigma_internal"] = float(condition["sigma_internal"])
        observed["sigma_target"] = float(condition["sigma_target"])
        observed["sigma_ext"] = float(condition["sigma_ext"])
        frames.append(observed)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize_choices(
    trials: pd.DataFrame,
    group_cols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Aggregate empirical choice statistics by experimental condition.

    Args:
        trials: Trial DataFrame containing response and correctness columns.
        group_cols: Optional grouping columns. Defaults to
            ``("context", "sigma_ext", "effective_delta")``.

    Returns:
        A condition-level summary DataFrame with choice rates, accuracy, and
        trial counts.
    """
    if group_cols is None:
        group_cols = ("context", "sigma_ext", "effective_delta")
    group_cols = list(group_cols)
    grouped = (
        trials.groupby(group_cols, as_index=False)
        .agg(
            p_choose_right=("choose_right", "mean"),
            p_correct=("is_correct", "mean"),
            n_trials=("choose_right", "size"),
        )
        .sort_values(group_cols)
    )
    return grouped


def render_example_pair(
    theta0: float,
    delta_theta: float,
    sigma_ext: float,
    rng: np.random.Generator | int | None = None,
    theta_min: float = -np.pi / 2,
    theta_max: float = 0,
    **stimulus_kwargs,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Render one example left-right stimulus pair for visualization.

    Args:
        theta0: Midpoint latent angle for the pair.
        delta_theta: Intended difference between right and left target angles.
        sigma_ext: Standard deviation of the external theta noise.
        rng: Optional NumPy random generator or seed.
        theta_min: Lower clipping bound for rendered latent angles.
        theta_max: Upper clipping bound for rendered latent angles.
        **stimulus_kwargs: Additional keyword arguments forwarded to
            ``get_theta_array``.

    Returns:
        A tuple ``(left, right, meta)`` containing the rendered left and right
        images plus a metadata dictionary with target angles, realized noisy
        angles, and noise draws.
    """
    rng = np.random.default_rng(rng)
    eps_left = rng.normal(scale=sigma_ext)
    eps_right = rng.normal(scale=sigma_ext)
    theta_left = np.clip(theta0 - delta_theta / 2.0 + eps_left, theta_min, theta_max)
    theta_right = np.clip(theta0 + delta_theta / 2.0 + eps_right, theta_min, theta_max)
    left = get_theta_array(
        theta_left,
        theta_min=theta_min,
        theta_max=theta_max,
        rng=rng,
        **stimulus_kwargs,
    )
    right = get_theta_array(
        theta_right,
        theta_min=theta_min,
        theta_max=theta_max,
        rng=rng,
        **stimulus_kwargs,
    )
    meta = {
        "theta_left": theta_left,
        "theta_right": theta_right,
        "theta_left_target": theta0 - delta_theta / 2.0,
        "theta_right_target": theta0 + delta_theta / 2.0,
        "eps_left": eps_left,
        "eps_right": eps_right,
    }
    return left, right, meta


def plot_example_pairs(
    theta0: float,
    delta_thetas: Iterable[float],
    sigma_ext_levels: Iterable[float],
    rng: np.random.Generator | int | None = None,
    **stimulus_kwargs,
):
    """Plot a grid of rendered example stimulus pairs.

    Args:
        theta0: Midpoint latent angle for all example pairs.
        delta_thetas: Iterable of intended left-right hue differences.
        sigma_ext_levels: Iterable of external noise levels to visualize.
        rng: Optional NumPy random generator or seed.
        **stimulus_kwargs: Additional keyword arguments forwarded to
            ``render_example_pair``.

    Returns:
        The Matplotlib ``(fig, axes)`` tuple for the example-pair grid.
    """
    rng = np.random.default_rng(rng)
    delta_thetas = list(delta_thetas)
    sigma_ext_levels = list(sigma_ext_levels)
    fig, axes = plt.subplots(
        len(sigma_ext_levels),
        len(delta_thetas),
        figsize=(3.5 * len(delta_thetas), 2.6 * len(sigma_ext_levels)),
        squeeze=False,
    )
    cmap = get_cmap()
    for i, sigma_ext in enumerate(sigma_ext_levels):
        for j, delta_theta in enumerate(delta_thetas):
            left, right, meta = render_example_pair(
                theta0,
                delta_theta,
                sigma_ext,
                rng=rng,
                **stimulus_kwargs,
            )
            ax = axes[i, j]
            ax.imshow(np.concatenate([left, right], axis=1), cmap=cmap, vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(
                rf"$\sigma_{{ext}}={sigma_ext:.2f}$, "
                rf"$\Delta\theta={delta_theta:.2f}$"
                "\n"
                rf"$\theta_L={meta['theta_left']:.2f}$, "
                rf"$\theta_R={meta['theta_right']:.2f}$"
            )
    fig.suptitle("Example side-by-side cue pairs", y=1.02)
    fig.tight_layout()
    return fig, axes


def plot_psychometrics(trials: pd.DataFrame, fits: pd.DataFrame):
    """Plot empirical psychometric points and fitted curves.

    Args:
        trials: Trial-level DataFrame of responses.
        fits: Condition-level DataFrame returned by ``fit_all_psychometrics``.

    Returns:
        The Matplotlib ``(fig, axes)`` tuple for the psychometric plots.
    """
    summary = summarize_choices(trials)
    contexts = list(summary["context"].unique())
    fig, axes = plt.subplots(
        1,
        len(contexts),
        figsize=(7 * len(contexts), 4.5),
        squeeze=False,
    )
    for ax, context in zip(axes[0], contexts):
        sub = summary[summary["context"] == context]
        subfits = fits[fits["context"] == context]
        for sigma_ext, group in sub.groupby("sigma_ext"):
            ax.scatter(
                group["effective_delta"],
                group["p_choose_right"],
                label=rf"$\sigma_{{ext}}={sigma_ext:.2f}$",
                alpha=0.75,
            )
            fit = subfits[subfits["sigma_ext"] == sigma_ext].iloc[0].to_dict()
            x_grid = np.linspace(
                group["effective_delta"].min(), group["effective_delta"].max(), 300
            )
            ax.plot(
                x_grid,
                psychometric(
                    x_grid,
                    fit["bias"],
                    fit["scale"],
                    fit["lapse_low"],
                    fit["lapse_high"],
                ),
                linewidth=2,
            )
        ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"{context.title()} context")
        ax.set_xlabel("Effective delta theta")
        ax.set_ylabel("P(choose right)")
        ax.legend()
    fig.tight_layout()
    return fig, axes


def plot_matched_noise_psychometrics(
    trials: pd.DataFrame,
    fits: pd.DataFrame | None = None,
    context: str | None = None,
):
    """Plot subject psychometric curves after total-noise matching.

    Args:
        trials: Trial-level DataFrame returned by
            ``simulate_matched_noise_observers`` or matching empirical data.
        fits: Optional Gaussian fits grouped by
            ``("subject_id", "context", "sigma_target")``. If omitted, they
            are computed automatically.
        context: Optional prompt context to display. Defaults to the first
            context present in ``trials``.

    Returns:
        The Matplotlib ``(fig, axes)`` tuple. Each panel is one target total
        noise level, with one curve per subject.
    """
    if context is None:
        context = str(trials["context"].iloc[0])
    subtrials = trials[trials["context"] == context]
    if fits is None:
        fits = fit_all_gaussian_psychometrics(
            subtrials,
            group_cols=("subject_id", "context", "sigma_target"),
        )
    else:
        fits = fits[fits["context"] == context]

    summary = summarize_choices(
        subtrials,
        group_cols=("subject_id", "context", "sigma_target", "effective_delta"),
    )
    targets = sorted(summary["sigma_target"].unique())
    fig, axes = plt.subplots(
        1,
        len(targets),
        figsize=(6 * len(targets), 4.5),
        squeeze=False,
    )
    for ax, sigma_target in zip(axes[0], targets):
        target_summary = summary[summary["sigma_target"] == sigma_target]
        for subject_id, group in target_summary.groupby("subject_id"):
            ax.scatter(
                group["effective_delta"],
                group["p_choose_right"],
                alpha=0.7,
                label=str(subject_id),
            )
            fit_rows = fits[
                (fits["subject_id"] == subject_id)
                & (fits["sigma_target"] == sigma_target)
            ]
            if not fit_rows.empty:
                fit = fit_rows.iloc[0].to_dict()
                x_grid = np.linspace(
                    group["effective_delta"].min(),
                    group["effective_delta"].max(),
                    300,
                )
                ax.plot(
                    x_grid,
                    gaussian_psychometric(
                        x_grid,
                        bias=fit["bias"],
                        sigma_stimulus=fit["sigma_stimulus"],
                        lapse_low=fit["lapse_low"],
                        lapse_high=fit["lapse_high"],
                        rho=fit.get("rho", 0.0),
                    ),
                    linewidth=2,
                )
        ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
        ax.set_title(rf"{context.title()}, $\sigma_{{target}}={sigma_target:.3f}$")
        ax.set_xlabel(r"Effective $\Delta\theta$")
        ax.set_ylabel("P(choose right)")
        ax.legend(title="Subject")
    fig.tight_layout()
    return fig, axes


def plot_thresholds(fits: pd.DataFrame):
    """Plot threshold growth with external noise and fit summaries.

    Args:
        fits: Condition-level DataFrame returned by ``fit_all_psychometrics``.

    Returns:
        A tuple ``(fig, axes, variance_fits)`` containing the threshold figure,
        axes array, and a dictionary of additive-variance fit results keyed by
        task context.
    """
    contexts = list(fits["context"].unique())
    fig, axes = plt.subplots(
        1,
        len(contexts),
        figsize=(6 * len(contexts), 4.5),
        squeeze=False,
    )
    variance_fits = {}
    for ax, context in zip(axes[0], contexts):
        sub = fits[fits["context"] == context].sort_values("sigma_ext")
        vf = fit_variance_model(sub)
        variance_fits[context] = vf
        ax.plot(
            sub["sigma_ext"], sub["threshold_75"], "o-", label="Recovered threshold"
        )
        x_grid = np.linspace(sub["sigma_ext"].min(), sub["sigma_ext"].max(), 200)
        y_grid = np.sqrt(vf["jnd0"] ** 2 + vf["a"] * x_grid**2)
        ax.plot(x_grid, y_grid, "--", label="Additive-variance fit")
        if np.isfinite(vf["sigma_swamp"]):
            ax.axvline(
                vf["sigma_swamp"], color="black", linestyle=":", label="Swamping point"
            )
        ax.set_title(
            f"{context.title()} context\n"
            rf"$JND_0={vf['jnd0']:.3f}$, $a={vf['a']:.3f}$"
        )
        ax.set_xlabel(r"$\sigma_{ext}$")
        ax.set_ylabel("Threshold (75% correct)")
        ax.legend()
    fig.tight_layout()
    return fig, axes, variance_fits


def plot_context_comparison(fits: pd.DataFrame):
    """Plot recovered thresholds for redder and bluer contexts together.

    Args:
        fits: Condition-level DataFrame returned by ``fit_all_psychometrics``.

    Returns:
        The Matplotlib ``(fig, ax)`` tuple for the context-comparison plot.
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for context, group in fits.groupby("context"):
        ax.plot(group["sigma_ext"], group["threshold_75"], "o-", label=context.title())
    ax.set_xlabel(r"$\sigma_{ext}$")
    ax.set_ylabel("Threshold (75% correct)")
    ax.set_title("Redder vs bluer threshold comparison")
    ax.legend()
    fig.tight_layout()
    return fig, ax
