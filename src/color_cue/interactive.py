from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .psychophysics import (
    compute_external_noise,
    fit_all_psychometrics,
    make_trial_table,
)
from .stimulus import get_cmap, get_theta_array


@dataclass
class InteractiveTaskConfig:
    """Configuration for the human-playable color-cue psychophysics task.

    Args:
        theta0: Midpoint latent angle around which left and right stimuli are
            constructed.
        delta_thetas: Signed target differences used across trials.
        sigma_ext_levels: External theta-noise standard deviations used across
            trials.
        n_repeats: Number of repeats for each unique
            ``(context, sigma_ext, delta_theta)`` condition.
        contexts: Task prompts to include.
        theta_min: Lower clipping bound for latent angles.
        theta_max: Upper clipping bound for latent angles.
        size: Stimulus image shape as ``(height, width)``.
        gap_pixels: Width of the blank separator inserted between the two
            stimuli on screen.
        border_pixels: Width of the white border around each stimulus panel.
        kappa: Deterministic signal magnitude in the complex plane.
        alpha: Temporal-correlation parameter forwarded to the stimulus
            generator.
        beta: Spatial power-spectrum exponent forwarded to the stimulus
            generator.
        trial_seed: Random seed used to generate the trial table.
        stimulus_seed: Random seed used to render the pixel textures.
        iti_seconds: Inter-trial pause in seconds after feedback is shown.
        show_feedback: Whether to display correctness feedback after each
            response.
        figure_size: Matplotlib figure size in inches.
    """

    theta0: float = -np.pi / 4
    delta_thetas: tuple[float, ...] = (
        -0.21,
        -0.14,
        -0.09,
        -0.05,
        0.05,
        0.09,
        0.14,
        0.21,
    )
    sigma_ext_levels: tuple[float, ...] = (0.0, 0.03, 0.06, 0.09, 0.12, 0.15)
    n_repeats: int = 10
    contexts: tuple[str, ...] = ("redder", "bluer")
    theta_min: float = -np.pi / 2
    theta_max: float = 0.0
    size: tuple[int, int] = (112, 112)
    gap_pixels: int = 28
    border_pixels: int = 14
    kappa: float = 0.1
    alpha: float = 1.0
    beta: float = -2.0
    trial_seed: int | None = 123
    stimulus_seed: int | None = 456
    iti_seconds: float = 0.3
    show_feedback: bool = True
    figure_size: tuple[float, float] = (9.5, 5.5)

    def to_dict(self) -> dict:
        """Serialize the task configuration into JSON-friendly Python types.

        Returns:
            A dictionary representation of the configuration, with tuples
            converted to lists so the result can be written to JSON directly.
        """
        return {
            "theta0": self.theta0,
            "delta_thetas": list(self.delta_thetas),
            "sigma_ext_levels": list(self.sigma_ext_levels),
            "n_repeats": self.n_repeats,
            "contexts": list(self.contexts),
            "theta_min": self.theta_min,
            "theta_max": self.theta_max,
            "size": list(self.size),
            "gap_pixels": self.gap_pixels,
            "border_pixels": self.border_pixels,
            "kappa": self.kappa,
            "alpha": self.alpha,
            "beta": self.beta,
            "trial_seed": self.trial_seed,
            "stimulus_seed": self.stimulus_seed,
            "iti_seconds": self.iti_seconds,
            "show_feedback": self.show_feedback,
            "figure_size": list(self.figure_size),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "InteractiveTaskConfig":
        """Build a configuration object from serialized metadata.

        Args:
            data: Dictionary produced by ``to_dict`` or an equivalent JSON
                object.

        Returns:
            A reconstructed ``InteractiveTaskConfig`` instance.
        """
        return cls(
            theta0=float(data["theta0"]),
            delta_thetas=tuple(float(x) for x in data["delta_thetas"]),
            sigma_ext_levels=tuple(float(x) for x in data["sigma_ext_levels"]),
            n_repeats=int(data["n_repeats"]),
            contexts=tuple(str(x) for x in data["contexts"]),
            theta_min=float(data["theta_min"]),
            theta_max=float(data["theta_max"]),
            size=tuple(int(x) for x in data["size"]),
            gap_pixels=int(data["gap_pixels"]),
            border_pixels=int(data["border_pixels"]),
            kappa=float(data["kappa"]),
            alpha=float(data["alpha"]),
            beta=float(data["beta"]),
            trial_seed=data["trial_seed"],
            stimulus_seed=data["stimulus_seed"],
            iti_seconds=float(data["iti_seconds"]),
            show_feedback=bool(data["show_feedback"]),
            figure_size=tuple(float(x) for x in data["figure_size"]),
        )

    @property
    def stimulus_kwargs(self) -> dict:
        """Return keyword arguments for ``get_theta_array``."""
        return {
            "size": self.size,
            "kappa": self.kappa,
            "alpha": self.alpha,
            "beta": self.beta,
            "theta_min": self.theta_min,
            "theta_max": self.theta_max,
        }


@dataclass
class StaircaseTaskConfig:
    """Configuration for an interleaved adaptive staircase session.

    Args:
        theta0: Midpoint latent angle around which both stimuli are
            constructed.
        sigma_ext_levels: External theta-noise levels; one staircase is created
            for each level.
        delta_grid: Monotone grid of positive ``|delta_theta|`` values ordered
            from hardest to easiest. Staircases move by index on this grid.
        start_index: Starting index into ``delta_grid`` for every staircase.
        context: Single prompt used throughout the session, either ``"redder"``
            or ``"bluer"``.
        shared_noise: Whether to use shared external noise instead of
            independent left/right external noise.
        n_down: Number of consecutive correct responses required before making
            the task harder.
        n_up: Number of incorrect responses required before making the task
            easier.
        max_trials_per_staircase: Maximum number of trials per staircase.
        max_reversals_per_staircase: Maximum number of reversals per staircase.
        threshold_reversal_count: Number of final reversals to average when
            estimating staircase-native thresholds.
        theta_min: Lower clipping bound for latent angles.
        theta_max: Upper clipping bound for latent angles.
        size: Stimulus image shape as ``(height, width)``.
        gap_pixels: Width of the blank separator inserted between the two
            stimuli on screen.
        border_pixels: Width of the white border around each stimulus panel.
        kappa: Deterministic signal magnitude in the complex plane.
        alpha: Temporal-correlation parameter forwarded to the stimulus
            generator.
        beta: Spatial power-spectrum exponent forwarded to the stimulus
            generator.
        trial_seed: Random seed used to interleave staircases and sample trial
            signs.
        stimulus_seed: Random seed used to render the pixel textures.
        iti_seconds: Inter-trial pause in seconds after feedback is shown.
        show_feedback: Whether to display correctness feedback after each
            response.
        figure_size: Matplotlib figure size in inches.
    """

    theta0: float = -np.pi / 4
    sigma_ext_levels: tuple[float, ...] = (0.0, 0.03, 0.06, 0.09, 0.12, 0.15)
    delta_grid: tuple[float, ...] = (0.03, 0.05, 0.07, 0.10, 0.14, 0.20, 0.28)
    start_index: int = 4
    context: str = "redder"
    shared_noise: bool = False
    n_down: int = 2
    n_up: int = 1
    max_trials_per_staircase: int = 40
    max_reversals_per_staircase: int = 8
    threshold_reversal_count: int = 6
    theta_min: float = -np.pi / 2
    theta_max: float = 0.0
    size: tuple[int, int] = (112, 112)
    gap_pixels: int = 28
    border_pixels: int = 14
    kappa: float = 0.1
    alpha: float = 1.0
    beta: float = -2.0
    trial_seed: int | None = 123
    stimulus_seed: int | None = 456
    iti_seconds: float = 0.3
    show_feedback: bool = True
    figure_size: tuple[float, float] = (9.5, 5.5)

    def to_dict(self) -> dict:
        """Serialize the staircase configuration into JSON-friendly types."""
        return {
            "theta0": self.theta0,
            "sigma_ext_levels": list(self.sigma_ext_levels),
            "delta_grid": list(self.delta_grid),
            "start_index": self.start_index,
            "context": self.context,
            "shared_noise": self.shared_noise,
            "n_down": self.n_down,
            "n_up": self.n_up,
            "max_trials_per_staircase": self.max_trials_per_staircase,
            "max_reversals_per_staircase": self.max_reversals_per_staircase,
            "threshold_reversal_count": self.threshold_reversal_count,
            "theta_min": self.theta_min,
            "theta_max": self.theta_max,
            "size": list(self.size),
            "gap_pixels": self.gap_pixels,
            "border_pixels": self.border_pixels,
            "kappa": self.kappa,
            "alpha": self.alpha,
            "beta": self.beta,
            "trial_seed": self.trial_seed,
            "stimulus_seed": self.stimulus_seed,
            "iti_seconds": self.iti_seconds,
            "show_feedback": self.show_feedback,
            "figure_size": list(self.figure_size),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StaircaseTaskConfig":
        """Reconstruct a staircase configuration from serialized metadata."""
        return cls(
            theta0=float(data["theta0"]),
            sigma_ext_levels=tuple(float(x) for x in data["sigma_ext_levels"]),
            delta_grid=tuple(float(x) for x in data["delta_grid"]),
            start_index=int(data["start_index"]),
            context=str(data["context"]),
            shared_noise=bool(data["shared_noise"]),
            n_down=int(data["n_down"]),
            n_up=int(data["n_up"]),
            max_trials_per_staircase=int(data["max_trials_per_staircase"]),
            max_reversals_per_staircase=int(data["max_reversals_per_staircase"]),
            threshold_reversal_count=int(data["threshold_reversal_count"]),
            theta_min=float(data["theta_min"]),
            theta_max=float(data["theta_max"]),
            size=tuple(int(x) for x in data["size"]),
            gap_pixels=int(data["gap_pixels"]),
            border_pixels=int(data["border_pixels"]),
            kappa=float(data["kappa"]),
            alpha=float(data["alpha"]),
            beta=float(data["beta"]),
            trial_seed=data["trial_seed"],
            stimulus_seed=data["stimulus_seed"],
            iti_seconds=float(data["iti_seconds"]),
            show_feedback=bool(data["show_feedback"]),
            figure_size=tuple(float(x) for x in data["figure_size"]),
        )

    @property
    def stimulus_kwargs(self) -> dict:
        """Return keyword arguments for ``get_theta_array``."""
        return {
            "size": self.size,
            "kappa": self.kappa,
            "alpha": self.alpha,
            "beta": self.beta,
            "theta_min": self.theta_min,
            "theta_max": self.theta_max,
        }


class InteractiveColorCueTask:
    """Run a human-playable side-by-side color-cue discrimination task.

    The task presents one left/right pair at a time in a Matplotlib window.
    Subjects respond with the left or right arrow keys according to the prompt
    shown above the stimuli.

    Example:
        >>> cfg = InteractiveTaskConfig(n_repeats=2)
        >>> task = InteractiveColorCueTask(cfg)
        >>> results = task.run()
        >>> results.head()
    """

    def __init__(self, config: InteractiveTaskConfig | None = None):
        """Initialize the task runner and pre-generate all trials.

        Args:
            config: Optional task configuration. If omitted, defaults from
                ``InteractiveTaskConfig`` are used.
        """
        self.config = config or InteractiveTaskConfig()
        self.trials = (
            make_trial_table(
                theta0=self.config.theta0,
                delta_thetas=self.config.delta_thetas,
                sigma_ext_levels=self.config.sigma_ext_levels,
                n_repeats=self.config.n_repeats,
                contexts=self.config.contexts,
                rng=self.config.trial_seed,
                theta_min=self.config.theta_min,
                theta_max=self.config.theta_max,
                shared_noise=False,
            )
            .sample(frac=1.0, random_state=self.config.trial_seed)
            .reset_index(drop=True)
        )
        self._stimulus_rng = np.random.default_rng(self.config.stimulus_seed)
        self._index = 0
        self._responses: list[dict] = []
        self._response_ready = False
        self._current_key: str | None = None

        self.fig: plt.Figure | None = None
        self.ax: plt.Axes | None = None
        self.image_artist = None
        self.header_text = None
        self.footer_text = None

    def _render_trial_image(self, trial: pd.Series) -> np.ndarray:
        """Render the left-right image shown for a single trial.

        Args:
            trial: One row from the trial table.

        Returns:
            An RGB array containing the left and right cue images embedded in a
            white background with a white separator and border padding.
        """
        left = get_theta_array(
            float(trial["theta_left"]),
            rng=self._stimulus_rng,
            **self.config.stimulus_kwargs,
        )
        right = get_theta_array(
            float(trial["theta_right"]),
            rng=self._stimulus_rng,
            **self.config.stimulus_kwargs,
        )
        cmap = get_cmap()
        left_rgb = cmap(left)[..., :3]
        right_rgb = cmap(right)[..., :3]

        border = self.config.border_pixels
        gap = self.config.gap_pixels
        panel_height, panel_width = left_rgb.shape[:2]
        canvas_height = panel_height + 2 * border
        canvas_width = 2 * panel_width + 2 * border + gap + 2 * border
        canvas = np.ones((canvas_height, canvas_width, 3), dtype=float)

        left_x0 = border
        left_x1 = left_x0 + panel_width
        right_x0 = left_x1 + border + gap + border
        right_x1 = right_x0 + panel_width
        y0 = border
        y1 = y0 + panel_height

        canvas[y0:y1, left_x0:left_x1, :] = left_rgb
        canvas[y0:y1, right_x0:right_x1, :] = right_rgb
        return canvas

    def _draw_trial(self, trial: pd.Series) -> None:
        """Draw the current trial and response instructions.

        Args:
            trial: One row from the trial table.
        """
        assert self.ax is not None
        image = self._render_trial_image(trial)
        if self.image_artist is None:
            self.image_artist = self.ax.imshow(
                image,
                interpolation="nearest",
            )
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.ax.set_facecolor("white")
            for spine in self.ax.spines.values():
                spine.set_visible(False)
        else:
            self.image_artist.set_data(image)

        total = len(self.trials)
        prompt = f"Which side is {trial['context']}?"
        header = f"Trial {self._index + 1}/{total}    {prompt}"
        footer = "Respond with LEFT or RIGHT arrow. Press q to quit."
        if self.header_text is None:
            self.header_text = self.ax.text(
                0.5,
                1.05,
                header,
                transform=self.ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
            )
        else:
            self.header_text.set_text(header)

        if self.footer_text is None:
            self.footer_text = self.ax.text(
                0.5,
                -0.06,
                footer,
                transform=self.ax.transAxes,
                ha="center",
                va="top",
                fontsize=11,
            )
        else:
            self.footer_text.set_text(footer)
        self.fig.canvas.draw_idle()

    def _on_key_press(self, event) -> None:
        """Record a response key for the current trial.

        Args:
            event: Matplotlib key-press event.
        """
        if event.key in {"left", "right", "q"}:
            self._current_key = event.key
            self._response_ready = True

    def _collect_single_response(self, trial: pd.Series) -> bool:
        """Wait for one response and store it.

        Args:
            trial: One row from the trial table.

        Returns:
            ``False`` if the task was quit early, otherwise ``True``.
        """
        self._response_ready = False
        self._current_key = None
        self._draw_trial(trial)
        while not self._response_ready:
            plt.pause(0.05)
            if not plt.fignum_exists(self.fig.number):
                return False

        if self._current_key == "q":
            return False

        choose_right = int(self._current_key == "right")
        is_correct = int(choose_right == int(trial["correct_right"]))
        response = trial.to_dict()
        response.update(
            {
                "response_key": self._current_key,
                "choose_right": choose_right,
                "is_correct": is_correct,
                "trial_index": self._index,
            }
        )
        self._responses.append(response)

        if self.config.show_feedback:
            assert self.footer_text is not None
            feedback = "Correct" if is_correct else "Incorrect"
            self.footer_text.set_text(
                f"{feedback}. Press LEFT or RIGHT on the next trial. Press q to quit."
            )
            self.fig.canvas.draw_idle()
            plt.pause(self.config.iti_seconds)
        return True

    def run(self) -> pd.DataFrame:
        """Run the interactive task until all trials are completed or quit.

        Returns:
            A DataFrame containing one row per completed trial with stimulus
            parameters, subject response, and correctness.
        """
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=self.config.figure_size)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        for idx in range(self._index, len(self.trials)):
            self._index = idx
            trial = self.trials.iloc[idx]
            keep_going = self._collect_single_response(trial)
            if not keep_going:
                break

        results = self.results
        if self.footer_text is not None and plt.fignum_exists(self.fig.number):
            self.footer_text.set_text("Task complete. You may close this window.")
            self.fig.canvas.draw_idle()
        return results

    @property
    def results(self) -> pd.DataFrame:
        """Return the completed-trial response table."""
        return pd.DataFrame(self._responses)

    def fit_completed_psychometrics(self) -> pd.DataFrame:
        """Fit psychometric curves to completed responses.

        Returns:
            A DataFrame of condition-level psychometric fits for the completed
            trials.

        Raises:
            ValueError: If no responses have been recorded yet.
        """
        if not self._responses:
            raise ValueError("No responses recorded yet.")
        return fit_all_psychometrics(self.results)

    def save_results(self, path: str | Path) -> Path:
        """Save completed trial data to CSV.

        Args:
            path: Output CSV path.

        Returns:
            The resolved output path.
        """
        path = Path(path).expanduser().resolve()
        self.results.to_csv(path, index=False)
        return path


class StaircaseColorCueTask:
    """Run an interleaved multi-staircase version of the color-cue task.

    One staircase is maintained per ``sigma_ext`` level. All staircases share a
    fixed prompt context and a fixed external-noise mode within a session.
    """

    def __init__(self, config: StaircaseTaskConfig | None = None):
        """Initialize the staircase task and its interleaved staircases.

        Args:
            config: Optional staircase task configuration.
        """
        self.config = config or StaircaseTaskConfig()
        self._trial_rng = np.random.default_rng(self.config.trial_seed)
        self._stimulus_rng = np.random.default_rng(self.config.stimulus_seed)
        self._responses: list[dict] = []
        self._response_ready = False
        self._current_key: str | None = None
        self._global_trial_index = 0

        self.fig: plt.Figure | None = None
        self.ax: plt.Axes | None = None
        self.image_artist = None
        self.header_text = None
        self.footer_text = None

        self._validate_config()
        self.staircases = self._init_staircases()

    def _validate_config(self) -> None:
        """Validate basic staircase configuration assumptions."""
        if self.config.context not in {"redder", "bluer"}:
            raise ValueError(f"Unsupported context: {self.config.context}")
        if len(self.config.delta_grid) == 0:
            raise ValueError("delta_grid must contain at least one positive value.")
        if any(x <= 0 for x in self.config.delta_grid):
            raise ValueError("delta_grid must contain only positive values.")
        if tuple(sorted(self.config.delta_grid)) != tuple(self.config.delta_grid):
            raise ValueError("delta_grid must be sorted from hardest to easiest.")
        if not (0 <= self.config.start_index < len(self.config.delta_grid)):
            raise ValueError("start_index must be a valid index into delta_grid.")

    def _init_staircases(self) -> dict[str, dict]:
        """Create one staircase state per sigma level."""
        staircases = {}
        for i, sigma_ext in enumerate(self.config.sigma_ext_levels):
            staircase_id = f"sigma_{i}"
            staircases[staircase_id] = {
                "staircase_id": staircase_id,
                "sigma_ext": float(sigma_ext),
                "current_index": int(self.config.start_index),
                "correct_streak": 0,
                "incorrect_streak": 0,
                "reversal_count": 0,
                "last_direction": None,
                "n_trials": 0,
                "active": True,
                "reversal_abs_deltas": [],
            }
        return staircases

    def _active_staircases(self) -> list[dict]:
        """Return the list of staircase states that are still active."""
        return [s for s in self.staircases.values() if s["active"]]

    def _render_trial_image(self, trial: dict) -> np.ndarray:
        """Render the left-right image shown for one staircase trial."""
        left = get_theta_array(
            float(trial["theta_left"]),
            rng=self._stimulus_rng,
            **self.config.stimulus_kwargs,
        )
        right = get_theta_array(
            float(trial["theta_right"]),
            rng=self._stimulus_rng,
            **self.config.stimulus_kwargs,
        )
        cmap = get_cmap()
        left_rgb = cmap(left)[..., :3]
        right_rgb = cmap(right)[..., :3]

        border = self.config.border_pixels
        gap = self.config.gap_pixels
        panel_height, panel_width = left_rgb.shape[:2]
        canvas_height = panel_height + 2 * border
        canvas_width = 2 * panel_width + 2 * border + gap + 2 * border
        canvas = np.ones((canvas_height, canvas_width, 3), dtype=float)

        left_x0 = border
        left_x1 = left_x0 + panel_width
        right_x0 = left_x1 + border + gap + border
        right_x1 = right_x0 + panel_width
        y0 = border
        y1 = y0 + panel_height

        canvas[y0:y1, left_x0:left_x1, :] = left_rgb
        canvas[y0:y1, right_x0:right_x1, :] = right_rgb
        return canvas

    def _make_trial(self, staircase: dict) -> dict:
        """Construct one staircase trial from the current staircase state."""
        abs_delta_theta = float(self.config.delta_grid[staircase["current_index"]])
        sign = self._trial_rng.choice([-1.0, 1.0])
        delta_theta = float(sign * abs_delta_theta)
        theta_left_target = self.config.theta0 - delta_theta / 2.0
        theta_right_target = self.config.theta0 + delta_theta / 2.0

        sigma_ext = float(staircase["sigma_ext"])
        if self.config.shared_noise:
            shared = self._trial_rng.normal(scale=sigma_ext)
            eps_left = shared
            eps_right = shared
        else:
            eps_left = self._trial_rng.normal(scale=sigma_ext)
            eps_right = self._trial_rng.normal(scale=sigma_ext)

        theta_left = float(
            np.clip(
                theta_left_target + eps_left,
                self.config.theta_min,
                self.config.theta_max,
            )
        )
        theta_right = float(
            np.clip(
                theta_right_target + eps_right,
                self.config.theta_min,
                self.config.theta_max,
            )
        )
        effective_delta = (
            delta_theta if self.config.context == "redder" else -delta_theta
        )
        correct_right = bool(effective_delta > 0)
        return {
            "context": self.config.context,
            "shared_noise": self.config.shared_noise,
            "sigma_ext": sigma_ext,
            "theta0": self.config.theta0,
            "delta_theta": delta_theta,
            "abs_delta_theta": abs_delta_theta,
            "theta_left_target": theta_left_target,
            "theta_right_target": theta_right_target,
            "eps_left": eps_left,
            "eps_right": eps_right,
            "theta_left": theta_left,
            "theta_right": theta_right,
            "effective_delta": effective_delta,
            "correct_right": correct_right,
            "staircase_id": staircase["staircase_id"],
            "step_index_before": int(staircase["current_index"]),
            "reversal_count_before": int(staircase["reversal_count"]),
            "global_trial_index": int(self._global_trial_index),
        }

    def _draw_trial(self, trial: dict) -> None:
        """Draw the current staircase trial and response instructions."""
        assert self.ax is not None
        image = self._render_trial_image(trial)
        if self.image_artist is None:
            self.image_artist = self.ax.imshow(image, interpolation="nearest")
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.ax.set_facecolor("white")
            for spine in self.ax.spines.values():
                spine.set_visible(False)
        else:
            self.image_artist.set_data(image)

        active = len(self._active_staircases())
        header = (
            f"Trial {self._global_trial_index + 1}    "
            f"Which side is {trial['context']}?    "
            f"sigma={trial['sigma_ext']:.2f}    active staircases={active}"
        )
        footer = "Respond with LEFT or RIGHT arrow. Press q to quit."
        if self.header_text is None:
            self.header_text = self.ax.text(
                0.5,
                1.05,
                header,
                transform=self.ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
            )
        else:
            self.header_text.set_text(header)

        if self.footer_text is None:
            self.footer_text = self.ax.text(
                0.5,
                -0.06,
                footer,
                transform=self.ax.transAxes,
                ha="center",
                va="top",
                fontsize=11,
            )
        else:
            self.footer_text.set_text(footer)
        self.fig.canvas.draw_idle()

    def _on_key_press(self, event) -> None:
        """Record a response key for the current staircase trial."""
        if event.key in {"left", "right", "q"}:
            self._current_key = event.key
            self._response_ready = True

    def _update_staircase(self, staircase: dict, is_correct: bool) -> tuple[int, bool]:
        """Update one staircase after a response.

        Args:
            staircase: Mutable staircase-state dictionary.
            is_correct: Whether the response was correct.

        Returns:
            A tuple ``(direction, reversal_flag)`` where direction is ``-1`` for
            harder, ``+1`` for easier, and ``0`` for no step.
        """
        direction = 0
        if is_correct:
            staircase["correct_streak"] += 1
            staircase["incorrect_streak"] = 0
            if staircase["correct_streak"] >= self.config.n_down:
                direction = -1
                staircase["correct_streak"] = 0
        else:
            staircase["incorrect_streak"] += 1
            staircase["correct_streak"] = 0
            if staircase["incorrect_streak"] >= self.config.n_up:
                direction = 1
                staircase["incorrect_streak"] = 0

        reversal_flag = False
        if direction != 0:
            if (
                staircase["last_direction"] is not None
                and staircase["last_direction"] != direction
            ):
                reversal_flag = True
                staircase["reversal_count"] += 1
                staircase["reversal_abs_deltas"].append(
                    float(self.config.delta_grid[staircase["current_index"]])
                )
            staircase["last_direction"] = direction
            staircase["current_index"] = int(
                np.clip(
                    staircase["current_index"] + direction,
                    0,
                    len(self.config.delta_grid) - 1,
                )
            )
        return direction, reversal_flag

    def _finalize_staircase_status(self, staircase: dict) -> None:
        """Deactivate a staircase when its stopping criteria are met."""
        if staircase["n_trials"] >= self.config.max_trials_per_staircase:
            staircase["active"] = False
        if staircase["reversal_count"] >= self.config.max_reversals_per_staircase:
            staircase["active"] = False

    def _collect_single_response(self, staircase: dict, trial: dict) -> bool:
        """Wait for one response, update the staircase, and log the trial."""
        self._response_ready = False
        self._current_key = None
        self._draw_trial(trial)
        while not self._response_ready:
            plt.pause(0.05)
            if not plt.fignum_exists(self.fig.number):
                return False

        if self._current_key == "q":
            return False

        choose_right = int(self._current_key == "right")
        is_correct = bool(choose_right == int(trial["correct_right"]))
        direction, reversal_flag = self._update_staircase(staircase, is_correct)
        staircase["n_trials"] += 1
        self._finalize_staircase_status(staircase)

        response = dict(trial)
        response.update(
            {
                "response_key": self._current_key,
                "choose_right": choose_right,
                "is_correct": int(is_correct),
                "step_direction": int(direction),
                "reversal": bool(reversal_flag),
                "step_index_after": int(staircase["current_index"]),
                "reversal_count_after": int(staircase["reversal_count"]),
                "staircase_trial_index": int(staircase["n_trials"]),
                "staircase_active_after": bool(staircase["active"]),
            }
        )
        self._responses.append(response)

        if self.config.show_feedback:
            assert self.footer_text is not None
            feedback = "Correct" if is_correct else "Incorrect"
            self.footer_text.set_text(
                f"{feedback}. Press LEFT or RIGHT on the next trial. Press q to quit."
            )
            self.fig.canvas.draw_idle()
            plt.pause(self.config.iti_seconds)
        return True

    def run(self) -> pd.DataFrame:
        """Run the interleaved staircase task until completion or quit.

        Returns:
            A DataFrame containing one row per completed staircase trial.
        """
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=self.config.figure_size)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)

        while True:
            active = self._active_staircases()
            if not active:
                break
            staircase = self._trial_rng.choice(active)
            trial = self._make_trial(staircase)
            keep_going = self._collect_single_response(staircase, trial)
            if not keep_going:
                break
            self._global_trial_index += 1

        results = self.results
        if self.footer_text is not None and plt.fignum_exists(self.fig.number):
            self.footer_text.set_text("Task complete. You may close this window.")
            self.fig.canvas.draw_idle()
        return results

    @property
    def results(self) -> pd.DataFrame:
        """Return the completed staircase-trial response table."""
        return pd.DataFrame(self._responses)

    def staircase_summary(self) -> pd.DataFrame:
        """Summarize per-staircase trial counts, reversals, and threshold estimate.

        Returns:
            A DataFrame with one row per staircase containing stopping status,
            reversal counts, and the mean of the last configured reversal
            magnitudes.
        """
        rows = []
        for staircase in self.staircases.values():
            reversals = list(staircase["reversal_abs_deltas"])
            tail = reversals[-self.config.threshold_reversal_count :]
            threshold_estimate = float(np.mean(tail)) if tail else np.nan
            rows.append(
                {
                    "staircase_id": staircase["staircase_id"],
                    "sigma_ext": staircase["sigma_ext"],
                    "context": self.config.context,
                    "shared_noise": self.config.shared_noise,
                    "n_trials": staircase["n_trials"],
                    "reversal_count": staircase["reversal_count"],
                    "active": staircase["active"],
                    "threshold_reversal_count_used": len(tail),
                    "threshold_estimate": threshold_estimate,
                }
            )
        return pd.DataFrame(rows).sort_values("sigma_ext").reset_index(drop=True)


def quickstart_task(
    n_repeats: int = 2,
    delta_thetas: Iterable[float] = (-0.14, -0.05, 0.05, 0.14),
    sigma_ext_levels: Iterable[float] = (0.0, 0.06, 0.12),
    trial_seed: int | None = 123,
    stimulus_seed: int | None = 456,
) -> InteractiveColorCueTask:
    """Create a short interactive task suitable for a first human demo.

    Args:
        n_repeats: Number of repeats per condition.
        delta_thetas: Signed hue differences to include.
        sigma_ext_levels: External-noise levels to include.
        trial_seed: Seed for trial randomization.
        stimulus_seed: Seed for pixel-texture generation.

    Returns:
        A configured ``InteractiveColorCueTask`` instance that can be run with
        ``task.run()``.
    """
    cfg = InteractiveTaskConfig(
        n_repeats=n_repeats,
        delta_thetas=tuple(float(x) for x in delta_thetas),
        sigma_ext_levels=tuple(float(x) for x in sigma_ext_levels),
        trial_seed=trial_seed,
        stimulus_seed=stimulus_seed,
    )
    return InteractiveColorCueTask(cfg)


def quickstart_calibrated_task(
    sigma_internal: float,
    sigma_targets: Iterable[float],
    n_repeats: int = 2,
    delta_thetas: Iterable[float] = (-0.14, -0.05, 0.05, 0.14),
    trial_seed: int | None = 123,
    stimulus_seed: int | None = 456,
) -> InteractiveColorCueTask:
    """Create an interactive task from per-stimulus target noise levels.

    Args:
        sigma_internal: Subject's estimated per-stimulus internal noise.
        sigma_targets: Desired per-stimulus total noise levels.
        n_repeats: Number of repeats per condition.
        delta_thetas: Signed hue differences to include.
        trial_seed: Seed for trial randomization.
        stimulus_seed: Seed for pixel-texture generation.

    Returns:
        A configured ``InteractiveColorCueTask`` whose ``sigma_ext_levels`` are
        computed from ``sigma_internal`` and ``sigma_targets``.
    """
    sigma_ext_levels = compute_external_noise(
        sigma_internal,
        np.asarray(tuple(float(x) for x in sigma_targets)),
        on_infeasible="raise",
    )
    cfg = InteractiveTaskConfig(
        n_repeats=n_repeats,
        delta_thetas=tuple(float(x) for x in delta_thetas),
        sigma_ext_levels=tuple(float(x) for x in sigma_ext_levels),
        trial_seed=trial_seed,
        stimulus_seed=stimulus_seed,
    )
    return InteractiveColorCueTask(cfg)


def quickstart_staircase_task(
    sigma_ext_levels: Iterable[float] = (0.0, 0.06, 0.12),
    delta_grid: Iterable[float] = (0.03, 0.05, 0.07, 0.10, 0.14, 0.20),
    context: str = "redder",
    shared_noise: bool = False,
    trial_seed: int | None = 123,
    stimulus_seed: int | None = 456,
) -> StaircaseColorCueTask:
    """Create a short staircase demo suitable for a first human pilot."""
    cfg = StaircaseTaskConfig(
        sigma_ext_levels=tuple(float(x) for x in sigma_ext_levels),
        delta_grid=tuple(float(x) for x in delta_grid),
        start_index=max(0, min(len(tuple(delta_grid)) - 1, 4)),
        context=context,
        shared_noise=shared_noise,
        trial_seed=trial_seed,
        stimulus_seed=stimulus_seed,
    )
    return StaircaseColorCueTask(cfg)


def _session_dir(root: str | Path, subject_id: str, session_id: str) -> Path:
    """Return the canonical directory for one saved session."""
    return Path(root).expanduser().resolve() / subject_id / session_id


def save_task_session(
    task: InteractiveColorCueTask,
    root: str | Path,
    subject_id: str,
    session_id: str,
    notes: str = "",
    overwrite: bool = False,
) -> Path:
    """Save one interactive-task run as a reusable session bundle.

    The bundle contains:
        - ``results.csv``: trial-level completed responses
        - ``metadata.json``: session identifiers, notes, and task configuration

    Args:
        task: Completed or partially completed interactive task.
        root: Root directory under which session folders are stored.
        subject_id: Subject identifier used as the first directory level.
        session_id: Session identifier used as the second directory level.
        notes: Optional free-text notes about the session.
        overwrite: Whether to overwrite an existing session directory.

    Returns:
        The resolved path of the saved session directory.

    Raises:
        ValueError: If the task has no recorded responses.
        FileExistsError: If the session already exists and ``overwrite`` is
            ``False``.
    """
    results = task.results
    if results.empty:
        raise ValueError("Cannot save a session with no recorded responses.")

    session_dir = _session_dir(root, subject_id, session_id)
    if session_dir.exists() and not overwrite:
        raise FileExistsError(f"Session already exists: {session_dir}")
    session_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "subject_id": subject_id,
        "session_id": session_id,
        "notes": notes,
        "n_completed_trials": int(len(results)),
        "config": task.config.to_dict(),
    }
    results.to_csv(session_dir / "results.csv", index=False)
    with open(session_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return session_dir


def load_session_metadata(root: str | Path, subject_id: str, session_id: str) -> dict:
    """Load the metadata dictionary for one saved session.

    Args:
        root: Root directory containing saved sessions.
        subject_id: Subject identifier.
        session_id: Session identifier.

    Returns:
        The parsed ``metadata.json`` object for the requested session.
    """
    session_dir = _session_dir(root, subject_id, session_id)
    with open(session_dir / "metadata.json", encoding="utf-8") as f:
        return json.load(f)


def load_session_results(
    root: str | Path, subject_id: str, session_id: str
) -> pd.DataFrame:
    """Load completed responses for one saved session.

    Args:
        root: Root directory containing saved sessions.
        subject_id: Subject identifier.
        session_id: Session identifier.

    Returns:
        A trial-level DataFrame annotated with ``subject_id`` and ``session_id``.
    """
    session_dir = _session_dir(root, subject_id, session_id)
    results = pd.read_csv(session_dir / "results.csv")
    results["subject_id"] = subject_id
    results["session_id"] = session_id
    return results


def list_saved_sessions(root: str | Path, subject_id: str) -> pd.DataFrame:
    """List all saved sessions for one subject.

    Args:
        root: Root directory containing saved sessions.
        subject_id: Subject identifier.

    Returns:
        A DataFrame with one row per saved session and summary metadata.
    """
    subject_dir = Path(root).expanduser().resolve() / subject_id
    if not subject_dir.exists():
        return pd.DataFrame(
            columns=["subject_id", "session_id", "n_completed_trials", "notes"]
        )

    rows = []
    for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
        meta_path = session_dir / "metadata.json"
        if not meta_path.exists():
            continue
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        rows.append(
            {
                "subject_id": meta["subject_id"],
                "session_id": meta["session_id"],
                "n_completed_trials": meta.get("n_completed_trials", np.nan),
                "notes": meta.get("notes", ""),
            }
        )
    return pd.DataFrame(rows)


def load_multiple_sessions(
    root: str | Path,
    subject_id: str,
    session_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Load and concatenate responses from multiple saved sessions.

    Args:
        root: Root directory containing saved sessions.
        subject_id: Subject identifier.
        session_ids: Optional iterable of session identifiers to load. If
            omitted, all saved sessions for the subject are loaded.

    Returns:
        A concatenated trial-level DataFrame across the requested sessions.

    Raises:
        ValueError: If no matching sessions are found.
    """
    if session_ids is None:
        listing = list_saved_sessions(root, subject_id)
        session_ids = listing["session_id"].tolist()
    session_ids = list(session_ids)
    if not session_ids:
        raise ValueError(f"No saved sessions found for subject {subject_id!r}.")
    frames = [
        load_session_results(root, subject_id, session_id) for session_id in session_ids
    ]
    return pd.concat(frames, ignore_index=True)


def _staircase_session_dir(root: str | Path, subject_id: str, session_id: str) -> Path:
    """Return the canonical directory for one saved staircase session."""
    return Path(root).expanduser().resolve() / subject_id / session_id


def save_staircase_session(
    task: StaircaseColorCueTask,
    root: str | Path,
    subject_id: str,
    session_id: str,
    notes: str = "",
    overwrite: bool = False,
) -> Path:
    """Save one staircase task run as a reusable session bundle.

    Args:
        task: Completed or partially completed staircase task.
        root: Root directory under which staircase sessions are stored.
        subject_id: Subject identifier used as the first directory level.
        session_id: Session identifier used as the second directory level.
        notes: Optional free-text notes about the session.
        overwrite: Whether to overwrite an existing session directory.

    Returns:
        The resolved path of the saved staircase session directory.
    """
    results = task.results
    if results.empty:
        raise ValueError("Cannot save a staircase session with no recorded responses.")

    session_dir = _staircase_session_dir(root, subject_id, session_id)
    if session_dir.exists() and not overwrite:
        raise FileExistsError(f"Staircase session already exists: {session_dir}")
    session_dir.mkdir(parents=True, exist_ok=True)

    summary = task.staircase_summary()
    metadata = {
        "subject_id": subject_id,
        "session_id": session_id,
        "notes": notes,
        "n_completed_trials": int(len(results)),
        "config": task.config.to_dict(),
        "staircase_summary": summary.to_dict(orient="records"),
    }
    results.to_csv(session_dir / "results.csv", index=False)
    summary.to_csv(session_dir / "staircase_summary.csv", index=False)
    with open(session_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return session_dir


def load_staircase_session_metadata(
    root: str | Path, subject_id: str, session_id: str
) -> dict:
    """Load the metadata dictionary for one saved staircase session."""
    session_dir = _staircase_session_dir(root, subject_id, session_id)
    with open(session_dir / "metadata.json", encoding="utf-8") as f:
        return json.load(f)


def load_staircase_session_results(
    root: str | Path, subject_id: str, session_id: str
) -> pd.DataFrame:
    """Load completed responses for one saved staircase session."""
    session_dir = _staircase_session_dir(root, subject_id, session_id)
    results = pd.read_csv(session_dir / "results.csv")
    results["subject_id"] = subject_id
    results["session_id"] = session_id
    return results


def list_saved_staircase_sessions(root: str | Path, subject_id: str) -> pd.DataFrame:
    """List all saved staircase sessions for one subject."""
    subject_dir = Path(root).expanduser().resolve() / subject_id
    if not subject_dir.exists():
        return pd.DataFrame(
            columns=["subject_id", "session_id", "n_completed_trials", "notes"]
        )

    rows = []
    for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
        meta_path = session_dir / "metadata.json"
        if not meta_path.exists():
            continue
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        rows.append(
            {
                "subject_id": meta["subject_id"],
                "session_id": meta["session_id"],
                "n_completed_trials": meta.get("n_completed_trials", np.nan),
                "notes": meta.get("notes", ""),
                "context": meta.get("config", {}).get("context", ""),
                "shared_noise": meta.get("config", {}).get("shared_noise", np.nan),
            }
        )
    return pd.DataFrame(rows)


def load_multiple_staircase_sessions(
    root: str | Path,
    subject_id: str,
    session_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Load and concatenate responses from multiple saved staircase sessions."""
    if session_ids is None:
        listing = list_saved_staircase_sessions(root, subject_id)
        session_ids = listing["session_id"].tolist()
    session_ids = list(session_ids)
    if not session_ids:
        raise ValueError(
            f"No saved staircase sessions found for subject {subject_id!r}."
        )
    frames = [
        load_staircase_session_results(root, subject_id, session_id)
        for session_id in session_ids
    ]
    return pd.concat(frames, ignore_index=True)


def summarize_loaded_staircase_sessions(
    trials: pd.DataFrame,
    threshold_reversal_count: int = 6,
) -> pd.DataFrame:
    """Summarize loaded staircase trials at the staircase level.

    Args:
        trials: Concatenated staircase-trial DataFrame from one or more saved
            sessions.
        threshold_reversal_count: Number of final reversal magnitudes to average
            when computing the staircase-native threshold estimate.

    Returns:
        A DataFrame with one row per loaded staircase.
    """
    rows = []
    group_cols = ["subject_id", "session_id", "staircase_id"]
    for keys, group in trials.groupby(group_cols, sort=True):
        reversal_abs = group.loc[
            group["reversal"].astype(bool), "abs_delta_theta"
        ].tolist()
        tail = reversal_abs[-threshold_reversal_count:]
        threshold_estimate = float(np.mean(tail)) if tail else np.nan
        rows.append(
            {
                "subject_id": keys[0],
                "session_id": keys[1],
                "staircase_id": keys[2],
                "sigma_ext": float(group["sigma_ext"].iloc[0]),
                "context": group["context"].iloc[0],
                "shared_noise": bool(group["shared_noise"].iloc[0]),
                "n_trials": int(len(group)),
                "reversal_count": int(group["reversal"].astype(bool).sum()),
                "threshold_reversal_count_used": len(tail),
                "threshold_estimate": threshold_estimate,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["subject_id", "session_id", "sigma_ext", "staircase_id"])
        .reset_index(drop=True)
    )
