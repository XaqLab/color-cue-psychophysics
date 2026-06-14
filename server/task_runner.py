"""Server-side task state for web-based psychophysics sessions.

Each task object is fully pickle-serialisable so it can be persisted between
HTTP requests.  The two concrete classes mirror the desktop tasks in
``color_cue.interactive`` but replace the Matplotlib event-loop with a
request/response interface:

* :class:`WebInteractiveTask` – fixed-level 2AFC (Experiment 2)
* :class:`WebStaircaseTask` – adaptive staircase (Experiment 1)
"""

from __future__ import annotations

import pickle
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from color_cue.interactive import (
    InteractiveColorCueTask,
    InteractiveTaskConfig,
    StaircaseColorCueTask,
    StaircaseTaskConfig,
    save_staircase_session,
    save_task_session,
)
from color_cue.psychophysics import make_trial_table, sample_bounded_gaussian_noise

from render import render_trial_image


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class _WebTaskBase:
    """Shared interface for web-hosted tasks."""

    task_type: str = ""

    def __init__(
        self,
        subject_id: str,
        session_id: str,
        notes: str = "",
    ) -> None:
        self.subject_id = subject_id
        self.session_id = session_id
        self.notes = notes
        self._responses: list[dict] = []

    def is_done(self) -> bool:
        raise NotImplementedError

    def get_current_trial_data(self) -> dict:
        """Return a dict with ``image_b64``, ``context``, and metadata."""
        raise NotImplementedError

    def record_response(self, choice: str) -> dict:
        """Record ``'left'`` or ``'right'`` and advance state.

        Returns a dict with at least ``is_correct`` and ``done`` keys.
        """
        raise NotImplementedError

    def save_results(self, root: Path) -> Path:
        """Persist completed (or partial) results to *root*."""
        raise NotImplementedError

    @property
    def results(self) -> pd.DataFrame:
        return pd.DataFrame(self._responses)


# ---------------------------------------------------------------------------
# Fixed-level task (Experiment 2)
# ---------------------------------------------------------------------------


class WebInteractiveTask(_WebTaskBase):
    """Web-hosted fixed-level color-cue 2AFC task.

    The full trial table is generated at construction time (deterministic given
    the config seeds).  Trials are served one at a time; stimulus images are
    rendered lazily and cached for the current trial so that page refreshes do
    not advance the random generator.
    """

    task_type = "calibrated"

    def __init__(
        self,
        config: InteractiveTaskConfig,
        subject_id: str,
        session_id: str,
        notes: str = "",
    ) -> None:
        super().__init__(subject_id, session_id, notes)
        self.config = config

        self.trials: pd.DataFrame = (
            make_trial_table(
                theta0=config.theta0,
                delta_thetas=config.delta_thetas,
                sigma_ext_levels=config.sigma_ext_levels,
                n_repeats=config.n_repeats,
                contexts=config.contexts,
                rng=config.trial_seed,
                theta_min=config.theta_min,
                theta_max=config.theta_max,
                shared_noise=False,
            )
            .sample(frac=1.0, random_state=config.trial_seed)
            .reset_index(drop=True)
        )

        self._index: int = 0
        self._stimulus_rng: np.random.Generator = np.random.default_rng(
            config.stimulus_seed
        )
        self._cached_image: tuple[int, str] | None = None  # (trial_index, b64)

        # Optional calibration annotations — set by app.py after construction.
        # Maps sigma_ext (float) -> sigma_target (float).
        self.sigma_target_map: dict[float, float] = {}
        self.sigma_internal_val: float | None = None

    # ------------------------------------------------------------------

    def is_done(self) -> bool:
        return self._index >= len(self.trials)

    def get_current_trial_data(self) -> dict:
        if self.is_done():
            return {
                "done": True,
                "total_trials": len(self.trials),
                "trials_remaining": 0,
            }

        # Return cached image for current trial to avoid advancing the RNG on
        # repeated requests (e.g. page refresh).
        if self._cached_image is not None and self._cached_image[0] == self._index:
            image_b64 = self._cached_image[1]
        else:
            trial = self.trials.iloc[self._index]
            image_b64 = render_trial_image(
                float(trial["theta_left"]),
                float(trial["theta_right"]),
                self.config.stimulus_kwargs,
                self._stimulus_rng,
                border_pixels=self.config.border_pixels,
                gap_pixels=self.config.gap_pixels,
            )
            self._cached_image = (self._index, image_b64)

        trial = self.trials.iloc[self._index]
        return {
            "done": False,
            "trial_index": self._index,
            "total_trials": len(self.trials),
            "trials_remaining": len(self.trials) - self._index,
            "context": str(trial["context"]),
            "image_b64": image_b64,
        }

    def record_response(self, choice: str) -> dict:
        if self.is_done():
            return {"error": "task already done"}

        trial = self.trials.iloc[self._index]
        choose_right = int(choice == "right")
        is_correct = int(choose_right == int(trial["correct_right"]))

        response = trial.to_dict()
        response.update(
            {
                "response_key": choice,
                "choose_right": choose_right,
                "is_correct": is_correct,
                "trial_index": self._index,
            }
        )
        # Annotate with calibration metadata if available.
        if self.sigma_target_map:
            sigma_ext = float(trial.get("sigma_ext", float("nan")))
            response["sigma_target"] = self.sigma_target_map.get(sigma_ext, float("nan"))
        if self.sigma_internal_val is not None:
            response["sigma_internal"] = self.sigma_internal_val
        self._responses.append(response)
        self._index += 1
        self._cached_image = None

        return {
            "is_correct": is_correct,
            "done": self.is_done(),
            "trials_remaining": len(self.trials) - self._index,
            "total_trials": len(self.trials),
        }

    def save_results(self, root: Path) -> Path:
        task = InteractiveColorCueTask(self.config)
        task._responses = list(self._responses)
        return save_task_session(
            task,
            root,
            self.subject_id,
            self.session_id,
            notes=self.notes,
            overwrite=True,
        )


# ---------------------------------------------------------------------------
# Staircase task (Experiment 1)
# ---------------------------------------------------------------------------


class WebStaircaseTask(_WebTaskBase):
    """Web-hosted adaptive staircase color-cue task.

    One 2-down-1-up staircase is maintained per ``sigma_ext`` level.  Trials
    are selected uniformly at random from active staircases.  The staircase
    state is fully embedded in the object and therefore survives pickle
    round-trips between requests.
    """

    task_type = "staircase"

    def __init__(
        self,
        config: StaircaseTaskConfig,
        subject_id: str,
        session_id: str,
        notes: str = "",
    ) -> None:
        super().__init__(subject_id, session_id, notes)
        self.config = config

        self._trial_rng = np.random.default_rng(config.trial_seed)
        self._stimulus_rng = np.random.default_rng(config.stimulus_seed)
        self._global_trial_index: int = 0
        self._cached_image: tuple[int, str] | None = None
        self._current_trial: dict | None = None
        self._current_staircase_id: str | None = None

        self.staircases: dict[str, dict] = {
            f"sigma_{i}": {
                "staircase_id": f"sigma_{i}",
                "sigma_ext": float(sigma_ext),
                "current_index": int(config.start_index),
                "correct_streak": 0,
                "incorrect_streak": 0,
                "reversal_count": 0,
                "last_direction": None,
                "n_trials": 0,
                "active": True,
                "reversal_abs_deltas": [],
            }
            for i, sigma_ext in enumerate(config.sigma_ext_levels)
        }

        self._advance_trial()

    # ------------------------------------------------------------------
    # Private helpers (replicate StaircaseColorCueTask logic)

    def _active_staircases(self) -> list[dict]:
        return [s for s in self.staircases.values() if s["active"]]

    def _advance_trial(self) -> None:
        active = self._active_staircases()
        if not active:
            self._current_trial = None
            self._current_staircase_id = None
            return
        staircase = self._trial_rng.choice(active)
        self._current_staircase_id = staircase["staircase_id"]
        self._current_trial = self._make_trial(staircase)

    def _make_trial(self, staircase: dict) -> dict:
        abs_delta = float(self.config.delta_grid[staircase["current_index"]])
        sign = self._trial_rng.choice([-1.0, 1.0])
        delta_theta = float(sign * abs_delta)
        theta_l_tgt = self.config.theta0 - delta_theta / 2.0
        theta_r_tgt = self.config.theta0 + delta_theta / 2.0

        sigma_ext = float(staircase["sigma_ext"])
        if self.config.shared_noise:
            eps_l = eps_r = sample_bounded_gaussian_noise(
                (theta_l_tgt, theta_r_tgt),
                sigma_ext,
                rng=self._trial_rng,
                theta_min=self.config.theta_min,
                theta_max=self.config.theta_max,
            )
        else:
            eps_l = sample_bounded_gaussian_noise(
                theta_l_tgt,
                sigma_ext,
                rng=self._trial_rng,
                theta_min=self.config.theta_min,
                theta_max=self.config.theta_max,
            )
            eps_r = sample_bounded_gaussian_noise(
                theta_r_tgt,
                sigma_ext,
                rng=self._trial_rng,
                theta_min=self.config.theta_min,
                theta_max=self.config.theta_max,
            )

        theta_l = float(theta_l_tgt + eps_l)
        theta_r = float(theta_r_tgt + eps_r)
        effective_delta = delta_theta if self.config.context == "redder" else -delta_theta
        return {
            "context": self.config.context,
            "shared_noise": self.config.shared_noise,
            "sigma_ext": sigma_ext,
            "theta0": self.config.theta0,
            "delta_theta": delta_theta,
            "abs_delta_theta": abs_delta,
            "theta_left_target": theta_l_tgt,
            "theta_right_target": theta_r_tgt,
            "eps_left": eps_l,
            "eps_right": eps_r,
            "theta_left": theta_l,
            "theta_right": theta_r,
            "effective_delta": effective_delta,
            "correct_right": bool(effective_delta > 0),
            "staircase_id": staircase["staircase_id"],
            "step_index_before": int(staircase["current_index"]),
            "reversal_count_before": int(staircase["reversal_count"]),
            "global_trial_index": int(self._global_trial_index),
        }

    def _update_staircase(self, staircase: dict, is_correct: bool) -> tuple[int, bool]:
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

    # ------------------------------------------------------------------

    def is_done(self) -> bool:
        return self._current_trial is None

    def get_current_trial_data(self) -> dict:
        if self.is_done():
            return {"done": True}

        cache_key = self._global_trial_index
        if self._cached_image is not None and self._cached_image[0] == cache_key:
            image_b64 = self._cached_image[1]
        else:
            trial = self._current_trial
            image_b64 = render_trial_image(
                float(trial["theta_left"]),
                float(trial["theta_right"]),
                self.config.stimulus_kwargs,
                self._stimulus_rng,
                border_pixels=self.config.border_pixels,
                gap_pixels=self.config.gap_pixels,
            )
            self._cached_image = (cache_key, image_b64)

        return {
            "done": False,
            "trial_index": self._global_trial_index,
            "context": str(self._current_trial["context"]),
            "sigma_ext": float(self._current_trial["sigma_ext"]),
            "active_staircases": len(self._active_staircases()),
            "image_b64": image_b64,
        }

    def record_response(self, choice: str) -> dict:
        if self.is_done():
            return {"error": "task already done"}

        trial = self._current_trial
        staircase = self.staircases[self._current_staircase_id]

        choose_right = int(choice == "right")
        is_correct = bool(choose_right == int(trial["correct_right"]))
        direction, reversal_flag = self._update_staircase(staircase, is_correct)
        staircase["n_trials"] += 1

        if (
            staircase["n_trials"] >= self.config.max_trials_per_staircase
            or staircase["reversal_count"] >= self.config.max_reversals_per_staircase
        ):
            staircase["active"] = False

        response = dict(trial)
        response.update(
            {
                "response_key": choice,
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
        self._global_trial_index += 1
        self._cached_image = None
        self._advance_trial()

        return {
            "is_correct": int(is_correct),
            "done": self.is_done(),
            "active_staircases": len(self._active_staircases()),
        }

    def staircase_summary(self) -> pd.DataFrame:
        rows = []
        for sc in self.staircases.values():
            tail = sc["reversal_abs_deltas"][-self.config.threshold_reversal_count :]
            rows.append(
                {
                    "staircase_id": sc["staircase_id"],
                    "sigma_ext": sc["sigma_ext"],
                    "context": self.config.context,
                    "n_trials": sc["n_trials"],
                    "reversal_count": sc["reversal_count"],
                    "active": sc["active"],
                    "threshold_estimate": float(np.mean(tail)) if tail else float("nan"),
                }
            )
        return pd.DataFrame(rows)

    def save_results(self, root: Path) -> Path:
        task = StaircaseColorCueTask(self.config)
        task._responses = list(self._responses)
        # Overwrite the freshly-initialised staircases with our real state.
        task.staircases = {k: dict(v) for k, v in self.staircases.items()}
        return save_staircase_session(
            task,
            root,
            self.subject_id,
            self.session_id,
            notes=self.notes,
            overwrite=True,
        )


# ---------------------------------------------------------------------------
# Active-session file store
# ---------------------------------------------------------------------------


class ActiveSessionStore:
    """Pickle-backed store for in-progress task sessions.

    Each session is stored as ``<root>/<token>.pkl``.  Tokens are random UUIDs
    generated at session creation.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, task: _WebTaskBase, token: str | None = None) -> str:
        """Persist *task* to disk and return its token."""
        if token is None:
            token = str(uuid.uuid4())
        with open(self.root / f"{token}.pkl", "wb") as f:
            pickle.dump(task, f)
        return token

    def load(self, token: str) -> _WebTaskBase | None:
        """Load and return a task by token, or ``None`` if not found."""
        path = self.root / f"{token}.pkl"
        if not path.exists():
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    def delete(self, token: str) -> None:
        """Remove the active-session file for *token*."""
        path = self.root / f"{token}.pkl"
        if path.exists():
            path.unlink()

    def list_for_subject(self, subject_id: str) -> list[dict]:
        """Return summary dicts for all active sessions belonging to *subject_id*."""
        sessions = []
        for pkl_file in sorted(self.root.glob("*.pkl")):
            try:
                with open(pkl_file, "rb") as f:
                    task = pickle.load(f)
                if task.subject_id == subject_id:
                    sessions.append(
                        {
                            "token": pkl_file.stem,
                            "session_id": task.session_id,
                            "task_type": task.task_type,
                            "trials_completed": len(task._responses),
                            "is_done": task.is_done(),
                        }
                    )
            except Exception:
                pass
        return sessions
