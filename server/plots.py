"""Server-side matplotlib plot generation for the admin data dashboard.

All public functions return a base64-encoded PNG string suitable for embedding
in an HTML ``<img src="data:image/png;base64,<value>">`` tag, or ``None`` when
there is insufficient data to produce the plot.
"""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / "src"))

from color_cue.interactive import (
    load_multiple_sessions,
    load_multiple_staircase_sessions,
    summarize_loaded_staircase_sessions,
)
from color_cue.psychophysics import (
    fit_all_gaussian_psychometrics,
    gaussian_psychometric,
    stimulus_sigma_from_jnd,
)

# Colour cycle shared across plots
_PALETTE = [
    "#2980b9", "#e74c3c", "#27ae60", "#8e44ad", "#e67e22",
    "#16a085", "#2c3e50", "#c0392b", "#d35400", "#1abc9c",
]


def _fig_to_b64(fig: plt.Figure) -> str:
    """Encode a matplotlib figure as a base64 PNG string and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _color(i: int) -> str:
    return _PALETTE[i % len(_PALETTE)]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_all_baseline(baseline_root: Path) -> pd.DataFrame:
    frames = []
    if not baseline_root.exists():
        return pd.DataFrame()
    for subj_dir in sorted(p for p in baseline_root.iterdir() if p.is_dir()):
        try:
            df = load_multiple_staircase_sessions(baseline_root, subj_dir.name)
            frames.append(df)
        except Exception:
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_all_calibrated(calibrated_root: Path) -> pd.DataFrame:
    frames = []
    if not calibrated_root.exists():
        return pd.DataFrame()
    for subj_dir in sorted(p for p in calibrated_root.iterdir() if p.is_dir()):
        try:
            df = load_multiple_sessions(calibrated_root, subj_dir.name)
            frames.append(df)
        except Exception:
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def sigma_internal_table(baseline_root: Path) -> pd.DataFrame:
    """Return per-(subject, session, staircase) sigma_internal estimates."""
    data = _load_all_baseline(baseline_root)
    if data.empty:
        return pd.DataFrame()
    summary = summarize_loaded_staircase_sessions(data, threshold_reversal_count=6)
    summary["sigma_internal"] = stimulus_sigma_from_jnd(
        summary["threshold_estimate"], p_correct=0.707
    )
    return summary


# ---------------------------------------------------------------------------
# Plot 1: sigma_internal per participant (Experiment 1 summary)
# ---------------------------------------------------------------------------


def sigma_internal_summary_plot(baseline_root: Path) -> str | None:
    """Bar chart of mean σ_internal ± std per participant.

    Each bar shows the mean across all staircases for that participant.
    Individual staircase estimates are overlaid as dots so that session-to-
    session variability is visible.
    """
    tbl = sigma_internal_table(baseline_root)
    if tbl.empty or "sigma_internal" not in tbl.columns:
        return None
    valid = tbl.dropna(subset=["sigma_internal"])
    if valid.empty:
        return None

    per_subj = (
        valid.groupby("subject_id")["sigma_internal"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("subject_id")
    )

    fig, ax = plt.subplots(figsize=(max(4.5, len(per_subj) * 1.0 + 1.5), 4))
    x = np.arange(len(per_subj))

    ax.bar(
        x,
        per_subj["mean"],
        yerr=per_subj["std"].fillna(0),
        capsize=5,
        color=[_color(i) for i in range(len(per_subj))],
        alpha=0.75,
        error_kw={"linewidth": 1.5},
    )

    # Individual staircase dots
    for xi, (_, row) in enumerate(per_subj.iterrows()):
        subj_vals = valid.loc[valid["subject_id"] == row["subject_id"], "sigma_internal"]
        ax.scatter(
            np.full(len(subj_vals), xi),
            subj_vals,
            color=_color(xi),
            s=30,
            zorder=5,
            alpha=0.8,
            edgecolors="white",
            linewidths=0.5,
        )

    grand_mean = per_subj["mean"].mean()
    ax.axhline(
        grand_mean,
        color="#555",
        linestyle="--",
        linewidth=1.5,
        label=f"Grand mean = {grand_mean:.4f}",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(per_subj["subject_id"], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("σ_internal estimate")
    ax.set_title("Estimated Internal Noise per Participant (Experiment 1)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _fig_to_b64(fig)


# ---------------------------------------------------------------------------
# Plot 2: Staircase traces for one participant
# ---------------------------------------------------------------------------


def staircase_trace_plot(subject_id: str, baseline_root: Path) -> str | None:
    """Staircase Δθ trace for every session/staircase of one participant.

    Each subplot shows |Δθ| vs trial number for one staircase.  Reversal
    points are highlighted in red and the threshold estimate (mean of the
    last N reversals) is shown as a dashed green line.
    """
    try:
        data = load_multiple_staircase_sessions(baseline_root, subject_id)
    except Exception:
        return None
    if data.empty:
        return None

    sessions = sorted(data["session_id"].unique())
    staircases = sorted(data["staircase_id"].unique())
    n_rows, n_cols = len(sessions), max(len(staircases), 1)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 4.5, n_rows * 3.2),
        squeeze=False,
    )

    for si, sess in enumerate(sessions):
        sess_data = data[data["session_id"] == sess]
        for sc_i, sc_id in enumerate(staircases):
            ax = axes[si][sc_i]
            sc_data = sess_data[sess_data["staircase_id"] == sc_id].copy()
            if sc_data.empty:
                ax.set_visible(False)
                continue

            ax.plot(
                sc_data["global_trial_index"],
                sc_data["abs_delta_theta"],
                "-o",
                color="#2980b9",
                markersize=3,
                linewidth=1,
                alpha=0.7,
            )

            rev = sc_data[sc_data["reversal"] == True]
            if not rev.empty:
                ax.scatter(
                    rev["global_trial_index"],
                    rev["abs_delta_theta"],
                    color="#e74c3c",
                    s=45,
                    zorder=5,
                    label="reversal",
                )

            tail = rev["abs_delta_theta"].values[-6:]
            if len(tail) >= 1:
                threshold = tail.mean()
                ax.axhline(
                    threshold,
                    color="#27ae60",
                    linestyle="--",
                    linewidth=1.5,
                    label=f"threshold ≈ {threshold:.3f}",
                )

            sigma = float(sc_data["sigma_ext"].iloc[0])
            ax.set_title(f"{sess}  |  σ_ext = {sigma:.2f}", fontsize=8)
            ax.set_xlabel("Trial", fontsize=8)
            ax.set_ylabel("|Δθ|", fontsize=8)
            ax.legend(fontsize=7, loc="upper right")
            ax.tick_params(labelsize=7)

        # Hide unused columns in this row
        for sc_i in range(len(staircases), n_cols):
            axes[si][sc_i].set_visible(False)

    fig.suptitle(f"Staircase Traces — {subject_id}", fontsize=11)
    fig.tight_layout()
    return _fig_to_b64(fig)


# ---------------------------------------------------------------------------
# Plot 3: Per-participant psychometric curves (Experiment 2)
# ---------------------------------------------------------------------------


def psychometric_per_participant_plot(
    subject_id: str, calibrated_root: Path
) -> str | None:
    """Psychometric curves per σ_target for one participant.

    Each subplot corresponds to one target noise level.  Data points are
    per-delta empirical proportions; the smooth curve is the fitted Gaussian
    2AFC psychometric function.
    """
    try:
        data = load_multiple_sessions(calibrated_root, subject_id)
    except Exception:
        return None
    if data.empty or "sigma_target" not in data.columns:
        return None

    data = data.dropna(subset=["sigma_target"])
    if data.empty:
        return None

    try:
        fits = fit_all_gaussian_psychometrics(
            data, group_cols=("subject_id", "context", "sigma_target")
        )
    except Exception:
        return None

    contexts = sorted(data["context"].unique())
    sigma_targets = sorted(data["sigma_target"].unique())
    n_cols = len(sigma_targets)
    n_rows = len(contexts)
    if n_cols == 0:
        return None

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 3.8, n_rows * 3.5), squeeze=False
    )
    x_range = np.linspace(
        data["effective_delta"].min(), data["effective_delta"].max(), 200
    )

    for ri, ctx in enumerate(contexts):
        ctx_data = data[data["context"] == ctx]
        for ci, st in enumerate(sigma_targets):
            ax = axes[ri][ci]
            sub = ctx_data[np.isclose(ctx_data["sigma_target"], st)]
            if not sub.empty:
                emp = (
                    sub.groupby("effective_delta")["choose_right"].mean().reset_index()
                )
                ax.scatter(
                    emp["effective_delta"],
                    emp["choose_right"],
                    color="#2980b9",
                    s=30,
                    zorder=5,
                    label="data",
                )
            fit_row = fits[
                (fits["subject_id"] == subject_id)
                & (fits["context"] == ctx)
                & np.isclose(fits["sigma_target"], st)
                & fits["success"]
            ]
            if not fit_row.empty:
                bias = float(fit_row["bias"].iloc[0])
                sigma_stim = float(fit_row["sigma_stimulus"].iloc[0])
                y = gaussian_psychometric(x_range, bias, sigma_stim)
                ax.plot(
                    x_range,
                    y,
                    color="#e74c3c",
                    linewidth=2,
                    label=f"fit σ={sigma_stim:.3f}",
                )
            ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
            ax.axvline(0.0, color="gray", linestyle=":", linewidth=1)
            ax.set_ylim(0, 1)
            ax.set_title(f"σ_target={st:.2f}  ctx={ctx}", fontsize=8)
            ax.set_xlabel("Effective Δθ", fontsize=8)
            if ci == 0:
                ax.set_ylabel("P(choose right)", fontsize=8)
            ax.legend(fontsize=7)
            ax.tick_params(labelsize=7)

    fig.suptitle(f"Psychometric Curves — {subject_id}", fontsize=11)
    fig.tight_layout()
    return _fig_to_b64(fig)


# ---------------------------------------------------------------------------
# Plot 4: Calibration validation scatter (all participants)
# ---------------------------------------------------------------------------


def calibration_validation_plot(calibrated_root: Path) -> str | None:
    """σ_stimulus vs σ_target scatter, one series per participant.

    Each point represents one (participant, context, σ_target) fit.  Points
    that fall on the identity line (dashed) indicate perfect calibration:
    the fitted per-stimulus noise matches the experimenter's target noise.
    Deviations above the line mean the participant's effective noise was
    higher than intended (under-calibrated); deviations below mean it was
    lower (over-calibrated).
    """
    data = _load_all_calibrated(calibrated_root)
    if data.empty or "sigma_target" not in data.columns:
        return None
    data = data.dropna(subset=["sigma_target"])
    if data.empty:
        return None

    try:
        fits = fit_all_gaussian_psychometrics(
            data, group_cols=("subject_id", "context", "sigma_target")
        )
    except Exception:
        return None

    fits = fits[fits["success"]].dropna(subset=["sigma_stimulus", "sigma_target"])
    if fits.empty:
        return None

    subjects = sorted(fits["subject_id"].unique())
    fig, ax = plt.subplots(figsize=(5.5, 5))

    for i, subj in enumerate(subjects):
        sub = fits[fits["subject_id"] == subj].sort_values("sigma_target")
        ax.scatter(
            sub["sigma_target"],
            sub["sigma_stimulus"],
            color=_color(i),
            s=60,
            label=subj,
            zorder=5,
        )
        ax.plot(
            sub["sigma_target"],
            sub["sigma_stimulus"],
            color=_color(i),
            linewidth=1.2,
            alpha=0.5,
        )

    lo = min(fits["sigma_target"].min(), fits["sigma_stimulus"].min()) * 0.85
    hi = max(fits["sigma_target"].max(), fits["sigma_stimulus"].max()) * 1.15
    ax.plot(
        [lo, hi],
        [lo, hi],
        "k--",
        linewidth=1.5,
        alpha=0.55,
        label="identity (perfect calibration)",
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("σ_target  (experimenter-set total noise)", fontsize=10)
    ax.set_ylabel("σ_stimulus  (fitted from data)", fontsize=10)
    ax.set_title("Calibration Validation: Fitted vs Target Noise", fontsize=11)
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return _fig_to_b64(fig)


# ---------------------------------------------------------------------------
# Plot 5: Psychometric overlay by σ_target (key calibration check)
# ---------------------------------------------------------------------------


def psychometric_overlay_plot(calibrated_root: Path) -> str | None:
    """All participants' psychometric curves overlaid per σ_target / context.

    Produces a grid of subplots (rows = contexts, columns = σ_target levels).
    Within each panel every participant's fitted Gaussian psychometric curve
    is plotted in a distinct colour, with empirical proportions as dots.

    **Interpretation:** if the noise-calibration worked, curves within each
    panel should collapse onto a single function — participants with higher
    internal noise received larger σ_ext, but their total noise σ_target is
    the same, so the difficulty (slope of the curve) should be equated.
    Divergent curves flag participants whose σ_internal was mis-estimated or
    whose performance changed between the baseline and calibration sessions.
    """
    data = _load_all_calibrated(calibrated_root)
    if data.empty or "sigma_target" not in data.columns:
        return None
    data = data.dropna(subset=["sigma_target"])
    if data.empty:
        return None

    try:
        fits = fit_all_gaussian_psychometrics(
            data, group_cols=("subject_id", "context", "sigma_target")
        )
    except Exception:
        return None

    contexts = sorted(data["context"].unique())
    sigma_targets = sorted(data["sigma_target"].unique())
    subjects = sorted(data["subject_id"].unique())
    n_rows, n_cols = len(contexts), len(sigma_targets)
    if n_cols == 0:
        return None

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 3.8, n_rows * 3.5),
        squeeze=False,
        sharey=True,
    )
    x_range = np.linspace(
        data["effective_delta"].min(), data["effective_delta"].max(), 200
    )

    for ri, ctx in enumerate(contexts):
        for ci, st in enumerate(sigma_targets):
            ax = axes[ri][ci]
            for pi, subj in enumerate(subjects):
                sub = data[
                    (data["subject_id"] == subj)
                    & (data["context"] == ctx)
                    & np.isclose(data["sigma_target"], st)
                ]
                if not sub.empty:
                    emp = (
                        sub.groupby("effective_delta")["choose_right"]
                        .mean()
                        .reset_index()
                    )
                    ax.scatter(
                        emp["effective_delta"],
                        emp["choose_right"],
                        color=_color(pi),
                        s=18,
                        alpha=0.55,
                        zorder=5,
                    )
                fit_row = fits[
                    (fits["subject_id"] == subj)
                    & (fits["context"] == ctx)
                    & np.isclose(fits["sigma_target"], st)
                    & fits["success"]
                ]
                if not fit_row.empty:
                    bias = float(fit_row["bias"].iloc[0])
                    sigma_stim = float(fit_row["sigma_stimulus"].iloc[0])
                    y = gaussian_psychometric(x_range, bias, sigma_stim)
                    ax.plot(
                        x_range,
                        y,
                        color=_color(pi),
                        linewidth=2,
                        label=subj,
                        alpha=0.85,
                    )

            ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
            ax.axvline(0.0, color="gray", linestyle=":", linewidth=0.8)
            ax.set_ylim(0, 1)
            ax.set_title(f"σ_target={st:.2f}  ctx={ctx}", fontsize=9)
            ax.set_xlabel("Effective Δθ", fontsize=8)
            if ci == 0:
                ax.set_ylabel("P(choose right)", fontsize=8)
            ax.tick_params(labelsize=7)
            if ri == 0 and ci == n_cols - 1:
                ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")

    fig.suptitle(
        "Psychometric Overlay by σ_target — Calibration Verification\n"
        "Curves should coincide within each panel if calibration succeeded.",
        fontsize=10,
    )
    fig.tight_layout()
    return _fig_to_b64(fig)
