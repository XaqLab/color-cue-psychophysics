"""Flask web application for the color-cue calibration psychophysics task.

Usage
-----
From the ``color-cue-psychophysics`` root::

    pip install -e ".[server]"
    python server/app.py

The first run creates a default admin account (admin / admin123).  Change the
password immediately via the admin panel or by editing server_data/users.json.

Environment variables
---------------------
SECRET_KEY : str
    Flask secret key for signing session cookies.  **Must** be changed for any
    deployment beyond localhost.
DATA_ROOT : str
    Override the default server data directory
    (``<repo>/server/server_data``).
"""

from __future__ import annotations

import os
import shutil
import sys
from functools import wraps
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from flask import (
    Flask,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

# ---------------------------------------------------------------------------
# Path setup: ensure color_cue and server modules are importable.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / "src"))
sys.path.insert(0, str(_HERE))

from auth import UserStore  # noqa: E402
from task_runner import ActiveSessionStore, WebInteractiveTask, WebStaircaseTask  # noqa: E402

# ---------------------------------------------------------------------------
# App and data directories
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-me-before-deploying")

_DATA_ROOT = Path(os.environ.get("DATA_ROOT", str(_HERE / "server_data")))
BASELINE_ROOT = _DATA_ROOT / "session_data" / "baseline_staircase"
CALIBRATED_ROOT = _DATA_ROOT / "session_data" / "calibrated_task"
ACTIVE_ROOT = _DATA_ROOT / "active_sessions"

for _d in [BASELINE_ROOT, CALIBRATED_ROOT, ACTIVE_ROOT]:
    _d.mkdir(parents=True, exist_ok=True)

user_store = UserStore(_DATA_ROOT / "users.json")
active_sessions = ActiveSessionStore(ACTIVE_ROOT)


# ---------------------------------------------------------------------------
# Auth decorators
# ---------------------------------------------------------------------------


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "subject_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)

    return decorated


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "subject_id" not in session:
            return redirect(url_for("login"))
        user = user_store.get(session["subject_id"])
        if not user or not user.get("is_admin"):
            abort(403)
        return f(*args, **kwargs)

    return decorated


# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    if "subject_id" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        subject_id = request.form.get("subject_id", "").strip()
        password = request.form.get("password", "")
        if user_store.verify(subject_id, password):
            session.clear()
            session["subject_id"] = subject_id
            return redirect(url_for("dashboard"))
        error = "Invalid credentials."
    return render_template("login.html", error=error)


@app.route("/register", methods=["GET", "POST"])
def register_public():
    """Public self-registration for non-admin participant accounts."""
    if "subject_id" in session:
        return redirect(url_for("dashboard"))
    error = success = None
    if request.method == "POST":
        subject_id = request.form.get("subject_id", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")
        if not subject_id or not password:
            error = "Participant ID and password are required."
        elif len(password) < 6:
            error = "Password must be at least 6 characters."
        elif password != confirm:
            error = "Passwords do not match."
        else:
            try:
                user_store.create(subject_id, password, is_admin=False)
                success = f"Account created. You can now log in as '{subject_id}'."
            except ValueError as exc:
                error = str(exc)
    return render_template("register_public.html", error=error, success=success)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/change-password", methods=["GET", "POST"])
@login_required
def change_password():
    error = success = None
    if request.method == "POST":
        current = request.form.get("current_password", "")
        new_pw = request.form.get("new_password", "")
        confirm = request.form.get("confirm_password", "")
        subject_id = session["subject_id"]
        if not user_store.verify(subject_id, current):
            error = "Current password is incorrect."
        elif new_pw != confirm:
            error = "New passwords do not match."
        elif len(new_pw) < 6:
            error = "Password must be at least 6 characters."
        else:
            user_store.change_password(subject_id, new_pw)
            success = "Password changed successfully."
    return render_template("change_password.html", error=error, success=success)


# ---------------------------------------------------------------------------
# Admin routes
# ---------------------------------------------------------------------------


@app.route("/admin")
@admin_required
def admin():
    users = user_store.list_users()
    return render_template("admin.html", users=users)


@app.route("/admin/register", methods=["GET", "POST"])
@admin_required
def register():
    error = success = None
    if request.method == "POST":
        subject_id = request.form.get("subject_id", "").strip()
        password = request.form.get("password", "")
        is_admin = bool(request.form.get("is_admin"))
        try:
            user_store.create(subject_id, password, is_admin=is_admin)
            success = f"Account created for participant '{subject_id}'."
        except ValueError as exc:
            error = str(exc)
    return render_template("register.html", error=error, success=success)


@app.route("/admin/user/<subject_id>")
@admin_required
def admin_user(subject_id):
    from color_cue.interactive import list_saved_sessions, list_saved_staircase_sessions
    from plots import psychometric_per_participant_plot, staircase_trace_plot

    user = user_store.get(subject_id)
    if user is None:
        abort(404)

    baseline_df = list_saved_staircase_sessions(BASELINE_ROOT, subject_id)
    calibrated_df = list_saved_sessions(CALIBRATED_ROOT, subject_id)
    pending = active_sessions.list_for_subject(subject_id)

    staircase_plot = staircase_trace_plot(subject_id, BASELINE_ROOT)
    psychometric_plot = psychometric_per_participant_plot(subject_id, CALIBRATED_ROOT)
    sigma_internal = _compute_sigma_internal(subject_id, baseline_df)

    return render_template(
        "admin_user.html",
        target_subject=subject_id,
        is_admin_account=user.get("is_admin", False),
        baseline_sessions=baseline_df.to_dict("records"),
        calibrated_sessions=calibrated_df.to_dict("records"),
        pending_sessions=pending,
        sigma_internal=sigma_internal,
        staircase_plot=staircase_plot,
        psychometric_plot=psychometric_plot,
    )


@app.route("/admin/reset-password", methods=["POST"])
@admin_required
def admin_reset_password():
    subject_id = request.form.get("subject_id", "").strip()
    new_password = request.form.get("new_password", "")
    if len(new_password) < 6:
        return jsonify({"error": "Password must be at least 6 characters."}), 400
    try:
        user_store.change_password(subject_id, new_password)
        return redirect(url_for("admin_user", subject_id=subject_id))
    except ValueError as exc:
        return render_template("error.html", message=str(exc)), 404


@app.route("/admin/data")
@admin_required
def admin_data():
    from plots import (
        calibration_validation_plot,
        psychometric_overlay_plot,
        sigma_internal_summary_plot,
        sigma_internal_table,
    )

    si_table = sigma_internal_table(BASELINE_ROOT)
    if not si_table.empty and "sigma_internal" in si_table.columns:
        valid = si_table.dropna(subset=["sigma_internal"])
        per_subj = (
            valid.groupby("subject_id")["sigma_internal"]
            .agg(mean="mean", std="std", count="count")
            .reset_index()
            .sort_values("subject_id")
        )
        summary_records = per_subj.to_dict("records")
    else:
        summary_records = []

    return render_template(
        "admin_data.html",
        summary_records=summary_records,
        si_plot=sigma_internal_summary_plot(BASELINE_ROOT),
        validation_plot=calibration_validation_plot(CALIBRATED_ROOT),
        overlay_plot=psychometric_overlay_plot(CALIBRATED_ROOT),
    )


@app.route("/admin/delete-user", methods=["POST"])
@admin_required
def delete_user():
    subject_id = request.form.get("subject_id", "").strip()
    if subject_id == session["subject_id"]:
        return jsonify({"error": "Cannot delete your own account."}), 400
    try:
        user_store.delete(subject_id)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    return redirect(url_for("admin"))


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


@app.route("/dashboard")
@login_required
def dashboard():
    from color_cue.interactive import (
        list_saved_sessions,
        list_saved_staircase_sessions,
    )

    subject_id = session["subject_id"]
    user = user_store.get(subject_id)

    baseline_df = list_saved_staircase_sessions(BASELINE_ROOT, subject_id)
    calibrated_df = list_saved_sessions(CALIBRATED_ROOT, subject_id)
    pending = active_sessions.list_for_subject(subject_id)

    # Estimate sigma_internal if baseline sessions exist
    sigma_internal = _compute_sigma_internal(subject_id, baseline_df)

    return render_template(
        "dashboard.html",
        subject_id=subject_id,
        is_admin=user.get("is_admin", False),
        baseline_sessions=baseline_df.to_dict("records"),
        calibrated_sessions=calibrated_df.to_dict("records"),
        pending_sessions=pending,
        sigma_internal=sigma_internal,
    )


@app.route("/session/delete-active", methods=["POST"])
@login_required
def delete_active_session():
    """Delete an in-progress (pickle) session.

    Any user may delete their own active sessions.  Admins may delete any.
    """
    token = request.form.get("token", "").strip()
    task = active_sessions.load(token)
    if task is None:
        return render_template("error.html", message="Session not found."), 404

    current_user = user_store.get(session["subject_id"])
    if task.subject_id != session["subject_id"] and not current_user.get("is_admin"):
        abort(403)

    active_sessions.delete(token)
    # Admin redirects back to the user page; participant to dashboard.
    if current_user.get("is_admin") and task.subject_id != session["subject_id"]:
        return redirect(url_for("admin_user", subject_id=task.subject_id))
    return redirect(url_for("dashboard"))


@app.route("/session/delete-saved", methods=["POST"])
@login_required
def delete_saved_session():
    """Permanently delete a saved session directory from disk.

    Any user may delete their own saved sessions.  Admins may delete any.
    """
    subject_id = request.form.get("subject_id", "").strip()
    session_id = request.form.get("session_id", "").strip()
    session_type = request.form.get("session_type", "")  # "staircase" or "calibrated"

    current_user = user_store.get(session["subject_id"])
    if subject_id != session["subject_id"] and not current_user.get("is_admin"):
        abort(403)

    root = BASELINE_ROOT if session_type == "staircase" else CALIBRATED_ROOT
    session_dir = root / subject_id / session_id
    if not session_dir.exists():
        return render_template("error.html", message="Session directory not found."), 404

    shutil.rmtree(session_dir)

    if current_user.get("is_admin") and subject_id != session["subject_id"]:
        return redirect(url_for("admin_user", subject_id=subject_id))
    return redirect(url_for("dashboard"))





@app.route("/task/start-staircase", methods=["POST"])
@login_required
def start_staircase():
    from color_cue.interactive import StaircaseTaskConfig

    subject_id = session["subject_id"]
    sess_id = request.form.get("session_id", "").strip()
    context = request.form.get("context", "redder")
    notes = request.form.get("notes", "").strip()

    if not sess_id:
        return redirect(url_for("dashboard"))

    cfg = StaircaseTaskConfig(
        theta0=-np.pi / 4,
        sigma_ext_levels=(0.0,),
        delta_grid=(0.02, 0.03, 0.045, 0.065, 0.09, 0.13, 0.18, 0.25),
        start_index=5,
        context=context,
        shared_noise=False,
        max_trials_per_staircase=60,
        max_reversals_per_staircase=10,
        threshold_reversal_count=6,
        trial_seed=None,
        stimulus_seed=None,
    )
    task = WebStaircaseTask(cfg, subject_id, sess_id, notes)
    token = active_sessions.save(task)
    return redirect(url_for("task_page", token=token))


@app.route("/task/start-calibrated", methods=["POST"])
@login_required
def start_calibrated():
    from color_cue.interactive import InteractiveTaskConfig, list_saved_staircase_sessions

    subject_id = session["subject_id"]
    sess_id = request.form.get("session_id", "").strip()
    context = request.form.get("context", "redder")
    notes = request.form.get("notes", "").strip()

    if not sess_id:
        return redirect(url_for("dashboard"))

    baseline_df = list_saved_staircase_sessions(BASELINE_ROOT, subject_id)
    sigma_internal = _compute_sigma_internal(subject_id, baseline_df)

    if sigma_internal is None or not np.isfinite(sigma_internal):
        return render_template(
            "error.html",
            message=(
                "No valid baseline staircase sessions found. "
                "Complete Experiment 1 first, then start a calibrated session."
            ),
        )

    from color_cue.psychophysics import make_noise_calibration_table

    sigma_targets = (0.10, 0.15, 0.20)
    subject_noise = pd.DataFrame(
        {"subject_id": [subject_id], "sigma_internal": [sigma_internal]}
    )
    cal_table = make_noise_calibration_table(subject_noise, sigma_targets)
    feasible = cal_table[cal_table["is_feasible"]]

    if feasible.empty:
        return render_template(
            "error.html",
            message=(
                f"All sigma_target values are below the estimated sigma_internal "
                f"({sigma_internal:.4f}).  Choose larger target noise levels."
            ),
        )

    sigma_ext_levels = tuple(float(x) for x in feasible["sigma_ext"])
    cfg = InteractiveTaskConfig(
        theta0=-np.pi / 4,
        delta_thetas=(-0.20, -0.14, -0.08, -0.04, 0.04, 0.08, 0.14, 0.20),
        sigma_ext_levels=sigma_ext_levels,
        n_repeats=6,
        contexts=(context,),
        trial_seed=None,
        stimulus_seed=None,
    )
    task = WebInteractiveTask(cfg, subject_id, sess_id, notes)
    task.sigma_target_map = dict(
        zip(feasible["sigma_ext"].astype(float), feasible["sigma_target"].astype(float))
    )
    task.sigma_internal_val = float(sigma_internal)
    token = active_sessions.save(task)
    return redirect(url_for("task_page", token=token))


@app.route("/task/<token>")
@login_required
def task_page(token):
    task = active_sessions.load(token)
    if task is None:
        return render_template("error.html", message="Session not found."), 404
    if task.subject_id != session["subject_id"]:
        abort(403)
    return render_template(
        "task.html",
        token=token,
        task_type=task.task_type,
        session_id=task.session_id,
        is_done=task.is_done(),
        iti_ms=int(getattr(task.config, "iti_seconds", 0.3) * 1000),
    )


# ---------------------------------------------------------------------------
# Task API
# ---------------------------------------------------------------------------


@app.route("/api/trial")
@login_required
def api_get_trial():
    token = request.args.get("token", "")
    task = active_sessions.load(token)
    if task is None or task.subject_id != session["subject_id"]:
        return jsonify({"error": "not found"}), 404

    data = task.get_current_trial_data()
    # Persist after rendering so the cached image survives a server restart.
    active_sessions.save(task, token=token)
    return jsonify(data)


@app.route("/api/respond", methods=["POST"])
@login_required
def api_respond():
    body = request.get_json(silent=True) or {}
    token = body.get("token", "")
    choice = body.get("choice", "")

    if choice not in {"left", "right"}:
        return jsonify({"error": "choice must be 'left' or 'right'"}), 400

    task = active_sessions.load(token)
    if task is None or task.subject_id != session["subject_id"]:
        return jsonify({"error": "not found"}), 404

    result = task.record_response(choice)
    active_sessions.save(task, token=token)
    return jsonify(result)


@app.route("/api/finalize", methods=["POST"])
@login_required
def api_finalize():
    body = request.get_json(silent=True) or {}
    token = body.get("token", "")

    task = active_sessions.load(token)
    if task is None or task.subject_id != session["subject_id"]:
        return jsonify({"error": "not found"}), 404

    if not task._responses:
        return jsonify({"error": "No responses recorded yet – nothing to save."}), 400

    # Staircase sessions are only written to disk once the stopping criteria
    # are met.  An early exit pauses the session in-place so it can be resumed.
    if task.task_type == "staircase" and not task.is_done():
        active_sessions.save(task, token=token)
        return jsonify({"paused": True})

    root = BASELINE_ROOT if task.task_type == "staircase" else CALIBRATED_ROOT
    try:
        saved = task.save_results(root)
        active_sessions.delete(token)
        return jsonify({"success": True, "saved_to": str(saved)})
    except (ValueError, FileExistsError) as exc:
        return jsonify({"error": str(exc)}), 400


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_sigma_internal(
    subject_id: str, baseline_df: pd.DataFrame
) -> float | None:
    """Return mean sigma_internal across all saved baseline sessions, or None."""
    if baseline_df.empty:
        return None

    try:
        from color_cue.interactive import load_multiple_staircase_sessions
        from color_cue.interactive import summarize_loaded_staircase_sessions
        from color_cue.psychophysics import stimulus_sigma_from_jnd

        loaded = load_multiple_staircase_sessions(BASELINE_ROOT, subject_id)
        summary = summarize_loaded_staircase_sessions(loaded, threshold_reversal_count=6)
        summary["sigma_internal"] = stimulus_sigma_from_jnd(
            summary["threshold_estimate"], p_correct=0.707
        )
        valid = summary.dropna(subset=["sigma_internal"])
        if valid.empty:
            return None
        return float(valid["sigma_internal"].mean())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not user_store.list_users():
        user_store.create("admin", "admin123", is_admin=True)
        print("=" * 60)
        print("Created default admin account:  admin / admin123")
        print("Change this password immediately via /change-password")
        print("=" * 60)

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("DEBUG", "0") == "1"
    print(f"Starting server on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)
