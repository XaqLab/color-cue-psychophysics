# Color-Cue Calibration — Web Server

A browser-based host for the interactive calibration protocol described in
`notebooks/interactive-calibration-sessions.ipynb`.  Participants complete
both experiments through a standard web browser; all results are saved
server-side in the same directory layout used by the notebook, so the
existing analysis cells work without modification.

---

## Overview

The original notebook drives a Matplotlib GUI window that must run on the
experimenter's local machine.  This server replaces that GUI with a Flask
web application: stimuli are rendered headlessly with Pillow and delivered
as images over HTTP.  Each participant logs in with a unique ID and
password, and their task state is automatically saved between page loads so
sessions can be paused and resumed.

### Workflow

```
Experiment 1  →  Baseline staircase (2-down-1-up, σ_ext = 0)
                   ↓
              σ_internal estimated from reversal thresholds
                   ↓
Experiment 2  →  Calibrated fixed-level 2AFC
                 (σ_ext computed per-subject from σ_targets)
                   ↓
Admin data dashboard  →  Calibration verification plots
```

Both experiments are accessible from the participant dashboard.  Experiment
2 is locked until at least one Experiment 1 session has been saved.

---

## File layout

```
server/
├── app.py            # Flask application and all route handlers
├── auth.py           # JSON-backed user store with hashed passwords
├── render.py         # Headless stimulus rendering → base64 PNG
├── task_runner.py    # Serialisable task state (staircase + fixed-level)
├── plots.py          # Server-side matplotlib plots → base64 PNG
├── templates/
│   ├── base.html
│   ├── login.html
│   ├── register_public.html   # self-registration
│   ├── register.html          # admin-only registration
│   ├── dashboard.html
│   ├── task.html
│   ├── change_password.html
│   ├── admin.html             # user list
│   ├── admin_user.html        # per-user session management + plots
│   ├── admin_data.html        # cross-participant data dashboard
│   └── error.html
└── server_data/          # created automatically on first run
    ├── users.json
    ├── active_sessions/  # in-progress task state (pickle files)
    └── session_data/
        ├── baseline_staircase/   # Experiment 1 results
        └── calibrated_task/      # Experiment 2 results
```

`server_data/` is created on first run and should **not** be committed to
version control.  Add it to `.gitignore` alongside any other data
directories.

---

## Installation

From the repository root, install the `server` extra:

```bash
pip install -e ".[server]"
```

This adds [Flask](https://flask.palletsprojects.com/) and
[Pillow](https://pillow.readthedocs.io/) on top of the existing
`color-cue-psychophysics` dependencies.

---

## Running the server

```bash
python server/app.py
```

The server starts on `http://127.0.0.1:5000` by default.

### Environment variables

| Variable     | Default               | Description                                              |
|--------------|-----------------------|----------------------------------------------------------|
| `SECRET_KEY` | `change-me-before-deploying` | Flask session-cookie signing key. **Must be changed** for any non-localhost deployment. |
| `DATA_ROOT`  | `server/server_data`  | Root directory for all persisted data.                   |
| `HOST`       | `127.0.0.1`           | Bind address.                                            |
| `PORT`       | `5000`                | Bind port.                                               |
| `DEBUG`      | `0`                   | Set to `1` to enable Flask debug/reload mode.            |

Example — expose the server on your local network:

```bash
SECRET_KEY="your-random-secret" HOST="0.0.0.0" PORT="8080" python server/app.py
```

---

## First-time setup

On the very first run, if no users exist, a default admin account is
created:

| Participant ID | Password   |
|----------------|------------|
| `admin`        | `admin123` |

**Change this password immediately** at `/change-password` before
registering any participants.

---

## Account management

### Self-registration (participants)

Participants can create their own non-admin account at `/register` (linked
from the login page).  Self-registered accounts have no admin privileges.

### Admin panel (`/admin`)

Navigate to `/admin` (admin account required).

- **📊 Data Dashboard** — opens the cross-participant visualisation
  dashboard (see below).
- **+ New Participant** — admin-created account with optional admin flag.
- **Manage** (per user) — opens the per-user page to view sessions, reset
  their password, and delete individual sessions.
- **Delete Account** — removes login credentials only; session data on disk
  is not affected.

### Per-user admin page (`/admin/user/<id>`)

- **Reset password** — set a new password for the participant without
  knowing their current one.
- **In-progress sessions** — list of resumable pickle sessions with a
  delete button.
- **Saved sessions** — list of all saved baseline and calibrated sessions
  with per-row delete buttons.
- **Per-participant plots** — staircase traces and psychometric curves
  rendered inline.

### Changing your own password

Any logged-in user can visit `/change-password` (also in the nav bar) to
update their own password.

---

## Participant workflow

### 1. Log in / register

Open the server URL.  Create an account at `/register` or log in with an
existing ID and password.

### 2. Dashboard

The dashboard shows:

- Your estimated **σ_internal** (once Experiment 1 data is available).
- Any **in-progress sessions** (with Resume and Delete buttons).
- Forms to **start new sessions** for both experiments.
- Tables of all **saved sessions** (with Delete buttons).

### 3. Running a session

**Experiment 1 — Baseline Staircase**

1. Choose a session ID (e.g. `baseline_redder_001`) and context
   (`redder` or `bluer`).
2. Click **Start Staircase**.
3. On the task page, press the **← Left** or **→ Right** arrow key to
   indicate which side matches the context.  Correctness feedback appears
   briefly between trials.
4. The staircase stops automatically when the stopping criteria are met
   (60 trials or 10 reversals per staircase), then the results are saved
   and you are returned to the dashboard.

**Experiment 2 — Calibrated Fixed-Level**

Requires at least one saved Experiment 1 session.  The server computes
σ_internal automatically and derives subject-specific σ_ext values for
σ_targets = (0.10, 0.15, 0.20).

1. Choose a session ID and context.
2. Click **Start Calibrated Session** and complete trials as in Experiment 1.

### 4. Pausing, saving, and deleting

**Staircase sessions (Experiment 1)**
A staircase is only written to disk once its stopping criteria are met
(60 trials or 10 reversals).  If you need to stop early, click
**Pause & Exit** — the session is kept as an in-progress entry on the
dashboard and can be resumed later.  Partial staircase data is *not*
written to `session_data/` because an incomplete reversal record produces
an unreliable threshold estimate.  The session auto-saves to disk and
disappears from "In-Progress" once it finishes naturally.

**Calibrated sessions (Experiment 2)**
Fixed-level trials are valid regardless of completion, so **Save & Exit**
is available at any time and writes results immediately to disk.  Saved
partial sessions can be pooled with other sessions in the notebook.

**Deleting sessions**
Both in-progress and saved sessions have a **Delete** button on the
dashboard.  Deleting a saved session permanently removes its directory from
disk — this cannot be undone.  Admins can delete any participant's sessions
from the per-user admin page.

---

## Admin data dashboard (`/admin/data`)

The data dashboard renders cross-participant plots server-side using
matplotlib (Agg backend) and embeds them as base64 PNGs in the page.

### Experiment 1 panel

**σ_internal summary table**
One row per participant showing mean and standard deviation of σ_internal
across all completed staircases.

**σ_internal bar chart**
Each bar is one participant; height = mean σ_internal; error bars = ±1 std.
Individual staircase estimates are overlaid as dots.  The dashed line marks
the grand mean across participants.

### Experiment 2 panel — Calibration verification

The calibration paradigm aims to **equalise total per-stimulus noise**
across participants.  A participant with high internal noise (large
σ_internal) receives a larger σ_ext so that

```
σ_total = sqrt(σ_internal² + σ_ext²) = σ_target
```

holds for everyone.  If this works, all participants should have the same
psychometric function for a given σ_target.  The two plots below verify
this.

#### Plot 1: Identity scatter (σ_stimulus vs σ_target)

- **X-axis:** σ_target — the experimenter-set total noise level.
- **Y-axis:** σ_stimulus — per-stimulus noise estimated by fitting a
  Gaussian 2AFC model to each participant's responses.
- **Each point:** one (participant, context, σ_target) condition.
- **Identity line (dashed):** perfect calibration means σ_stimulus = σ_target,
  so all points should cluster along the diagonal.
- **Interpretation:** points above the line indicate the participant
  experienced more noise than intended (σ_internal may be underestimated);
  points below indicate less noise than intended.

#### Plot 2: Psychometric overlay by σ_target

A grid of subplots with **contexts as rows** and **σ_target levels as
columns**.  Within each panel:

- **Dots:** empirical proportion of "choose right" responses at each Δθ
  level for each participant (colour-coded by participant).
- **Curves:** fitted Gaussian psychometric functions, one per participant.

**How to read it:** if calibration succeeded, all curves within a panel
should collapse onto one — participants experience the same difficulty
despite having different internal noise levels.  Divergent curves flag
specific participants or conditions where calibration may need revisiting.

These two plots are complementary: the identity scatter gives a quick
numerical summary of how well σ_stimulus matches σ_target; the overlay
provides a direct visual check of whether the psychometric functions
themselves are equalised.

---

## Compatibility with the notebook

Saved data uses exactly the same directory structure and file format as
`save_task_session` and `save_staircase_session`.  Calibrated-session
results additionally include `sigma_target` and `sigma_internal` columns so
the notebook's `fit_all_gaussian_psychometrics` call can group by
`sigma_target` directly, without a manual merge step.

```
session_data/
└── <data_root>/
    └── <subject_id>/
        └── <session_id>/
            ├── results.csv
            ├── metadata.json
            └── staircase_summary.csv   # staircase sessions only
```

Point the notebook variables at the server data:

```python
BASELINE_DATA_ROOT   = "server/server_data/session_data/baseline_staircase"
CALIBRATED_DATA_ROOT = "server/server_data/session_data/calibrated_task"
```

---

## Security notes

- Passwords are stored as [Werkzeug](https://werkzeug.palletsprojects.com/)
  password hashes (PBKDF2-SHA256); plaintext passwords are never written to
  disk.
- The default `SECRET_KEY` is **not secure**.  Generate a strong random key
  for any deployment reachable beyond localhost:
  ```bash
  python -c "import secrets; print(secrets.token_hex(32))"
  ```
- Session cookies are signed but **not encrypted**.  Use HTTPS (e.g. via an
  nginx reverse proxy) when running over a network.
- Self-registration is open by default — anyone who can reach the server can
  create a participant account.  If this is undesirable, remove or protect
  the `/register` route in `app.py`.
- This server is designed for **internal lab use** with a small number of
  known participants.  It is not hardened for public internet exposure.

