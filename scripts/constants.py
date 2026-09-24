"""Shared paths and constants for the analysis / evaluation / plotting scripts.

Import from the scripts directory, e.g.:

    from constants import DATA_DIR, MODELS_DIR, OBS_KEY

(Display-name mappings live in display_names.py; experiment tags live in
web_app/constants.py.)
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Prolific study data (gameplay/survey msgpack + json files).  One subdirectory per
# data collection: "newflydata" is the original 4-arm study, "resub" the 6-arm
# resubmission study (which also adds comedi_br / pace_br and records the human's
# game slot).  Outputs are namespaced by the directory's basename so runs on
# different collections don't overwrite each other.
PROLIFIC_DATA_ROOT = os.path.join(PROJ_ROOT, "prolific_data")
DEFAULT_DATASET = "resub"
DATA_DIR = os.path.join(PROLIFIC_DATA_ROOT, DEFAULT_DATASET)

# Per-dataset list of participants judged not to have really played (e.g. holding
# one key down), with user_id and prolific_id columns.  Scripts ignore these users
# when run with --exclude-bad-users.
BAD_USERS_FILENAME = "bad_users.csv"

RESULTS_DIR = os.path.join(PROJ_ROOT, "results")
FIGURES_DIR = os.path.join(PROJ_ROOT, "figures")

# Model checkpoints; overridable for runs outside the container.
MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")

# Companion repo holding the AI-side evaluation outputs (skill axes, stage
# classifiers, cross-play summaries).
SEP_REP_ROOT = "/sep-rep-learning"
SEP_REP_RESULTS_DIR = os.path.join(SEP_REP_ROOT, "results")


def resolve_data_dir(arg=None):
    """Resolve a --data-dir argument to an absolute path.

    Accepts a bare dataset name ("resub"), a path relative to the project root
    ("prolific_data/resub"), a path relative to the current directory, or an
    absolute path.  Returns DATA_DIR when `arg` is None.
    """
    if not arg:
        return DATA_DIR
    if os.path.isabs(arg):
        return os.path.normpath(arg)
    for candidate in (
        os.path.join(PROLIFIC_DATA_ROOT, arg),   # bare dataset name
        os.path.join(PROJ_ROOT, arg),            # repo-relative
        os.path.abspath(arg),                    # cwd-relative
    ):
        if os.path.isdir(candidate):
            return os.path.normpath(candidate)
    # Nothing exists yet; fall back to the repo-relative reading so the error
    # message downstream names a sensible path.
    return os.path.normpath(os.path.join(PROJ_ROOT, arg))


def dataset_name(data_dir):
    """Basename of a data directory, used as the per-dataset output namespace."""
    return os.path.basename(os.path.normpath(data_dir))


def dataset_results_dir(*parts, data_dir):
    """RESULTS_DIR/<parts...>/<dataset>."""
    return os.path.join(RESULTS_DIR, *parts, dataset_name(data_dir))


def dataset_figures_dir(*parts, data_dir):
    """FIGURES_DIR/<dataset>/<parts...>."""
    return os.path.join(FIGURES_DIR, dataset_name(data_dir), *parts)


# ---------------------------------------------------------------------------
# Trajectory replay
# ---------------------------------------------------------------------------

OBS_KEY = 'grid_2d'

# Keyboard-index → game-engine action mapping (from web_app/experiment.py).
# action_idx in records is a keyboard index 0-5 ('up','down','left','right',
# 'stay','interact'); env.step expects the remapped game action via this array.
ACTION_ARRAY = [3, 1, 2, 0, 4, 5]

# Tolerance for comparing simulated vs. recorded reward during slot inference.
REWARD_MATCH_TOL = 0.5

# ---------------------------------------------------------------------------
# Survey response scales
# ---------------------------------------------------------------------------

LIKERT_SCALE = {
    "Strongly disagree": 1,
    "Disagree": 2,
    "Neutral": 3,
    "Agree": 4,
    "Strongly agree": 5,
}
COORD_SCALE = {
    "Very poor": 1,
    "Poor": 2,
    "Neutral": 3,
    "Good": 4,
    "Very good": 5,
}
