"""Analyze gameplay and survey data from app/data, producing two bar plots:
  1. Average return by layout, hued by tag.
  2. Average survey rating (1-5) by question, hued by tag.
"""

import argparse
import os
import re
import struct
import sys
import msgpack
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon, ttest_rel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import (
    BAD_USERS_FILENAME,
    COORD_SCALE,
    DATA_DIR,
    LIKERT_SCALE,
    PROJ_ROOT,
    dataset_figures_dir,
    dataset_results_dir,
    resolve_data_dir,
)
from display_names import (
    DISPLAY_COLORS,
    DISPLAY_NAMES,
    DISPLAY_ORDERING,
    OC_ORIGINAL_LAYOUT_NAMES,
)

sys.path.insert(0, PROJ_ROOT)
from web_app.constants import EXPERIMENT_TAGS

TUTORIAL_DESC = b"Instructions & Tutorial"

# Short labels for survey questions so x-axis is readable
QUESTION_LABELS = {
    "The agent adapted to me when making decisions.": "Adapted to me",
    "The agent was consistent in its actions.": "Consistent",
    "The agent's actions were human-like.": "Human-like",
    "The agent frequently got in my way.": "Got in my way",
    "The agent's behavior was frustrating.": "Frustrating",
    "Overall, I enjoyed playing with the agent.": "Enjoyed playing",
    "Overall, I felt that the agent's ability to coordinate with me was:": "Coordination",
}

# Text size for the survey figure
FONTSIZE = 18


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ext_hook(code, data):
    return (code, data)


def load_user_ids(data_dir):
    """Return a sorted list of unique user IDs found in gameplay/survey filenames."""
    user_ids = set()
    for fname in os.listdir(data_dir):
        m = re.search(r"user=(\d+)", fname)
        if m:
            user_ids.add(m.group(1))
    return sorted(user_ids)


def load_overcooked_experience(data_dir):
    """Return a dict mapping user_id -> overcooked_experience from user_data msgpack files."""
    result = {}
    for fname in os.listdir(data_dir):
        if not fname.startswith("user_data_") or not fname.endswith(".msgpack"):
            continue
        user_id = fname[len("user_data_"):-len(".msgpack")]
        path = os.path.join(data_dir, fname)
        try:
            records = _msgpack_records(path)
            last = next((r for r in reversed(records) if isinstance(r, dict)), None)
            if last is None:
                result[user_id] = "N/A"
                continue
            exp = (
                last.get(b"user_storage", {}).get(b"overcooked_experience", b"")
                or last.get(b"overcooked_experience", b"")
            )
            if isinstance(exp, bytes):
                exp = exp.decode(errors="replace")
            result[user_id] = exp or "N/A"
        except Exception:
            result[user_id] = "N/A"
    return result


def load_prolific_ids(data_dir):
    """Return a dict mapping user_id -> prolific_id (empty string if unavailable).

    Handles two storage layouts:
      - Newer sessions: prolific_id nested inside episode_metadata
      - Older sessions: prolific_id as a top-level key in the last dict record
    Users whose msgpack file exists but contains no dict record are included
    with an empty string so they appear in reporting.
    """
    result = {}
    for fname in os.listdir(data_dir):
        if not fname.startswith("user_data_") or not fname.endswith(".msgpack"):
            continue
        user_id = fname[len("user_data_"):-len(".msgpack")]
        path = os.path.join(data_dir, fname)
        try:
            records = _msgpack_records(path)
            last = next((r for r in reversed(records) if isinstance(r, dict)), None)
            if last is None:
                result[user_id] = ""
                continue
            prolific_id = (
                last.get(b"episode_metadata", {}).get(b"prolific_id", b"")
                or last.get(b"prolific_id", b"")
            )
            if isinstance(prolific_id, bytes):
                prolific_id = prolific_id.decode(errors="replace")
            result[user_id] = prolific_id
        except Exception:
            result[user_id] = ""
    return result


def load_valid_users(data_dir):
    """Return {user_id: prolific_id} for all confirmed participants.

    A user is a confirmed participant if they have a user_data_*.msgpack file,
    even if their prolific_id is missing or empty.  Prolific IDs that are present
    are included for downstream reporting; an empty string signals an unknown ID.
    """
    return load_prolific_ids(data_dir)


def load_users_with_msgpack(data_dir):
    """Return the set of user IDs that have a user_data_*.msgpack file."""
    ids = set()
    for fname in os.listdir(data_dir):
        if fname.startswith("user_data_") and fname.endswith(".msgpack"):
            ids.add(fname[len("user_data_"):-len(".msgpack")])
    return ids


def load_test_user_ids(data_dir):
    """Return a set of user IDs whose prolific ID contains 'test' (case-insensitive)."""
    prolific_ids = load_prolific_ids(data_dir)
    return {uid for uid, pid in prolific_ids.items() if "test" in pid.lower()}


# A user is complete if they finished every tag in the current experiment, or every
# tag in the original 4-tag experiment (data collected before comedi_br / pace_br
# were added on 2026-09-21).
EXPECTED_TAGS = set(EXPERIMENT_TAGS)
LEGACY_EXPECTED_TAGS = {"fcp", "mep", "oc_cec_v3", "oc_cecp_pred_1000"}


def load_incomplete_user_ids(data_dir):
    """Return a set of user IDs that are missing one or more of the expected survey tags."""
    all_user_ids = set(load_user_ids(data_dir))
    tags_per_user = {}
    for fname in os.listdir(data_dir):
        if not fname.startswith("survey_"):
            continue
        m_uid = re.search(r"user=(\d+)", fname)
        m_tag = re.search(r"tag=(.+?)(?:_coord_ring|_counter_circuit)", fname)
        if m_uid and m_tag:
            tags_per_user.setdefault(m_uid.group(1), set()).add(m_tag.group(1))
    return {
        uid for uid in all_user_ids
        if tags_per_user.get(uid, set()) not in (EXPECTED_TAGS, LEGACY_EXPECTED_TAGS)
    }


def add_exclude_bad_users_arg(parser):
    """Add the shared --exclude-bad-users option to an argparse parser."""
    parser.add_argument(
        "--exclude-bad-users", nargs="?", const=True, default=None, metavar="CSV",
        help=f"Ignore users listed in a CSV with a user_id column. With no value, reads "
             f"<data-dir>/{BAD_USERS_FILENAME} (participants judged not to have really "
             f"played; see analyze_action_distributions.py).",
    )


def load_bad_user_ids(data_dir, exclude_bad_users, verbose=True):
    """Return the user IDs to ignore for an --exclude-bad-users value (empty if unset).

    `exclude_bad_users` is None (option not given), True (use the dataset's
    bad_users.csv) or a path to another CSV with a user_id column.
    """
    if not exclude_bad_users:
        return set()
    path = (os.path.join(data_dir, BAD_USERS_FILENAME) if exclude_bad_users is True
            else exclude_bad_users)
    if not os.path.exists(path):
        raise FileNotFoundError(f"--exclude-bad-users: no such file {path}")
    ids = set(pd.read_csv(path, dtype={"user_id": str})["user_id"].str.strip())
    if verbose:
        print(f"Excluding {len(ids)} bad user(s) listed in {path}: {sorted(ids)}")
    return ids


def load_excluded_user_ids(data_dir, exclude_test=False, exclude_bad_users=None, verbose=True):
    """Return the set of user IDs to drop from analysis, printing why when verbose.

    Excludes users with no prolific ID who also lack a user_data msgpack file (users
    whose msgpack exists but lost its data are still confirmed participants), users
    missing one or more expected survey tags, and optionally test users and the users
    in a bad-user list (see load_bad_user_ids).
    """
    user_ids = load_user_ids(data_dir)
    prolific_ids = load_prolific_ids(data_dir)
    users_with_msgpack = load_users_with_msgpack(data_dir)
    no_prolific_ids = {uid for uid in user_ids if not prolific_ids.get(uid) and uid not in users_with_msgpack}
    if no_prolific_ids and verbose:
        print(f"Excluding {len(no_prolific_ids)} user(s) with no prolific ID: {sorted(no_prolific_ids)}")

    incomplete_ids = load_incomplete_user_ids(data_dir)
    if incomplete_ids and verbose:
        print(f"Excluding {len(incomplete_ids)} incomplete user(s) (missing survey tags): {sorted(incomplete_ids)}")

    exclude_ids = no_prolific_ids | incomplete_ids
    if exclude_test:
        test_ids = load_test_user_ids(data_dir)
        if test_ids and verbose:
            print(f"Excluding {len(test_ids)} test user(s): {sorted(test_ids)}")
        exclude_ids |= test_ids
    exclude_ids |= load_bad_user_ids(data_dir, exclude_bad_users, verbose=verbose)
    return exclude_ids


def _msgpack_records(path):
    """Return every record in a nicewebrl data file, in write order.

    The files are a sequence of frames: a 4-byte big-endian length followed by that
    many bytes of msgpack (see nicewebrl.utils.read_msgpack_records_sync).  Reading
    them as a bare msgpack stream instead — as this module used to — usually works by
    luck, because the length bytes happen to decode as small throwaway integers, but
    desyncs whenever a length byte is itself a type marker (0xc1 raises, 0xca/0xcf
    silently swallow payload).  Five resub survey files and one gameplay file hit that.

    Records are unpacked with raw=True, so keys are bytes, and with the ext hook that
    leaves float32 payloads as (code, data) tuples for _decode_float32_ext.
    """
    with open(path, "rb") as f:
        content = f.read()

    records = []
    pos = 0
    while pos < len(content):
        size_bytes = content[pos:pos + 4]
        if len(size_bytes) < 4:
            break
        size = struct.unpack(">I", size_bytes)[0]
        pos += 4
        if pos + size > len(content):
            print(f"  WARNING: incomplete final record in {os.path.basename(path)}")
            break
        try:
            records.append(msgpack.unpackb(
                content[pos:pos + size],
                raw=True,
                strict_map_key=False,
                max_array_len=2**32,
                max_map_len=2**32,
                max_str_len=2**32,
                ext_hook=_ext_hook,
            ))
        except Exception as e:
            print(f"  WARNING: unreadable record in {os.path.basename(path)}: {e}")
            break
        pos += size
    return records


def _decode_float32_ext(ext_tuple):
    """Decode a (code, bytes) tuple encoding a float32 numpy scalar/array."""
    _, data = ext_tuple
    # inner msgpack: [shape, dtype, raw_bytes]
    inner = msgpack.unpackb(data, raw=True)
    if isinstance(inner, list) and len(inner) == 3:
        _, dtype, raw = inner
        if dtype == b"float32":
            n = len(raw) // 4
            return sum(struct.unpack(f"{n}f", raw))
    return 0.0


def _decode_uint8_ext(ext_tuple):
    """Decode a (code, bytes) tuple encoding a uint8 numpy scalar (e.g. step_type)."""
    _, data = ext_tuple
    inner = msgpack.unpackb(data, raw=True)
    if isinstance(inner, list) and len(inner) == 3:
        _, dtype, raw = inner
        if dtype == b"uint8" and len(raw) == 1:
            return raw[0]
    return None


# dm_env-style step types stored in each recorded timestep.
STEP_FIRST = 0
STEP_LAST = 2


# ---------------------------------------------------------------------------
# Gameplay parsing
# ---------------------------------------------------------------------------

def parse_gameplay_files(data_dir, exclude_ids=None):
    """Return one row per gameplay file with the return of a single episode.

    A file can hold several episodes with the same agent when a participant restarted
    the session and replayed a round.  Episodes are split on FIRST timesteps (a page
    refresh mid-episode resumes the same episode in a new block, so block names are
    not episode boundaries).  The latest complete episode is kept; if none reached a
    LAST timestep, the latest partial episode is used instead.
    """
    rows = []
    n_repeated = 0
    for fname in os.listdir(data_dir):
        if not fname.startswith("gameplay_"):
            continue
        m_uid = re.search(r"user=(\d+)", fname)
        if exclude_ids and m_uid and m_uid.group(1) in exclude_ids:
            continue
        path = os.path.join(data_dir, fname)
        episodes = []  # each: {"return": float, "complete": bool}
        tag = None
        layout = None

        for item in _msgpack_records(path):
            if not isinstance(item, dict):
                continue
            meta = item.get(b"metadata", {})
            if meta.get(b"type") != b"EnvStage":
                continue

            # Skip tutorial blocks
            bm = meta.get(b"block_metadata", {})
            if bm.get(b"desc") == TUTORIAL_DESC or bm.get(b"tag") is None:
                tag = None
                break

            tag = bm.get(b"tag", b"").decode()
            agent_id = bm.get(b"agent_id")
            desc = bm.get(b"desc", b"").decode()
            # desc format: "{tag} on {layout}"
            m = re.match(r".+ on (.+)", desc)
            layout = m.group(1) if m else desc

            # Decode this step's reward and step type
            reward = 0.0
            step_type = None
            data = item.get(b"data", {})
            if isinstance(data, dict):
                ts_bytes = data.get(b"timestep")
                if isinstance(ts_bytes, bytes) and ts_bytes:
                    try:
                        ts = msgpack.unpackb(
                            ts_bytes, raw=True, strict_map_key=False, ext_hook=_ext_hook
                        )
                        reward_field = ts.get(b"reward")
                        if isinstance(reward_field, tuple):
                            reward = _decode_float32_ext(reward_field)
                        step_type_field = ts.get(b"step_type")
                        if isinstance(step_type_field, tuple):
                            step_type = _decode_uint8_ext(step_type_field)
                    except Exception:
                        pass

            # The first recorded episode starts mid-stream (its reset step isn't
            # logged); later episodes start with a FIRST step.
            if (not episodes or step_type == STEP_FIRST
                    or (episodes[-1]["complete"] and step_type != STEP_LAST)):
                episodes.append({"return": 0.0, "complete": False})
            ep = episodes[-1]
            if ep["complete"]:
                # The trailing timer record repeats the LAST timestep; don't count
                # its reward twice.
                continue
            ep["return"] += reward
            if step_type == STEP_LAST:
                ep["complete"] = True

        if tag is not None and layout is not None and episodes:
            complete = [ep for ep in episodes if ep["complete"]]
            chosen = complete[-1] if complete else episodes[-1]
            if len(episodes) > 1:
                n_repeated += 1
            user_id = m_uid.group(1) if m_uid else None
            rows.append({"user_id": user_id, "tag": tag, "agent_id": agent_id, "layout": layout, "total_return": chosen["return"]})

    if n_repeated:
        print(f"Note: {n_repeated} gameplay file(s) contained repeated episodes; "
              f"kept the latest complete episode for each.")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Survey parsing
# ---------------------------------------------------------------------------

def parse_survey_files(data_dir, exclude_ids=None):
    rows = []
    for fname in os.listdir(data_dir):
        if not fname.startswith("survey_"):
            continue
        m_uid = re.search(r"user=(\d+)", fname)
        if exclude_ids and m_uid and m_uid.group(1) in exclude_ids:
            continue
        path = os.path.join(data_dir, fname)
        records = _msgpack_records(path)

        # Take the last record that carries both the answers and the block metadata.
        # A survey file holds one record per submission; resub files can hold two (an
        # initial write plus the final one), and the last is the completed response.
        data_dict = None
        meta_dict = None
        for r in records:
            if not isinstance(r, dict):
                continue
            data = r.get(b"data")
            meta = r.get(b"metadata")
            if isinstance(data, dict) and isinstance(meta, dict):
                data_dict, meta_dict = data, meta

        if data_dict is None or meta_dict is None:
            continue

        bm = meta_dict.get(b"block_metadata", {})
        tag = bm.get(b"tag", b"").decode()
        desc = bm.get(b"desc", b"").decode()
        m = re.match(r".+ on (.+)", desc)
        layout = m.group(1) if m else desc

        questions = list(data_dict.keys())
        for i, q_bytes in enumerate(questions):
            ans_bytes = data_dict[q_bytes]
            q = q_bytes.decode() if isinstance(q_bytes, bytes) else str(q_bytes)
            ans = ans_bytes.decode() if isinstance(ans_bytes, bytes) else str(ans_bytes)
            scale = COORD_SCALE if i == len(questions) - 1 else LIKERT_SCALE
            score = scale.get(ans)
            if score is None:
                continue
            label = QUESTION_LABELS.get(q, q[:30])
            rows.append({"user_id": m_uid.group(1) if m_uid else None, "tag": tag, "layout": layout, "question": label, "score": score})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistical testing
# ---------------------------------------------------------------------------

CECP_RAW = "oc_cecp_pred_1000"


def _holm_bonferroni(p_values):
    """Return Holm-Bonferroni adjusted p-values (same order as input)."""
    n = len(p_values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: p_values[i])
    adjusted = [0.0] * n
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = min(p_values[idx] * (n - rank), 1.0)
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return adjusted


def _sig_label(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _paired_test(a, b, alternative, test):
    if test == "ttest":
        return ttest_rel(a, b, alternative=alternative)
    return wilcoxon(a, b, alternative=alternative)


def _run_wilcoxon_tests(df, test="wilcoxon"):
    """
    Paired one-sided Wilcoxon signed-rank tests: CECP > each other method,
    within each layout group and pooled across all layouts.

    df must have columns: user_id, tag (raw), layout (raw), total_return.
    Returns list of dicts with layout_display, other_display, stat, p_raw, label.
    """
    def _tests_for_pivot(pivot, layout_raw):
        rows = []
        if CECP_RAW not in pivot.columns:
            return rows
        for other_raw in pivot.columns:
            if other_raw == CECP_RAW:
                continue
            paired = pivot[[CECP_RAW, other_raw]].dropna()
            if len(paired) < 3:
                continue
            stat, p = _paired_test(paired[CECP_RAW], paired[other_raw], "two-sided", test)
            rows.append({"layout_raw": layout_raw, "other_raw": other_raw, "stat": stat, "p_raw": p})
        return rows

    all_raw = []

    for layout_raw in df["layout"].unique():
        sub = (
            df[df["layout"] == layout_raw]
            .groupby(["user_id", "tag"])["total_return"]
            .mean()
            .reset_index()
        )
        all_raw.extend(_tests_for_pivot(
            sub.pivot(index="user_id", columns="tag", values="total_return"),
            layout_raw,
        ))

    # Pooled across all layouts
    sub_all = df.groupby(["user_id", "tag"])["total_return"].mean().reset_index()
    all_raw.extend(_tests_for_pivot(
        sub_all.pivot(index="user_id", columns="tag", values="total_return"),
        "_all",
    ))

    if not all_raw:
        return []

    results = []
    for r in all_raw:
        layout_display = (
            "Both Layouts" if r["layout_raw"] == "_all"
            else OC_ORIGINAL_LAYOUT_NAMES.get(r["layout_raw"], r["layout_raw"])
        )
        results.append({
            **r,
            "label": _sig_label(r["p_raw"]),
            "layout_display": layout_display,
            "other_display": DISPLAY_NAMES.get(r["other_raw"], r["other_raw"]),
        })
    return results


LOWER_QUESTIONS = {"Got in my way", "Frustrating"}


def _run_wilcoxon_tests_survey(df, test="wilcoxon"):
    """
    Paired one-sided Wilcoxon tests per survey question, pooled across layouts.
    Tests CECP < other for negative questions; CECP > other for all others.

    df must have columns: user_id, tag (raw), question (label), score.
    Returns list of dicts with question, other_display, stat, p_raw, label.
    """
    results = []
    for question in df["question"].unique():
        alt = "two-sided"
        sub = (
            df[df["question"] == question]
            .groupby(["user_id", "tag"])["score"]
            .mean()
            .reset_index()
        )
        pivot = sub.pivot(index="user_id", columns="tag", values="score")
        if CECP_RAW not in pivot.columns:
            continue
        for other_raw in pivot.columns:
            if other_raw == CECP_RAW:
                continue
            paired = pivot[[CECP_RAW, other_raw]].dropna()
            if len(paired) < 3:
                continue
            stat, p = _paired_test(paired[CECP_RAW], paired[other_raw], alt, test)
            results.append({
                "question": question,
                "other_raw": other_raw,
                "stat": stat,
                "p_raw": p,
                "label": _sig_label(p),
                "other_display": DISPLAY_NAMES.get(other_raw, other_raw),
            })
    return results


def _vertical_whisker_tops(ax):
    """(x, top_y) for every error bar seaborn drew, so brackets can clear them."""
    tops = []
    for line in ax.lines:
        xd = list(line.get_xdata())
        yd = list(line.get_ydata())
        if len(xd) >= 2 and max(xd) - min(xd) < 1e-9 and yd:
            tops.append((float(xd[0]), float(max(yd))))
    return tops


def _text_height_in_data(ax, fontsize):
    """Height of one line of `fontsize` text, in y-data units at the current limits."""
    fig = ax.get_figure()
    ax_height_px = ax.get_window_extent().height
    y0, y1 = ax.get_ylim()
    return (fontsize * fig.dpi / 72.0) * (y1 - y0) / ax_height_px


def _layout_significance_brackets(ax, sig_results, bar_positions, x_display_order, hue_order,
                                  group_key, fontsize, whisker_tops):
    """Place every bracket without drawing it.

    Brackets in one x group are stacked shortest-span first, each a full text height
    (plus its downward ticks and a gap) above the one below, starting clear of the
    tallest bar *and error bar* in the group.  Returns the placements and the axis
    top they need; spacing is derived from the rendered text height, so the stack
    stays collision-free at any font size.
    """
    cecp_display = DISPLAY_NAMES.get(CECP_RAW, CECP_RAW)
    text_h = _text_height_in_data(ax, fontsize)
    gap = 0.35 * text_h
    tick_len = 0.45 * text_h
    step = text_h + tick_len + gap

    placements = []
    needed_top = ax.get_ylim()[1]

    for x_idx, x_display in enumerate(x_display_order):
        group_results = [r for r in sig_results if r[group_key] == x_display]
        if not group_results:
            continue

        cecp_x = bar_positions.get((x_idx, cecp_display), (None, None))[0]
        if cecp_x is None:
            continue

        # Sort by distance between bars so shorter spans are drawn lower
        group_results = sorted(
            group_results,
            key=lambda r: abs(cecp_x - bar_positions.get((x_idx, r["other_display"]), (cecp_x, 0))[0]),
        )

        # Clear the tallest bar in this x group, and any error bar rising above it
        group_heights = [bar_positions.get((x_idx, t), (0, 0))[1] for t in hue_order]
        group_heights += [top for x, top in whisker_tops if abs(x - x_idx) < 0.5]
        base_y = max(group_heights) + tick_len + gap

        for bracket_idx, r in enumerate(group_results):
            other_x = bar_positions.get((x_idx, r["other_display"]), (None, None))[0]
            if other_x is None:
                continue
            y_bracket = base_y + bracket_idx * step
            placements.append((cecp_x, other_x, y_bracket, r["label"]))
            needed_top = max(needed_top, y_bracket + text_h + gap)

    return placements, needed_top, tick_len


def _draw_significance_brackets(ax, sig_results, bar_positions, x_display_order, hue_order,
                                group_key="layout_display", fontsize=None):
    """Overlay significance brackets for CECP vs each other method."""
    # "ns" labels are drawn a little smaller, so size the stack by the larger label.
    label_fontsize = fontsize if fontsize is not None else 11
    whisker_tops = _vertical_whisker_tops(ax)

    # Raising the top to fit the stack shrinks a data unit, which grows the text in data
    # units, which can need a little more room again — so re-place until it settles.
    for _ in range(5):
        placements, needed_top, tick_len = _layout_significance_brackets(
            ax, sig_results, bar_positions, x_display_order, hue_order,
            group_key, label_fontsize, whisker_tops,
        )
        if needed_top <= ax.get_ylim()[1] + 1e-9:
            break
        ax.set_ylim(top=needed_top)

    for cecp_x, other_x, y_bracket, label in placements:
        color = "black" if label != "ns" else "#888888"
        ls = "-" if label != "ns" else "--"
        lw = 1.2
        # Short downward ticks from the horizontal bar
        ax.plot([cecp_x, cecp_x], [y_bracket - tick_len, y_bracket], color=color, lw=lw, ls=ls, clip_on=False)
        ax.plot([other_x, other_x], [y_bracket - tick_len, y_bracket], color=color, lw=lw, ls=ls, clip_on=False)
        ax.plot([cecp_x, other_x], [y_bracket, y_bracket], color=color, lw=lw, ls=ls, clip_on=False)
        ax.text(
            (cecp_x + other_x) / 2, y_bracket, label,
            ha="center", va="bottom",
            fontsize=fontsize if fontsize is not None else (9 if label == "ns" else 11),
            color=color,
        )


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------

def generate_summary_csv(gameplay_df, prolific_ids, overcooked_exp, out_path=None,
                         data_dir=DATA_DIR):
    """Build and return a per-episode summary DataFrame, also saving it as a CSV."""
    if out_path is None:
        out_path = os.path.join(
            dataset_results_dir(data_dir=data_dir), "user_episode_summary.csv"
        )
    df = gameplay_df.copy()
    df["prolific_id"] = df["user_id"].map(prolific_ids).fillna("")
    df["overcooked_experience"] = df["user_id"].map(overcooked_exp).fillna("N/A")
    df = df.rename(columns={"tag": "agent_tag", "total_return": "episode_return"})
    df = df[["user_id", "prolific_id", "agent_tag", "agent_id", "layout", "overcooked_experience", "episode_return"]]
    df = df.sort_values(["user_id", "agent_tag", "layout"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _build_palette(labels):
    """DISPLAY_COLORS where defined, tab10 fallbacks for anything else."""
    fallback = sns.color_palette("tab10")
    palette = {}
    fallback_idx = 0
    for label in labels:
        if label in DISPLAY_COLORS:
            palette[label] = DISPLAY_COLORS[label]
        else:
            palette[label] = fallback[fallback_idx % len(fallback)]
            fallback_idx += 1
    return palette


def plot_return(df, out_path=None, test="ttest",
                show_both_layouts=True, show_significance=True, data_dir=DATA_DIR):
    if out_path is None:
        out_path = os.path.join(
            dataset_figures_dir(data_dir=data_dir), "avg_return_by_layout.png"
        )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if show_significance:
        sig_results = _run_wilcoxon_tests(df, test=test)
        sig_results = [r for r in sig_results if r["layout_display"] == "Both Layouts"]
        for r in sig_results:
            print(
                f"  {r['layout_display']:20s}  CECP vs {r['other_display']:10s}"
                f"  p={r['p_raw']:.4f}  {r['label']}"
            )
    else:
        sig_results = []

    df = df.copy()
    df["tag"] = df["tag"].map(lambda t: DISPLAY_NAMES.get(t, t))
    df["layout"] = df["layout"].map(lambda l: OC_ORIGINAL_LAYOUT_NAMES.get(l, l))
    hue_order = [t for t in DISPLAY_ORDERING if t in df["tag"].unique()]
    per_layout_order = sorted(df["layout"].unique())
    layout_order = per_layout_order + (["Both Layouts"] if show_both_layouts else [])

    if show_both_layouts:
        df_all = df.copy()
        df_all["layout"] = "Both Layouts"
        df = pd.concat([df, df_all], ignore_index=True)

    # Save data needed to reproduce this figure
    out_dir = dataset_results_dir("analyze_data", data_dir=data_dir)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "return_data.csv"), index=False)
    print(f"Saved: {os.path.join(out_dir, 'return_data.csv')}")
    if sig_results:
        pd.DataFrame(sig_results).to_csv(os.path.join(out_dir, "return_significance.csv"), index=False)
        print(f"Saved: {os.path.join(out_dir, 'return_significance.csv')}")

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.barplot(
        data=df, x="layout", y="total_return", hue="tag",
        hue_order=hue_order, order=layout_order, palette=_build_palette(hue_order), ax=ax,
    )

    bar_positions = {}
    for hue_idx, container in enumerate(ax.containers):
        tag_display = hue_order[hue_idx]
        for layout_idx, bar in enumerate(container):
            bar_positions[(layout_idx, tag_display)] = (
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
            )

    ax.set_title("Average Return by Layout")
    ax.set_xlabel("Layout")
    ax.set_ylabel("Average Return")
    ax.legend(title="Tag", loc="lower left")
    # Lay the figure out before placing brackets: their spacing is measured in pixels,
    # so the axes must already be at its final size.
    fig.tight_layout()

    if show_significance:
        _draw_significance_brackets(ax, sig_results, bar_positions, layout_order, hue_order, group_key="layout_display")

    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.replace(".png", ".svg"))
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_survey(df, out_path=None, test="wilcoxon",
                show_significance=True, data_dir=DATA_DIR):
    if out_path is None:
        out_path = os.path.join(
            dataset_figures_dir(data_dir=data_dir), "avg_survey_by_question.png"
        )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if show_significance:
        sig_results = _run_wilcoxon_tests_survey(df, test=test)
        for r in sig_results:
            print(
                f"  {r['question']:20s}  CECP vs {r['other_display']:10s}"
                f"  p={r['p_raw']:.4f}  {r['label']}"
            )
    else:
        sig_results = []

    df = df.copy()
    df["tag"] = df["tag"].map(lambda t: DISPLAY_NAMES.get(t, t))
    df["layout"] = df["layout"].map(lambda l: OC_ORIGINAL_LAYOUT_NAMES.get(l, l))
    hue_order = [t for t in DISPLAY_ORDERING if t in df["tag"].unique()]
    question_order = list(QUESTION_LABELS.values())

    # Save data needed to reproduce this figure
    out_dir = dataset_results_dir("analyze_data", data_dir=data_dir)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "survey_data.csv"), index=False)
    print(f"Saved: {os.path.join(out_dir, 'survey_data.csv')}")
    if sig_results:
        pd.DataFrame(sig_results).to_csv(os.path.join(out_dir, "survey_significance.csv"), index=False)
        print(f"Saved: {os.path.join(out_dir, 'survey_significance.csv')}")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=df,
        x="question",
        y="score",
        hue="tag",
        hue_order=hue_order,
        order=question_order,
        palette=_build_palette(hue_order),
        ax=ax,
    )

    bar_positions = {}
    for hue_idx, container in enumerate(ax.containers):
        tag_display = hue_order[hue_idx]
        for q_idx, bar in enumerate(container):
            bar_positions[(q_idx, tag_display)] = (
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
            )

    ax.set_title("Average Survey Rating by Question", fontsize=FONTSIZE)
    ax.set_xlabel("Question", fontsize=FONTSIZE)
    ax.set_ylabel("Rating (1–5)", fontsize=FONTSIZE)
    # Extra headroom above the bars for the legend and brackets; ratings only go to 5,
    # so the ticks stop there even though the axis runs higher.
    ax.set_ylim(0, 7.0)
    ax.set_yticks(range(0, 6))
    # Outside the axes: at this font size there is no free space left inside, and an
    # inset legend covers the right-hand groups' bars and brackets.
    ax.legend(title=None, fontsize=FONTSIZE, handlelength=1.0, handleheight=1.0,
              handletextpad=0.5, labelspacing=0.25, borderpad=0.3,
              loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.tick_params(axis="both", labelsize=FONTSIZE)
    plt.xticks(rotation=30, ha="right")
    # Lay the figure out before placing brackets: their spacing is measured in pixels,
    # so the axes must already be at its final size.
    fig.tight_layout()

    if show_significance:
        # Only mark the comparisons that reached significance; "ns" brackets add clutter.
        _draw_significance_brackets(
            ax, [r for r in sig_results if r["label"] != "ns"], bar_positions,
            question_order, hue_order, group_key="question", fontsize=FONTSIZE,
        )

    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.replace(".png", ".svg"))
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Dataset name ('resub', 'newflydata') or path to a data directory "
             "(default: prolific_data/resub/). Outputs are namespaced by its basename.",
    )
    parser.add_argument(
        "--exclude-test",
        action="store_true",
        help="Exclude participants whose prolific ID contains 'test'.",
    )
    add_exclude_bad_users_arg(parser)
    parser.add_argument(
        "--return-test",
        choices=["wilcoxon", "ttest"],
        default="ttest",
        help="Statistical test for return significance brackets (default: ttest).",
    )
    parser.add_argument(
        "--survey-test",
        choices=["wilcoxon", "ttest"],
        default="wilcoxon",
        help="Statistical test for survey significance brackets (default: wilcoxon).",
    )
    parser.add_argument(
        "--both-layouts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show/hide the pooled 'Both Layouts' column in the return plot (default: show).",
    )
    parser.add_argument(
        "--significance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show/hide significance brackets (default: show).",
    )
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    print(f"Data directory: {data_dir}")

    user_ids = load_user_ids(data_dir)
    overcooked_exp = load_overcooked_experience(data_dir)
    prolific_ids = load_prolific_ids(data_dir)

    print("Users (user_id -> prolific_id, overcooked_experience):")
    for uid in user_ids:
        prolific = prolific_ids.get(uid, "")
        exp = overcooked_exp.get(uid, "N/A")
        print(f"  user={uid}: prolific_id={prolific!r}, overcooked_exp={exp}")

    exclude_ids = load_excluded_user_ids(data_dir, exclude_test=args.exclude_test,
                                         exclude_bad_users=args.exclude_bad_users)

    n_total = len(user_ids)
    n_excluded = len(exclude_ids)
    print(f"Users: {n_total} loaded, {n_excluded} excluded, {n_total - n_excluded} analyzed")

    gameplay_df = parse_gameplay_files(data_dir, exclude_ids=exclude_ids)
    print(f"Gameplay rows: {len(gameplay_df)}")
    print(gameplay_df.groupby(["tag", "layout"])["total_return"].mean().to_string())

    summary_df = generate_summary_csv(
        gameplay_df, prolific_ids, overcooked_exp, data_dir=data_dir
    )
    print(f"\nUser episode summary ({len(summary_df)} rows):")
    print(summary_df.to_string())

    survey_df = parse_survey_files(data_dir, exclude_ids=exclude_ids)
    print(f"\nSurvey rows: {len(survey_df)}")
    print(survey_df.groupby(["tag", "question"])["score"].mean().to_string())

    plot_return(gameplay_df, test=args.return_test, show_both_layouts=args.both_layouts,
                show_significance=args.significance, data_dir=data_dir)
    plot_survey(survey_df, test=args.survey_test, show_significance=args.significance,
                data_dir=data_dir)
