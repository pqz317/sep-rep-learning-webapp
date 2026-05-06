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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
from display_names import DISPLAY_NAMES, OC_ORIGINAL_LAYOUT_NAMES, DISPLAY_ORDERING

DATA_DIR = os.path.join(os.path.dirname(__file__), "newflydata")

TUTORIAL_DESC = b"Instructions & Tutorial"

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


EXPECTED_TAGS = {"fcp", "mep", "oc_cec_v3", "oc_cecp_pred_1000"}


def load_incomplete_user_ids(data_dir):
    """Return a set of user IDs that are missing one or more of the 4 expected survey tags."""
    all_user_ids = set(load_user_ids(data_dir))
    tags_per_user = {}
    for fname in os.listdir(data_dir):
        if not fname.startswith("survey_"):
            continue
        m_uid = re.search(r"user=(\d+)", fname)
        m_tag = re.search(r"tag=(.+?)(?:_coord_ring|_counter_circuit)", fname)
        if m_uid and m_tag:
            tags_per_user.setdefault(m_uid.group(1), set()).add(m_tag.group(1))
    return {uid for uid in all_user_ids if tags_per_user.get(uid, set()) != EXPECTED_TAGS}


def _msgpack_records(path):
    with open(path, "rb") as f:
        data = f.read()
    unpacker = msgpack.Unpacker(
        raw=True,
        strict_map_key=False,
        max_array_len=2**32,
        max_map_len=2**32,
        max_str_len=2**32,
        ext_hook=_ext_hook,
    )
    unpacker.feed(data)
    try:
        return list(unpacker)
    except msgpack.exceptions.FormatError:
        # Some files have a reserved 0xc1 byte at offset 3 (after a 3-byte header)
        # before the actual msgpack map payload; skip it and parse as a single object.
        obj = msgpack.unpackb(
            data[4:],
            raw=True,
            strict_map_key=False,
            max_array_len=2**32,
            max_map_len=2**32,
            max_str_len=2**32,
            ext_hook=_ext_hook,
        )
        if isinstance(obj, dict):
            records = []
            for k, v in obj.items():
                records.append(k)
                records.append(v)
            return records
        raise


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


# ---------------------------------------------------------------------------
# Gameplay parsing
# ---------------------------------------------------------------------------

def parse_gameplay_files(data_dir, exclude_ids=None):
    rows = []
    for fname in os.listdir(data_dir):
        if not fname.startswith("gameplay_"):
            continue
        m_uid = re.search(r"user=(\d+)", fname)
        if exclude_ids and m_uid and m_uid.group(1) in exclude_ids:
            continue
        path = os.path.join(data_dir, fname)
        total_reward = 0.0
        tag = None
        layout = None

        with open(path, "rb") as f:
            unpacker = msgpack.Unpacker(
                f,
                raw=True,
                max_array_len=2**32,
                max_map_len=2**32,
                max_str_len=2**32,
                ext_hook=_ext_hook,
            )
            for item in unpacker:
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

                # Accumulate reward from this step
                data = item.get(b"data", {})
                if isinstance(data, dict):
                    ts_bytes = data.get(b"timestep")
                    if isinstance(ts_bytes, bytes) and ts_bytes:
                        try:
                            ts = msgpack.unpackb(ts_bytes, raw=True, ext_hook=_ext_hook)
                            reward_field = ts.get(b"reward")
                            if isinstance(reward_field, tuple):
                                total_reward += _decode_float32_ext(reward_field)
                        except Exception:
                            pass

        if tag is not None and layout is not None:
            user_id = m_uid.group(1) if m_uid else None
            rows.append({"user_id": user_id, "tag": tag, "agent_id": agent_id, "layout": layout, "total_return": total_reward})

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

        # Locate the data dict and metadata dict by their preceding label
        data_dict = None
        meta_dict = None
        for i, r in enumerate(records):
            if r == b"data" and i + 1 < len(records) and isinstance(records[i + 1], dict):
                data_dict = records[i + 1]
            if r == b"metadata" and i + 1 < len(records) and isinstance(records[i + 1], dict):
                meta_dict = records[i + 1]

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
            stat, p = _paired_test(paired[CECP_RAW], paired[other_raw], "greater", test)
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
        alt = "less" if question in LOWER_QUESTIONS else "greater"
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


def _draw_significance_brackets(ax, sig_results, bar_positions, x_display_order, hue_order, group_key="layout_display", tick_frac=0.04):
    """Overlay significance brackets for CECP vs each other method."""
    cecp_display = DISPLAY_NAMES.get(CECP_RAW, CECP_RAW)
    y_top = ax.get_ylim()[1]
    step = 0.09 * y_top
    tick_len = tick_frac * y_top
    max_bracket_y = y_top

    for x_idx, x_display in enumerate(x_display_order):
        group_results = [r for r in sig_results if r[group_key] == x_display]
        if not group_results:
            continue

        cecp_key = (x_idx, cecp_display)
        cecp_x = bar_positions.get(cecp_key, (None, None))[0]
        if cecp_x is None:
            continue

        # Sort by distance between bars so shorter spans are drawn lower
        group_results = sorted(
            group_results,
            key=lambda r: abs(cecp_x - bar_positions.get((x_idx, r["other_display"]), (cecp_x, 0))[0]),
        )

        # Base height = tallest bar in this x group
        group_bar_heights = [bar_positions.get((x_idx, t), (0, 0))[1] for t in hue_order]
        base_y = max(group_bar_heights) + 0.5 * step

        for bracket_idx, r in enumerate(group_results):
            other_display = r["other_display"]
            label = r["label"]
            other_key = (x_idx, other_display)
            other_x = bar_positions.get(other_key, (None, None))[0]
            if other_x is None:
                continue

            y_bracket = base_y + bracket_idx * step
            max_bracket_y = max(max_bracket_y, y_bracket)

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
                fontsize=9 if label == "ns" else 11,
                color=color,
            )

    ax.set_ylim(top=max_bracket_y + 1.5 * step)


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------

def generate_summary_csv(gameplay_df, prolific_ids, overcooked_exp, out_path="results/user_episode_summary.csv"):
    """Build and return a per-episode summary DataFrame, also saving it as a CSV."""
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

def plot_return(df, out_path="figures/avg_return_by_layout.png", test="wilcoxon",
                show_both_layouts=True, show_significance=True):
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

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.barplot(
        data=df, x="layout", y="total_return", hue="tag",
        hue_order=hue_order, order=layout_order, ax=ax,
    )

    bar_positions = {}
    for hue_idx, container in enumerate(ax.containers):
        tag_display = hue_order[hue_idx]
        for layout_idx, bar in enumerate(container):
            bar_positions[(layout_idx, tag_display)] = (
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
            )

    if show_significance:
        _draw_significance_brackets(ax, sig_results, bar_positions, layout_order, hue_order, group_key="layout_display")

    ax.set_title("Average Return by Layout")
    ax.set_xlabel("Layout")
    ax.set_ylabel("Average Return")
    ax.legend(title="Tag", loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.replace(".png", ".svg"))
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_survey(df, out_path="figures/avg_survey_by_question.png", test="wilcoxon",
                show_significance=True):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if show_significance:
        sig_results = _run_wilcoxon_tests_survey(df, test=test)
        for r in sig_results:
            direction = "lower" if r["question"] in LOWER_QUESTIONS else "higher"
            print(
                f"  {r['question']:20s}  CECP {direction} than {r['other_display']:10s}"
                f"  p={r['p_raw']:.4f}  {r['label']}"
            )
    else:
        sig_results = []

    df = df.copy()
    df["tag"] = df["tag"].map(lambda t: DISPLAY_NAMES.get(t, t))
    df["layout"] = df["layout"].map(lambda l: OC_ORIGINAL_LAYOUT_NAMES.get(l, l))
    hue_order = [t for t in DISPLAY_ORDERING if t in df["tag"].unique()]
    question_order = list(QUESTION_LABELS.values())
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(
        data=df,
        x="question",
        y="score",
        hue="tag",
        hue_order=hue_order,
        order=question_order,
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

    if show_significance:
        _draw_significance_brackets(
            ax, sig_results, bar_positions, question_order, hue_order, group_key="question",
        )

    ax.set_title("Average Survey Rating by Question")
    ax.set_xlabel("Question")
    ax.set_ylabel("Rating (1–5)")
    ax.set_ylim(0, 5.5)
    ax.legend(title="Tag")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
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
        help="Path to data directory (default: flydata/ next to this script).",
    )
    parser.add_argument(
        "--exclude-test",
        action="store_true",
        help="Exclude participants whose prolific ID contains 'test'.",
    )
    parser.add_argument(
        "--test",
        choices=["wilcoxon", "ttest"],
        default="wilcoxon",
        help="Statistical test to use for significance brackets (default: wilcoxon).",
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

    data_dir = (
        os.path.join(os.path.dirname(__file__), args.data_dir)
        if args.data_dir
        else DATA_DIR
    )

    user_ids = load_user_ids(data_dir)
    overcooked_exp = load_overcooked_experience(data_dir)
    prolific_ids = load_prolific_ids(data_dir)

    print("Users (user_id -> prolific_id, overcooked_experience):")
    for uid in user_ids:
        prolific = prolific_ids.get(uid, "")
        exp = overcooked_exp.get(uid, "N/A")
        print(f"  user={uid}: prolific_id={prolific!r}, overcooked_exp={exp}")

    users_with_msgpack = load_users_with_msgpack(data_dir)
    # Exclude users with no prolific ID only if they also lack a user_data msgpack file.
    # Users whose msgpack exists but lost its data are still confirmed participants.
    no_prolific_ids = {uid for uid in user_ids if not prolific_ids.get(uid) and uid not in users_with_msgpack}
    if no_prolific_ids:
        print(f"Excluding {len(no_prolific_ids)} user(s) with no prolific ID: {sorted(no_prolific_ids)}")

    incomplete_ids = load_incomplete_user_ids(data_dir)
    if incomplete_ids:
        print(f"Excluding {len(incomplete_ids)} incomplete user(s) (missing survey tags): {sorted(incomplete_ids)}")

    exclude_ids = no_prolific_ids | incomplete_ids
    if args.exclude_test:
        test_ids = load_test_user_ids(data_dir)
        if test_ids:
            print(f"Excluding {len(test_ids)} test user(s): {sorted(test_ids)}")
        exclude_ids |= test_ids

    n_total = len(user_ids)
    n_excluded = len(exclude_ids)
    print(f"Users: {n_total} loaded, {n_excluded} excluded, {n_total - n_excluded} analyzed")

    gameplay_df = parse_gameplay_files(data_dir, exclude_ids=exclude_ids)
    print(f"Gameplay rows: {len(gameplay_df)}")
    print(gameplay_df.groupby(["tag", "layout"])["total_return"].mean().to_string())

    summary_df = generate_summary_csv(gameplay_df, prolific_ids, overcooked_exp)
    print(f"\nUser episode summary ({len(summary_df)} rows):")
    print(summary_df.to_string())

    survey_df = parse_survey_files(data_dir, exclude_ids=exclude_ids)
    print(f"\nSurvey rows: {len(survey_df)}")
    print(survey_df.groupby(["tag", "question"])["score"].mean().to_string())

    plot_return(gameplay_df, test=args.test, show_both_layouts=args.both_layouts, show_significance=args.significance)
    plot_survey(survey_df, test=args.test, show_significance=args.significance)
