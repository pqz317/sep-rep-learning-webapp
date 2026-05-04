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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
from display_names import DISPLAY_NAMES, OC_ORIGINAL_LAYOUT_NAMES, DISPLAY_ORDERING

DATA_DIR = os.path.join(os.path.dirname(__file__), "flydata")

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
            last = records[-1] if records else {}
            if not isinstance(last, dict):
                continue
            user_storage = last.get(b"user_storage", {})
            exp = user_storage.get(b"overcooked_experience", b"")
            if isinstance(exp, bytes):
                exp = exp.decode(errors="replace")
            result[user_id] = exp or "N/A"
        except Exception:
            pass
    return result


def load_prolific_ids(data_dir):
    """Return a dict mapping user_id -> prolific_id (empty string if unavailable)."""
    result = {}
    for fname in os.listdir(data_dir):
        if not fname.startswith("user_data_") or not fname.endswith(".msgpack"):
            continue
        user_id = fname[len("user_data_"):-len(".msgpack")]
        path = os.path.join(data_dir, fname)
        try:
            records = _msgpack_records(path)
            last = records[-1] if records else {}
            if not isinstance(last, dict):
                continue
            prolific_id = last.get(b"episode_metadata", {}).get(b"prolific_id", b"")
            if isinstance(prolific_id, bytes):
                prolific_id = prolific_id.decode(errors="replace")
            result[user_id] = prolific_id
        except Exception:
            pass
    return result


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
            rows.append({"tag": tag, "layout": layout, "total_return": total_reward})

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
            rows.append({"tag": tag, "layout": layout, "question": label, "score": score})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_return(df, out_path="figures/avg_return_by_layout.png"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df = df.copy()
    df["tag"] = df["tag"].map(lambda t: DISPLAY_NAMES.get(t, t))
    df["layout"] = df["layout"].map(lambda l: OC_ORIGINAL_LAYOUT_NAMES.get(l, l))
    hue_order = [t for t in DISPLAY_ORDERING if t in df["tag"].unique()]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df, x="layout", y="total_return", hue="tag", hue_order=hue_order, ax=ax)
    ax.set_title("Average Return by Layout")
    ax.set_xlabel("Layout")
    ax.set_ylabel("Average Return")
    ax.legend(title="Tag")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_survey(df, out_path="figures/avg_survey_by_question.png"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
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
    ax.set_title("Average Survey Rating by Question")
    ax.set_xlabel("Question")
    ax.set_ylabel("Rating (1–5)")
    ax.set_ylim(0, 5.5)
    ax.legend(title="Tag")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
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

    no_prolific_ids = {uid for uid in user_ids if not prolific_ids.get(uid)}
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

    survey_df = parse_survey_files(data_dir, exclude_ids=exclude_ids)
    print(f"\nSurvey rows: {len(survey_df)}")
    print(survey_df.groupby(["tag", "question"])["score"].mean().to_string())

    plot_return(gameplay_df)
    plot_survey(survey_df)
