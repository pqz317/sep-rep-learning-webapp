"""Does return improve across the rounds a participant plays on their layout?

Each participant is assigned one layout and plays every agent tag on it, in a
per-user randomized order (see web_app/web_app.py: random.shuffle(tags)). That
order is recorded in each user's episode_metadata.tag_agent_pairs. This script
joins that play-order "slot" (1st, 2nd, ... round) onto the per-episode
returns already parsed by analyze_data.py, then checks for a within-session
learning/practice effect: do returns trend up from the first slot to the last?

The number of rounds follows the dataset: 4 for prolific_data/newflydata, 6 for
prolific_data/resub (which adds comedi_br and pace_br), so nothing here assumes a
fixed count.

Outputs (results/analyze_order_effect/<dataset>/):
  user_slot_returns.csv  - one row per (user, slot): layout, tag, agent_id, return
  order_summary.csv      - mean/SEM/n of return by (layout, slot)
  return_by_slot.png     - line plot of mean return vs. slot, by layout
  summary.md             - tables + trend statistics
"""

import argparse
import os
import re
import sys

import msgpack
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, ttest_1samp, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_data as ad
from constants import PROJ_ROOT, dataset_results_dir, resolve_data_dir
from display_names import DISPLAY_NAMES, OC_ORIGINAL_LAYOUT_NAMES

# Fixed categorical color order (validated pair, dataviz skill palette slots 1/2)
LAYOUT_COLORS = {
    "coord_ring_9": "#2a78d6",       # slot 1 blue
    "counter_circuit_9": "#eb6834",  # slot 2 orange
}


# ---------------------------------------------------------------------------
# Play-order extraction
# ---------------------------------------------------------------------------

def _find_order_meta(records):
    """Locate (layout, pairs) within a file's parsed msgpack records.

    Each record is normally a nested dict, e.g.
    {..., 'episode_metadata': {'session_layout':..., 'tag_agent_pairs':...}, ...}.
    Some saves put the keys at the top level of a record instead, and some carry
    `user_storage`'s `session_tag_agent_pairs` (same content, different key) when
    `episode_metadata` never made it into the save.  Scan for all three shapes and
    keep the last match, which is the most complete.
    """
    layout, pairs = None, None
    for r in records:
        if not isinstance(r, dict):
            continue
        if b"tag_agent_pairs" in r:
            layout, pairs = r.get(b"session_layout"), r.get(b"tag_agent_pairs")
        elif isinstance(r.get(b"episode_metadata"), dict) and b"tag_agent_pairs" in r[b"episode_metadata"]:
            em = r[b"episode_metadata"]
            layout, pairs = em.get(b"session_layout"), em.get(b"tag_agent_pairs")
        elif b"session_tag_agent_pairs" in r:
            layout, pairs = r.get(b"session_layout"), r.get(b"session_tag_agent_pairs")
    return layout, pairs


def load_play_order(data_dir, exclude_ids=None):
    """Return a DataFrame: user_id, layout, tag, agent_id, slot (1-indexed).

    Reads the per-user play order (tag_agent_pairs) from each
    user_data_*.msgpack file -- the order the experiment rounds were
    actually presented in.
    """
    rows = []
    for fname in os.listdir(data_dir):
        if not fname.startswith("user_data_") or not fname.endswith(".msgpack"):
            continue
        user_id = fname[len("user_data_"):-len(".msgpack")]
        if exclude_ids and user_id in exclude_ids:
            continue
        path = os.path.join(data_dir, fname)
        try:
            records = ad._msgpack_records(path)
            layout_bytes, pairs = _find_order_meta(records)
            if not pairs:
                print(f"  No play order found in {fname}")
                continue
            layout = layout_bytes.decode() if isinstance(layout_bytes, bytes) else layout_bytes
            for slot, pair in enumerate(pairs, start=1):
                tag_bytes, agent_id = pair[0], pair[1]
                tag = tag_bytes.decode() if isinstance(tag_bytes, bytes) else str(tag_bytes)
                rows.append({
                    "user_id": user_id,
                    "layout": layout,
                    "tag": tag,
                    "agent_id": agent_id,
                    "slot": slot,
                })
        except Exception as e:
            print(f"  Skipping {fname}: {e}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Trend statistics
# ---------------------------------------------------------------------------

def _ordinal(n):
    """1 -> '1st', 2 -> '2nd', 6 -> '6th'."""
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def per_user_slopes(df):
    """OLS slope of return on slot, fit separately per user. One episode per
    (user, slot) is expected since a user plays each tag exactly once."""
    slopes = []
    for user_id, sub in df.groupby("user_id"):
        sub = sub.sort_values("slot")
        if len(sub) < 2:
            continue
        x = sub["slot"].to_numpy(dtype=float)
        y = sub["episode_return"].to_numpy(dtype=float)
        slope = np.polyfit(x, y, 1)[0]
        layout = sub["layout"].iloc[0]
        slopes.append({"user_id": user_id, "layout": layout, "slope": slope})
    return pd.DataFrame(slopes)


def trend_tests(df, label):
    """One-sample tests (slope > 0) and paired first-vs-last-slot tests."""
    out = []
    slopes = per_user_slopes(df)
    if len(slopes) >= 3:
        t_stat, t_p = ttest_1samp(slopes["slope"], 0.0, alternative="greater")
        out.append({
            "group": label, "test": "one-sample t (mean slope > 0)",
            "n": len(slopes), "stat": t_stat, "p": t_p,
            "mean_slope": slopes["slope"].mean(),
        })

    pivot = df.pivot_table(index="user_id", columns="slot", values="episode_return")
    # First vs. last round, whatever the round count is for this dataset.
    if len(pivot.columns) >= 2:
        first, last = min(pivot.columns), max(pivot.columns)
        paired = pivot[[first, last]].dropna()
        if len(paired) >= 3:
            t_stat, t_p = ttest_rel(paired[last], paired[first], alternative="greater")
            w_stat, w_p = wilcoxon(paired[last], paired[first], alternative="greater")
            # Per-round change, so it stays comparable to the OLS slope above.
            mean_slope = (paired[last] - paired[first]).mean() / (last - first)
            out.append({
                "group": label, "test": f"paired t (slot{last} > slot{first})",
                "n": len(paired), "stat": t_stat, "p": t_p,
                "mean_slope": mean_slope,
            })
            out.append({
                "group": label, "test": f"wilcoxon (slot{last} > slot{first})",
                "n": len(paired), "stat": w_stat, "p": w_p,
                "mean_slope": mean_slope,
            })
    return out


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_return_by_slot(summary, out_path):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for layout in sorted(summary["layout"].unique()):
        sub = summary[summary["layout"] == layout].sort_values("slot")
        color = LAYOUT_COLORS.get(layout, "#52514e")
        label = OC_ORIGINAL_LAYOUT_NAMES.get(layout, layout)
        ax.errorbar(
            sub["slot"], sub["mean_return"], yerr=sub["sem"],
            marker="o", markersize=8, linewidth=2, capsize=4,
            color=color, label=label,
        )

    slots = sorted(int(s) for s in summary["slot"].unique())
    ax.set_xticks(slots)
    ax.set_xlabel(f"Round played (1st → {_ordinal(slots[-1])})")
    ax.set_ylabel("Mean episode return")
    ax.set_title("Return by play order, per layout")
    ax.grid(axis="y", color="#dddddd", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Dataset name ('resub', 'newflydata') or path to a data directory "
             "(default: prolific_data/resub/). Outputs are namespaced by its basename.",
    )
    ad.add_exclude_bad_users_arg(parser)
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    out_dir = dataset_results_dir("analyze_order_effect", data_dir=data_dir)
    print(f"Data directory: {data_dir}")
    os.makedirs(out_dir, exist_ok=True)

    user_ids = ad.load_user_ids(data_dir)
    exclude_ids = ad.load_excluded_user_ids(data_dir, exclude_bad_users=args.exclude_bad_users)
    print(f"Users: {len(user_ids)} loaded, {len(exclude_ids)} excluded, {len(user_ids) - len(exclude_ids)} analyzed")

    gameplay_df = ad.parse_gameplay_files(data_dir, exclude_ids=exclude_ids)
    order_df = load_play_order(data_dir, exclude_ids=exclude_ids)

    merged = gameplay_df.merge(
        order_df[["user_id", "tag", "agent_id", "slot"]],
        on=["user_id", "tag", "agent_id"], how="inner",
    )
    dropped = len(gameplay_df) - len(merged)
    if dropped:
        print(f"Warning: {dropped} gameplay rows had no matching play-order record and were dropped.")
    merged = merged.rename(columns={"total_return": "episode_return"})

    user_slot_path = os.path.join(out_dir, "user_slot_returns.csv")
    merged.sort_values(["user_id", "slot"]).to_csv(user_slot_path, index=False)
    print(f"Saved: {user_slot_path}")

    summary = (
        merged.groupby(["layout", "slot"])["episode_return"]
        .agg(mean_return="mean", sem=lambda s: s.std(ddof=1) / np.sqrt(len(s)), n="count")
        .reset_index()
    )
    summary_path = os.path.join(out_dir, "order_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")
    print(summary.to_string(index=False))

    plot_return_by_slot(summary, os.path.join(out_dir, "return_by_slot.png"))

    stats_rows = trend_tests(merged, "Both Layouts")
    for layout in sorted(merged["layout"].unique()):
        stats_rows.extend(trend_tests(merged[merged["layout"] == layout], OC_ORIGINAL_LAYOUT_NAMES.get(layout, layout)))
    stats_df = pd.DataFrame(stats_rows)
    stats_path = os.path.join(out_dir, "trend_stats.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"\nTrend statistics:\n{stats_df.to_string(index=False)}")

    write_summary_md(summary, stats_df, merged, os.path.join(out_dir, "summary.md"),
                     source_dir=out_dir)


def write_summary_md(summary, stats_df, merged, out_path, source_dir):
    n_tags = merged["tag"].nunique()
    last_slot = int(merged["slot"].max())
    lines = ["# Return by Play Order (Learning Effect Within a Session)", ""]
    lines.append(
        f"Each participant plays all {n_tags} agent tags on a single, randomly assigned "
        f"layout, in a per-user randomized order. `slot` = round number "
        f"(1st-{_ordinal(last_slot)}) that tag was played in."
    )
    n_users = merged["user_id"].nunique()
    rel_source = os.path.relpath(os.path.join(source_dir, "user_slot_returns.csv"), PROJ_ROOT)
    lines.append(f"Participants: {n_users} total. Source: `{rel_source}`.\n")

    for layout in sorted(summary["layout"].unique()):
        name = OC_ORIGINAL_LAYOUT_NAMES.get(layout, layout)
        lines.append(f"## {name}\n")
        lines.append("| Slot | Mean Return | SEM | n |")
        lines.append("|---|---:|---:|---:|")
        sub = summary[summary["layout"] == layout].sort_values("slot")
        for _, r in sub.iterrows():
            lines.append(f"| {int(r['slot'])} | {r['mean_return']:.1f} | {r['sem']:.1f} | {int(r['n'])} |")
        lines.append("")

    lines.append("## Trend statistics\n")
    lines.append(
        "`slope`/`mean_slope` is the average per-round change in return (units of return per round). "
        "One-sample t-test: is the mean per-user OLS slope of return-on-slot greater than 0? "
        "Paired tests: is round-4 return greater than round-1 return? One-sided (alternative='greater') "
        "since the question is specifically about improvement, not any change.\n"
    )
    lines.append("| Group | Test | n | stat | p | mean slope/round |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for _, r in stats_df.iterrows():
        lines.append(f"| {r['group']} | {r['test']} | {int(r['n'])} | {r['stat']:.3f} | {r['p']:.4f} | {r['mean_slope']:.2f} |")
    lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
