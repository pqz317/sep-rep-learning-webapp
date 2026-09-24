"""Per-user key-press distributions, and the list of participants who did not really play.

For every valid participant (same exclusions as analyze_data.py), computes the
distribution over the six keys in each episode.  An episode is flagged when a single
key makes up more than --top-key-threshold (default 45%) of its presses, and a user
is flagged if any of their episodes is.  Flagged users are written to
prolific_data/<dataset>/bad_users.csv, which the other analysis scripts drop when run
with --exclude-bad-users; rerunning this script regenerates that list.

The 45% cutoff reproduces a manual review of the resub figures exactly: every user
judged bad had an episode above 51%, and no other user had one above 41%.

Outputs:
  prolific_data/<dataset>/bad_users.csv                    user_id, prolific_id
  figures/<dataset>/action_distributions/user=<uid>.png   one per user
  figures/<dataset>/action_distributions/bad_users.png    flagged users, one row each
  figures/<dataset>/action_distributions/top_key_summary.png
  results/action_distributions/<dataset>/episode_metrics.csv
  results/action_distributions/<dataset>/user_flags.csv

Usage:
    python scripts/analyze_action_distributions.py --data-dir resub
"""

import argparse
import math
import os
import re
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_data import (
    STEP_FIRST,
    STEP_LAST,
    TUTORIAL_DESC,
    _decode_float32_ext,
    _decode_uint8_ext,
    _ext_hook,
    _msgpack_records,
    load_excluded_user_ids,
    load_prolific_ids,
    load_user_ids,
)
from constants import BAD_USERS_FILENAME, dataset_figures_dir, dataset_results_dir, resolve_data_dir
from display_names import DISPLAY_NAMES, OC_ORIGINAL_LAYOUT_NAMES

import msgpack

# action_idx indexes the web app's action_keys (web_app/experiment.py):
# [ArrowLeft, ArrowDown, ArrowRight, ArrowUp, s, space].  Label by the physical key;
# the logged action_name is the game-internal name and does not match the arrows.
KEY_LABELS = ["←", "↓", "→", "↑", "S\nstay", "Space\ninteract"]
N_KEYS = len(KEY_LABELS)
STAY_IDX = 4
SHARE_COLS = [f"share_{i}" for i in range(N_KEYS)]

# Flag an episode when one key makes up more than this share of its presses.
DEFAULT_TOP_KEY_THRESHOLD = 0.45

# Colors (reference palette): one series, plus the reserved status color for flags.
BAR_COLOR = "#2a78d6"
REF_COLOR = "#0b0b0b"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
NEUTRAL_DOT = "#9a9994"
FLAG_COLOR = "#d03b3b"  # status: critical
GRID_COLOR = "#e4e3df"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_episodes(path):
    """Split one gameplay file into episodes of key presses.

    Uses the same episode boundaries as analyze_data.parse_gameplay_files: split on
    FIRST timesteps, close an episode on its LAST timestep, and skip the trailing
    timer record that repeats the LAST timestep.  Timer records (action_idx < 0) are
    not key presses and are dropped.

    Returns (tag, agent_id, layout, episodes), where each episode is a dict with
    keys actions (list[int]), return, complete, start_time; or None for tutorial
    or unparseable files.
    """
    episodes = []
    tag = agent_id = layout = None
    for item in _msgpack_records(path):
        if not isinstance(item, dict):
            continue
        meta = item.get(b"metadata", {})
        if meta.get(b"type") != b"EnvStage":
            continue
        bm = meta.get(b"block_metadata", {})
        if bm.get(b"desc") == TUTORIAL_DESC or bm.get(b"tag") is None:
            return None
        tag = bm[b"tag"].decode()
        agent_id = bm.get(b"agent_id")
        m = re.match(r".+ on (.+)", bm.get(b"desc", b"").decode())
        layout = m.group(1) if m else None

        data = item.get(b"data", {})
        if not isinstance(data, dict):
            continue
        reward = 0.0
        step_type = None
        ts_bytes = data.get(b"timestep")
        if isinstance(ts_bytes, bytes) and ts_bytes:
            try:
                ts = msgpack.unpackb(ts_bytes, raw=True, strict_map_key=False, ext_hook=_ext_hook)
                if isinstance(ts.get(b"reward"), tuple):
                    reward = _decode_float32_ext(ts[b"reward"])
                if isinstance(ts.get(b"step_type"), tuple):
                    step_type = _decode_uint8_ext(ts[b"step_type"])
            except Exception:
                pass

        if (not episodes or step_type == STEP_FIRST
                or (episodes[-1]["complete"] and step_type != STEP_LAST)):
            seen = data.get(b"image_seen_time")
            episodes.append({
                "actions": [], "return": 0.0, "complete": False,
                "start_time": seen.decode() if isinstance(seen, bytes) else None,
            })
        ep = episodes[-1]
        if ep["complete"]:
            continue
        ep["return"] += reward
        action_idx = data.get(b"action_idx")
        if isinstance(action_idx, int) and 0 <= action_idx < N_KEYS:
            ep["actions"].append(action_idx)
        if step_type == STEP_LAST:
            ep["complete"] = True

    if tag is None or layout is None or not episodes:
        return None
    return tag, agent_id, layout, episodes


def episode_metrics(actions):
    """Key-share vector and summary statistics for one episode's key presses."""
    a = np.asarray(actions, dtype=int)
    counts = np.bincount(a, minlength=N_KEYS)
    share = counts / max(len(a), 1)
    nz = share[share > 0]
    entropy = float(-(nz * np.log(nz)).sum() / np.log(N_KEYS)) if len(a) else float("nan")
    return {
        "n_presses": int(len(a)),
        "entropy": entropy,
        "top_key_share": float(share.max()) if len(a) else float("nan"),
        "repeat_rate": float((a[1:] == a[:-1]).mean()) if len(a) > 1 else float("nan"),
        "stay_share": float(share[STAY_IDX]),
        **{f"share_{i}": float(share[i]) for i in range(N_KEYS)},
    }


def collect_episode_metrics(data_dir, user_ids):
    """One row per (user, gameplay file) with key-press metrics.

    Keeps only the latest complete episode per file (the latest partial episode if
    none completed), matching the episode analyze_data.py scores.
    """
    rows = []
    for fname in sorted(os.listdir(data_dir)):
        m = re.match(r"gameplay_user=(\d+)_", fname)
        if not m or m.group(1) not in user_ids:
            continue
        parsed = parse_episodes(os.path.join(data_dir, fname))
        if parsed is None:
            continue
        tag, agent_id, layout, episodes = parsed
        complete = [i for i, ep in enumerate(episodes) if ep["complete"]]
        i = complete[-1] if complete else len(episodes) - 1
        ep = episodes[i]
        if not ep["actions"]:
            continue
        rows.append({
            "user_id": m.group(1),
            "tag": tag,
            "agent_id": agent_id,
            "layout": layout,
            "attempt": i,
            "n_attempts_in_file": len(episodes),
            "complete": ep["complete"],
            "start_time": ep["start_time"],
            "episode_return": ep["return"],
            **episode_metrics(ep["actions"]),
        })
    df = pd.DataFrame(rows)
    df["play_order"] = (
        df.sort_values("start_time").groupby("user_id").cumcount() + 1
    ).reindex(df.index)
    return df.sort_values(["user_id", "play_order"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Flagging
# ---------------------------------------------------------------------------

def flag_users(ep_df, top_key_threshold, prolific_ids):
    """Mark episodes where one key exceeds the threshold, and users with any such episode."""
    ep_df = ep_df.copy()
    ep_df["flagged"] = ep_df["top_key_share"] > top_key_threshold
    g = ep_df.groupby("user_id")
    users = pd.DataFrame({
        "prolific_id": [prolific_ids.get(u, "") for u in g.groups],
        "layout": g["layout"].first(),
        "n_episodes": g.size(),
        "n_flagged_episodes": g["flagged"].sum(),
        "max_top_key_share": g["top_key_share"].max(),
        "median_entropy": g["entropy"].median(),
        "mean_return": g["episode_return"].mean(),
    })
    users["flagged"] = users["n_flagged_episodes"] > 0
    users = users.reset_index().rename(columns={"index": "user_id"})
    # Numeric user order, so the written list is stable across runs.
    users = users.sort_values("user_id", key=lambda s: s.astype(int)).reset_index(drop=True)
    return ep_df, users


def write_bad_users(users, path):
    """Write flagged users to `path`, reporting how the list changed from the one on disk."""
    bad = users.loc[users["flagged"], ["user_id", "prolific_id"]]
    if os.path.exists(path):
        old = set(pd.read_csv(path, dtype={"user_id": str})["user_id"].str.strip())
        new = set(bad["user_id"])
        if old == new:
            print(f"Bad-user list unchanged ({len(new)} users): {path}")
        else:
            print(f"Bad-user list changed: added {sorted(new - old) or 'none'}, "
                  f"removed {sorted(old - new) or 'none'}")
    bad.to_csv(path, index=False)
    print(f"Wrote {len(bad)} bad users to {path}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _style_axis(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)


def _legend_handles():
    return (
        [plt.Rectangle((0, 0), 1, 1, color=BAR_COLOR),
         Line2D([], [], color=REF_COLOR, linewidth=1.5, linestyle=(0, (2, 1.5)))],
        ["This episode", "All users (mean)"],
    )


def _draw_episode(ax, ep, ref_share, ymax, top_key_threshold, compact=False):
    """Bar chart of one episode's key shares, with the all-user mean as reference ticks."""
    x = np.arange(N_KEYS)
    ax.bar(x, ep[SHARE_COLS].values.astype(float), width=0.72, color=BAR_COLOR,
           edgecolor="white", linewidth=2)
    ax.hlines(ref_share, x - 0.36, x + 0.36, colors=REF_COLOR, linewidth=1.5,
              linestyles=(0, (2, 1.5)))
    ax.set_xticks(x, KEY_LABELS)
    ax.set_ylim(0, ymax)
    _style_axis(ax)

    agent = DISPLAY_NAMES.get(ep["tag"], ep["tag"])
    if compact:
        title = f"{ep['play_order']}. {agent}  ·  return {ep['episode_return']:.0f}"
        stats = f"top key {ep['top_key_share']:.2f}   H {ep['entropy']:.2f}   repeat {ep['repeat_rate']:.2f}"
    else:
        attempt = (f" (attempt {ep['attempt'] + 1}/{ep['n_attempts_in_file']})"
                   if ep["n_attempts_in_file"] > 1 else "")
        title = f"{ep['play_order']}. {agent}{attempt}  ·  return {ep['episode_return']:.0f}"
        stats = (f"top key {ep['top_key_share']:.2f}   H {ep['entropy']:.2f}   "
                 f"repeat {ep['repeat_rate']:.2f}   n {ep['n_presses']}")
    ax.set_title(title, fontsize=10.5, color=TEXT_PRIMARY, loc="left", pad=18)
    ax.text(0, 1.02, stats, transform=ax.transAxes, fontsize=8.5, color=TEXT_SECONDARY)
    if ep["flagged"]:
        ax.text(1.0, 1.13, f"▲ top key > {top_key_threshold:.0%}", transform=ax.transAxes,
                ha="right", fontsize=9, fontweight="bold", color=FLAG_COLOR)


def plot_user(user, user_eps, ref_share, top_key_threshold, out_path):
    n = len(user_eps)
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.6 * nrows + 1.0),
                             sharey=True, squeeze=False)
    ymax = max(0.5, float(user_eps[SHARE_COLS].values.max()) + 0.08)

    for ax, (_, ep) in zip(axes.flat, user_eps.iterrows()):
        _draw_episode(ax, ep, ref_share, ymax, top_key_threshold)

    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Share of key presses", fontsize=9.5, color=TEXT_SECONDARY)

    layout = OC_ORIGINAL_LAYOUT_NAMES.get(user["layout"], user["layout"])
    pid = user["prolific_id"] or "unknown prolific ID"
    fig.suptitle(f"User {user['user_id']}  ·  {layout}", x=0.01, ha="left",
                 fontsize=14, fontweight="bold", color=TEXT_PRIMARY)
    fig.text(0.01, 1 - 0.55 / fig.get_figheight(),
             f"{pid}   max top key {user['max_top_key_share']:.2f}   "
             f"mean return {user['mean_return']:.0f}",
             fontsize=9.5, color=TEXT_SECONDARY)
    if user["flagged"]:
        fig.text(0.99, 1 - 0.55 / fig.get_figheight(),
                 f"▲ BAD USER ({user['n_flagged_episodes']}/{user['n_episodes']} episodes "
                 f"with top key > {top_key_threshold:.0%})",
                 ha="right", fontsize=10.5, fontweight="bold", color=FLAG_COLOR)
    handles, labels = _legend_handles()
    fig.legend(handles=handles, labels=labels, loc="lower center", ncol=2,
               frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1 - 0.75 / fig.get_figheight()))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_user_grid(users, ep_df, ref_share, top_key_threshold, title, out_path):
    """One row per user, one column per episode in play order, on a shared y-scale."""
    rows = [(user, ep_df[ep_df["user_id"] == user["user_id"]]) for _, user in users.iterrows()]
    ncols = max(len(eps) for _, eps in rows)
    fig, axes = plt.subplots(len(rows), ncols, figsize=(3.4 * ncols + 2.2, 2.9 * len(rows) + 1.2),
                             sharey=True, squeeze=False)
    ymax = max(0.5, float(ep_df[ep_df["user_id"].isin(users["user_id"])][SHARE_COLS].values.max()) + 0.08)

    for row_axes, (user, user_eps) in zip(axes, rows):
        for ax, (_, ep) in zip(row_axes, user_eps.iterrows()):
            _draw_episode(ax, ep, ref_share, ymax, top_key_threshold, compact=True)
        for ax in row_axes[len(user_eps):]:
            ax.set_visible(False)
        layout = OC_ORIGINAL_LAYOUT_NAMES.get(user["layout"], user["layout"])
        row_axes[0].text(-0.32, 0.5, f"{user['user_id']}", transform=row_axes[0].transAxes,
                         ha="right", va="bottom", fontsize=11, fontweight="bold", color=TEXT_PRIMARY)
        row_axes[0].text(-0.32, 0.46,
                         f"{layout}\n{user['prolific_id'] or 'unknown prolific ID'}\n"
                         f"mean return {user['mean_return']:.0f}",
                         transform=row_axes[0].transAxes, ha="right", va="top",
                         fontsize=8.5, color=TEXT_SECONDARY, linespacing=1.4)
        row_axes[0].set_ylabel("Key share", fontsize=9, color=TEXT_SECONDARY)

    fig.suptitle(title, x=0.01, ha="left", fontsize=14, fontweight="bold", color=TEXT_PRIMARY)
    handles, labels = _legend_handles()
    fig.legend(handles=handles, labels=labels, loc="lower center", ncol=2,
               frameon=False, fontsize=9.5)
    fig.tight_layout(rect=(0.1, 0.5 / fig.get_figheight(), 1, 1 - 0.5 / fig.get_figheight()),
                     h_pad=2.2)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_summary(ep_df, users, top_key_threshold, out_path):
    """Top-key share of every episode, one column per user, sorted by each user's maximum."""
    order = users.sort_values("max_top_key_share", ascending=False)["user_id"].tolist()
    pos = {u: i for i, u in enumerate(order)}
    fig, ax = plt.subplots(figsize=(16, 6))
    rng = np.random.default_rng(0)
    xs = ep_df["user_id"].map(pos).values + rng.uniform(-0.18, 0.18, len(ep_df))
    flagged_ep = ep_df["flagged"].values
    ax.scatter(xs[~flagged_ep], ep_df["top_key_share"].values[~flagged_ep], s=36,
               color=NEUTRAL_DOT, edgecolor="white", linewidth=1, zorder=3, label="Episode")
    ax.scatter(xs[flagged_ep], ep_df["top_key_share"].values[flagged_ep], s=44, marker="^",
               color=FLAG_COLOR, edgecolor="white", linewidth=1, zorder=3,
               label=f"Episode with top key > {top_key_threshold:.0%}")
    ax.axhline(top_key_threshold, color=TEXT_SECONDARY, linewidth=1, linestyle="--", zorder=2)
    ax.text(1.005, top_key_threshold, f"threshold {top_key_threshold:g}", va="center",
            fontsize=8.5, color=TEXT_SECONDARY, transform=ax.get_yaxis_transform(), clip_on=False)
    _style_axis(ax)

    ax.set_ylabel("Share of presses on the most-pressed key", fontsize=10, color=TEXT_SECONDARY)
    ax.set_title(
        f"Most-pressed key's share per episode, {len(order)} users sorted by their maximum  ·  "
        f"▲ users are flagged (any episode above {top_key_threshold:.0%})",
        loc="left", fontsize=12, color=TEXT_PRIMARY,
    )
    ax.legend(loc="upper right", frameon=False, fontsize=9)

    flagged = set(users.loc[users["flagged"], "user_id"])
    ax.set_xticks(range(len(order)), [f"▲ {u}" if u in flagged else u for u in order],
                  rotation=90, fontsize=8)
    for tick, u in zip(ax.get_xticklabels(), order):
        if u in flagged:
            tick.set_color(TEXT_PRIMARY)
            tick.set_fontweight("bold")
    ax.set_xlim(-0.7, len(order) - 0.3)
    ax.set_xlabel("User", fontsize=10, color=TEXT_SECONDARY)
    fig.tight_layout(rect=(0, 0, 0.94, 1))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", default=None,
                        help="Dataset name ('resub', 'newflydata') or path (default: resub).")
    parser.add_argument("--exclude-test", action="store_true",
                        help="Exclude participants whose prolific ID contains 'test'.")
    parser.add_argument("--top-key-threshold", type=float, default=DEFAULT_TOP_KEY_THRESHOLD,
                        help="Flag a user if, in at least one episode, a single key makes up more "
                             f"than this share of their presses (default: {DEFAULT_TOP_KEY_THRESHOLD}).")
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    fig_dir = dataset_figures_dir("action_distributions", data_dir=data_dir)
    res_dir = dataset_results_dir("action_distributions", data_dir=data_dir)
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    print(f"Data directory: {data_dir}")

    exclude_ids = load_excluded_user_ids(data_dir, exclude_test=args.exclude_test)
    user_ids = set(load_user_ids(data_dir)) - exclude_ids
    prolific_ids = load_prolific_ids(data_dir)
    print(f"Valid users: {len(user_ids)}")

    ep_df = collect_episode_metrics(data_dir, user_ids)
    ep_df, users = flag_users(ep_df, args.top_key_threshold, prolific_ids)
    print(f"Episodes: {len(ep_df)} across {ep_df['user_id'].nunique()} users")

    ep_df.to_csv(os.path.join(res_dir, "episode_metrics.csv"), index=False)
    users.to_csv(os.path.join(res_dir, "user_flags.csv"), index=False)
    write_bad_users(users, os.path.join(data_dir, BAD_USERS_FILENAME))

    ref_share = ep_df[SHARE_COLS].mean().values
    for _, user in users.iterrows():
        user_eps = ep_df[ep_df["user_id"] == user["user_id"]]
        plot_user(user, user_eps, ref_share, args.top_key_threshold,
                  os.path.join(fig_dir, f"user={user['user_id']}.png"))
    plot_summary(ep_df, users, args.top_key_threshold, os.path.join(fig_dir, "top_key_summary.png"))

    bad = users[users["flagged"]]
    if len(bad):
        plot_user_grid(bad, ep_df, ref_share, args.top_key_threshold,
                       f"Bad users ({len(bad)}): key-press share per episode  ·  "
                       f"flagged when one key > {args.top_key_threshold:.0%} of presses in any episode",
                       os.path.join(fig_dir, "bad_users.png"))

    cols = ["user_id", "prolific_id", "layout", "n_flagged_episodes", "n_episodes",
            "max_top_key_share", "median_entropy", "mean_return"]
    print(f"\nBad users: one key > {args.top_key_threshold:.0%} of presses in any episode ({len(bad)}):")
    print(bad[cols].round(3).to_string(index=False) if len(bad) else "  none")
    closest = users[~users["flagged"]].nlargest(3, "max_top_key_share")
    print("\nClosest unflagged users:")
    print(closest[cols].round(3).to_string(index=False))
    print(f"\nFigures: {fig_dir}\nTables:  {res_dir}")


if __name__ == "__main__":
    main()
