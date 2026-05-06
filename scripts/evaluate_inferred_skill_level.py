#!/usr/bin/env python3
"""
Evaluate inferred partner skill level from model readout latents during human gameplay.

For each user gameplay with the given model tag, replays the trajectory through the
AI model to read out the 'readout' layer hidden state at each timestep.  The readout
is projected onto a pre-computed skill axis (slope + intercept) to produce a per-step
inferred skill level, which is then averaged over a configurable timestep window.

Two scatter / correlation plots are produced (matching the style of
evaluate_human_action_prediction._plot_return_vs_accuracy_scatter):
  1. Normalized per-episode return (per layout) vs. inferred skill level.
  2. Self-reported Overcooked experience (1–5) vs. inferred skill level.

Skill axis files are loaded from:
  /sep-rep-learning/results/evaluate_vib_latents/hksyr2i5/agent_<id>/skill_axis.npy
  /sep-rep-learning/results/evaluate_vib_latents/hksyr2i5/agent_<id>/skill_axis_intercept.npy

Usage:
    python scripts/evaluate_inferred_skill_level.py --tag oc_cecp_pred_1000
    python scripts/evaluate_inferred_skill_level.py --tag oc_cecp_pred_1000 --latent-start-t 50 --latent-end-t 100
"""

import argparse
import glob
import json
import os
import re
import sys

import jax
if not hasattr(jax, "tree_map"):
    jax.tree_map = jax.tree_util.tree_map

import jax.numpy as jnp
import matplotlib
import numpy as np
from flax import serialization
from scipy.stats import pearsonr, linregress

matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analyze_data import load_valid_users, load_overcooked_experience, LIKERT_SCALE
from coop_foraging_scripts.evaluation_utils import load_model_checkpoint, write_csv
from nicewebrl.utils import read_all_records_sync
from nicewebrl.nicejax import TimestepWrapper
from web_app.constants import ORIGINAL_5_TAGS
from web_app.experiment import create_environment

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "newflydata")
DEFAULT_MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")
DEFAULT_OUTPUT_DIR = "./results/evaluate_inferred_skill_level"
SKILL_AXIS_BASE_DIR = "/sep-rep-learning/results/evaluate_vib_latents/hksyr2i5"

OBS_KEY = 'grid_2d'
REWARD_MATCH_TOL = 0.5
ACTION_ARRAY = [3, 1, 2, 0, 4, 5]


# ---------------------------------------------------------------------------
# Gameplay file helpers
# ---------------------------------------------------------------------------

def find_gameplay_files(user_id: str, tag: str, data_dir: str) -> list[str]:
    pattern = os.path.join(data_dir, f"gameplay_user={user_id}_{tag}_*.json")
    return sorted(glob.glob(pattern))


def parse_filename(filepath: str, tag: str) -> dict:
    name = os.path.basename(filepath)
    match = re.match(
        rf"gameplay_user=(\w+)_{re.escape(tag)}_(.+)_agent(\d+)\.json$",
        name,
    )
    if not match:
        raise ValueError(f"Could not parse filename: {name}")
    return {
        "user_id": match.group(1),
        "layout": match.group(2),
        "agent_id": int(match.group(3)),
    }


# ---------------------------------------------------------------------------
# Per-layout environment cache
# ---------------------------------------------------------------------------

_ENV_CACHE: dict[str, tuple] = {}


def _get_env(layout: str) -> tuple:
    if layout in _ENV_CACHE:
        return _ENV_CACHE[layout]
    env, _ = create_environment(layout, "overcooked")
    jax_env = TimestepWrapper(env, autoreset=True, reset_w_batch_dim=False, use_params=False)
    template_ts = jax_env.reset(jax.random.key(0), {})
    step_fn = jax.jit(jax_env.step)
    dummy_action = {'agent_0': jnp.array(4), 'agent_1': jnp.array(4)}
    _ = step_fn(jax.random.key(0), template_ts, dummy_action)
    _ENV_CACHE[layout] = (template_ts, step_fn)
    return template_ts, step_fn


# ---------------------------------------------------------------------------
# Model loading with per-session cache
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict[str, tuple] = {}


def _load_model(tag: str, layout: str, agent_id: int, models_dir: str) -> tuple:
    cache_key = f"{tag}_{layout}_{agent_id}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    load_tag = f"{tag}_{layout}" if tag in ORIGINAL_5_TAGS else tag
    tag_dir = os.path.join(models_dir, load_tag)
    matches = glob.glob(os.path.join(tag_dir, f"model_*-{agent_id}-best"))
    if not matches:
        raise FileNotFoundError(f"No checkpoint for agent_id={agent_id} in {tag_dir}")

    model_fn, model_state = load_model_checkpoint(matches[0])
    model_fn.get_action_distribution = jax.jit(model_fn.get_action_distribution)
    model_fn.forward_with_aux = jax.jit(model_fn.forward_with_aux)

    _MODEL_CACHE[cache_key] = (model_fn, model_state)
    return model_fn, model_state


# ---------------------------------------------------------------------------
# Skill axis loading with cache
# ---------------------------------------------------------------------------

_SKILL_AXIS_CACHE: dict[int, tuple] = {}


def _load_skill_axis(agent_id: int) -> tuple[np.ndarray, np.ndarray]:
    if agent_id in _SKILL_AXIS_CACHE:
        return _SKILL_AXIS_CACHE[agent_id]

    agent_dir = os.path.join(SKILL_AXIS_BASE_DIR, f"agent_{agent_id}")
    axis_path = os.path.join(agent_dir, "skill_axis_per_timestep.npy")
    intercept_path = os.path.join(agent_dir, "skill_axis_intercept_per_timestep.npy")

    if not os.path.exists(axis_path) or not os.path.exists(intercept_path):
        raise FileNotFoundError(
            f"Per-timestep skill axis files not found for agent_id={agent_id} at {agent_dir}"
        )

    skill_axis = np.load(axis_path)       # (T_axis, d)
    intercepts = np.load(intercept_path)  # (T_axis,)

    _SKILL_AXIS_CACHE[agent_id] = (skill_axis, intercepts)
    return skill_axis, intercepts


# ---------------------------------------------------------------------------
# Human-agent slot inference via reward matching (same as evaluate_human_action_prediction)
# ---------------------------------------------------------------------------

def _recorded_reward(records: list, template_ts) -> float:
    return sum(
        float(serialization.from_bytes(template_ts, r['data']['timestep']).reward)
        for r in records
    )


def _simulate_reward(
    records: list,
    template_ts,
    step_fn,
    model_fn,
    model_state,
    human_agent: int,
) -> float:
    ac = model_fn.actor_critic_fn
    obs_h, obs_w, obs_c = ac.feature_extractor_config.obs_space[OBS_KEY].shape
    model_key = f"agent_{1 - human_agent}"

    rnn_state = model_fn.init_rnn_state(jax.random.key(0), batch_size=1)
    rng = jax.random.key(42)
    total_reward = 0.0

    for i in range(len(records) - 1):
        rec = records[i]
        next_rec = records[i + 1]

        ts = serialization.from_bytes(template_ts, rec['data']['timestep'])
        obs_raw = np.array(ts.observation[model_key])
        obs_grid = obs_raw.reshape(1, 1, obs_h, obs_w, -1).astype(np.float32)
        runtime_c = obs_grid.shape[-1]
        if runtime_c < obs_c:
            obs_grid = jnp.pad(obs_grid, [(0, 0), (0, 0), (0, 0), (0, 0), (0, obs_c - runtime_c)])
        elif runtime_c > obs_c:
            obs_grid = obs_grid[..., :obs_c]

        rng, rng_act = jax.random.split(rng)
        next_rnn, pi = model_fn.get_action_distribution(
            rng_act, model_state, rnn_state, {OBS_KEY: obs_grid}
        )
        model_action = int(pi.mode()['all'].flatten()[0])

        raw_ak = int(next_rec['data']['action_idx'])
        human_action = ACTION_ARRAY[raw_ak] if raw_ak >= 0 else 4

        a0 = human_action if human_agent == 0 else model_action
        a1 = model_action if human_agent == 0 else human_action
        action_dict = {'agent_0': jnp.array(a0), 'agent_1': jnp.array(a1)}

        rng, rng_step = jax.random.split(rng)
        new_ts = step_fn(rng_step, ts, action_dict)
        total_reward += float(new_ts.reward)
        rnn_state = next_rnn

    return total_reward


def _infer_human_agent(
    filepath: str,
    records: list,
    template_ts,
    step_fn,
    model_fn,
    model_state,
) -> tuple[int | None, bool]:
    fname = os.path.basename(filepath)
    recorded = _recorded_reward(records, template_ts)
    reward_0 = _simulate_reward(records, template_ts, step_fn, model_fn, model_state, human_agent=0)
    reward_1 = _simulate_reward(records, template_ts, step_fn, model_fn, model_state, human_agent=1)

    match_0 = abs(reward_0 - recorded) < REWARD_MATCH_TOL
    match_1 = abs(reward_1 - recorded) < REWARD_MATCH_TOL

    if match_0 and match_1:
        print(
            f"  AMBIGUOUS {fname}: recorded={recorded:.1f}, "
            f"h=0={reward_0:.1f}, h=1={reward_1:.1f} — skipping."
        )
        return None, True
    elif not match_0 and not match_1:
        print(
            f"  NO MATCH {fname}: recorded={recorded:.1f}, "
            f"h=0={reward_0:.1f}, h=1={reward_1:.1f} — skipping."
        )
        return None, True
    else:
        inferred = 0 if match_0 else 1
        matched_r = reward_0 if match_0 else reward_1
        print(f"  Inferred human_agent={inferred} (recorded={recorded:.1f}, matched={matched_r:.1f})")
        return inferred, False


# ---------------------------------------------------------------------------
# Episode replay with readout-latent extraction
# ---------------------------------------------------------------------------

def replay_episode_for_skill(
    filepath: str,
    tag: str,
    models_dir: str,
    latent_start_t: int,
    latent_end_t: int,
    skill_outlier_threshold: float | None,
    verbose: bool,
) -> dict | None:
    """Replay one episode and return per-step inferred skill levels.

    Returns dict with 'mean_skill', 'episode_return', etc., or None on failure.
    """
    parsed = parse_filename(filepath, tag)
    layout = parsed["layout"]
    agent_id = parsed["agent_id"]

    model_fn, model_state = _load_model(tag, layout, agent_id, models_dir)
    skill_axis, intercepts = _load_skill_axis(agent_id)

    template_ts, step_fn = _get_env(layout)
    records = [
        r for r in read_all_records_sync(filepath)
        if "data" in r and "timestep" in r.get("data", {})
    ]

    human_agent, is_ambiguous = _infer_human_agent(
        filepath, records, template_ts, step_fn, model_fn, model_state
    )
    if is_ambiguous:
        return None

    model_agent_key = f"agent_{1 - human_agent}"
    ac = model_fn.actor_critic_fn
    model_obs_h, model_obs_w, model_obs_c = ac.feature_extractor_config.obs_space[OBS_KEY].shape

    rnn_state = model_fn.init_rnn_state(jax.random.key(0), batch_size=1)
    rng = jax.random.key(42)

    skill_levels: list[float] = []

    for i in range(len(records) - 1):
        rec = records[i]
        ts = serialization.from_bytes(template_ts, rec["data"]["timestep"])

        obs_raw = np.array(ts.observation[model_agent_key])
        obs_grid = obs_raw.reshape(1, 1, model_obs_h, model_obs_w, -1).astype(np.float32)
        runtime_c = obs_grid.shape[-1]
        if runtime_c < model_obs_c:
            obs_grid = jnp.pad(
                obs_grid, [(0, 0), (0, 0), (0, 0), (0, 0), (0, model_obs_c - runtime_c)]
            )
        elif runtime_c > model_obs_c:
            obs_grid = obs_grid[..., :model_obs_c]

        rng, rng_fwd = jax.random.split(rng)
        next_rnn, _, _, info = model_fn.forward_with_aux(
            rng_fwd, model_state, rnn_state, {OBS_KEY: obs_grid}
        )

        if 'readout' in info and i < len(skill_axis):
            readout = np.array(info['readout']).reshape(-1)      # (d,)
            skill_levels.append(float(readout @ skill_axis[i] + intercepts[i]))
        else:
            skill_levels.append(float('nan'))

        rnn_state = next_rnn

        if verbose and (i % 50 == 0 or i == len(records) - 2):
            print(f"    step {i + 1}/{len(records) - 1}  skill={skill_levels[-1]:.3f}")

    skill_arr = np.array(skill_levels)
    T = len(skill_arr)
    t0 = min(latent_start_t, T)
    t1 = min(latent_end_t, T)
    if t0 >= t1:
        print(f"  WARNING: episode has {T} steps, window [{latent_start_t},{latent_end_t}) clamps to empty; using full episode.")
        mean_skill = float(np.nanmean(skill_arr))
    else:
        mean_skill = float(np.nanmean(skill_arr[t0:t1]))

    if skill_outlier_threshold is not None and mean_skill > skill_outlier_threshold:
        print(f"  Outlier skill={mean_skill:.3f} > {skill_outlier_threshold}, excluding episode.")
        return None

    episode_return = _recorded_reward(records, template_ts)

    return {
        "user_id": parsed["user_id"],
        "layout": layout,
        "agent_id": agent_id,
        "human_agent": human_agent,
        "mean_skill": mean_skill,
        "episode_return": episode_return,
        "skill_levels": skill_arr,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_scatter(
    x_vals: np.ndarray,
    skill_vals: np.ndarray,
    x_label: str,
    title: str,
    output_path: str,
) -> None:
    valid = ~(np.isnan(x_vals) | np.isnan(skill_vals))
    x = x_vals[valid]
    y = skill_vals[valid]
    if len(x) < 3:
        print(f"  Skipping plot (only {len(x)} valid points): {output_path}")
        return

    r, p = pearsonr(x, y)
    slope, intercept, _, _, _ = linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 200)
    p_str = f'p = {p:.3f}' if p >= 0.001 else 'p < 0.001'

    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    ax.scatter(x, y, alpha=0.75, edgecolors='white', linewidth=0.5)
    ax.plot(x_line, slope * x_line + intercept, color='firebrick', linewidth=1.5)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Inferred Skill Level")
    ax.set_title(f"{title}\nPearson's r = {r:.3f}, {p_str}")
    ax.grid(alpha=0.25)

    fig.savefig(output_path, dpi=150)
    fig.savefig(os.path.splitext(output_path)[0] + '.svg')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def _plot_skill_by_timestep(
    skill_trajectories: list[np.ndarray],
    labels: list[str],
    latent_start_t: int,
    latent_end_t: int,
    output_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    cmap = plt.get_cmap('tab10')
    for i, (traj, label) in enumerate(zip(skill_trajectories, labels)):
        timesteps = np.arange(len(traj))
        ax.plot(timesteps, traj, color=cmap(i % cmap.N), linewidth=1.5, alpha=0.85, label=label)

    ax.axvspan(latent_start_t, latent_end_t, color='gray', alpha=0.12, label=f'averaging window [{latent_start_t}, {latent_end_t})')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Inferred Skill Level')
    ax.set_title('Inferred partner skill level over time (example users)')
    ax.legend(loc='best', fontsize='small')
    ax.grid(alpha=0.25)

    fig.savefig(output_path, dpi=150)
    fig.savefig(os.path.splitext(output_path)[0] + '.svg')
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate inferred partner skill level from readout latents."
    )
    parser.add_argument("--tag", required=True,
                        help="Experiment tag, e.g. oc_cecp_pred_1000.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--models-dir", default=DEFAULT_MODELS_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--latent-start-t", type=int, default=50,
                        help="Start of timestep window for averaging inferred skill (inclusive, default 50).")
    parser.add_argument("--latent-end-t", type=int, default=100,
                        help="End of timestep window for averaging inferred skill (exclusive, default 100).")
    parser.add_argument("--skill-outlier-threshold", type=float, default=10.0,
                        help="Exclude episodes whose mean inferred skill exceeds this value (default 10). Pass 'inf' to disable.")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    valid_users = load_valid_users(args.data_dir)
    if not valid_users:
        print(f"No users with msgpack files found in {args.data_dir}")
        sys.exit(1)
    print(f"Found {len(valid_users)} users with msgpack files.")

    oc_exp_raw = load_overcooked_experience(args.data_dir)

    user_rows: list[dict] = []
    skipped_rows: list[dict] = []

    for user_id, prolific_id in sorted(valid_users.items()):
        files = find_gameplay_files(user_id, args.tag, args.data_dir)
        if not files:
            continue

        for filepath in files:
            print(f"\nReplaying {os.path.basename(filepath)} ...")
            try:
                result = replay_episode_for_skill(
                    filepath=filepath,
                    tag=args.tag,
                    models_dir=args.models_dir,
                    latent_start_t=args.latent_start_t,
                    latent_end_t=args.latent_end_t,
                    skill_outlier_threshold=args.skill_outlier_threshold,
                    verbose=args.verbose,
                )
            except FileNotFoundError as e:
                print(f"  SKIPPED ({e})")
                result = None

            if result is None:
                skipped_rows.append({
                    "user_id": user_id,
                    "prolific_id": prolific_id,
                    "file": os.path.basename(filepath),
                })
                continue

            exp_raw = oc_exp_raw.get(user_id, "N/A")
            exp_score = LIKERT_SCALE.get(exp_raw) if isinstance(exp_raw, str) else None

            user_rows.append({
                "user_id": user_id,
                "prolific_id": prolific_id,
                "layout": result["layout"],
                "agent_id": result["agent_id"],
                "human_agent": result["human_agent"],
                "episode_return": result["episode_return"],
                "mean_skill": result["mean_skill"],
                "skill_levels": result["skill_levels"],
                "overcooked_experience_raw": exp_raw,
                "overcooked_experience_score": exp_score,
            })
            print(
                f"  return={result['episode_return']:.1f}  "
                f"mean_skill={result['mean_skill']:.3f}  "
                f"oc_exp={exp_raw!r}"
            )

    if skipped_rows:
        print(f"\nSkipped {len(skipped_rows)} episode(s):")
        for row in skipped_rows:
            print(f"  user={row['user_id']}  {row['file']}")

    if not user_rows:
        print("No episodes processed.")
        sys.exit(1)

    # Min-max normalize returns within each layout separately.
    ep_returns = np.array([r["episode_return"] for r in user_rows], dtype=float)
    mean_skills = np.array([r["mean_skill"] for r in user_rows], dtype=float)
    layouts = [r["layout"] for r in user_rows]

    normalized_returns = np.empty_like(ep_returns)
    for layout in set(layouts):
        idx = np.array([i for i, l in enumerate(layouts) if l == layout])
        r = ep_returns[idx]
        r_min, r_max = r.min(), r.max()
        normalized_returns[idx] = (
            (r - r_min) / (r_max - r_min) if r_max > r_min else np.zeros_like(r)
        )

    oc_scores = np.array(
        [
            r["overcooked_experience_score"]
            if r["overcooked_experience_score"] is not None
            else float('nan')
            for r in user_rows
        ],
        dtype=float,
    )

    output_dir = os.path.join(args.output_dir, args.tag)
    os.makedirs(output_dir, exist_ok=True)

    for row, nr in zip(user_rows, normalized_returns):
        row["normalized_return"] = float(nr)

    csv_rows = [{k: v for k, v in row.items() if k != "skill_levels"} for row in user_rows]
    write_csv(os.path.join(output_dir, "episodes.csv"), csv_rows)
    if skipped_rows:
        write_csv(os.path.join(output_dir, "skipped.csv"), skipped_rows)

    summary = {
        "tag": args.tag,
        "num_episodes_processed": len(user_rows),
        "num_episodes_skipped": len(skipped_rows),
        "latent_window": [args.latent_start_t, args.latent_end_t],
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if not args.no_plots:
        _plot_scatter(
            x_vals=normalized_returns,
            skill_vals=mean_skills,
            x_label="Normalized Per-Episode Return",
            title="Inferred partner skill vs. episode return",
            output_path=os.path.join(output_dir, "skill_vs_return_scatter.png"),
        )
        _plot_scatter(
            x_vals=oc_scores,
            skill_vals=mean_skills,
            x_label="Self-Reported Overcooked Experience (1–5)",
            title="Inferred partner skill vs. self-reported experience",
            output_path=os.path.join(output_dir, "skill_vs_experience_scatter.png"),
        )

        n_example = min(5, len(user_rows))
        step = max(1, len(user_rows) // n_example)
        example_rows = user_rows[::step][:n_example]
        _plot_skill_by_timestep(
            skill_trajectories=[r["skill_levels"] for r in example_rows],
            labels=[f"{r['user_id']} / {r['layout']}" for r in example_rows],
            latent_start_t=args.latent_start_t,
            latent_end_t=args.latent_end_t,
            output_path=os.path.join(output_dir, "skill_by_timestep_examples.png"),
        )

    print(f"\nDone. Results written to: {output_dir}")
    print(f"  Episodes processed: {len(user_rows)}, skipped: {len(skipped_rows)}")


if __name__ == "__main__":
    main()
