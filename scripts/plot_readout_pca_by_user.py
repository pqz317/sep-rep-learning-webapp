#!/usr/bin/env python3
"""
Replay human gameplay episodes with a specific agent, collect readout-layer
hidden states at each timestep, perform PCA, and plot trajectories in PC space
colored by each user's self-reported Overcooked experience.

Usage:
    python scripts/plot_readout_pca_by_user.py
    python scripts/plot_readout_pca_by_user.py --tag oc_cecp_pred_1000 --agent-id 2
"""

import argparse
import csv
import glob
import os
import re
import sys
from math import ceil, sqrt

import jax
if not hasattr(jax, "tree_map"):
    jax.tree_map = jax.tree_util.tree_map

import jax.numpy as jnp
import matplotlib
import numpy as np
from flax import serialization

matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analyze_data import LIKERT_SCALE
from coop_foraging_scripts.evaluation_utils import load_model_checkpoint
from nicewebrl.utils import read_all_records_sync
from nicewebrl.nicejax import TimestepWrapper
from web_app.constants import ORIGINAL_5_TAGS
from web_app.experiment import create_environment

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "newflydata")
DEFAULT_MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")
DEFAULT_SUMMARY_CSV = os.path.join(os.path.dirname(__file__), "..", "results", "user_episode_summary.csv")
DEFAULT_OUTPUT_DIR = "./results/readout_pca_by_user"

OBS_KEY = 'grid_2d'
REWARD_MATCH_TOL = 0.5
ACTION_ARRAY = [3, 1, 2, 0, 4, 5]

# Ordered experience labels for consistent coloring
EXPERIENCE_ORDER = ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree", "N/A"]


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def load_summary_csv(csv_path: str) -> list[dict]:
    with open(csv_path, newline='') as f:
        return list(csv.DictReader(f))


def filter_summary(rows: list[dict], tag: str, agent_id: int) -> list[dict]:
    return [
        r for r in rows
        if r['agent_tag'] == tag and int(r['agent_id']) == agent_id
    ]


# ---------------------------------------------------------------------------
# Environment and model cache (same as evaluate_inferred_skill_level.py)
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
# Gameplay file helpers (same as evaluate_inferred_skill_level.py)
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
# Human slot inference (same as evaluate_inferred_skill_level.py)
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
        print(f"  AMBIGUOUS {fname}: recorded={recorded:.1f}, h=0={reward_0:.1f}, h=1={reward_1:.1f} — skipping.")
        return None, True
    elif not match_0 and not match_1:
        print(f"  NO MATCH {fname}: recorded={recorded:.1f}, h=0={reward_0:.1f}, h=1={reward_1:.1f} — skipping.")
        return None, True
    else:
        inferred = 0 if match_0 else 1
        matched_r = reward_0 if match_0 else reward_1
        print(f"  Inferred human_agent={inferred} (recorded={recorded:.1f}, matched={matched_r:.1f})")
        return inferred, False


# ---------------------------------------------------------------------------
# Episode replay: collect readout hidden states
# (same replay logic as evaluate_inferred_skill_level.py, returns raw readouts)
# ---------------------------------------------------------------------------

def replay_episode_for_readouts(
    filepath: str,
    tag: str,
    models_dir: str,
    verbose: bool,
) -> np.ndarray | None:
    """Replay one episode and return readout states as (T, d) array, or None on failure."""
    parsed = parse_filename(filepath, tag)
    layout = parsed["layout"]
    agent_id = parsed["agent_id"]

    model_fn, model_state = _load_model(tag, layout, agent_id, models_dir)
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

    readouts: list[np.ndarray] = []

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

        if 'readout' in info:
            readouts.append(np.array(info['readout']).reshape(-1))
        else:
            print(f"  WARNING: no 'readout' key in info at step {i}; skipping episode.")
            return None

        rnn_state = next_rnn

        if verbose and (i % 50 == 0 or i == len(records) - 2):
            print(f"    step {i + 1}/{len(records) - 1}")

    if not readouts:
        return None
    return np.stack(readouts, axis=0)  # (T, d)


# ---------------------------------------------------------------------------
# PCA (same approach as evaluate_vib_latents.py)
# ---------------------------------------------------------------------------

def fit_pca(latents_by_run: np.ndarray, num_pcs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    latents_by_run: (n_runs, T, d)
    Returns: pc_scores_traj (T, n_runs, num_pcs), pc_axes (d, num_pcs),
             explained_var_ratio (d,), latent_centered (d, T*n_runs)
    """
    n_runs, n_timesteps, n_units = latents_by_run.shape
    latent_matrix = np.transpose(latents_by_run, (2, 1, 0)).reshape(n_units, n_timesteps * n_runs)
    latent_centered = latent_matrix - latent_matrix.mean(axis=1, keepdims=True)

    print(f'Performing PCA on centered latent matrix of shape {latent_centered.shape}...')
    U, S, _ = np.linalg.svd(latent_centered, full_matrices=False)
    num_pcs = min(max(1, num_pcs), n_units)
    pc_axes = U[:, :num_pcs]
    pc_scores = pc_axes.T @ latent_centered
    pc_scores_traj = pc_scores.T.reshape(n_timesteps, n_runs, num_pcs)

    explained_var_ratio = (S ** 2) / np.maximum(np.sum(S ** 2), 1e-12)
    return pc_scores_traj, pc_axes, explained_var_ratio, latent_centered


# ---------------------------------------------------------------------------
# Plotting (same style as evaluate_vib_latents.py _plot_pca_trajectories_mean_with_subsample)
# ---------------------------------------------------------------------------

def _subsample_trajectories_per_class(
    pc_scores_traj: np.ndarray,
    label_ids: np.ndarray,
    n_per_class: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    classes = np.unique(label_ids)
    selected: list[int] = []
    for cls in classes:
        cls_indices = np.where(label_ids == cls)[0]
        n_sel = min(n_per_class, len(cls_indices))
        chosen = rng.choice(cls_indices, size=n_sel, replace=False)
        selected.extend(chosen.tolist())
    idx = np.asarray(selected)
    return pc_scores_traj[:, idx, :], label_ids[idx]


def plot_pca_trajectories_mean_with_subsample(
    pc_scores_traj: np.ndarray,
    label_ids: np.ndarray,
    label_names: list[str],
    output_path: str,
    subsample_n: int = 30,
    cmap: str = 'tab10',
) -> None:
    sub_traj, sub_y = _subsample_trajectories_per_class(pc_scores_traj, label_ids, subsample_n)

    n_timesteps, n_runs, num_pcs = sub_traj.shape
    n_pairs = min(max(num_pcs - 1, 0), 9)
    if n_pairs == 0:
        return

    n_cols = int(ceil(sqrt(n_pairs)))
    n_rows = int(ceil(n_pairs / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), constrained_layout=True)
    if isinstance(axes, np.ndarray):
        axes = axes.ravel()
    else:
        axes = np.asarray([axes])

    cmap_obj = plt.get_cmap(cmap)
    n_labels = max(len(label_names), 1)
    if cmap_obj.N <= 20:
        class_colors = [cmap_obj(i % cmap_obj.N) for i in range(len(label_names))]
    else:
        class_colors = [cmap_obj(0.2 + 0.8 * i / max(n_labels - 1, 1)) for i in range(len(label_names))]

    class_mean_trajs = []
    for cls_id in range(len(label_names)):
        mask = label_ids == cls_id
        if mask.sum() == 0:
            class_mean_trajs.append(None)
        else:
            class_mean_trajs.append(pc_scores_traj[:, mask, :].mean(axis=1))

    for pair_idx in range(n_pairs):
        ax = axes[pair_idx]
        pc_x = pair_idx
        pc_y = pair_idx + 1

        for run_idx in range(n_runs):
            cls = int(sub_y[run_idx])
            color = class_colors[cls]
            traj = sub_traj[:, run_idx, :]
            ax.plot(traj[:, pc_x], traj[:, pc_y], color=color, alpha=0.3, linewidth=1.0)

        for cls_id in range(len(label_names)):
            mean_traj = class_mean_trajs[cls_id]
            if mean_traj is None:
                continue
            color = class_colors[cls_id]
            ax.plot(mean_traj[:, pc_x], mean_traj[:, pc_y], color=color, linewidth=4.0)
            ax.scatter(mean_traj[0, pc_x], mean_traj[0, pc_y], color="black", s=70, zorder=5)

        ax.set_xlabel(f'PC{pc_x}')
        ax.set_ylabel(f'PC{pc_y}')
        ax.set_title(f'PC{pc_x} vs PC{pc_y}')
        ax.grid(alpha=0.25)

    for idx in range(n_pairs, len(axes)):
        axes[idx].axis('off')

    legend_handles = [
        plt.Line2D([0], [0], color=class_colors[i], lw=2, label=label_names[i])
        for i in range(len(label_names))
    ]
    axes[0].legend(handles=legend_handles, loc='best')

    fig.savefig(output_path, dpi=150)
    fig.savefig(os.path.splitext(output_path)[0] + '.svg')
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot readout-layer PCA trajectories per user, colored by Overcooked experience."
    )
    parser.add_argument("--tag", default="oc_cecp_pred_1000",
                        help="Agent tag to analyze (default: oc_cecp_pred_1000).")
    parser.add_argument("--agent-id", type=int, default=2,
                        help="Agent ID within the tag (default: 2).")
    parser.add_argument("--summary-csv", default=DEFAULT_SUMMARY_CSV,
                        help="Path to user_episode_summary.csv.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help="Directory containing gameplay JSON files.")
    parser.add_argument("--models-dir", default=DEFAULT_MODELS_DIR,
                        help="Directory containing model checkpoints.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-pcs", type=int, default=10,
                        help="Number of principal components to compute (default: 10).")
    parser.add_argument("--subsample-n", type=int, default=30,
                        help="Max trajectories per class to show in individual-trace plot (default: 30).")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Load summary CSV and filter to target tag + agent_id
    summary_rows = load_summary_csv(args.summary_csv)
    target_rows = filter_summary(summary_rows, args.tag, args.agent_id)
    if not target_rows:
        print(f"No rows in CSV for tag={args.tag!r} agent_id={args.agent_id}.")
        sys.exit(1)
    print(f"Found {len(target_rows)} CSV rows for tag={args.tag!r} agent_id={args.agent_id}.")

    # Build user_id -> overcooked_experience mapping
    user_experience: dict[str, str] = {
        r['user_id']: r['overcooked_experience'] for r in target_rows
    }

    # Collect readout trajectories
    run_readouts: list[np.ndarray] = []
    run_experience: list[str] = []

    for row in target_rows:
        user_id = row['user_id']
        experience = row['overcooked_experience']

        files = find_gameplay_files(user_id, args.tag, args.data_dir)
        # Filter to files matching this specific agent_id
        files = [f for f in files if parse_filename(f, args.tag)['agent_id'] == args.agent_id]

        if not files:
            print(f"  No gameplay file for user={user_id} tag={args.tag} agent_id={args.agent_id}")
            continue

        filepath = files[0]
        print(f"\nReplaying {os.path.basename(filepath)} (experience={experience!r}) ...")
        try:
            readouts = replay_episode_for_readouts(
                filepath=filepath,
                tag=args.tag,
                models_dir=args.models_dir,
                verbose=args.verbose,
            )
        except FileNotFoundError as e:
            print(f"  SKIPPED ({e})")
            readouts = None

        if readouts is None:
            continue

        run_readouts.append(readouts)
        run_experience.append(experience)
        print(f"  Collected {readouts.shape[0]} timesteps, d={readouts.shape[1]}")

    if len(run_readouts) < 2:
        print("Need at least 2 episodes for PCA.")
        sys.exit(1)

    # Truncate all trajectories to the minimum episode length
    min_T = min(r.shape[0] for r in run_readouts)
    print(f"\nTruncating all {len(run_readouts)} trajectories to min length T={min_T}.")
    run_readouts = [r[:min_T] for r in run_readouts]

    # Stack into (n_runs, T, d)
    latents_by_run = np.stack(run_readouts, axis=0)
    print(f"Latents shape: {latents_by_run.shape}")

    # Build class labels from overcooked_experience
    present_experiences = [e for e in EXPERIENCE_ORDER if e in set(run_experience)]
    exp_to_id = {name: i for i, name in enumerate(present_experiences)}
    label_ids = np.array([exp_to_id[e] for e in run_experience], dtype=int)

    # PCA
    pc_scores_traj, pc_axes, explained_var_ratio, _ = fit_pca(latents_by_run, args.num_pcs)
    num_pcs_used = pc_scores_traj.shape[2]
    print(f"Top-{num_pcs_used} explained variance ratios: "
          f"{[f'{v:.3f}' for v in explained_var_ratio[:num_pcs_used]]}")

    output_dir = os.path.join(args.output_dir, args.tag, f"agent_{args.agent_id}")
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, 'latents_by_run.npy'), latents_by_run)
    np.save(os.path.join(output_dir, 'pc_axes.npy'), pc_axes)
    np.save(os.path.join(output_dir, 'pc_scores_traj.npy'), pc_scores_traj)
    np.save(os.path.join(output_dir, 'explained_variance_ratio.npy'), explained_var_ratio)
    np.save(os.path.join(output_dir, 'label_ids.npy'), label_ids)

    import json
    with open(os.path.join(output_dir, 'label_names.json'), 'w') as f:
        json.dump(present_experiences, f, indent=2)

    meta_rows = [
        {'user_id': row['user_id'], 'overcooked_experience': exp, 'label_id': int(lid)}
        for row, exp, lid in zip(target_rows[:len(run_experience)], run_experience, label_ids)
    ]
    with open(os.path.join(output_dir, 'run_metadata.json'), 'w') as f:
        json.dump(meta_rows, f, indent=2)

    if not args.no_plots:
        plot_pca_trajectories_mean_with_subsample(
            pc_scores_traj=pc_scores_traj,
            label_ids=label_ids,
            label_names=present_experiences,
            output_path=os.path.join(output_dir, 'pca_trajectories_by_experience.png'),
            subsample_n=args.subsample_n,
        )

    print(f"\nDone. Results written to: {output_dir}")
    print(f"  Episodes: {len(run_readouts)}, experience labels: {present_experiences}")


if __name__ == "__main__":
    main()
