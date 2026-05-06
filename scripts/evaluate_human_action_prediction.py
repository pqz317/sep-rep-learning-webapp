#!/usr/bin/env python3
"""
Evaluate how accurately a model's next-state predictions capture human partner actions.

For all users with Prolific IDs, replays recorded gameplay episodes with the given model
tag and evaluates per-timestep action prediction accuracy using the model's predictive head.

The human's game-slot (agent 0 or 1) was randomly assigned each session and not stored in
the data.  This script infers it by replaying each episode twice — once assuming the human
was agent 0, once assuming agent 1 — and keeping the hypothesis whose simulated cumulative
reward matches the recorded reward.  Episodes where neither or both hypotheses match are
flagged and excluded from downstream analysis.

Usage:
    python scripts/evaluate_human_action_prediction.py --tag oc_cecp_pred_1000
    python scripts/evaluate_human_action_prediction.py --tag oc_cecp_pred_1000 --data-dir newflydata
"""

import argparse
import glob
import json
import os
import re
import sys

import jax
# JAX 0.6.0 removed jax.tree_map; restore it for jaxmarl which hasn't migrated.
if not hasattr(jax, "tree_map"):
    jax.tree_map = jax.tree_util.tree_map

import jax.numpy as jnp
import matplotlib
import numpy as np
from flax import serialization
from scipy.stats import pearsonr, linregress

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analyze_data import load_valid_users
from coop_foraging_scripts.evaluation_utils import load_model_checkpoint, write_csv
from nicewebrl.utils import read_all_records_sync
from nicewebrl.nicejax import TimestepWrapper
from web_app.constants import ORIGINAL_5_TAGS
from web_app.experiment import create_environment

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "newflydata")
DEFAULT_MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")
DEFAULT_OUTPUT_DIR = "./results/evaluate_human_action_prediction"

ACTION_NAMES_5CLASS = ['right', 'down', 'left', 'up', 'no-move']
NUM_ACTION_CLASSES = len(ACTION_NAMES_5CLASS)
OBS_KEY = 'grid_2d'
REWARD_MATCH_TOL = 0.5   # tolerance for comparing simulated vs recorded reward

# Keyboard-index → game-engine action mapping (from web_app/experiment.py).
# action_idx in records is a keyboard index 0-5 ('up','down','left','right','stay','interact');
# env.step expects the remapped game action via this array.
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
# Per-layout environment cache (template timestep + compiled step function)
# ---------------------------------------------------------------------------

_ENV_CACHE: dict[str, tuple] = {}   # layout -> (template_ts, step_fn)


def _get_env(layout: str) -> tuple:
    """Return (template_ts, step_fn) for a layout, building and JIT-compiling once."""
    if layout in _ENV_CACHE:
        return _ENV_CACHE[layout]
    env, _ = create_environment(layout, "overcooked")
    jax_env = TimestepWrapper(env, autoreset=True, reset_w_batch_dim=False, use_params=False)
    template_ts = jax_env.reset(jax.random.key(0), {})
    step_fn = jax.jit(jax_env.step)
    # Warm up JIT so the first real call is fast
    dummy_action = {'agent_0': jnp.array(4), 'agent_1': jnp.array(4)}
    _ = step_fn(jax.random.key(0), template_ts, dummy_action)
    _ENV_CACHE[layout] = (template_ts, step_fn)
    return template_ts, step_fn


# ---------------------------------------------------------------------------
# Model loading with per-session cache
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict[str, tuple] = {}


def load_model_for_prediction(tag: str, layout: str, agent_id: int, models_dir: str):
    """Load full model_fn and model_state needed for forward_with_prediction."""
    cache_key = f"{tag}_{layout}_{agent_id}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    load_tag = f"{tag}_{layout}" if tag in ORIGINAL_5_TAGS else tag
    tag_dir = os.path.join(models_dir, load_tag)
    matches = glob.glob(os.path.join(tag_dir, f"model_*-{agent_id}-best"))
    if not matches:
        raise FileNotFoundError(f"No checkpoint for agent_id={agent_id} in {tag_dir}")

    model_fn, model_state = load_model_checkpoint(matches[0])

    # Wrap forward methods in jax.jit once per model instance.  Subsequent
    # calls hit JAX's trace cache (keyed on the jit object + input shapes)
    # instead of re-entering XLA/LLVM compilation, preventing mmap region
    # accumulation across many episodes — same pattern as experiment.py's
    # _JitModelWrapper.
    model_fn.get_action_distribution = jax.jit(model_fn.get_action_distribution)
    if model_fn.has_predictive_head():
        model_fn.forward_with_prediction = jax.jit(model_fn.forward_with_prediction)

    _MODEL_CACHE[cache_key] = (model_fn, model_state)
    return model_fn, model_state


# ---------------------------------------------------------------------------
# Human-agent inference via reward matching
# ---------------------------------------------------------------------------

def _recorded_reward(records: list, template_ts) -> float:
    """Sum of per-step rewards from the actual gameplay records."""
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
    """Simulate the episode assuming the human was `human_agent` (0 or 1).

    Record[i] stores the pre-action game state for step i.  The human's action
    for step i is stored in record[i+1].action_idx (offset by one).  At each
    step the model's greedy action is computed from the state in record[i], and
    the environment is stepped with (human_action, model_greedy_action).

    For the correct human-agent hypothesis the model sees the same observations
    as in the real game, so it reproduces the same greedy actions and the
    accumulated reward matches the recorded total.
    """
    ac = model_fn.actor_critic_fn
    obs_h, obs_w, obs_c = ac.feature_extractor_config.obs_space[OBS_KEY].shape
    model_key = f"agent_{1 - human_agent}"

    rnn_state = model_fn.init_rnn_state(jax.random.key(0), batch_size=1)
    rng = jax.random.key(42)
    total_reward = 0.0

    # Only simulate steps 0..N-2; step i uses records[i] (state) + records[i+1] (human action).
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
        next_rnn, pi = model_fn.get_action_distribution(rng_act, model_state, rnn_state, {OBS_KEY: obs_grid})
        model_action = int(pi.mode()['all'].flatten()[0])

        # Human action for step i is stored in records[i+1].action_idx.
        # Negative action_idx signals a timer event (episode end) → treat as stay.
        raw_ak = int(next_rec['data']['action_idx'])
        human_action = ACTION_ARRAY[raw_ak] if raw_ak >= 0 else 4  # 4 = stay

        a0 = human_action if human_agent == 0 else model_action
        a1 = model_action if human_agent == 0 else human_action
        action_dict = {'agent_0': jnp.array(a0), 'agent_1': jnp.array(a1)}

        rng, rng_step = jax.random.split(rng)
        new_ts = step_fn(rng_step, ts, action_dict)
        total_reward += float(new_ts.reward)

        rnn_state = next_rnn

    return total_reward


def infer_human_agent(
    filepath: str,
    records: list,
    template_ts,
    step_fn,
    model_fn,
    model_state,
) -> tuple[int | None, bool]:
    """Infer which game slot the human occupied by reward matching.

    Simulates the episode under both hypotheses (human=0, human=1) with greedy
    model actions and compares each simulated reward to the recorded reward.

    Returns:
        (human_agent, is_ambiguous) where human_agent is 0, 1, or None.
        is_ambiguous is True when neither or both hypotheses match.
    """
    fname = os.path.basename(filepath)
    recorded = _recorded_reward(records, template_ts)
    reward_0 = _simulate_reward(records, template_ts, step_fn, model_fn, model_state, human_agent=0)
    reward_1 = _simulate_reward(records, template_ts, step_fn, model_fn, model_state, human_agent=1)

    match_0 = abs(reward_0 - recorded) < REWARD_MATCH_TOL
    match_1 = abs(reward_1 - recorded) < REWARD_MATCH_TOL

    if match_0 and match_1:
        print(
            f"  AMBIGUOUS {fname}: recorded={recorded:.1f}, "
            f"h=0 gives {reward_0:.1f}, h=1 gives {reward_1:.1f} — skipping."
        )
        return None, True
    elif not match_0 and not match_1:
        print(
            f"  NO MATCH {fname}: recorded={recorded:.1f}, "
            f"h=0 gives {reward_0:.1f}, h=1 gives {reward_1:.1f} — skipping."
        )
        return None, True
    else:
        inferred = 0 if match_0 else 1
        print(f"  Inferred human_agent={inferred} (recorded={recorded:.1f}, matched reward={reward_0 if match_0 else reward_1:.1f})")
        return inferred, False


# ---------------------------------------------------------------------------
# Action inference from predicted next-observation
# ---------------------------------------------------------------------------

def _infer_partner_action_from_obs(
    current_obs: np.ndarray,
    pred_next_obs: np.ndarray,
    partner_dir_channels: tuple[int, int] = (6, 10),
) -> np.ndarray:
    """Infer predicted partner action by comparing direction channels.

    Movement actions (0-3) change the partner's direction to the action index.
    Stay/interact leave direction unchanged and map to class 4 (no-move).

    Args:
        current_obs: shape (..., H, W, C)
        pred_next_obs: shape (..., H, W, C)

    Returns:
        predicted actions, shape (...) with values in {0, 1, 2, 3, 4}
    """
    ch_start, ch_end = partner_dir_channels
    current_dir_act = current_obs[..., ch_start:ch_end].sum(axis=(-3, -2))
    pred_dir_act = pred_next_obs[..., ch_start:ch_end].sum(axis=(-3, -2))
    current_dir = np.argmax(current_dir_act, axis=-1)
    pred_dir = np.argmax(pred_dir_act, axis=-1)
    return np.where(current_dir != pred_dir, pred_dir, 4)


def _map_action_to_5class(action: np.ndarray) -> np.ndarray:
    """Map 6-action space to 5-class: actions 0-4 unchanged, action 5 -> 4."""
    return np.minimum(action, 4)


# ---------------------------------------------------------------------------
# Accuracy computation
# ---------------------------------------------------------------------------

def _compute_accuracy_metrics(
    predicted_actions: np.ndarray,
    ground_truth_actions: np.ndarray,
) -> dict:
    """Compute accuracy metrics from (T, B) predicted vs ground truth arrays."""
    T, B = predicted_actions.shape
    correct = predicted_actions == ground_truth_actions

    per_timestep = correct.mean(axis=1)

    per_class_per_timestep = np.full((T, NUM_ACTION_CLASSES), np.nan)
    for cls in range(NUM_ACTION_CLASSES):
        cls_mask = ground_truth_actions == cls
        cls_counts = cls_mask.sum(axis=1)
        cls_correct = (correct & cls_mask).sum(axis=1)
        valid = cls_counts > 0
        per_class_per_timestep[valid, cls] = cls_correct[valid] / cls_counts[valid]

    confusion = np.zeros((NUM_ACTION_CLASSES, NUM_ACTION_CLASSES), dtype=np.int64)
    for true_cls in range(NUM_ACTION_CLASSES):
        for pred_cls in range(NUM_ACTION_CLASSES):
            confusion[true_cls, pred_cls] = int(
                ((ground_truth_actions == true_cls) & (predicted_actions == pred_cls)).sum()
            )

    return {
        'per_timestep': per_timestep,
        'per_class_per_timestep': per_class_per_timestep,
        'confusion_matrix': confusion,
        'overall_accuracy': float(correct.mean()),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_accuracy_vs_timestep(
    overall_accuracy_by_timestep: np.ndarray,
    per_user_accuracy: np.ndarray | None,
    user_names: list[str] | None,
    output_path: str,
) -> None:
    T = len(overall_accuracy_by_timestep)
    timesteps = np.arange(T)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

    if per_user_accuracy is not None and user_names is not None:
        cmap = plt.get_cmap('tab10')
        for i in range(per_user_accuracy.shape[0]):
            ax.plot(timesteps, per_user_accuracy[i], color=cmap(i % cmap.N),
                    alpha=0.4, linewidth=0.8, label=user_names[i])

    ax.plot(timesteps, overall_accuracy_by_timestep, color='black', linewidth=2.0, label='Overall')

    chance = 1.0 / NUM_ACTION_CLASSES
    ax.axhline(chance, linestyle='--', color='gray', linewidth=1.0,
               label=f'Chance ({chance:.2f})')

    ax.set_xlim(0, T - 1)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Prediction Accuracy')
    ax.set_title('Human Partner Action Prediction Accuracy vs Timestep')
    ax.grid(alpha=0.25)
    ax.legend(loc='best', fontsize='small', ncol=2)

    fig.savefig(output_path, dpi=150)
    fig.savefig(os.path.splitext(output_path)[0] + '.svg')
    plt.close(fig)


def _plot_confusion_matrix(confusion: np.ndarray, output_path: str) -> None:
    row_sums = confusion.sum(axis=1, keepdims=True)
    normalized = np.where(row_sums > 0, confusion / row_sums, 0.0)

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    sns.heatmap(
        normalized, annot=True, fmt='.2f', cmap='Blues',
        xticklabels=ACTION_NAMES_5CLASS, yticklabels=ACTION_NAMES_5CLASS,
        ax=ax, vmin=0.0, vmax=1.0,
    )
    ax.set_xlabel('Predicted Action')
    ax.set_ylabel('True Action')
    ax.set_title('Human Action Prediction Confusion Matrix (row-normalized)')
    fig.savefig(output_path, dpi=150)
    fig.savefig(os.path.splitext(output_path)[0] + '.svg')
    plt.close(fig)


def _plot_per_class_accuracy(per_class_per_timestep: np.ndarray, output_path: str) -> None:
    T = per_class_per_timestep.shape[0]
    timesteps = np.arange(T)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    cmap = plt.get_cmap('Set1')
    for cls in range(NUM_ACTION_CLASSES):
        vals = per_class_per_timestep[:, cls]
        valid = ~np.isnan(vals)
        if valid.any():
            ax.plot(timesteps[valid], vals[valid], color=cmap(cls),
                    linewidth=1.5, alpha=0.8, label=ACTION_NAMES_5CLASS[cls])

    chance = 1.0 / NUM_ACTION_CLASSES
    ax.axhline(chance, linestyle='--', color='gray', linewidth=1.0,
               label=f'Chance ({chance:.2f})')

    ax.set_xlim(0, T - 1)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Prediction Accuracy')
    ax.set_title('Per-Action-Class Human Prediction Accuracy')
    ax.grid(alpha=0.25)
    ax.legend(loc='best')

    fig.savefig(output_path, dpi=150)
    fig.savefig(os.path.splitext(output_path)[0] + '.svg')
    plt.close(fig)


def _plot_return_vs_accuracy_scatter(
    normalized_returns: np.ndarray,
    accuracies: np.ndarray,
    output_path: str,
) -> None:
    r, p = pearsonr(normalized_returns, accuracies)
    slope, intercept, _, _, _ = linregress(normalized_returns, accuracies)

    x_line = np.linspace(normalized_returns.min(), normalized_returns.max(), 200)

    p_str = f'p = {p:.3f}' if p >= 0.001 else 'p < 0.001'

    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    ax.scatter(normalized_returns, accuracies, alpha=0.75, edgecolors='white', linewidth=0.5)
    ax.plot(x_line, slope * x_line + intercept, color='firebrick', linewidth=1.5)
    ax.set_xlabel('Normalized Per-Epsidoe Return')
    ax.set_ylabel('Prediction Accuracy')
    ax.set_title(f"Human subject performance vs. CECP's movement prediction\nPearson's r = {r:.3f}, {p_str}")
    ax.grid(alpha=0.25)

    fig.savefig(output_path, dpi=150)
    fig.savefig(os.path.splitext(output_path)[0] + '.svg')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-episode replay with prediction
# ---------------------------------------------------------------------------

def replay_episode_for_prediction(
    filepath: str,
    tag: str,
    models_dir: str,
    verbose: bool,
) -> dict | None:
    """Replay one gameplay episode and collect predicted vs actual partner actions.

    Infers the human's game slot automatically via reward matching, then at each
    timestep:
      1. Samples the model's action from its policy.
      2. Calls forward_with_prediction(current_obs, model_action) -> pred_next_obs.
      3. Infers the predicted partner action via direction-channel comparison.
      4. Records the human's actual action from the gameplay file.

    Returns dict with 'predicted_actions' and 'gt_actions' arrays of shape (T,),
    or None if the model has no predictive head or the human slot is ambiguous.
    """
    parsed = parse_filename(filepath, tag)
    layout = parsed["layout"]
    agent_id = parsed["agent_id"]

    model_fn, model_state = load_model_for_prediction(tag, layout, agent_id, models_dir)

    if not model_fn.has_predictive_head():
        print(f"  WARNING: model for tag={tag} agent_id={agent_id} has no predictive head; skipping.")
        return None

    template_ts, step_fn = _get_env(layout)
    records = [
        r for r in read_all_records_sync(filepath)
        if "data" in r and "timestep" in r.get("data", {})
    ]

    # --- Infer which game slot the human occupied ---
    human_agent, is_ambiguous = infer_human_agent(
        filepath, records, template_ts, step_fn, model_fn, model_state
    )
    if is_ambiguous:
        return None

    model_agent_key = f"agent_{1 - human_agent}"
    ac = model_fn.actor_critic_fn
    model_obs_h, model_obs_w, model_obs_c = ac.feature_extractor_config.obs_space[OBS_KEY].shape

    rnn_state = model_fn.init_rnn_state(jax.random.key(0), batch_size=1)
    rng = jax.random.key(42)

    current_obs_list = []
    pred_next_obs_list = []
    gt_actions_list = []

    # Step i uses records[i] (model state/obs) + records[i+1].action_idx (human action).
    for i in range(len(records) - 1):
        rec = records[i]
        next_rec = records[i + 1]

        ts = serialization.from_bytes(template_ts, rec["data"]["timestep"])

        obs_raw = np.array(ts.observation[model_agent_key])  # (H, W, C)
        obs_grid = obs_raw.reshape(1, 1, model_obs_h, model_obs_w, -1).astype(np.float32)

        runtime_c = obs_grid.shape[-1]
        if runtime_c < model_obs_c:
            obs_grid = jnp.pad(
                obs_grid,
                [(0, 0), (0, 0), (0, 0), (0, 0), (0, model_obs_c - runtime_c)],
            )
        elif runtime_c > model_obs_c:
            obs_grid = obs_grid[..., :model_obs_c]

        obs_dict = {OBS_KEY: obs_grid}

        rng, rng_act, rng_sample, rng_pred = jax.random.split(rng, 4)

        # Advance RNN and sample model action
        next_rnn, pi = model_fn.get_action_distribution(rng_act, model_state, rnn_state, obs_dict)
        model_action = pi.sample(seed=rng_sample)

        # Predict next obs conditioned on model action (from same rnn_state, not next_rnn)
        _, _, _, pred_next_obs = model_fn.forward_with_prediction(
            rng_pred, model_state, rnn_state, obs_dict, model_action
        )

        current_obs_list.append(np.array(obs_grid[0, 0]))
        pred_next_obs_list.append(np.array(pred_next_obs[OBS_KEY][0, 0]))
        # Human action for step i is in records[i+1].action_idx (offset by one).
        # Map keyboard index → game action; negative = timer/stay.
        raw_ak = int(next_rec["data"]["action_idx"])
        gt_actions_list.append(ACTION_ARRAY[raw_ak] if raw_ak >= 0 else 4)

        rnn_state = next_rnn

        if verbose and (i % 50 == 0 or i == len(records) - 1):
            print(f"    step {i + 1}/{len(records) - 1}")

    current_obs_arr = np.stack(current_obs_list)      # (T, H, W, C)
    pred_next_obs_arr = np.stack(pred_next_obs_list)   # (T, H, W, C)
    gt_actions_arr = np.array(gt_actions_list, dtype=np.int32)  # (T,)

    predicted_actions = _infer_partner_action_from_obs(current_obs_arr, pred_next_obs_arr)
    gt_actions_5class = _map_action_to_5class(gt_actions_arr)

    episode_return = _recorded_reward(records, template_ts)

    return {
        "user_id": parsed["user_id"],
        "layout": layout,
        "agent_id": agent_id,
        "human_agent": human_agent,
        "predicted_actions": predicted_actions,
        "gt_actions": gt_actions_5class,
        "episode_return": episode_return,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model next-state prediction accuracy for human partner actions."
    )
    parser.add_argument(
        "--tag", required=True,
        help="Experiment tag identifying the model, e.g. oc_cecp_pred_1000.",
    )
    parser.add_argument(
        "--data-dir", default=DEFAULT_DATA_DIR,
        help=f"Directory containing gameplay files (default: {DEFAULT_DATA_DIR}).",
    )
    parser.add_argument(
        "--models-dir", default=DEFAULT_MODELS_DIR,
        help=f"Directory containing model checkpoints (default: {DEFAULT_MODELS_DIR}).",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help=f"Root directory for output files (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip saving plots.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    prolific_users = load_valid_users(args.data_dir)
    if not prolific_users:
        print(f"No users with msgpack files found in {args.data_dir}")
        sys.exit(1)
    print(f"Found {len(prolific_users)} users with msgpack files.")

    all_predicted: list[np.ndarray] = []
    all_gt: list[np.ndarray] = []
    per_user_accuracy: list[np.ndarray] = []
    user_rows: list[dict] = []
    user_names: list[str] = []
    skipped_rows: list[dict] = []

    for user_id, prolific_id in sorted(prolific_users.items()):
        files = find_gameplay_files(user_id, args.tag, args.data_dir)
        if not files:
            print(f"  No gameplay files for user {user_id} with tag={args.tag}, skipping.")
            continue

        for filepath in files:
            print(f"\nReplaying {os.path.basename(filepath)} ...")
            result = replay_episode_for_prediction(
                filepath=filepath,
                tag=args.tag,
                models_dir=args.models_dir,
                verbose=args.verbose,
            )
            if result is None:
                skipped_rows.append({"user_id": user_id, "prolific_id": prolific_id,
                                     "file": os.path.basename(filepath)})
                continue

            pred = result["predicted_actions"]
            gt = result["gt_actions"]
            ep_acc = float((pred == gt).mean())

            all_predicted.append(pred)
            all_gt.append(gt)
            per_user_accuracy.append((pred == gt).astype(float))

            label = f"{user_id}_{result['layout']}"
            user_names.append(label)
            user_rows.append({
                "user_id": user_id,
                "prolific_id": prolific_id,
                "layout": result["layout"],
                "agent_id": result["agent_id"],
                "human_agent": result["human_agent"],
                "label": label,
                "overall_accuracy": ep_acc,
                "episode_return": result["episode_return"],
            })
            print(f"  {len(pred)} steps | accuracy: {ep_acc:.4f}")

    if skipped_rows:
        print(f"\nSkipped {len(skipped_rows)} episode(s) due to ambiguous/unmatched human-agent inference:")
        for row in skipped_rows:
            print(f"  user={row['user_id']}  {row['file']}")

    if not all_predicted:
        print("No episodes processed.")
        sys.exit(1)

    lengths = [len(p) for p in all_predicted]
    if len(set(lengths)) > 1:
        min_len = min(lengths)
        print(f"Warning: varying episode lengths {set(lengths)}, truncating to {min_len}.")
        all_predicted = [p[:min_len] for p in all_predicted]
        all_gt = [g[:min_len] for g in all_gt]
        per_user_accuracy = [a[:min_len] for a in per_user_accuracy]

    predicted_stacked = np.stack(all_predicted, axis=1)   # (T, N)
    gt_stacked = np.stack(all_gt, axis=1)                 # (T, N)
    per_user_acc_arr = np.stack(per_user_accuracy, axis=0) # (N, T)

    overall_metrics = _compute_accuracy_metrics(predicted_stacked, gt_stacked)

    output_dir = os.path.join(args.output_dir, args.tag)
    os.makedirs(output_dir, exist_ok=True)

    T = overall_metrics['per_timestep'].shape[0]
    timestep_rows = []
    for t in range(T):
        row = {'timestep': int(t), 'overall_accuracy': float(overall_metrics['per_timestep'][t])}
        for cls in range(NUM_ACTION_CLASSES):
            val = overall_metrics['per_class_per_timestep'][t, cls]
            row[f'accuracy_{ACTION_NAMES_5CLASS[cls]}'] = float(val) if not np.isnan(val) else None
        timestep_rows.append(row)
    write_csv(os.path.join(output_dir, 'accuracy_by_timestep.csv'), timestep_rows)
    write_csv(os.path.join(output_dir, 'users.csv'), user_rows)
    if skipped_rows:
        write_csv(os.path.join(output_dir, 'skipped_episodes.csv'), skipped_rows)

    np.save(os.path.join(output_dir, 'confusion_matrix.npy'), overall_metrics['confusion_matrix'])
    np.save(os.path.join(output_dir, 'accuracy_by_timestep.npy'), overall_metrics['per_timestep'])
    np.save(os.path.join(output_dir, 'per_user_accuracy_by_timestep.npy'), per_user_acc_arr)

    # Min-max normalize episode returns within each layout separately
    ep_returns = np.array([row['episode_return'] for row in user_rows], dtype=float)
    ep_accuracies = np.array([row['overall_accuracy'] for row in user_rows], dtype=float)
    layouts = [row['layout'] for row in user_rows]
    normalized_returns = np.empty_like(ep_returns)
    for layout in set(layouts):
        idx = np.array([i for i, l in enumerate(layouts) if l == layout])
        r = ep_returns[idx]
        r_min, r_max = r.min(), r.max()
        normalized_returns[idx] = (r - r_min) / (r_max - r_min) if r_max > r_min else np.zeros_like(r)

    pearson_r, pearson_p = pearsonr(normalized_returns, ep_accuracies)

    if not args.no_plots:
        _plot_accuracy_vs_timestep(
            overall_accuracy_by_timestep=overall_metrics['per_timestep'],
            per_user_accuracy=per_user_acc_arr,
            user_names=user_names,
            output_path=os.path.join(output_dir, 'accuracy_vs_timestep.png'),
        )
        _plot_confusion_matrix(
            confusion=overall_metrics['confusion_matrix'],
            output_path=os.path.join(output_dir, 'confusion_matrix.png'),
        )
        _plot_per_class_accuracy(
            per_class_per_timestep=overall_metrics['per_class_per_timestep'],
            output_path=os.path.join(output_dir, 'per_class_accuracy.png'),
        )
        _plot_return_vs_accuracy_scatter(
            normalized_returns=normalized_returns,
            accuracies=ep_accuracies,
            output_path=os.path.join(output_dir, 'return_vs_accuracy_scatter.png'),
        )

    summary = {
        'tag': args.tag,
        'num_users': len(prolific_users),
        'num_episodes_processed': len(all_predicted),
        'num_episodes_skipped': len(skipped_rows),
        'episode_length': T,
        'num_action_classes': NUM_ACTION_CLASSES,
        'action_class_names': ACTION_NAMES_5CLASS,
        'chance_level': 1.0 / NUM_ACTION_CLASSES,
        'overall_accuracy': overall_metrics['overall_accuracy'],
        'mean_per_timestep_accuracy': float(overall_metrics['per_timestep'].mean()),
        'per_user_accuracy': {
            name: float(acc.mean())
            for name, acc in zip(user_names, per_user_accuracy)
        },
        'return_vs_accuracy_pearson_r': float(pearson_r),
        'return_vs_accuracy_pearson_p': float(pearson_p),
    }
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print('\nHuman action prediction evaluation complete.')
    print(f'  Tag: {args.tag}')
    print(f'  Users with msgpack files: {len(prolific_users)}')
    print(f'  Episodes processed: {len(all_predicted)}')
    print(f'  Episodes skipped (ambiguous): {len(skipped_rows)}')
    print(f'  Overall accuracy: {overall_metrics["overall_accuracy"]:.4f}')
    print(f'  Chance level: {1.0 / NUM_ACTION_CLASSES:.4f}')
    print(f'  Results written to: {output_dir}')


if __name__ == "__main__":
    main()
