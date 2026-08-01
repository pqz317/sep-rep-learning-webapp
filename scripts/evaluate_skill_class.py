#!/usr/bin/env python3
"""
Evaluate the model's inferred partner *skill class* during human gameplay.

For each user gameplay with the given model tag, replays the trajectory through the
AI model to read out the 'readout' layer hidden state at each timestep.  The readout
is pushed through a pre-computed per-timestep stage classifier (LDA weights +
intercepts) that predicts whether the partner is in the early / mid / late stage of
experience with the task.  Class scores are averaged over a configurable timestep
window and the argmax gives the model's best guess of the subject's skill class for
that (layout, subject) episode.

Averaged human return is then plotted per predicted skill class, both on
layout-normalized returns and on raw per-layout returns.

Stage classifier files are loaded from:
  /sep-rep-learning/results/evaluate_skill_decoding/hksyr2i5/agent_<id>/
    stage_classifier_weights_per_timestep.npy     (T, d, C)
    stage_classifier_intercepts_per_timestep.npy  (T, C)
    stage_label_names.json                        (C,)

Usage:
    python scripts/evaluate_skill_class.py --tag oc_cecp_pred_1000
    python scripts/evaluate_skill_class.py --tag oc_cecp_pred_1000 --latent-start-t 50 --latent-end-t 100
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

matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analyze_data import load_valid_users, load_overcooked_experience, LIKERT_SCALE
from coop_foraging_scripts.evaluation_utils import load_model_checkpoint, write_csv
from nicewebrl.utils import read_all_records_sync
from nicewebrl.nicejax import TimestepWrapper
from web_app.constants import EXPERIMENT_TAGS, ORIGINAL_5_TAGS
from web_app.experiment import create_environment

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "newflydata")
DEFAULT_MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")
DEFAULT_OUTPUT_DIR = "./results/evaluate_skill_class"
STAGE_CLASSIFIER_BASE_DIR = "/sep-rep-learning/results/evaluate_skill_decoding/hksyr2i5"

OBS_KEY = 'grid_2d'
ACTION_ARRAY = [3, 1, 2, 0, 4, 5]

# Slot inference: state fields compared, and how well the winning hypothesis must do.
# Static fields (wall_map, goal_pos, pot_pos) and fields that move identically under both
# hypotheses (time) are excluded — they only dilute the signal.
STATE_MATCH_FIELDS = ("agent_pos", "agent_dir_idx", "agent_inv", "maze_map")
STATE_MATCH_MIN_FRAC = 0.9
STATE_MATCH_MIN_INFORMATIVE = 5

# Ordinal blue ramp (steps 250 / 450 / 650) on a light surface.
CLASS_COLORS = ['#86b6ef', '#2a78d6', '#104281']


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
# Stage classifier loading with cache
# ---------------------------------------------------------------------------

_CLASSIFIER_CACHE: dict[int, tuple] = {}


def _load_stage_classifier(agent_id: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if agent_id in _CLASSIFIER_CACHE:
        return _CLASSIFIER_CACHE[agent_id]

    agent_dir = os.path.join(STAGE_CLASSIFIER_BASE_DIR, f"agent_{agent_id}")
    weights_path = os.path.join(agent_dir, "stage_classifier_weights_per_timestep.npy")
    intercepts_path = os.path.join(agent_dir, "stage_classifier_intercepts_per_timestep.npy")
    names_path = os.path.join(agent_dir, "stage_label_names.json")

    if not os.path.exists(weights_path) or not os.path.exists(intercepts_path):
        raise FileNotFoundError(
            f"Per-timestep stage classifier files not found for agent_id={agent_id} at {agent_dir}"
        )

    weights = np.load(weights_path)        # (T, d, C)
    intercepts = np.load(intercepts_path)  # (T, C)
    with open(names_path) as f:
        class_names = json.load(f)

    _CLASSIFIER_CACHE[agent_id] = (weights, intercepts, class_names)
    return weights, intercepts, class_names


# ---------------------------------------------------------------------------
# Human-agent slot inference via next-state matching
#
# The gameplay logs never record which slot the human occupied — nicewebrl's
# MultiAgentEnvStage.activate() draws human_id ~ Uniform{0,1} per stage and keeps it
# in memory only.  We recover it by teacher-forced one-step prediction: from each
# logged timestep, apply (logged human action, model action) under both slot
# hypotheses and check which predicted successor state matches the next logged state.
# Steps where the two hypotheses predict the same successor carry no information and
# are excluded from the denominator.
# ---------------------------------------------------------------------------

def _recorded_reward(records: list, template_ts) -> float:
    return sum(
        float(serialization.from_bytes(template_ts, r['data']['timestep']).reward)
        for r in records
    )


def _other_agent_avg_return(user_id: str, tag: str, data_dir: str) -> tuple[float | None, int]:
    """Average recorded return for `user_id` over episodes played with every model
    (tag) other than `tag`, instead of the return from the `tag` episode itself."""
    returns = []
    for other_tag in EXPERIMENT_TAGS:
        if other_tag == tag:
            continue
        for filepath in find_gameplay_files(user_id, other_tag, data_dir):
            try:
                parsed = parse_filename(filepath, other_tag)
            except ValueError:
                continue
            template_ts, _ = _get_env(parsed["layout"])
            records = [
                r for r in read_all_records_sync(filepath)
                if "data" in r and "timestep" in r.get("data", {})
            ]
            if not records:
                continue
            returns.append(_recorded_reward(records, template_ts))

    if not returns:
        return None, 0
    return float(np.mean(returns)), len(returns)


def _state_fingerprint(state) -> tuple:
    """The state fields that actually move when the two slot hypotheses differ."""
    return tuple(np.asarray(getattr(state, f)) for f in STATE_MATCH_FIELDS)


def _fingerprints_equal(a: tuple, b: tuple) -> bool:
    return all(np.array_equal(x, y) for x, y in zip(a, b))


def _prep_obs(obs_raw: np.ndarray, obs_h: int, obs_w: int, obs_c: int):
    obs_grid = np.asarray(obs_raw).reshape(1, 1, obs_h, obs_w, -1).astype(np.float32)
    runtime_c = obs_grid.shape[-1]
    if runtime_c < obs_c:
        obs_grid = jnp.pad(obs_grid, [(0, 0), (0, 0), (0, 0), (0, 0), (0, obs_c - runtime_c)])
    elif runtime_c > obs_c:
        obs_grid = obs_grid[..., :obs_c]
    return obs_grid


def _score_slot_hypotheses(
    records: list,
    template_ts,
    step_fn,
    model_fn,
    model_state,
) -> dict:
    """One pass scoring both slot hypotheses by next-state agreement."""
    ac = model_fn.actor_critic_fn
    obs_h, obs_w, obs_c = ac.feature_extractor_config.obs_space[OBS_KEY].shape

    rnn_states = {h: model_fn.init_rnn_state(jax.random.key(0), batch_size=1) for h in (0, 1)}
    rng = jax.random.key(42)

    matches = {0: 0, 1: 0}
    sim_reward = {0: 0.0, 1: 0.0}
    n_informative = 0

    ts = serialization.from_bytes(template_ts, records[0]['data']['timestep'])
    for i in range(len(records) - 1):
        next_ts = serialization.from_bytes(template_ts, records[i + 1]['data']['timestep'])
        actual = _state_fingerprint(next_ts.state)

        raw_ak = int(records[i + 1]['data']['action_idx'])
        human_action = ACTION_ARRAY[raw_ak] if raw_ak >= 0 else 4

        # Share one key per timestep across hypotheses so they are scored on equal terms.
        rng, rng_act, rng_step = jax.random.split(rng, 3)

        predicted = {}
        for h in (0, 1):
            obs_grid = _prep_obs(ts.observation[f"agent_{1 - h}"], obs_h, obs_w, obs_c)
            next_rnn, pi = model_fn.get_action_distribution(
                rng_act, model_state, rnn_states[h], {OBS_KEY: obs_grid}
            )
            rnn_states[h] = next_rnn
            model_action = int(pi.mode()['all'].flatten()[0])

            a0 = human_action if h == 0 else model_action
            a1 = model_action if h == 0 else human_action
            new_ts = step_fn(rng_step, ts, {'agent_0': jnp.array(a0), 'agent_1': jnp.array(a1)})
            sim_reward[h] += float(new_ts.reward)
            predicted[h] = _state_fingerprint(new_ts.state)

        # Both hypotheses agree on the successor — this step cannot discriminate.
        if _fingerprints_equal(predicted[0], predicted[1]):
            ts = next_ts
            continue

        n_informative += 1
        for h in (0, 1):
            if _fingerprints_equal(predicted[h], actual):
                matches[h] += 1

        ts = next_ts

    return {
        "n_informative": n_informative,
        "matches": matches,
        "match_frac": {
            h: (matches[h] / n_informative if n_informative else float('nan'))
            for h in (0, 1)
        },
        "sim_reward": sim_reward,
    }


def _infer_human_agent(
    filepath: str,
    records: list,
    template_ts,
    step_fn,
    model_fn,
    model_state,
) -> tuple[int | None, bool]:
    fname = os.path.basename(filepath)
    scores = _score_slot_hypotheses(records, template_ts, step_fn, model_fn, model_state)

    n_inf = scores["n_informative"]
    frac_0, frac_1 = scores["match_frac"][0], scores["match_frac"][1]
    detail = f"informative={n_inf}, h=0={frac_0:.3f}, h=1={frac_1:.3f}"

    if n_inf < STATE_MATCH_MIN_INFORMATIVE:
        print(f"  UNDETERMINED {fname}: only {n_inf} discriminating step(s) — skipping.")
        return None, True

    match_0 = frac_0 >= STATE_MATCH_MIN_FRAC
    match_1 = frac_1 >= STATE_MATCH_MIN_FRAC

    if match_0 and match_1:
        print(f"  AMBIGUOUS {fname}: {detail} — skipping.")
        return None, True
    elif not match_0 and not match_1:
        print(f"  NO MATCH {fname}: {detail} — skipping.")
        return None, True
    else:
        inferred = 0 if match_0 else 1
        print(f"  Inferred human_agent={inferred} ({detail})")
        return inferred, False


# ---------------------------------------------------------------------------
# Episode replay with readout-latent extraction
# ---------------------------------------------------------------------------

def replay_episode_for_skill_class(
    filepath: str,
    tag: str,
    models_dir: str,
    latent_start_t: int,
    latent_end_t: int,
    verbose: bool,
) -> dict | None:
    """Replay one episode and classify the human partner's skill stage.

    Returns dict with 'predicted_class', 'episode_return', etc., or None on failure.
    """
    parsed = parse_filename(filepath, tag)
    layout = parsed["layout"]
    agent_id = parsed["agent_id"]

    model_fn, model_state = _load_model(tag, layout, agent_id, models_dir)
    weights, intercepts, class_names = _load_stage_classifier(agent_id)

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

    num_classes = len(class_names)
    class_scores: list[np.ndarray] = []

    for i in range(len(records) - 1):
        rec = records[i]
        ts = serialization.from_bytes(template_ts, rec["data"]["timestep"])

        obs_grid = _prep_obs(
            ts.observation[model_agent_key], model_obs_h, model_obs_w, model_obs_c
        )

        rng, rng_fwd = jax.random.split(rng)
        next_rnn, _, _, info = model_fn.forward_with_aux(
            rng_fwd, model_state, rnn_state, {OBS_KEY: obs_grid}
        )

        if 'readout' in info and i < len(weights):
            readout = np.array(info['readout']).reshape(-1)            # (d,)
            class_scores.append(readout @ weights[i] + intercepts[i])  # (C,)
        else:
            class_scores.append(np.full(num_classes, np.nan))

        rnn_state = next_rnn

        if verbose and (i % 50 == 0 or i == len(records) - 2):
            latest = class_scores[-1]
            step_pred = "n/a" if np.isnan(latest).any() else class_names[int(np.argmax(latest))]
            print(f"    step {i + 1}/{len(records) - 1}  pred={step_pred}")

    scores_arr = np.array(class_scores)  # (T, C)
    T = len(scores_arr)
    t0 = min(latent_start_t, T)
    t1 = min(latent_end_t, T)
    if t0 >= t1:
        print(
            f"  WARNING: episode has {T} steps, window [{latent_start_t},{latent_end_t}) "
            f"clamps to empty; using full episode."
        )
        t0, t1 = 0, T

    window_scores = np.nanmean(scores_arr[t0:t1], axis=0)  # (C,)
    if np.isnan(window_scores).all():
        print("  No readout latents available in the window — skipping episode.")
        return None

    predicted_idx = int(np.nanargmax(window_scores))

    valid = ~np.isnan(scores_arr[t0:t1]).any(axis=1)
    window_votes = np.argmax(scores_arr[t0:t1][valid], axis=1)
    vote_fractions = np.array([
        float(np.mean(window_votes == c)) if len(window_votes) else float('nan')
        for c in range(num_classes)
    ])

    episode_return = _recorded_reward(records, template_ts)

    return {
        "user_id": parsed["user_id"],
        "layout": layout,
        "agent_id": agent_id,
        "human_agent": human_agent,
        "class_names": class_names,
        "predicted_class": class_names[predicted_idx],
        "window_scores": window_scores,
        "vote_fractions": vote_fractions,
        "episode_return": episode_return,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _mean_sem(values: np.ndarray) -> tuple[float, float]:
    if len(values) == 0:
        return float('nan'), float('nan')
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(np.mean(values)), float(np.std(values, ddof=1) / np.sqrt(len(values)))


def _plot_metric_by_class(
    class_names: list[str],
    values_by_class: dict[str, np.ndarray],
    y_label: str,
    title: str,
    output_path: str,
    form: str = "bar",
    ylim: tuple[float, float] | None = None,
) -> None:
    """Mean +/- SEM of a per-episode metric, split by predicted skill class.

    form='bar' for magnitudes measured from zero (returns); form='dot' for bounded
    ordinal scales like the 1-5 Likert, where a zero baseline is meaningless and a
    truncated bar would misstate the effect size.
    """
    stats = [_mean_sem(values_by_class.get(n, np.array([]))) for n in class_names]
    means = [s[0] for s in stats]
    sems = [s[1] for s in stats]
    x = np.arange(len(class_names))
    colors = [CLASS_COLORS[i % len(CLASS_COLORS)] for i in range(len(class_names))]

    fig, ax = plt.subplots(figsize=(5, 4.5), constrained_layout=True)
    if form == "bar":
        ax.bar(
            x, means, width=0.62, yerr=sems, capsize=4,
            color=colors, edgecolor='white', linewidth=1.0,
        )
    else:
        for xi, m, s, c in zip(x, means, sems, colors):
            ax.errorbar(xi, m, yerr=s, fmt='o', markersize=10, color=c,
                        ecolor=c, elinewidth=2, capsize=5,
                        markeredgecolor='white', markeredgewidth=1.2, zorder=4)

    rng = np.random.default_rng(0)
    for xi, name in zip(x, class_names):
        vals = values_by_class.get(name, np.array([]))
        if len(vals):
            jitter = (rng.random(len(vals)) - 0.5) * 0.3
            ax.scatter(xi + jitter, vals, s=14, color='#52514e', alpha=0.45, zorder=3, linewidths=0)

    # Anchor the label above the error-bar cap so it never sits on the whisker.
    for xi, name, m, s in zip(x, class_names, means, sems):
        n = len(values_by_class.get(name, np.array([])))
        if not np.isnan(m):
            top = m + (s if not np.isnan(s) else 0.0)
            ax.annotate(f"{m:.2f}  n={n}", (xi, top), textcoords="offset points",
                        xytext=(0, 9), ha='center', fontsize=9, color='#0b0b0b')

    ax.set_xticks(x)
    ax.set_xticklabels([n.capitalize() for n in class_names])
    ax.set_xlim(-0.6, len(class_names) - 0.4)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Model-predicted partner skill class")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.25)
    ax.set_axisbelow(True)

    fig.savefig(output_path, dpi=150)
    fig.savefig(os.path.splitext(output_path)[0] + '.svg')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def _plot_metric_by_class_per_layout(
    class_names: list[str],
    rows: list[dict],
    value_key: str,
    y_label: str,
    title: str,
    output_path: str,
    form: str = "bar",
    ylim: tuple[float, float] | None = None,
) -> None:
    rows = [r for r in rows if r.get(value_key) is not None]
    layouts = sorted({r["layout"] for r in rows})
    if not layouts:
        print(f"  Skipping plot (no rows with '{value_key}'): {output_path}")
        return
    x = np.arange(len(layouts))
    width = 0.8 / len(class_names)

    fig, ax = plt.subplots(figsize=(max(6.0, 1.7 * len(layouts)), 4.5), constrained_layout=True)
    for ci, name in enumerate(class_names):
        stats = [
            _mean_sem(np.array([
                r[value_key] for r in rows
                if r["layout"] == layout and r["predicted_class"] == name
            ], dtype=float))
            for layout in layouts
        ]
        offsets = x + (ci - (len(class_names) - 1) / 2) * width
        color = CLASS_COLORS[ci % len(CLASS_COLORS)]
        if form == "bar":
            ax.bar(
                offsets, [s[0] for s in stats], width=width * 0.9,
                yerr=[s[1] for s in stats], capsize=3, label=name.capitalize(),
                color=color, edgecolor='white', linewidth=1.0,
            )
        else:
            ax.errorbar(
                offsets, [s[0] for s in stats], yerr=[s[1] for s in stats],
                fmt='o', markersize=8, color=color, ecolor=color, elinewidth=2,
                capsize=4, linestyle='none', label=name.capitalize(),
                markeredgecolor='white', markeredgewidth=1.2,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(layouts, rotation=20, ha='right')
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(title="Predicted class", fontsize='small')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.25)
    ax.set_axisbelow(True)

    fig.savefig(output_path, dpi=150)
    fig.savefig(os.path.splitext(output_path)[0] + '.svg')
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the model's inferred partner skill class from readout latents."
    )
    parser.add_argument("--tag", required=True,
                        help="Experiment tag, e.g. oc_cecp_pred_1000.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--models-dir", default=DEFAULT_MODELS_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--latent-start-t", type=int, default=50,
                        help="Start of timestep window for averaging class scores (inclusive, default 50).")
    parser.add_argument("--latent-end-t", type=int, default=100,
                        help="End of timestep window for averaging class scores (exclusive, default 100).")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--other-agent-return", action="store_true",
        help=(
            "Use each user's average return across episodes played with every model "
            "other than --tag, instead of the return from the --tag episode itself. "
            "The skill class is still predicted from the --tag episode's replay."
        ),
    )
    args = parser.parse_args()

    valid_users = load_valid_users(args.data_dir)
    if not valid_users:
        print(f"No users with msgpack files found in {args.data_dir}")
        sys.exit(1)
    print(f"Found {len(valid_users)} users with msgpack files.")

    oc_exp_raw = load_overcooked_experience(args.data_dir)

    user_rows: list[dict] = []
    skipped_rows: list[dict] = []
    class_names: list[str] | None = None

    for user_id, prolific_id in sorted(valid_users.items()):
        files = find_gameplay_files(user_id, args.tag, args.data_dir)
        if not files:
            continue

        for filepath in files:
            print(f"\nReplaying {os.path.basename(filepath)} ...")
            try:
                result = replay_episode_for_skill_class(
                    filepath=filepath,
                    tag=args.tag,
                    models_dir=args.models_dir,
                    latent_start_t=args.latent_start_t,
                    latent_end_t=args.latent_end_t,
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

            episode_return = result["episode_return"]
            if args.other_agent_return:
                episode_return, n_other = _other_agent_avg_return(user_id, args.tag, args.data_dir)
                if episode_return is None:
                    print(f"  SKIPPED (no episodes with other-agent models found for user {user_id})")
                    skipped_rows.append({
                        "user_id": user_id,
                        "prolific_id": prolific_id,
                        "file": os.path.basename(filepath),
                    })
                    continue

            class_names = result["class_names"]
            exp_raw = oc_exp_raw.get(user_id, "N/A")
            exp_score = LIKERT_SCALE.get(exp_raw) if isinstance(exp_raw, str) else None

            row = {
                "user_id": user_id,
                "prolific_id": prolific_id,
                "layout": result["layout"],
                "agent_id": result["agent_id"],
                "human_agent": result["human_agent"],
                "episode_return": episode_return,
                "predicted_class": result["predicted_class"],
                "overcooked_experience_raw": exp_raw,
                "overcooked_experience_score": exp_score,
            }
            for ci, name in enumerate(class_names):
                row[f"score_{name}"] = float(result["window_scores"][ci])
                row[f"vote_frac_{name}"] = float(result["vote_fractions"][ci])
            user_rows.append(row)

            return_note = f" (avg over {n_other} other-agent episode(s))" if args.other_agent_return else ""
            print(
                f"  return={episode_return:.1f}{return_note}  "
                f"predicted_class={result['predicted_class']}  "
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
    layouts = [r["layout"] for r in user_rows]

    normalized_returns = np.empty_like(ep_returns)
    for layout in set(layouts):
        idx = np.array([i for i, l in enumerate(layouts) if l == layout])
        r = ep_returns[idx]
        r_min, r_max = r.min(), r.max()
        normalized_returns[idx] = (
            (r - r_min) / (r_max - r_min) if r_max > r_min else np.zeros_like(r)
        )
    for row, nr in zip(user_rows, normalized_returns):
        row["normalized_return"] = float(nr)

    output_dir = os.path.join(args.output_dir, args.tag)
    os.makedirs(output_dir, exist_ok=True)

    write_csv(os.path.join(output_dir, "episodes.csv"), user_rows)
    if skipped_rows:
        write_csv(os.path.join(output_dir, "skipped.csv"), skipped_rows)

    returns_by_class = {
        name: np.array([r["normalized_return"] for r in user_rows if r["predicted_class"] == name])
        for name in class_names
    }
    raw_by_class = {
        name: np.array([r["episode_return"] for r in user_rows if r["predicted_class"] == name])
        for name in class_names
    }
    # Self-reported experience is missing for subjects who skipped the survey question.
    experience_by_class = {
        name: np.array([
            r["overcooked_experience_score"] for r in user_rows
            if r["predicted_class"] == name and r["overcooked_experience_score"] is not None
        ], dtype=float)
        for name in class_names
    }
    n_missing_exp = sum(1 for r in user_rows if r["overcooked_experience_score"] is None)

    summary = {
        "tag": args.tag,
        "other_agent_return": args.other_agent_return,
        "num_episodes_processed": len(user_rows),
        "num_episodes_skipped": len(skipped_rows),
        "latent_window": [args.latent_start_t, args.latent_end_t],
        "class_names": class_names,
        "class_counts": {n: int(len(returns_by_class[n])) for n in class_names},
        "mean_normalized_return_by_class": {
            n: float(np.mean(returns_by_class[n])) if len(returns_by_class[n]) else None
            for n in class_names
        },
        "mean_raw_return_by_class": {
            n: float(np.mean(raw_by_class[n])) if len(raw_by_class[n]) else None
            for n in class_names
        },
        "experience_class_counts": {n: int(len(experience_by_class[n])) for n in class_names},
        "mean_experience_by_class": {
            n: float(np.mean(experience_by_class[n])) if len(experience_by_class[n]) else None
            for n in class_names
        },
        "num_episodes_missing_experience": n_missing_exp,
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nPredicted class counts: " + ", ".join(
        f"{n}={len(returns_by_class[n])}" for n in class_names
    ))

    if n_missing_exp:
        print(f"Episodes without a self-reported experience score: {n_missing_exp}")
    print("Experience class counts: " + ", ".join(
        f"{n}={len(experience_by_class[n])}" for n in class_names
    ))

    return_title_suffix = " (avg. over other-agent episodes)" if args.other_agent_return else ""

    if not args.no_plots:
        _plot_metric_by_class(
            class_names=class_names,
            values_by_class=returns_by_class,
            y_label="Normalized per-episode return",
            title=f"Human return by model-predicted skill class{return_title_suffix}",
            output_path=os.path.join(output_dir, "return_by_skill_class.png"),
        )
        _plot_metric_by_class_per_layout(
            class_names=class_names,
            rows=user_rows,
            value_key="episode_return",
            y_label="Mean episode return",
            title=f"Human return by model-predicted skill class, per layout{return_title_suffix}",
            output_path=os.path.join(output_dir, "return_by_skill_class_per_layout.png"),
        )
        _plot_metric_by_class(
            class_names=class_names,
            values_by_class=experience_by_class,
            y_label="Self-reported Overcooked experience (1–5)",
            title="Self-reported experience by model-predicted skill class",
            output_path=os.path.join(output_dir, "experience_by_skill_class.png"),
            form="dot",
            ylim=(0.7, 5.3),
        )
        _plot_metric_by_class_per_layout(
            class_names=class_names,
            rows=user_rows,
            value_key="overcooked_experience_score",
            y_label="Self-reported Overcooked experience (1–5)",
            title="Self-reported experience by model-predicted skill class, per layout",
            output_path=os.path.join(output_dir, "experience_by_skill_class_per_layout.png"),
            form="dot",
            ylim=(0.7, 5.3),
        )

    print(f"\nDone. Results written to: {output_dir}")
    print(f"  Episodes processed: {len(user_rows)}, skipped: {len(skipped_rows)}")


if __name__ == "__main__":
    main()
