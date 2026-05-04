#!/usr/bin/env python3
"""
Replay a recorded gameplay trajectory through the model.

Given a user_id and tag, finds the gameplay JSON file(s), deserializes each
timestep, and runs it through the model step-by-step. Records the model's
hidden (RNN) state and readout latent z at every step if the model has a
readout layer (use_readout_layer=True) or VIB (use_vib=True).

Usage:
    python scripts/replay_trajectory.py --user-id 3399402009 --tag oc_cecp_pred_1000
    python scripts/replay_trajectory.py --user-id 3399402009 --tag fcp --human-agent 1
"""

import argparse
import glob
import os
import pickle
import re
import sys

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from coop_foraging_scripts.evaluation_utils import load_model_checkpoint
from nicewebrl.utils import read_all_records_sync
from nicewebrl.nicejax import TimestepWrapper
from web_app.constants import ORIGINAL_5_TAGS
from web_app.experiment import create_environment

DEFAULT_DATA_DIR = os.environ.get("DATA_DIR", "data")
DEFAULT_MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")


def find_gameplay_files(user_id: str, tag: str, data_dir: str) -> list[str]:
    pattern = os.path.join(data_dir, f"gameplay_user={user_id}_{tag}_*.json")
    return sorted(glob.glob(pattern))


def parse_filename(filepath: str, tag: str) -> dict:
    """Extract layout and agent_id from a gameplay filename.

    Filename format: gameplay_user={user_id}_{tag}_{layout}_agent{agent_id}.json
    """
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


def load_model(tag: str, layout: str, agent_id: int, models_dir: str):
    """Load actor_critic_fn and its params for the given tag/layout/agent_id."""
    load_tag = f"{tag}_{layout}" if tag in ORIGINAL_5_TAGS else tag
    tag_dir = os.path.join(models_dir, load_tag)
    matches = glob.glob(os.path.join(tag_dir, f"model_*-{agent_id}-best"))
    if not matches:
        raise FileNotFoundError(
            f"No checkpoint for agent_id={agent_id} in {tag_dir}"
        )
    model_fn, model_state = load_model_checkpoint(matches[0])
    return model_fn.actor_critic_fn, model_state["actor_critic_params"]


def make_template_timestep(layout: str):
    """Return a sample Timestep for use as a flax deserialization template."""
    env, _ = create_environment(layout, "overcooked")
    jax_env = TimestepWrapper(
        env, autoreset=True, reset_w_batch_dim=False, use_params=False
    )
    return jax_env.reset(jax.random.key(0), {})


def replay_file(
    filepath: str,
    tag: str,
    human_agent: int,
    models_dir: str,
    verbose: bool,
) -> dict:
    """Replay one gameplay file through the model.

    Returns a dict containing per-step arrays of actions, RNN states,
    post-LSTM features, and (if present) readout latent z vectors.
    """
    parsed = parse_filename(filepath, tag)
    layout = parsed["layout"]
    agent_id = parsed["agent_id"]
    model_agent = 1 - human_agent
    model_agent_key = f"agent_{model_agent}"

    if verbose:
        print(
            f"  layout={layout}  agent_id={agent_id}"
            f"  model observes {model_agent_key}"
        )

    ac, params = load_model(tag, layout, agent_id, models_dir)

    has_readout = getattr(ac, "use_readout_layer", False)
    has_vib = getattr(ac, "use_vib", False)
    has_latent = has_readout or has_vib
    model_obs_h, model_obs_w, model_obs_c = (
        ac.feature_extractor_config.obs_space["grid_2d"].shape
    )

    rnn_state = ac.init_rnn_state(jax.random.key(0), batch_size=1)
    template_ts = make_template_timestep(layout)
    records = list(read_all_records_sync(filepath))

    actions = []
    action_names = []
    rnn_states = []
    features = []
    readouts = [] if has_latent else None

    step_records = [r for r in records if "data" in r and "timestep" in r.get("data", {})]

    for i, rec in enumerate(step_records):
        ts = serialization.from_bytes(template_ts, rec["data"]["timestep"])

        obs = ts.observation[model_agent_key]  # (H, W, C)
        obs_flat = obs.flatten()
        obs_grid = obs_flat.reshape(1, 1, model_obs_h, model_obs_w, -1)

        # Match channel dim to what the checkpoint expects (pad or crop).
        runtime_c = obs_grid.shape[-1]
        if runtime_c < model_obs_c:
            obs_grid = jnp.pad(
                obs_grid,
                [(0, 0), (0, 0), (0, 0), (0, 0), (0, model_obs_c - runtime_c)],
            )
        elif runtime_c > model_obs_c:
            obs_grid = obs_grid[..., :model_obs_c]

        rnn_state, _pi, _val, info = ac.apply(
            params, rnn_state, {"grid_2d": obs_grid}
        )

        actions.append(int(rec["data"]["action_idx"]))
        action_names.append(rec["data"]["action_name"])
        rnn_states.append(
            jax.tree_util.tree_map(lambda x: np.array(x), rnn_state)
        )
        features.append(np.array(info["feature"]))

        if has_latent:
            if has_vib:
                # Use the VIB posterior mean as the latent z.
                readouts.append(np.array(info["vib_mu"]))
            else:
                readouts.append(np.array(info["readout"]))

        if verbose and (i % 50 == 0 or i == len(step_records) - 1):
            print(f"    step {i + 1}/{len(records)}")

    return {
        "user_id": parsed["user_id"],
        "tag": tag,
        "layout": layout,
        "agent_id": agent_id,
        "model_agent": model_agent,
        "has_readout": has_readout,
        "has_vib": has_vib,
        # (T,) — the human's action index at each step
        "actions": np.array(actions, dtype=np.int32),
        "action_names": action_names,
        # list of T RNN-state pytrees, each leaf shape (1, hidden_size)
        "rnn_states": rnn_states,
        # (T, 1, 1, feature_dim)
        "features": np.stack(features, axis=0),
        # (T, 1, 1, readout_size) or None
        "readouts": np.stack(readouts, axis=0) if readouts is not None else None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Replay gameplay trajectory through the model and record latent states."
    )
    parser.add_argument(
        "--user-id", required=True,
        help="User ID (seed value) from the gameplay filename.",
    )
    parser.add_argument(
        "--tag", required=True,
        help="Experiment tag, e.g. oc_cecp_pred_1000, fcp, mep, oc_cec_v3.",
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
        "--human-agent", type=int, default=0, choices=[0, 1],
        help=(
            "Which agent index (0 or 1) was the human player. "
            "The model's obs is taken from the other agent. "
            "Default: 0. Note: this was randomly assigned per session and "
            "is not stored in the gameplay file — verify if it matters for your analysis."
        ),
    )
    parser.add_argument(
        "--output", default=None,
        help="Output pickle path. Defaults to replay_user={user_id}_{tag}.pkl.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    files = find_gameplay_files(args.user_id, args.tag, args.data_dir)
    if not files:
        print(
            f"No gameplay files found for user_id={args.user_id}, tag={args.tag}"
            f" in {args.data_dir}"
        )
        sys.exit(1)

    print(f"Found {len(files)} file(s):")
    for f in files:
        print(f"  {f}")

    all_results = []
    for filepath in files:
        print(f"\nReplaying {os.path.basename(filepath)} ...")
        result = replay_file(
            filepath,
            tag=args.tag,
            human_agent=args.human_agent,
            models_dir=args.models_dir,
            verbose=args.verbose,
        )
        all_results.append(result)
        n = len(result["actions"])
        has_z = result["readouts"] is not None
        print(
            f"  {n} steps  |  feature shape: {result['features'].shape}"
            + (f"  |  readout shape: {result['readouts'].shape}" if has_z else "  |  no readout layer")
        )

    output_path = args.output or f"replay_user={args.user_id}_{args.tag}.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(all_results, f)
    print(f"\nSaved {len(all_results)} episode(s) → {output_path}")


if __name__ == "__main__":
    main()
