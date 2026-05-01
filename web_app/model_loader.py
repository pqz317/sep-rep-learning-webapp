"""Model discovery and loading for the web app.

Resolves W&B tags to run IDs, discovers model checkpoints, and loads them.
Reuses existing utilities from coop_foraging_scripts/.
"""

import sys
import os

# Ensure coop_foraging_scripts is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from coop_foraging_scripts.wandb_utils import (
    resolve_unique_run_id_for_tag,
    get_run_config,
)
from coop_foraging_scripts.evaluation_utils import (
    discover_model_dirs,
    extract_agent_id_from_model_dir,
    load_model_checkpoint,
)

WANDB_ENTITY = 'pqz317-university-of-washington'
WANDB_PROJECT = 'sep-rep-learning'
CHECKPOINT_GLOB = './results/**/model_*-*-best'


def discover_agents_for_tag(tag: str) -> tuple[str, list[int]]:
    """Resolve a W&B tag and return (run_id, available_agent_ids).

    This is lightweight — only does tag resolution + filesystem glob,
    does NOT load model checkpoints.
    """
    run_id = resolve_unique_run_id_for_tag(tag, WANDB_ENTITY, WANDB_PROJECT)
    return discover_agents_for_run_id(run_id)


def discover_agents_for_run_id(run_id: str) -> tuple[str, list[int]]:
    """Return (run_id, available_agent_ids) via filesystem glob."""
    model_dirs = discover_model_dirs(CHECKPOINT_GLOB, run_id)
    agent_ids = sorted(extract_agent_id_from_model_dir(d) for d in model_dirs)
    return run_id, agent_ids


def load_models_for_tag(tag: str, agent_id: int = 0) -> dict:
    """Load model checkpoint for a given W&B tag.

    Args:
        tag: W&B run tag to resolve.
        agent_id: Which agent's model to load (0 or 1).

    Returns:
        Dictionary with model_fn, model_state, actor_critic_fn,
        actor_critic_params, layout, run_id, obs_shape, pad_obs_shape_to.
    """
    run_id = resolve_unique_run_id_for_tag(tag, WANDB_ENTITY, WANDB_PROJECT)
    return load_models_for_run_id(run_id, agent_id=agent_id)


def load_models_for_run_id(run_id: str, agent_id: int = 0) -> dict:
    """Load model checkpoint for a given W&B run ID.

    Args:
        run_id: W&B run ID.
        agent_id: Which agent's model to load (0 or 1).

    Returns:
        Dictionary with model_fn, model_state, actor_critic_fn,
        actor_critic_params, env_name, layout, run_id, obs_shape, pad_obs_shape_to.
    """
    # Get training config from W&B
    run_config = get_run_config(run_id, WANDB_ENTITY, WANDB_PROJECT)
    env_name = run_config.get('env', {}).get('name', 'overcooked')
    layout = run_config.get('env', {}).get('layout', 'cramped_room')
    pad_obs_shape_to = run_config.get('env', {}).get('pad_obs_shape_to', None)

    # Discover model directories
    model_dirs = discover_model_dirs(CHECKPOINT_GLOB, run_id)
    available_agent_ids = [extract_agent_id_from_model_dir(d) for d in model_dirs]
    print(f"Available agents for run '{run_id}': {available_agent_ids}")

    # Find the directory for the requested agent_id
    target_dir = None
    for d, aid in zip(model_dirs, available_agent_ids):
        if aid == agent_id:
            target_dir = d
            break
    if target_dir is None:
        raise ValueError(
            f"Agent ID {agent_id} not found for run '{run_id}'. "
            f"Available: {available_agent_ids}"
        )

    print(f"Loading model from: {target_dir}")
    model_fn, model_state = load_model_checkpoint(target_dir)

    # Extract inner Flax module and params
    actor_critic_fn = model_fn.actor_critic_fn
    actor_critic_params = model_state

    # Get obs shape from the feature extractor config
    obs_space = actor_critic_fn.feature_extractor_config.obs_space
    obs_shape = obs_space['grid_2d'].shape  # e.g. (9, 9, 26)

    return {
        'model_fn': model_fn,
        'model_state': model_state,
        'actor_critic_fn': actor_critic_fn,
        'actor_critic_params': actor_critic_params,
        'env_name': env_name,
        'layout': layout,
        'run_id': run_id,
        'model_dirs': model_dirs,
        'available_agent_ids': available_agent_ids,
        'obs_shape': obs_shape,
        'pad_obs_shape_to': tuple(pad_obs_shape_to) if pad_obs_shape_to else None,
    }
