"""Model loading for the web app.

Loads model checkpoints directly from /app/models/<tag>/ on the local filesystem.
Always assumes 8 agents per model (agent_id 0–7); no W&B resolution needed.
"""

import glob
import os

MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")


def load_models_for_tag(tag: str, agent_id: int = 0) -> dict:
    """Load a model checkpoint from /app/models/<tag>/.

    Args:
        tag: Model tag (name of subdirectory under MODELS_DIR).
        agent_id: Which agent's model to load (0–7).

    Returns:
        Dictionary with actor_critic_fn, model_state, obs_shape, env_name, etc.
    """
    from coop_foraging_scripts.evaluation_utils import load_model_checkpoint

    tag_dir = os.path.join(MODELS_DIR, tag)
    if not os.path.isdir(tag_dir):
        raise FileNotFoundError(f"Model directory not found: {tag_dir}")

    matches = glob.glob(os.path.join(tag_dir, f"model_*-{agent_id}-best"))
    if not matches:
        raise FileNotFoundError(
            f"No checkpoint found for agent_id={agent_id} in {tag_dir}."
        )
    model_dir = matches[0]
    print(f"Loading model from: {model_dir}")

    model_fn, model_state = load_model_checkpoint(model_dir)
    actor_critic_fn = model_fn.actor_critic_fn
    obs_shape = actor_critic_fn.feature_extractor_config.obs_space['grid_2d'].shape

    return {
        'model_fn': model_fn,
        'model_state': model_state,
        'actor_critic_fn': actor_critic_fn,
        'actor_critic_params': model_state,
        'env_name': 'overcooked',
        'layout': 'unknown',
        'tag': tag,
        'obs_shape': obs_shape,
        'pad_obs_shape_to': None,
    }
