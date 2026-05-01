"""Flax adapter module bridging sep-rep-learning models to nicewebrl's interface.

nicewebrl's MultiAgentEnvStage calls:
    action_res = self.model.apply(model_params, hidden_state, (obs_flat, sim_done, agent_pos))
    hidden_state, pi = action_res[0], action_res[1]
    other_agent_action = jnp.argmax(pi.probs, 2)[0]

Sep-rep-learning's SharedActorCriticNet.__call__ expects:
    (rnn_states, obs_dict, return_prediction=False)
    -> (rnn_states, ActionDistribution({'all': Categorical}), value, info)

This adapter reshapes the flat observation, forwards through the inner model,
and extracts the raw Categorical distribution that nicewebrl expects.
"""

import flax.linen as nn
import jax.numpy as jnp


class SepRepModelAdapter(nn.Module):
    """Wraps a sep-rep-learning SharedActorCriticNet to match nicewebrl's model interface."""
    inner_model: nn.Module  # SharedActorCriticNet instance
    obs_shape: tuple  # (H, W, 26) — the grid observation shape
    pad_obs_shape_to: tuple | None = None  # Optional padding target shape

    @nn.compact
    def __call__(self, hidden_state, x):
        obs_flat, dones, agent_pos = x

        # Reshape flat obs (1, 1, H*W*C) -> (1, 1, H, W, C)
        # Infer C from runtime flat size to tolerate env/model channel mismatches.
        h, w, _ = self.obs_shape
        hw = h * w
        if obs_flat.shape[-1] % hw != 0:
            raise ValueError(
                f"Observation flat size {obs_flat.shape[-1]} is not divisible by H*W={hw}"
            )
        runtime_c = obs_flat.shape[-1] // hw
        grid_obs = obs_flat.reshape(obs_flat.shape[:2] + (h, w, runtime_c))

        # Align to target observation shape used by the checkpoint.
        # Supports both padding (target > runtime) and cropping (target < runtime).
        if self.pad_obs_shape_to is not None:
            target_h, target_w, target_c = self.pad_obs_shape_to

            # Crop first if runtime shape is larger than target.
            grid_obs = grid_obs[..., :target_h, :target_w, :target_c]

            # Then pad if runtime shape is smaller than target.
            cur_h, cur_w, cur_c = grid_obs.shape[-3:]
            pad_h = max(0, target_h - cur_h)
            pad_w = max(0, target_w - cur_w)
            pad_c = max(0, target_c - cur_c)

            if pad_h > 0 or pad_w > 0 or pad_c > 0:
                grid_obs = jnp.pad(
                    grid_obs,
                    ((0, 0), (0, 0), (0, pad_h), (0, pad_w), (0, pad_c)),
                    mode='constant',
                    constant_values=0,
                )

        obs = {'grid_2d': grid_obs}

        # Forward through the inner SharedActorCriticNet
        # __call__ signature: (rnn_states, obs, action=None, return_prediction=False)
        # Returns: (rnn_states, ActionDistribution({'all': Categorical}), value, info)
        rnn_state, action_dist, val, _info = self.inner_model(
            hidden_state, obs, return_prediction=False
        )

        # Extract the raw Categorical distribution (nicewebrl expects pi.probs)
        pi = action_dist['all']

        return rnn_state, pi, val
