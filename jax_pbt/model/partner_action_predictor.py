"""Partner action prediction module for E3T (Yan et al. 2023).

Predicts the partner's next action distribution from the partner's last `k`
state-action pairs and the current state. Architecture follows Table 4 of the
E3T paper:

    Conv2D(25, 5x5, leaky_relu)                            # row 1
    Conv2D(25, 3x3, leaky_relu) x 2                        # rows 2-3
    Concatenate state-embedding with action-embedding      # row 4
    (FC(64) + leaky_relu) x 3                              # row 5
    Concatenate the k state-action embeddings              # row 6, 64 * k
    (FC(64) + leaky_relu) x 3                              # row 7
    FC(64) + tanh                                          # row 8
    FC(|A|) + L2 normalize                                 # row 9

The output of row 9 is treated as logits: a softmax produces the predicted
action distribution that is fed (as `z`) into the ego policy at rollout time,
and a softmax cross-entropy against the actual partner action is used as the
training loss.

Only `DiscreteSpace` action spaces are supported (Overcooked / coop_foraging).
A flattened/concatenated action over multiple discrete keys is allowed; we
pick the first discrete key to supervise (`action_key`).
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.struct import PyTreeNode

from ..env.spaces import (
    ActionSpace,
    ContinuousSpace,
    DiscreteSpace,
    ObservationSpace,
)
from .nn_blocks import CNN, MLP


def _select_partner_action_key(action_space: ActionSpace) -> str:
    """Pick the discrete action key the predictor will model.

    Coop_foraging and Overcooked both expose a single discrete key (`'all'`)
    in the registered ActionSpace. If multiple discrete keys are present, the
    lexicographically-first one is used; non-discrete keys raise.
    """
    discrete_keys = sorted(
        k for k, space in action_space.items() if isinstance(space, DiscreteSpace)
    )
    if not discrete_keys:
        raise ValueError(
            "PartnerActionPredictor requires at least one DiscreteSpace action key; "
            f"got action_space={dict(action_space.items())}"
        )
    return discrete_keys[0]


class PartnerActionPredictor(nn.Module):
    """Flax module implementing Table 4 of E3T.

    Inputs and shapes (B = batch, k = history_length):
        - current_obs: pytree of `[B, *obs_shape]`
        - history_obs: pytree of `[B, k, *obs_shape]`
        - history_action: int array of `[B, k]`

    Output:
        - logits: `[B, |A|]` (post-L2-normalization)
    """
    obs_space: ObservationSpace
    action_space: ActionSpace
    action_key: str
    history_length: int = 5
    cnn_features: tuple[int, ...] = (25, 25, 25)
    cnn_kernels: tuple[tuple[int, int], ...] = ((5, 5), (3, 3), (3, 3))
    cnn_strides: tuple[int, ...] = (1, 1, 1)
    per_step_mlp_hidden: int = 64
    per_step_mlp_depth: int = 3
    aggregate_mlp_hidden: int = 64
    aggregate_mlp_depth: int = 3
    pre_output_size: int = 64
    activation: str = 'leaky_relu'

    def setup(self) -> None:
        if self.action_key not in self.action_space:
            raise ValueError(
                f"action_key={self.action_key!r} not in action_space={list(self.action_space.keys())}"
            )
        action_subspace = self.action_space[self.action_key]
        if not isinstance(action_subspace, DiscreteSpace):
            raise ValueError(
                f"PartnerActionPredictor only supports DiscreteSpace; "
                f"got {type(action_subspace).__name__} for key {self.action_key!r}"
            )
        self._num_actions = action_subspace.n

        # Per-key encoders: CNN for image obs, MLP for vector obs. We share
        # the encoder across the current obs and each of the k history slots
        # by reshape+stack rather than constructing k copies.
        encoders: dict[str, nn.Module] = {}
        for key in sorted(self.obs_space.keys()):
            space = self.obs_space[key]
            if not isinstance(space, ContinuousSpace):
                raise NotImplementedError(
                    f"PartnerActionPredictor only supports ContinuousSpace observations; "
                    f"got {type(space).__name__} for key {key!r}"
                )
            if len(space.shape) == 3:
                encoders[key] = CNN(
                    feature_lst=tuple(self.cnn_features),
                    kernel_lst=tuple(self.cnn_kernels),
                    stride_lst=tuple(self.cnn_strides),
                    activation_type=self.activation,
                )
            elif len(space.shape) == 1:
                encoders[key] = MLP(
                    hidden_size_lst=tuple(self.cnn_features),
                    activation_type=self.activation,
                )
            else:
                raise NotImplementedError(
                    f"PartnerActionPredictor only supports 1D or 3D obs; "
                    f"got shape {space.shape} for key {key!r}"
                )
        self.obs_encoders = encoders
        self.per_step_mlp = MLP(
            hidden_size_lst=tuple(
                self.per_step_mlp_hidden for _ in range(self.per_step_mlp_depth)
            ),
            activation_type=self.activation,
        )
        self.aggregate_mlp = MLP(
            hidden_size_lst=tuple(
                self.aggregate_mlp_hidden for _ in range(self.aggregate_mlp_depth)
            ),
            activation_type=self.activation,
        )
        self.pre_output_layer = nn.Dense(self.pre_output_size)
        self.output_layer = nn.Dense(self._num_actions)

    @property
    def num_actions(self) -> int:
        return self._num_actions

    def _encode_obs(self, obs_value: jax.Array, key: str) -> jax.Array:
        encoder = self.obs_encoders[key]
        encoded = encoder(obs_value)
        # Flatten any spatial dims so each example becomes a vector.
        return encoded.reshape(*encoded.shape[: obs_value.ndim - len(self.obs_space[key].shape)], -1)

    def _encode_obs_dict(self, obs: dict[str, jax.Array]) -> jax.Array:
        # Concatenate per-key embeddings along the feature axis.
        feats = []
        for key in sorted(self.obs_space.keys()):
            if key not in obs:
                continue
            feats.append(self._encode_obs(obs[key], key))
        if not feats:
            raise ValueError("PartnerActionPredictor received no matching observation keys")
        return jnp.concatenate(feats, axis=-1)

    def _encode_action(self, action_int: jax.Array) -> jax.Array:
        # action_int: [..., ] -> one-hot [..., |A|]
        return jax.nn.one_hot(action_int.astype(jnp.int32), self._num_actions, dtype=jnp.float32)

    def __call__(
        self,
        current_obs: dict[str, jax.Array],
        history_obs: dict[str, jax.Array],
        history_action: jax.Array,
    ) -> jax.Array:
        # 1. Encode each of the k history (obs, action) pairs into a per-step
        #    embedding via the shared CNN/MLP -> concat -> per-step MLP.
        #    history_obs[key] is [B, k, *obs_shape]. We fold (B, k) into a
        #    leading batch axis so the encoder sees a flat batch.
        if not isinstance(history_action, jax.Array) and not hasattr(history_action, 'shape'):
            history_action = jnp.asarray(history_action)
        batch_size = history_action.shape[0]
        k = history_action.shape[1]
        if k != self.history_length:
            raise ValueError(
                f"history length mismatch: history_action.shape[1]={k}, "
                f"history_length={self.history_length}"
            )

        flat_history_obs = {}
        for key in sorted(self.obs_space.keys()):
            if key not in history_obs:
                continue
            arr = history_obs[key]  # [B, k, *obs_shape]
            obs_shape = self.obs_space[key].shape
            flat_history_obs[key] = arr.reshape((batch_size * k,) + tuple(obs_shape))

        flat_history_obs_emb = self._encode_obs_dict(flat_history_obs)  # [B*k, F_obs]

        flat_history_action = history_action.reshape((batch_size * k,))
        flat_history_action_emb = self._encode_action(flat_history_action)  # [B*k, |A|]

        flat_pair_emb = jnp.concatenate(
            [flat_history_obs_emb, flat_history_action_emb], axis=-1
        )  # [B*k, F_obs + |A|]

        flat_pair_feat = self.per_step_mlp(flat_pair_emb)  # [B*k, per_step_mlp_hidden]
        # Concatenate the k state-action features along the feature axis -> [B, k * H]
        history_feat = flat_pair_feat.reshape((batch_size, k * self.per_step_mlp_hidden))

        # 2. Aggregate MLP over the concatenated history.
        history_summary = self.aggregate_mlp(history_feat)  # [B, aggregate_mlp_hidden]

        # 3. Combine with current_obs embedding.
        current_obs_emb = self._encode_obs_dict(current_obs)  # [B, F_obs]
        combined = jnp.concatenate([current_obs_emb, history_summary], axis=-1)

        # 4. Pre-output FC + tanh, then output FC + L2-normalize (paper Table 4
        #    rows 8-9). The L2-normalized vector is treated as logits.
        pre = nn.tanh(self.pre_output_layer(combined))
        raw_logits = self.output_layer(pre)
        norm = jnp.linalg.norm(raw_logits, axis=-1, keepdims=True)
        logits = raw_logits / jnp.maximum(norm, 1e-8)
        return logits


class PartnerHistoryBuffer(PyTreeNode):
    """FIFO sliding window of the partner's last k (obs, action) pairs.

    Stored as `obs: dict[str, [B, k, *obs_shape]]` and `action: [B, k]`. The
    most-recent step is at index k-1; index 0 is the oldest.
    """
    obs: dict[str, jax.Array]
    action: jax.Array  # [B, k] int


def init_partner_history(
    obs_space: ObservationSpace,
    history_length: int,
    batch_size: int,
    default_action: int = 0,
) -> PartnerHistoryBuffer:
    """Return a zeroed `PartnerHistoryBuffer` of shape `[batch_size, history_length, ...]`.

    For each obs key, fills with the space's `example()` value. Actions default
    to `default_action` (paper's reference implementation uses action index 4
    = 'noop' / 'stay' on Overcooked; coop_foraging has no such index, so we
    use 0 as a safe default).
    """
    obs = {}
    for key in sorted(obs_space.keys()):
        space = obs_space[key]
        # space.example() returns float32 zeros for ContinuousSpace
        sample = space.example(batch_shape=())  # [*obs_shape]
        sample = jnp.broadcast_to(sample, (batch_size, history_length) + sample.shape)
        obs[key] = jnp.copy(sample)
    action = jnp.full((batch_size, history_length), default_action, dtype=jnp.int32)
    return PartnerHistoryBuffer(obs=obs, action=action)


def push_partner_history(
    history: PartnerHistoryBuffer,
    new_obs: dict[str, jax.Array],
    new_action: jax.Array,
    done: jax.Array,
    obs_space: ObservationSpace,
    default_action: int = 0,
) -> PartnerHistoryBuffer:
    """Shift the window left by 1 and write (new_obs, new_action) at index k-1.

    For rows where `done=True`, the entire history (after the shift) is reset
    to the default zeroed/default-action state, since the next episode is a
    fresh trajectory and history should not leak across the boundary.
    """
    batch_size = new_action.shape[0]
    k = history.action.shape[1]
    done_b = done.astype(jnp.float32).reshape(batch_size, 1)

    # Default state to reset to (broadcast over the k axis after the new step
    # is appended at index k-1).
    fresh = init_partner_history(obs_space, k, batch_size, default_action=default_action)

    # Action: shift left, append new at last index, then reset rows where done.
    shifted_action = jnp.concatenate(
        [history.action[:, 1:], new_action.astype(jnp.int32).reshape(batch_size, 1)],
        axis=1,
    )  # [B, k]
    new_action_buf = (1.0 - done_b) * shifted_action + done_b * fresh.action.astype(jnp.float32)
    new_action_buf = new_action_buf.astype(jnp.int32)

    new_obs_dict: dict[str, jax.Array] = {}
    for key in sorted(obs_space.keys()):
        if key not in history.obs or key not in new_obs:
            continue
        old = history.obs[key]  # [B, k, *obs_shape]
        new_step = jnp.expand_dims(new_obs[key], axis=1)  # [B, 1, *obs_shape]
        shifted = jnp.concatenate([old[:, 1:], new_step], axis=1)  # [B, k, *obs_shape]
        # Reset rows where done.
        done_mask = done_b.reshape((batch_size,) + (1,) * (shifted.ndim - 1))
        new_obs_dict[key] = (1.0 - done_mask) * shifted + done_mask * fresh.obs[key]

    return PartnerHistoryBuffer(obs=new_obs_dict, action=new_action_buf)


def split_role_into_player_halves(role_array: jax.Array, num_players: int) -> list[jax.Array]:
    """Split a role-batched tensor `[role_batch, ...]` into per-player chunks
    along axis 0.

    Mirrors the role layout used by `FrozenAssignmentEnv` in this codebase
    (with `env_agent_to_role[env_idx] = [role_id, role_id]`): role obs is laid
    out as [agent 0 across envs, agent 1 across envs] = [player 0, player 1].
    """
    return jnp.split(role_array, num_players, axis=0)


def split_role_obs_into_player_halves(
    role_obs: dict[str, jax.Array], num_players: int
) -> list[dict[str, jax.Array]]:
    """Like `split_role_into_player_halves`, but for an Observation dict."""
    parts: list[dict[str, jax.Array]] = [{} for _ in range(num_players)]
    for key, arr in role_obs.items():
        chunks = jnp.split(arr, num_players, axis=0)
        for i, chunk in enumerate(chunks):
            parts[i][key] = chunk
    return parts
