"""
Utilities for the latent-convention-discovery training paradigm.

Contains:
- `sample_z_bank`: draws a fresh batch of convention latents z ~ N(0, I).
- `compute_structured_pe_reward`: structured population-entropy reward (global
  diversity minus local smoothness via a Gaussian kernel over z).
- `log_convention_structure_metrics`: diagnostic metrics for the z -> pi manifold.

The PE reward is applied *on top* of the extrinsic task reward, only to the
P1 half of each role's batch. Its gradient is stopped before entering PPO.
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp


def sample_z_bank(rng: jax.Array, batch_size: int, latent_dim: int) -> jax.Array:
    """Sample a per-episode z-bank.

    Returns shape [batch_size, latent_dim], each row drawn from N(0, I_d).
    """
    return jax.random.normal(rng, shape=(batch_size, latent_dim))


def _log_prob_of_action(action_logits: jax.Array, action: jax.Array) -> jax.Array:
    """Log probability of a discrete action under categorical logits.

    [Input]
        - action_logits: [..., A] unnormalized logits.
        - action: [...] integer action indices.
    [Output]
        - log_p: [...] log probability of the taken action.
    """
    log_probs = jax.nn.log_softmax(action_logits, axis=-1)
    return jnp.take_along_axis(log_probs, action[..., None], axis=-1).squeeze(-1)


def _prob_of_action(action_logits: jax.Array, action: jax.Array) -> jax.Array:
    probs = jax.nn.softmax(action_logits, axis=-1)
    return jnp.take_along_axis(probs, action[..., None], axis=-1).squeeze(-1)


def compute_structured_pe_reward(
    policy_fn: Callable[[jax.Array, jax.Array], jax.Array],
    features: jax.Array,
    actions: jax.Array,
    z_true: jax.Array,
    z_bank: jax.Array,
    alpha_global: float = 0.0,
    alpha_local: float = 0.0,
    sigma: float | None = None,
) -> jax.Array:
    """Compute the structured population-entropy reward for the P1 slice.

    Implements, per (timestep, P1 env) index (t, b):

        r_PE(s, a; z_b) =
            + alpha_global * ( -log pi_bar_global(a | s) )
            + alpha_local  * (  log pi_bar_local(a | s; z_b) )

    where
        pi_bar_global(a | s)       = mean_j pi(a | s, z_j)
        pi_bar_local(a | s; z_b)   = sum_j w_{b,j} pi(a | s, z_j),
        w_{b,j} proportional to exp(-||z_b - z_j||^2 / (2 sigma^2)),
        Sum_j w_{b,j} = 1.

    [Input]
        - policy_fn: f(feature, z) -> action logits of shape [..., num_actions].
          Expected to take a feature batch of shape [B, F] and a z batch of
          shape [B, d] and return logits of shape [B, num_actions]. Must be
          vmap-friendly in the z dimension.
        - features: [T, B_p1, F] policy-branch features for the P1 slice.
          For the dual-extractor latent-convention model these are the
          feedforward (no-RNN) CNN/MLP encoder outputs; policy_fn should
          consume them directly together with z (no RNN pass required).
        - actions: [T, B_p1] integer actions taken by P1.
        - z_true: [B_p1, d] latents assigned to the P1 envs this iteration
          (only used to compute the kernel weights for each P1 env).
        - z_bank: [K, d] latents used for the global / local averaging.
        - alpha_global, alpha_local: reward coefficients.
        - sigma: kernel bandwidth for the local term. If None, defaults to
          sqrt(latent_dim).

    [Output]
        - pe_reward: [T, B_p1] shaped additive reward (stop-gradient applied).
    """
    if alpha_global == 0.0 and alpha_local == 0.0:
        return jnp.zeros(actions.shape, dtype=features.dtype)

    T, B_p1, F = features.shape
    K, d = z_bank.shape

    if sigma is None:
        sigma = jnp.sqrt(jnp.asarray(d, dtype=features.dtype))
    else:
        sigma = jnp.asarray(sigma, dtype=features.dtype)

    flat_features = features.reshape(T * B_p1, F)  # [T*B_p1, F]
    flat_actions = actions.reshape(T * B_p1)        # [T*B_p1]

    def _pi_for_z(z_single: jax.Array) -> jax.Array:
        # broadcast single z across all features
        z_broadcast = jnp.broadcast_to(z_single[None, :], (flat_features.shape[0], d))
        return policy_fn(flat_features, z_broadcast)  # [T*B_p1, A]

    logits_bank = jax.vmap(_pi_for_z)(z_bank)  # [K, T*B_p1, A]
    probs_bank = jax.nn.softmax(logits_bank, axis=-1)  # [K, T*B_p1, A]

    # pick probability of the taken action under each z_j -> [K, T*B_p1]
    flat_actions_exp = jnp.broadcast_to(flat_actions[None, :], (K, flat_actions.shape[0]))
    prob_at_action = jnp.take_along_axis(
        probs_bank, flat_actions_exp[..., None], axis=-1
    ).squeeze(-1)

    reward_flat = jnp.zeros(flat_actions.shape, dtype=features.dtype)

    if alpha_global != 0.0:
        pi_bar_global = prob_at_action.mean(axis=0)  # [T*B_p1]
        reward_flat = reward_flat + alpha_global * (
            -jnp.log(jnp.clip(pi_bar_global, a_min=1e-12))
        )

    if alpha_local != 0.0:
        # Kernel weights: w_{b,j} over the bank, per P1 env.
        # z_true: [B_p1, d], z_bank: [K, d]
        sq_dist = jnp.sum(
            (z_true[:, None, :] - z_bank[None, :, :]) ** 2, axis=-1
        )  # [B_p1, K]
        log_w = -sq_dist / (2.0 * sigma**2)
        w = jax.nn.softmax(log_w, axis=-1)  # [B_p1, K], rows sum to 1

        # Broadcast weights over T: final weights per (t, b, j). [T*B_p1, K]
        w_tb = jnp.broadcast_to(w[None, :, :], (T, B_p1, K)).reshape(T * B_p1, K)
        # prob_at_action is [K, T*B_p1]; align to [T*B_p1, K].
        prob_at_action_bk = prob_at_action.transpose(1, 0)  # [T*B_p1, K]
        pi_bar_local = jnp.sum(w_tb * prob_at_action_bk, axis=-1)  # [T*B_p1]
        reward_flat = reward_flat + alpha_local * jnp.log(
            jnp.clip(pi_bar_local, a_min=1e-12)
        )

    pe_reward = reward_flat.reshape(T, B_p1)
    return jax.lax.stop_gradient(pe_reward)


def log_convention_structure_metrics(
    policy_fn: Callable[[jax.Array, jax.Array], jax.Array],
    features: jax.Array,
    z_true: jax.Array,
    z_bank: jax.Array,
    sigma: float | None = None,
) -> dict[str, jax.Array]:
    """Cheap diagnostics for the z -> pi manifold.

    Returns a dict of scalar metrics:
        - H_pi_bar_global: entropy of the globally averaged policy at each state
        - H_pi_bar_local:  entropy of the locally averaged policy
        - H_gap: H_pi_bar_global - H_pi_bar_local (structured surplus)
        - kl_z_dist_spearman: Spearman corr between KL(pi(z_i)||pi(z_j)) and ||z_i-z_j||
    """
    T, B_p1, F = features.shape
    K, d = z_bank.shape
    if sigma is None:
        sigma = jnp.sqrt(jnp.asarray(d, dtype=features.dtype))

    flat_features = features.reshape(T * B_p1, F)

    def _logits_for_z(z_single: jax.Array) -> jax.Array:
        z_broadcast = jnp.broadcast_to(z_single[None, :], (flat_features.shape[0], d))
        return policy_fn(flat_features, z_broadcast)

    logits_bank = jax.vmap(_logits_for_z)(z_bank)  # [K, T*B_p1, A]
    probs_bank = jax.nn.softmax(logits_bank, axis=-1)

    pi_bar_global = probs_bank.mean(axis=0)  # [T*B_p1, A]
    H_global = -jnp.sum(
        pi_bar_global * jnp.log(jnp.clip(pi_bar_global, 1e-12)), axis=-1
    ).mean()

    sq_dist = jnp.sum((z_true[:, None, :] - z_bank[None, :, :]) ** 2, axis=-1)
    log_w = -sq_dist / (2.0 * sigma**2)
    w = jax.nn.softmax(log_w, axis=-1)  # [B_p1, K]
    w_tb = jnp.broadcast_to(w[None, :, :], (T, B_p1, K)).reshape(T * B_p1, K)
    # Weighted average of probabilities across bank
    probs_bank_tb = probs_bank.transpose(1, 0, 2)  # [T*B_p1, K, A]
    pi_bar_local = jnp.sum(w_tb[..., None] * probs_bank_tb, axis=1)  # [T*B_p1, A]
    H_local = -jnp.sum(
        pi_bar_local * jnp.log(jnp.clip(pi_bar_local, 1e-12)), axis=-1
    ).mean()

    return {
        'H_pi_bar_global': H_global,
        'H_pi_bar_local': H_local,
        'H_gap': H_global - H_local,
    }
