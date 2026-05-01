"""Rule-based agents for the cooperative foraging gridworld.

This module provides two fixed (non-learned) policies that implement the
RecurrentAgent interface so they can be used with existing controller rollout
code without any trainer changes.

Agents in this module assume the observation layout produced by
CoopForagingEnv.get_agent_obs:
    - grid channel 0: wall mask
    - grid channel 1: this agent position (one-hot)
    - grid channels [2 : 2 + num_goals): goal positions (one channel per goal)
    - next num_agents channels: all agent positions (one channel per agent)
"""

from typing import Tuple

from flax.struct import PyTreeNode
import jax
import jax.numpy as jnp

from ..env import Action, Observation
from .agent_wrapper import RecurrentAgent


class _BaseCoopForagingFixedAgent(RecurrentAgent):
    """Shared utilities for fixed coop-foraging policies.

    The base class provides:
        - action ids matching coop_foraging_env.Actions
        - stateless recurrent-agent scaffolding (dummy state)
        - helpers for parsing one-hot grid channels
        - a greedy wall-aware single-step motion primitive
    """

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4

    def __init__(self, num_goals: int = 2, num_agents: int = 2) -> None:
        """Create a fixed-policy agent.

        Args:
            num_goals: Number of goal channels in the observation grid.
            num_agents: Number of all-agent position channels in the grid.
        """
        self.num_goals = num_goals
        self.num_agents = num_agents
        self._delta = jnp.array([
            [-1, 0],
            [1, 0],
            [0, -1],
            [0, 1],
            [0, 0],
        ], dtype=jnp.int32)

    def init_agent_state(self, rng: jax.Array, batch_size: int = 1) -> PyTreeNode:
        """Initialize a placeholder recurrent state.

        Fixed policies are memoryless in behavior, but this keeps compatibility
        with rollout code expecting RecurrentAgent state.
        """
        return {'prev_pos': -jnp.ones((batch_size, 2), dtype=jnp.int32)}

    def reset_agent_state(self, rng: jax.Array, agent_state: PyTreeNode, done: jax.Array) -> PyTreeNode:
        """Reset dummy state entries where done is True."""
        default_state = self.init_agent_state(rng, batch_size=done.shape[0])
        return jax.tree_util.tree_map(
            lambda x, y: x * (1 - done.reshape(done.shape + (1,) * (x.ndim - done.ndim))) + y,
            agent_state,
            default_state,
        )

    @staticmethod
    def _coords_from_onehot(map_batch: jax.Array) -> jax.Array:
        """Convert one-hot spatial maps [B, H, W] to integer coordinates [B, 2]."""
        bsz, height, width = map_batch.shape
        flat_idx = jnp.argmax(map_batch.reshape(bsz, height * width), axis=-1)
        return jnp.stack([flat_idx // width, flat_idx % width], axis=-1)

    def _choose_move(self, my_pos: jax.Array, target_pos: jax.Array, wall_map: jax.Array, prev_pos: jax.Array) -> jax.Array:
        """Choose one step toward target with obstacle-aware detours.

        Preference order:
          1) Any valid move that decreases Manhattan distance.
          2) Otherwise, any valid move with smallest resulting distance
             (permits temporary side/backtrack moves around obstacles).
          3) Avoid immediate backtracking to previous position when possible.
          4) STAY if all four movement directions are blocked.
        """
        move_actions = jnp.array([self.UP, self.DOWN, self.LEFT, self.RIGHT], dtype=jnp.int32)
        next_pos = my_pos[None, :] + self._delta[move_actions]

        blocked = wall_map[next_pos[:, 0], next_pos[:, 1]]
        valid_move = jnp.logical_not(blocked)

        curr_dist = jnp.abs(target_pos - my_pos).sum()
        next_dist = jnp.abs(target_pos[None, :] - next_pos).sum(-1)
        improving = next_dist < curr_dist

        prefer_mask = jnp.logical_and(valid_move, improving)
        has_improving = jnp.any(prefer_mask)
        candidate_mask = jnp.where(has_improving, prefer_mask, valid_move)

        has_prev_pos = prev_pos[0] >= 0
        is_backtrack = jnp.logical_and(
            has_prev_pos,
            jnp.all(next_pos == prev_pos[None, :], axis=-1),
        )
        non_backtrack_mask = jnp.logical_and(candidate_mask, jnp.logical_not(is_backtrack))
        candidate_mask = jnp.where(jnp.any(non_backtrack_mask), non_backtrack_mask, candidate_mask)

        large = jnp.array(10 ** 6, dtype=next_dist.dtype)
        masked_dist = jnp.where(candidate_mask, next_dist, large)
        best_idx = jnp.argmin(masked_dist)
        best_action = move_actions[best_idx]

        return jnp.where(jnp.any(valid_move), best_action, jnp.int32(self.STAY))

    def _prepare_grid(self, obs: Observation) -> Tuple[jax.Array, Tuple[int, ...], int, int, int]:
        """Normalize observation grid to flattened batch shape.

        Returns:
            grid: Reshaped tensor [B_flat, H, W, C].
            batch_shape: Original batch prefix to restore output shape.
            height: Grid height.
            width: Grid width.
            channels: Number of grid channels.
        """
        grid = obs['grid']
        if grid.ndim == 3:
            grid = grid[None, ...]
        batch_shape = grid.shape[:-3]
        height, width, channels = grid.shape[-3], grid.shape[-2], grid.shape[-1]
        grid = grid.reshape((-1, height, width, channels))
        return grid, batch_shape, height, width, channels


class GoalLeaderAgent(_BaseCoopForagingFixedAgent):
    """Fixed policy that moves toward a sampled goal for each episode.

    At initialization/reset, each batch element samples a goal index uniformly
    from [0, num_goals). The agent then follows that goal until the episode
    ends, at which point a new target is sampled for the next episode.
    """

    def init_agent_state(self, rng: jax.Array, batch_size: int = 1) -> PyTreeNode:
        """Sample one persistent goal target per batch element."""
        target_goal_idx = jax.random.randint(
            rng,
            shape=(batch_size,),
            minval=0,
            maxval=self.num_goals,
            dtype=jnp.int32,
        )
        return {
            'target_goal_idx': target_goal_idx,
            'prev_pos': -jnp.ones((batch_size, 2), dtype=jnp.int32),
        }

    def reset_agent_state(self, rng: jax.Array, agent_state: PyTreeNode, done: jax.Array) -> PyTreeNode:
        """Resample goal targets for done episodes; keep others unchanged."""
        done_bool = done.astype(jnp.bool_)
        new_targets = jax.random.randint(
            rng,
            shape=done_bool.shape,
            minval=0,
            maxval=self.num_goals,
            dtype=jnp.int32,
        )
        target_goal_idx = jnp.where(done_bool, new_targets, agent_state['target_goal_idx'])
        prev_pos = jnp.where(done_bool[:, None], -jnp.ones_like(agent_state['prev_pos']), agent_state['prev_pos'])
        return {'target_goal_idx': target_goal_idx, 'prev_pos': prev_pos}

    def step(
        self,
        rng: jax.Array,
        model_state: PyTreeNode,
        agent_state: PyTreeNode,
        obs: Observation,
    ) -> tuple[PyTreeNode, Action, dict]:
        """Compute one action step.

        The target is the sampled goal index stored in agent_state.
        """
        grid, batch_shape, _, _, _ = self._prepare_grid(obs)
        wall_map = grid[..., 0] > 0.5
        my_pos = self._coords_from_onehot(grid[..., 1])

        goal_channels = grid[..., 2:2 + self.num_goals]
        goal_pos = jax.vmap(self._coords_from_onehot, in_axes=-1, out_axes=1)(goal_channels)

        target_goal_idx = agent_state['target_goal_idx']
        if target_goal_idx.shape != batch_shape:
            target_goal_idx = target_goal_idx.reshape(batch_shape)
        target_goal_idx = target_goal_idx.reshape((-1,)).astype(jnp.int32)
        target_pos = goal_pos[jnp.arange(goal_pos.shape[0]), target_goal_idx]

        prev_pos = agent_state['prev_pos']
        if prev_pos.shape != batch_shape + (2,):
            prev_pos = prev_pos.reshape(batch_shape + (2,))
        prev_pos = prev_pos.reshape((-1, 2)).astype(jnp.int32)

        move_action = jax.vmap(self._choose_move)(my_pos, target_pos, wall_map, prev_pos)
        reached_target = jnp.all(my_pos == target_pos, axis=-1)
        stay_action = jnp.full_like(move_action, self.STAY)
        action = jnp.where(reached_target, stay_action, move_action)
        action = action.reshape(batch_shape)
        next_agent_state = {
            'target_goal_idx': target_goal_idx.reshape(batch_shape),
            'prev_pos': my_pos.reshape(batch_shape + (2,)),
        }
        return next_agent_state, Action({'all': action}), {}


class FollowerAgent(_BaseCoopForagingFixedAgent):
    """Fixed policy that greedily moves toward the nearest other agent."""

    def step(
        self,
        rng: jax.Array,
        model_state: PyTreeNode,
        agent_state: PyTreeNode,
        obs: Observation,
    ) -> tuple[PyTreeNode, Action, dict]:
        """Compute one action step.

        The target is the nearest non-self agent by Manhattan distance. If no
        other agent can be identified, the policy stays in place.
        """
        grid, batch_shape, _, _, _ = self._prepare_grid(obs)
        wall_map = grid[..., 0] > 0.5
        my_pos = self._coords_from_onehot(grid[..., 1])

        agent_channels = grid[..., 2 + self.num_goals:2 + self.num_goals + self.num_agents]
        all_agent_pos = jax.vmap(self._coords_from_onehot, in_axes=-1, out_axes=1)(agent_channels)
        all_dist = jnp.abs(all_agent_pos - my_pos[:, None, :]).sum(-1)

        large_number = jnp.array(10 ** 6, dtype=all_dist.dtype)
        non_self_dist = jnp.where(all_dist > 0, all_dist, large_number)
        target_idx = jnp.argmin(non_self_dist, axis=-1)
        has_other_agent = jnp.any(all_dist > 0, axis=-1)
        target_pos = all_agent_pos[jnp.arange(all_agent_pos.shape[0]), target_idx]

        prev_pos = agent_state['prev_pos']
        if prev_pos.shape != batch_shape + (2,):
            prev_pos = prev_pos.reshape(batch_shape + (2,))
        prev_pos = prev_pos.reshape((-1, 2)).astype(jnp.int32)

        move_action = jax.vmap(self._choose_move)(my_pos, target_pos, wall_map, prev_pos)
        stay_action = jnp.full_like(move_action, self.STAY)
        action = jnp.where(has_other_agent, move_action, stay_action)

        action = action.reshape(batch_shape)
        next_agent_state = {
            'prev_pos': my_pos.reshape(batch_shape + (2,)),
        }
        return next_agent_state, Action({'all': action}), {}
