from typing import Any, Sequence

import jax
import jax.numpy as jnp
from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2 as _OvercookedV2, State as _OvercookedV2State
from jaxmarl.environments.overcooked_v2.layouts import overcooked_v2_layouts

from ..base_env import BaseEnvConst, BaseEnvState, BaseEnv
from ..spaces import Observation, Action, ObservationSpace, ActionSpace


class OvercookedV2Const(BaseEnvConst):
    shaped_reward_factor: float = 0.0
    use_reserved_layouts: bool = False


class OvercookedV2State(BaseEnvState):
    _state: _OvercookedV2State


class OvercookedV2Env(BaseEnv[OvercookedV2Const, OvercookedV2State]):
    def __init__(
        self,
        layout: str = 'cramped_room',
        max_steps: int = 400,
        pad_obs_shape_to: Sequence[int] | None = None,
        heldout_layout_seed: int = 0,
        heldout_layout_num_samples: int = 0,
        random_reset: bool = False,
        agent_view_size: int | None = None,
        negative_rewards: bool = False,
        random_agent_positions: bool = False,
        sample_recipe_on_delivery: bool = False,
        indicate_successful_delivery: bool = False,
    ):
        self.random_reset = random_reset
        self.agent_view_size = agent_view_size
        self.negative_rewards = negative_rewards
        self.random_agent_positions = random_agent_positions
        self.sample_recipe_on_delivery = sample_recipe_on_delivery
        self.indicate_successful_delivery = indicate_successful_delivery
        self._env: _OvercookedV2 = _OvercookedV2(
            layout=layout,
            max_steps=max_steps,
            random_reset=random_reset,
            agent_view_size=agent_view_size,
            negative_rewards=negative_rewards,
            random_agent_positions=random_agent_positions,
            sample_recipe_on_delivery=sample_recipe_on_delivery,
            indicate_successful_delivery=indicate_successful_delivery,
        )
        self.max_steps: int = max_steps

        self.heldout_layout_seed = heldout_layout_seed
        self.heldout_layout_num_samples = heldout_layout_num_samples

        self._heldout_layout_state_pool: _OvercookedV2State | None = None
        self.num_heldout_layouts: int = 0

        if heldout_layout_num_samples > 0:
            self._initialize_layout_state_pools()

        self.pad_obs_shape_to = tuple(pad_obs_shape_to) if pad_obs_shape_to is not None else None

    # ------------------------------------------------------------------
    # Held-out layout pool
    # ------------------------------------------------------------------

    def _stack_state_list(self, state_list: list[_OvercookedV2State]) -> _OvercookedV2State:
        return jax.tree_util.tree_map(lambda *x: jnp.stack(x, axis=0), *state_list)

    def _sample_state_from_pool(self, rng: jax.Array, state_pool: _OvercookedV2State) -> _OvercookedV2State:
        pool_size = state_pool.time.shape[0]
        idx = jax.random.randint(rng, shape=(), minval=0, maxval=pool_size)
        return jax.tree_util.tree_map(lambda x: x[idx], state_pool)

    def _initialize_layout_state_pools(self) -> None:
        """Build a pool of initial states from layouts sharing the same (H, W) as the primary layout."""
        h, w = self._env.height, self._env.width
        key = jax.random.PRNGKey(self.heldout_layout_seed)
        collected: list[_OvercookedV2State] = []

        layout_names = list(overcooked_v2_layouts.keys())
        samples_per_layout = max(1, self.heldout_layout_num_samples // max(1, len(layout_names)))

        for layout_name in layout_names:
            if len(collected) >= self.heldout_layout_num_samples:
                break
            candidate_layout = overcooked_v2_layouts[layout_name]
            if candidate_layout.height != h or candidate_layout.width != w:
                continue
            try:
                tmp_env = _OvercookedV2(
                    layout=layout_name,
                    max_steps=self.max_steps,
                    random_reset=self.random_reset,
                    agent_view_size=self.agent_view_size,
                    negative_rewards=self.negative_rewards,
                    random_agent_positions=self.random_agent_positions,
                    sample_recipe_on_delivery=self.sample_recipe_on_delivery,
                    indicate_successful_delivery=self.indicate_successful_delivery,
                )
            except Exception:
                continue
            for _ in range(samples_per_layout):
                if len(collected) >= self.heldout_layout_num_samples:
                    break
                key, subkey = jax.random.split(key)
                _, state = tmp_env.reset(subkey)
                collected.append(state)

        if collected:
            self._heldout_layout_state_pool = self._stack_state_list(collected)
        self.num_heldout_layouts = len(collected)

    # ------------------------------------------------------------------
    # BaseEnv interface
    # ------------------------------------------------------------------

    @property
    def default_const(self) -> OvercookedV2Const:
        return OvercookedV2Const(
            max_steps=self._env.max_steps,
            use_reserved_layouts=False,
        )

    @property
    def env_observation_space(self) -> tuple[ObservationSpace, dict[str, tuple[int]]]:
        # obs_shape is the per-agent shape: (H, W, num_channels)
        if self.pad_obs_shape_to is not None:
            obs_shape = self.pad_obs_shape_to
        else:
            obs_shape = self._env.obs_shape  # (H, W, num_channels)
        n = self._env.num_agents
        return ObservationSpace({'grid_2d': obs_shape}), {'grid_2d': (n,)}

    @property
    def env_action_space(self) -> tuple[ActionSpace, dict[str, tuple[int]]]:
        num_actions = len(self._env.action_set)
        n = self._env.num_agents
        return ActionSpace({'all': num_actions}), {'all': (n,)}

    def env_reset(
        self,
        rng: jax.Array,
        const: OvercookedV2Const,
    ) -> tuple[OvercookedV2State, Observation]:
        if self._heldout_layout_state_pool is None:
            obs_dict, _state = self._env.reset(rng)
        else:
            use_reserved_layouts = jnp.asarray(const.use_reserved_layouts, dtype=jnp.bool_)

            def _reset_from_reserved(_: None) -> tuple[dict[str, jax.Array], _OvercookedV2State]:
                sampled_state = self._sample_state_from_pool(rng, self._heldout_layout_state_pool)
                obs, new_state = self._env.reset_from_state(sampled_state, rng)
                return obs, new_state

            def _reset_from_default(_: None) -> tuple[dict[str, jax.Array], _OvercookedV2State]:
                return self._env.reset(rng)

            obs_dict, _state = jax.lax.cond(use_reserved_layouts, _reset_from_reserved, _reset_from_default, operand=None)

        state = OvercookedV2State(_state=_state, _step=0)
        # Stack per-agent observations: (num_agents, H, W, num_channels)
        grid_2d = jnp.stack([obs_dict[f'agent_{i}'] for i in range(self._env.num_agents)])
        obs = Observation({'grid_2d': grid_2d})

        if self.pad_obs_shape_to is not None:
            obs = self._pad_obs(obs)

        return jax.lax.stop_gradient(state), jax.lax.stop_gradient(obs)

    def env_step(
        self,
        rng: jax.Array,
        const: OvercookedV2Const,
        state: OvercookedV2State,
        action: Action,
    ) -> tuple[OvercookedV2State, Observation, jax.Array, jax.Array, dict[Any, Any]]:
        env_action = {f'agent_{i}': action['all'][i] for i in range(self._env.num_agents)}
        # env.step auto-resets on episode end (MultiAgentEnv base class)
        obs_dict, _state, rewards, dones, infos = self._env.step(
            key=rng,
            state=state._state,
            actions=env_action,
        )
        grid_2d = jnp.stack([obs_dict[f'agent_{i}'] for i in range(self._env.num_agents)])
        obs = Observation({'grid_2d': grid_2d})
        state = OvercookedV2State(_state=_state, _step=state._step + 1)
        reward = jnp.stack([rewards[f'agent_{i}'] for i in range(self._env.num_agents)])
        shaped_reward = infos['shaped_reward']
        shaped_reward = jnp.stack([shaped_reward[f'agent_{i}'] for i in range(self._env.num_agents)])
        reward = reward + const.shaped_reward_factor * shaped_reward
        done = jnp.stack([dones[f'agent_{i}'] for i in range(self._env.num_agents)])
        info = {'_overcooked_v2_env_info': infos}

        if self.pad_obs_shape_to is not None:
            obs = self._pad_obs(obs)

        return jax.lax.stop_gradient(state), jax.lax.stop_gradient(obs), jax.lax.stop_gradient(reward), jax.lax.stop_gradient(done), info

    def _pad_obs(self, obs: Observation) -> Observation:
        def _padding(x: jax.Array) -> jax.Array:
            # x has shape (num_agents, H, W, C); pad the last 3 dims
            return jnp.pad(
                x,
                pad_width=[(0, 0) for _ in range(x.ndim - 3)] + [(0, pad_d - x_d) for pad_d, x_d in zip(self.pad_obs_shape_to, x.shape[-3:])],
                mode='constant',
                constant_values=0,
            )
        return jax.tree_util.tree_map(_padding, obs)

    # ------------------------------------------------------------------
    # Public MARL API
    # ------------------------------------------------------------------

    @property
    def num_agents(self) -> int:
        return self._env.num_agents

    def get_agent_batch_size(self) -> list[int]:
        return [1 for _ in range(self.num_agents)]

    def get_observation_space(self) -> list[ObservationSpace]:
        return [self.env_observation_space[0] for _ in range(self.num_agents)]

    def get_action_space(self) -> list[ActionSpace]:
        return [self.env_action_space[0] for _ in range(self.num_agents)]

    def reset(
        self,
        rng: jax.Array,
        const: OvercookedV2Const,
    ) -> tuple[OvercookedV2State, list[Observation]]:
        state, env_obs = self.env_reset(rng, const)
        # env_obs['grid_2d'] shape: (num_agents, H, W, C); slice per agent → (1, H, W, C)
        obs_lst = [env_obs[i:i+1] for i in range(self.num_agents)]
        return state, obs_lst

    def step(
        self,
        rng: jax.Array,
        const: OvercookedV2Const,
        state: OvercookedV2State,
        action_lst: Sequence[Action],
    ) -> tuple[OvercookedV2State, list[Observation], list[jax.Array], list[jax.Array], dict[Any, Any]]:
        env_action = Action({
            'all': jnp.concatenate([action_lst[i]['all'] for i in range(self.num_agents)])
        })
        state, env_obs, env_reward, env_done, info = self.env_step(rng, const, state, env_action)
        obs_lst = [env_obs[i:i+1] for i in range(self.num_agents)]
        reward_lst = [env_reward[i:i+1] for i in range(self.num_agents)]
        done_lst = [env_done[i:i+1] for i in range(self.num_agents)]
        return state, obs_lst, reward_lst, done_lst, info
