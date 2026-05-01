from enum import IntEnum
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import seaborn as sns
import matplotlib.patches as patches

from ..base_env import BaseEnvConst, BaseEnvState, BaseEnv
from ..spaces import Observation, Action, ObservationSpace, ActionSpace

class GridWorldConst(BaseEnvConst):
    goal_detect_reward: float

class GridWorldState(BaseEnvState):
    agent_pos: jax.Array
    agent_allow_move: jax.Array
    agent_trial_best_score: jax.Array
    agent_discount_trial_ret: jax.Array
    goal_pos_map: jax.Array
    goal_score_map: jax.Array
    wall_map: jax.Array

class Actions(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2 
    RIGHT = 3 
    STAY = 4

class GridWorldEnv(BaseEnv[GridWorldConst, GridWorldState]):
    def __init__(
        self, 
        num_agents: int = 1,
        num_goals: int = 3,
        grid_size: int = 12,
        view_range: int = 4,
        view_upper_left_offset: int = 0, 
        clutter_density: float = 0.0,
        trial_steps: int = 20,
        max_steps: int = 60, 
        goal_score_scale: float = 1.0,
        stop_after_goal_reach: bool = True,
        last_ret_scale: float = 0.4,
        history_ret_scale: float = 0.6
    ) -> None:
        """        
        [Param]
            num_agents: Number of players
            num_goals: Number of goals.
            grid_size: The size of the grid (grid_size x grid_size).
            clutter_density: Ratio of number_of_clutters / grid_world_size for clutter generation
            view_range: Range of view (to right and down) to four directions.
            view_upper_left_offset: Additional view to upper and left (usually used to pad observation).
            trial_steps: Number of steps for each trial
            max_steps: Maximum steps before the environment terminates.
            distance_penalty: Penalty for each step based on Manhattan distance to the goal.
            goal_sore_scale: Scale of how goal scores are randomized.
            stop_after_goal_reach: Agents will stop after reaching a goal until the end of a trial.
            last_ret_scale, history_ret_scale: Determine how trial returns are cumulated as prestige
        """
        self.n_agent = num_agents
        self.n_goal = num_goals
        self.grid_size = grid_size
        self.n_clutter = int(clutter_density * (grid_size - 2) * (grid_size - 2))
        self.trial_steps = trial_steps
        self.max_steps = max_steps
        self.view_range = view_range
        self.view_upper_left_offset = view_upper_left_offset
        self.goal_score_scale = goal_score_scale
        self.stop_after_goal_reach = stop_after_goal_reach

        self.last_ret_scale = last_ret_scale
        self.history_ret_scale = history_ret_scale

        h, w = self.grid_size, self.grid_size
        self.available_positions = jnp.array([
            (x, y) for x in range(1, h - 1) for y in range(1, w - 1)
        ])
        self.obs_size = 2 * self.view_range + 1 + view_upper_left_offset
    
    @property
    def default_const(self) -> GridWorldConst:
        return GridWorldConst(max_steps=self.max_steps, goal_detect_reward=0.01)
    
    @property
    def env_observation_space(self) -> tuple[ObservationSpace, dict[str, tuple[int]]]:
        map_n_feature = 1
        my_pos_n_feature = 1
        other_pos_n_feature = self.n_agent - 1
        goal_pos_n_feature = 1
        total_n_feature = map_n_feature + my_pos_n_feature + goal_pos_n_feature + other_pos_n_feature
        obs_space = (self.obs_size, self.obs_size, total_n_feature)
        return ObservationSpace({'grid': obs_space}), {'grid': (self.n_agent,)}
    
    @property
    def env_action_space(self) -> tuple[ActionSpace, dict[str, tuple[int]]]:
        num_actions = len(Actions)
        return ActionSpace({'all': num_actions}), {'all': (self.n_agent,)}

    def _reset_positions(self, rng: jax.Array, const: GridWorldConst) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Reset the map, and the positions of all agents and goals
        """
        h, w = self.grid_size, self.grid_size
        n_agent, n_goal, n_clutter = self.n_agent, self.n_goal, self.n_clutter
        available_positions = self.available_positions

        rng, rng_index = jax.random.split(rng)
        indices = jax.random.choice(rng_index, len(available_positions), shape=(n_agent + n_goal + n_clutter,), replace=False)
        agent_indicies, goal_indicies, clutter_indicies = indices[:n_agent], indices[n_agent:n_agent+n_goal], indices[n_agent+n_goal:]
        
        agent_pos = available_positions[agent_indicies]
        goal_pos = available_positions[goal_indicies]
        clutter_pos = available_positions[clutter_indicies]
        return agent_pos, goal_pos, clutter_pos

    def env_reset(
        self,
        rng: jax.Array,
        const: GridWorldConst
    ) -> tuple[GridWorldState, Observation]:
        """Resets the environment to an initial state."""
        h, w = self.grid_size, self.grid_size

        rng, rng_reset_pos = jax.random.split(rng)
        agent_pos, goal_pos, clutter_pos = self._reset_positions(rng_reset_pos, const)
        
        rng, rng_score = jax.random.split(rng)
        goal_scores = self.goal_score_scale * jax.random.uniform(rng_score, shape=(self.n_goal,), minval=-1, maxval=1)
        goal_pos_map = jnp.zeros((h, w), dtype=jnp.bool_)
        goal_pos_map = goal_pos_map.at[goal_pos[:, 0], goal_pos[:, 1]].set(True)
        goal_score_map = jnp.zeros((h, w), dtype=jnp.float32)
        goal_score_map = goal_score_map.at[goal_pos[:, 0], goal_pos[:, 1]].set(goal_scores)

        wall_map = jnp.zeros((h, w), dtype=jnp.bool_)
        wall_map = wall_map.at[0, :].set(True)
        wall_map = wall_map.at[-1, :].set(True)
        wall_map = wall_map.at[:, 0].set(True)
        wall_map = wall_map.at[:, -1].set(True)
        wall_map = wall_map.at[clutter_pos[:, 0], clutter_pos[:, 1]].set(True)

        state = GridWorldState(
            _step=0,
            agent_pos=agent_pos,
            agent_allow_move=jnp.ones((self.n_agent,), dtype=jnp.bool_),
            agent_trial_best_score=-jnp.ones((self.n_agent,), dtype=jnp.float32),
            agent_discount_trial_ret=jnp.zeros((self.n_agent,), dtype=jnp.float32),
            goal_pos_map=goal_pos_map,
            goal_score_map=goal_score_map,
            wall_map=wall_map,
        )
        obs = self.get_obs(state)
        return jax.lax.stop_gradient(state), jax.lax.stop_gradient(obs)

    def _crop_map(self, full_map: jax.Array, pos: jax.Array) -> jax.Array:
        full_map = jnp.pad(full_map, self.view_range + self.view_upper_left_offset)
        return jax.lax.dynamic_slice(full_map, pos, (self.obs_size, self.obs_size))
    
    def get_agent_obs(self, state: GridWorldState, agent_id: int) -> jax.Array:
        agent_pos = state.agent_pos[agent_id]
        agent_trial_best_score = state.agent_trial_best_score[agent_id]
        h, w = self.grid_size, self.grid_size
        empty_map = jnp.zeros((h, w), dtype=jnp.float32)

        pos_map = empty_map.at[agent_pos[0], agent_pos[1]].set(2 + agent_trial_best_score)
        pos_map = self._crop_map(pos_map, agent_pos)
        goal_map = self._crop_map(state.goal_pos_map, agent_pos)
        wall_map = self._crop_map(state.wall_map, agent_pos)

        return jnp.concatenate([
            wall_map.reshape(self.obs_size, self.obs_size, 1),
            pos_map.reshape(self.obs_size, self.obs_size, 1),
            goal_map.reshape(self.obs_size, self.obs_size, 1),
        ], axis=-1)

    def get_obs(self, state: GridWorldState) -> Observation:
        my_obs = jax.vmap(self.get_agent_obs, in_axes=(None, 0))(state, jnp.arange(self.n_agent))

        h, w = self.grid_size, self.grid_size
        empty_map = jnp.zeros((h, w), dtype=jnp.float32)
        other_obs = jnp.stack([
            jnp.stack([
                self._crop_map(empty_map.at[state.agent_pos[j, 0], state.agent_pos[j, 1]].set(2 + state.agent_discount_trial_ret[j]), state.agent_pos[i])
                for j in range(self.n_agent) if i != j
            ], axis=-1)
            for i in range(self.n_agent)
        ], axis=0)

        return Observation({
            'grid': jnp.concatenate([my_obs, other_obs], axis=-1)
        })

    def agent_step(self, rng: jax.Array, const: GridWorldConst, state: GridWorldState, agent_id: int, action: int) -> dict[str, jax.Array]:
        agent_pos = state.agent_pos[agent_id]
        agent_allow_move = state.agent_allow_move[agent_id]
        agent_trial_best_score = state.agent_trial_best_score[agent_id]

        move = jnp.array([0, 0])
        move = jax.lax.cond(action == Actions.UP, lambda _: jnp.array([-1, 0]), lambda _: move, None)
        move = jax.lax.cond(action == Actions.DOWN, lambda _: jnp.array([1, 0]), lambda _: move, None)
        move = jax.lax.cond(action == Actions.LEFT, lambda _: jnp.array([0, -1]), lambda _: move, None)
        move = jax.lax.cond(action == Actions.RIGHT, lambda _: jnp.array([0, 1]), lambda _: move, None)

        move = jax.lax.cond(agent_allow_move, lambda _: move, lambda _: jnp.array([0, 0]), None)

        new_pos = agent_pos + move

        collide_wall = state.wall_map[new_pos[0], new_pos[1]]
        new_pos = jax.lax.cond(collide_wall, lambda _: agent_pos, lambda _: new_pos, None)

        reach_goal = state.goal_pos_map[new_pos[0], new_pos[1]]

        score = reach_goal * state.goal_score_map[new_pos[0], new_pos[1]] + (1 - reach_goal) * -1

        goal_pos_in_view = self._crop_map(state.goal_pos_map, agent_pos)
        goal_detected = jnp.any(goal_pos_in_view)

        reward = score - agent_trial_best_score
        reward += const.goal_detect_reward * goal_detected
        
        return {
            'new_pos': new_pos,
            'reach_goal': reach_goal,
            'score': score,
            'goal_detected': goal_detected,
            'best_score': jnp.maximum(score, agent_trial_best_score),
            'reward': reward
        }
    
    def _reset_for_trial(self, rng: jax.Array, const: GridWorldConst, state: GridWorldState) -> GridWorldState:
        rng, rng_reset_pos = jax.random.split(rng)
        agent_pos, goal_pos, clutter_pos = self._reset_positions(rng_reset_pos, const)
        return state.replace(
            agent_pos=agent_pos,
            agent_allow_move=jnp.ones((self.n_agent,), dtype=jnp.bool_),
            agent_trial_best_score=-jnp.ones((self.n_agent,), dtype=jnp.float32),
            agent_discount_trial_ret=self.last_ret_scale * state.agent_discount_trial_ret + self.history_ret_scale * state.agent_trial_best_score 
        )

    def env_step(
        self,
        rng: jax.Array,
        const: GridWorldConst,
        state: GridWorldState,
        action: Action
    ) -> tuple[GridWorldState, Observation, jax.Array, jax.Array, dict[Any, Any]]:
        env_action = action['all']
        rng, rng_agent_step_batch = jax.random.split(rng)
        rng_agent_step_batch = jax.random.split(rng_agent_step_batch, self.n_agent)
        data = jax.vmap(self.agent_step, in_axes=(0, None, None, 0, 0))(rng_agent_step_batch, const, state, jnp.arange(self.n_agent), env_action)

        reward = data['reward']

        num_steps = state._step + 1
        done = num_steps >= const.max_steps
        new_state = state.replace(
            _step=state._step + 1,
            agent_pos=data['new_pos'],
            agent_trial_best_score=data['best_score'],
        )
        if self.stop_after_goal_reach:
            agent_allow_move = jnp.logical_and(
                state.agent_allow_move,
                jnp.logical_not(data['reach_goal'])
            )
            new_state = new_state.replace(agent_allow_move=agent_allow_move)        
        
        # Reset for each trial
        rng, rng_trial = jax.random.split(rng)
        state_new_trial = self._reset_for_trial(rng_trial, const, new_state)
        new_state = jax.lax.cond(num_steps % self.trial_steps == 0, lambda _: state_new_trial, lambda _: new_state, None)

        rng, rng_reset = jax.random.split(rng)
        state_reset, obs_reset = self.env_reset(rng_reset, const)

        return jax.lax.cond(
            done,
            lambda _: (state_reset, obs_reset, reward, jnp.repeat(done, self.n_agent), {}),
            lambda _: (new_state, self.get_obs(new_state), reward, jnp.repeat(done, self.n_agent), {}),
            None
        )
    
    @property
    def num_agents(self) -> int:
        return self.n_agent

    def get_agent_batch_size(self) -> list[int]:
        return [1 for _ in range(self.num_agents)]

    def get_observation_space(self) -> list[ObservationSpace]:
        return [self.env_observation_space[0] for _ in range(self.n_agent)]
    
    def get_action_space(self) -> list[ActionSpace]:
        return [self.env_action_space[0] for _ in range(self.n_agent)]

    def reset(
        self,
        rng: jax.Array,
        const: GridWorldConst
    ) -> tuple[GridWorldState, list[Observation]]:
        state, env_obs = self.env_reset(rng, const)
        obs_lst = [env_obs[i:i+1] for i in range(self.n_agent)]
        return state, obs_lst
    
    def step(
        self,
        rng: jax.Array,
        const: GridWorldConst,
        state: GridWorldState,
        action: Sequence[Action]
    ) -> tuple[GridWorldState, list[Observation], list[jax.Array], list[jax.Array], dict[Any, Any]]:
        env_action = Action({
            'all': jnp.concatenate([agent_action['all'] for agent_action in action])
        })
        state, env_obs, env_reward, env_done, info = self.env_step(rng, const, state, env_action)
        obs_lst = [env_obs[i:i+1] for i in range(self.n_agent)]
        reward_lst = [env_reward[i:i+1] for i in range(self.n_agent)]
        done_lst = [env_done[i:i+1] for i in range(self.n_agent)]
        return state, obs_lst, reward_lst, done_lst, info
    
    def export_gif(self, states: Sequence[GridWorldState], filename: str = "gridworld.gif", agent_id: int = 0):
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        from tqdm import tqdm
        import seaborn as sns
        import jax.numpy as jnp

        TILE_SIZE = 80 
        W = self.grid_size * TILE_SIZE
        font_size = 52
        try:
            font = ImageFont.truetype("LiberationSerif-Regular.ttf", font_size)
        except:
            font = ImageFont.load_default(size=font_size)

        sns_palette = (np.array(sns.color_palette("tab10", n_colors=10)) * 255).astype(np.uint8)
        COLOR_GRAY, COLOR_BLACK, COLOR_GREEN, COLOR_RED = (60, 60, 60), (20, 20, 20), (0, 255, 0), (255, 0, 0)

        wall_mask = np.array(states[0].wall_map) 
        base_layer = Image.new("RGBA", (W, W), (0, 0, 0, 0))
        d_base = ImageDraw.Draw(base_layer)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                rect = [j*TILE_SIZE, i*TILE_SIZE, (j+1)*TILE_SIZE, (i+1)*TILE_SIZE]
                if wall_mask[i, j]:
                    d_base.rectangle(rect, fill=(*COLOR_GRAY, 255))
                d_base.rectangle(rect, outline=(0, 0, 0, 255), width=1)

        frames = []
        for env_step, state in enumerate(states):
            bg_map = np.full((self.grid_size, self.grid_size), 76.0)
            for i in range(self.n_agent):
                x, y = int(state.agent_pos[i][0]), int(state.agent_pos[i][1])
                bg_map[max(0, x-self.view_range-self.view_upper_left_offset):min(self.grid_size, x+self.view_range+1), 
                       max(0, y-self.view_range-self.view_upper_left_offset):min(self.grid_size, y+self.view_range+1)] = 255.0

            pixel_bg = np.repeat(np.repeat(bg_map, TILE_SIZE, axis=0), TILE_SIZE, axis=1).astype(np.uint8)
            img = Image.fromarray(np.stack([pixel_bg]*3, axis=-1), mode="RGB")
            
            img.paste(base_layer, (0, 0), base_layer)
            draw = ImageDraw.Draw(img)

            black_vec, green_vec, red_vec = jnp.array(COLOR_BLACK), jnp.array(COLOR_GREEN), jnp.array(COLOR_RED)
            
            goal_coords = np.argwhere(np.array(state.goal_pos_map))
            for i, j in goal_coords:
                s_goal = jnp.clip(state.goal_score_map[i, j] / self.goal_score_scale, -1, 1)
                c_vec = (black_vec + (green_vec - black_vec) * jnp.clip(s_goal, 0, 1) + 
                         (red_vec - black_vec) * jnp.clip(-s_goal, 0, 1)).clip(0, 255)
                c_goal = tuple(np.array(c_vec).astype(np.uint8).tolist())
                draw.rectangle([j*TILE_SIZE+10, i*TILE_SIZE+10, (j+1)*TILE_SIZE-10, (i+1)*TILE_SIZE-10], fill=c_goal)
                
            for i in range(self.n_agent):
                s = jnp.clip(state.agent_discount_trial_ret[i], -1, 1)
                color_vec = (black_vec + (green_vec - black_vec) * jnp.clip(s, 0, 1) + 
                             (red_vec - black_vec) * jnp.clip(-s, 0, 1)).clip(0, 255)
                agent_fill = tuple(np.array(color_vec).astype(np.uint8).tolist())
                
                pos_i, pos_j = int(state.agent_pos[i][0]), int(state.agent_pos[i][1])
                rect = [pos_j*TILE_SIZE, pos_i*TILE_SIZE, (pos_j+1)*TILE_SIZE, (pos_i+1)*TILE_SIZE]
                
                draw.rectangle(rect, fill=agent_fill, outline=(0, 0, 0), width=2)
                
                text = str(i)
                bbox = draw.textbbox((0, 0), text, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                draw.text((rect[0]+(TILE_SIZE-tw)//2, rect[1]+(TILE_SIZE-th)//2-10), text, fill=(255, 255, 255), font=font)
            
            step_text = f"Step: {env_step + 1} / {len(states)}"
            t_bbox = draw.textbbox((0, 0), step_text, font=font)
            t_w = t_bbox[2] - t_bbox[0]
            draw.text((W - t_w - 20, W - TILE_SIZE + 10), step_text, fill=(255, 255, 255), font=font)

            frames.append(img)
        
        sample_img = Image.fromarray(np.uint8(np.concatenate([np.array(f) for f in frames], axis=0)))
        global_palette = sample_img.convert("P", palette=Image.ADAPTIVE, colors=256)
        frames = [f.quantize(palette=global_palette) for f in frames]

        frames[0].save(filename, save_all=True, append_images=frames[1:], duration=500, loop=0, optimize=True)
        tqdm.write(f"\nVisualizing: GIF successfully exported to {filename}\n")

class NaiveFourPlayerGame(GridWorldEnv):
    def __init__(self, num_agents = 4, num_goals = 4, grid_size = 20, view_range = 5, clutter_density = 0, trial_steps = 20, max_steps = 120, goal_score_scale = 1, stop_after_goal_reach = True, last_ret_scale = 0.4, history_ret_scale = 0.6):
        assert num_agents == 4
        assert num_goals == 4
        assert int(clutter_density * (grid_size - 2) * (grid_size - 2)) == 0
        super().__init__(num_agents, num_goals, grid_size, view_range, clutter_density, trial_steps, max_steps, goal_score_scale, stop_after_goal_reach, last_ret_scale, history_ret_scale)
    
    def _reset_positions(self, rng: jax.Array, const: GridWorldConst) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Reset the map, and the positions of all agents and goals
        """
        h, w = self.grid_size, self.grid_size
        n_agent, n_goal, n_clutter = self.n_agent, self.n_goal, self.n_clutter
        available_positions = self.available_positions

        rng, rng_index = jax.random.split(rng)
        indices = jax.random.choice(rng_index, len(available_positions), shape=(n_agent + n_goal + n_clutter,), replace=False)
        agent_indicies, goal_indicies, clutter_indicies = indices[:n_agent], indices[n_agent:n_agent+n_goal], indices[n_agent+n_goal:]
        
        agent_pos = available_positions[agent_indicies]
        goal_pos = available_positions[goal_indicies]
        clutter_pos = available_positions[clutter_indicies]

        agent_area_size = self.view_range // 2 + 1
        goal_area_size = (self.grid_size - self.view_range) // 2

        agent_pos %= agent_area_size
        agent_pos += 1
        agent_pos = jnp.array([
            [h // 2 - agent_pos[0, 0], w // 2 - agent_pos[0, 1]],
            [h // 2 - agent_pos[1, 0], w // 2 + agent_pos[1, 1]],
            [h // 2 + agent_pos[2, 0], w // 2 - agent_pos[2, 1]],
            [h // 2 + agent_pos[3, 0], w // 2 + agent_pos[3, 1]],
        ])

        goal_pos %= goal_area_size
        goal_pos = jnp.array([
            [1 + goal_pos[0, 0], 1 + goal_pos[0, 1]],
            [1 + goal_pos[1, 0], w - 2 - goal_pos[1, 1]],
            [h - 2 - goal_pos[2, 0], 1 + goal_pos[2, 1]],
            [h - 2 - goal_pos[3, 0], w - 2 - goal_pos[3, 1]],
        ])
        return agent_pos, goal_pos, clutter_pos
    
    def env_reset(
        self,
        rng: jax.Array,
        const: GridWorldConst
    ) -> tuple[GridWorldState, Observation]:
        """Resets the environment to an initial state."""
        h, w = self.grid_size, self.grid_size

        rng, rng_reset_pos = jax.random.split(rng)
        agent_pos, goal_pos, clutter_pos = self._reset_positions(rng_reset_pos, const)
        
        rng, rng_score = jax.random.split(rng)
        goal_scores = 0.10 * self.goal_score_scale * jax.random.uniform(rng_score, shape=(self.n_goal,), minval=-1, maxval=1)
        rng, rng_base_perm = jax.random.split(rng)
        goal_scores += jax.random.permutation(rng_base_perm, jnp.array([-0.8, -0.2, 0.2, 0.8], dtype=jnp.float32))
        goal_pos_map = jnp.zeros((h, w), dtype=jnp.bool_)
        goal_pos_map = goal_pos_map.at[goal_pos[:, 0], goal_pos[:, 1]].set(True)
        goal_score_map = jnp.zeros((h, w), dtype=jnp.float32)
        goal_score_map = goal_score_map.at[goal_pos[:, 0], goal_pos[:, 1]].set(goal_scores)

        wall_map = jnp.zeros((h, w), dtype=jnp.bool_)
        wall_map = wall_map.at[0, :].set(True)
        wall_map = wall_map.at[-1, :].set(True)
        wall_map = wall_map.at[:, 0].set(True)
        wall_map = wall_map.at[:, -1].set(True)
        wall_map = wall_map.at[clutter_pos[:, 0], clutter_pos[:, 1]].set(True)

        state = GridWorldState(
            _step=0,
            agent_pos=agent_pos,
            agent_allow_move=jnp.ones((self.n_agent,), dtype=jnp.bool_),
            agent_trial_best_score=-jnp.ones((self.n_agent,), dtype=jnp.float32),
            agent_discount_trial_ret=jnp.zeros((self.n_agent,), dtype=jnp.float32),
            goal_pos_map=goal_pos_map,
            goal_score_map=goal_score_map,
            wall_map=wall_map,
        )
        obs = self.get_obs(state)
        return jax.lax.stop_gradient(state), jax.lax.stop_gradient(obs)

class TwoRoundFourPlayerGame(NaiveFourPlayerGame):
    def __init__(self, num_agents = 4, num_goals = 4, grid_size = 20, view_range = 5, clutter_density = 0, trial_steps = 20, max_steps = 40, goal_score_scale = 1, stop_after_goal_reach = True, last_ret_scale = 0.4, history_ret_scale = 0.6):
        assert num_agents == 4
        assert num_goals == 4
        assert int(clutter_density * (grid_size - 2) * (grid_size - 2)) == 0
        assert max_steps == 2 * trial_steps
        super().__init__(num_agents, num_goals, grid_size, view_range, clutter_density, trial_steps, max_steps, goal_score_scale, stop_after_goal_reach, last_ret_scale, history_ret_scale)
        
    def env_step(
        self,
        rng: jax.Array,
        const: GridWorldConst,
        state: GridWorldState,
        action: Action
    ) -> tuple[GridWorldState, Observation, jax.Array, jax.Array, dict[Any, Any]]:
        env_action = action['all']
        rng, rng_agent_step_batch = jax.random.split(rng)
        rng_agent_step_batch = jax.random.split(rng_agent_step_batch, self.n_agent)
        data = jax.vmap(self.agent_step, in_axes=(0, None, None, 0, 0))(rng_agent_step_batch, const, state, jnp.arange(self.n_agent), env_action)

        reward = data['reward']

        num_steps = state._step + 1

        # Additional rewards at the second trial
        second_trial_reward = jax.lax.cond(
            num_steps == 2 * self.trial_steps,
            lambda _: 10 * data['best_score'],
            lambda _: 0 * data['best_score'],
            None
        )
        agent_occupy_map = jnp.zeros((self.grid_size, self.grid_size), dtype=jnp.bool_).at[state.agent_pos[:, 0], state.agent_pos[:, 1]].set(True)
        num_goal_covered = (state.goal_pos_map * agent_occupy_map).astype(jnp.float32).sum()
        is_hit_best = second_trial_reward >= 10 * 0.5
        reward += second_trial_reward


        done = num_steps >= const.max_steps
        new_state = state.replace(
            _step=state._step + 1,
            agent_pos=data['new_pos'],
            agent_trial_best_score=data['best_score'],
        )
        if self.stop_after_goal_reach:
            agent_allow_move = jnp.logical_and(
                state.agent_allow_move,
                jnp.logical_not(data['reach_goal'])
            )
            new_state = new_state.replace(agent_allow_move=agent_allow_move)        
        
        # Reset for each trial
        rng, rng_trial = jax.random.split(rng)
        state_new_trial = self._reset_for_trial(rng_trial, const, new_state)
        new_state = jax.lax.cond(num_steps % self.trial_steps == 0, lambda _: state_new_trial, lambda _: new_state, None)

        rng, rng_reset = jax.random.split(rng)
        state_reset, obs_reset = self.env_reset(rng_reset, const)

        return jax.lax.cond(
            done,
            lambda _: (state_reset, obs_reset, reward, jnp.repeat(done, self.n_agent), {'num_goal_covered': jnp.zeros_like(num_goal_covered), 'is_hit_best': jnp.zeros_like(is_hit_best)}),
            lambda _: (new_state, self.get_obs(new_state), reward, jnp.repeat(done, self.n_agent), {'num_goal_covered': num_goal_covered, 'is_hit_best': is_hit_best}),
            None
        )

class TwoRoundSinglePlayerGame(TwoRoundFourPlayerGame):
    def get_obs(self, state: GridWorldState) -> Observation:
        my_obs = jax.vmap(self.get_agent_obs, in_axes=(None, 0))(state, jnp.arange(self.n_agent))

        h, w = self.grid_size, self.grid_size
        empty_map = jnp.zeros((h, w), dtype=jnp.float32)
        other_obs = jnp.stack([
            jnp.stack([
                self._crop_map(empty_map, state.agent_pos[i]) # Zero out the position of other agents
                for j in range(self.n_agent) if i != j
            ], axis=-1)
            for i in range(self.n_agent)
        ], axis=0)

        return Observation({
            'grid': jnp.concatenate([my_obs, other_obs], axis=-1)
        })