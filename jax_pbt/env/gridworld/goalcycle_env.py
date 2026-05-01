from enum import IntEnum
from typing import Any, Sequence

from flax.struct import PyTreeNode
import jax
import jax.numpy as jnp
from jax.lax import stop_gradient

from ..base_env import BaseEnvConst, BaseEnvState, BaseEnv
from ..spaces import Observation, Action, ObservationSpace, ActionSpace

class GoalCycleConst(BaseEnvConst):
    goal_reward: float
    goal_penalty: float
    distance_penalty: float

class AgentState(PyTreeNode):
    pos: jax.Array
    last_visited_goal: jax.Array
    last_step_reward: jax.Array
    prestige: jax.Array

class GoalCycleState(BaseEnvState):
    agent_state: AgentState
    goal_pos: jax.Array
    goal_channel_assignment: jax.Array
    wall_map: jax.Array

class Actions(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2 
    RIGHT = 3 
    STAY = 4

class GoalCycleEnv(BaseEnv[GoalCycleConst, GoalCycleState]):
    def __init__(
        self, 
        num_agents: int = 1,
        num_goals: int = 3,
        grid_size: int = 10, 
        max_steps: int = 60, 
        distance_penalty: float = 0.0, 
        goal_reward: float = 1.0,
        goal_penalty: float = -1.0, 
        obstacle_density: float = 0.05,
        is_obs_prestige: bool = True,
        is_obs_last_step_r: bool = True,
        is_obs_all_agent_pos: bool = True,
    ) -> None:
        """
        [Param]
            num_agents: Number of players
            num_goals: NUmber of goals.
            grid_size: The size of the grid (grid_size x grid_size).
            max_steps: Maximum steps before the environment terminates.
            distance_penalty: Penalty for each step based on Manhattan distance to the goal.
            goal_reward: Reward for reaching the goal.
            goal_penalty: Penalty for incorrect goal order visiting.
            obstacle_density: Ratio of number_of_obstacles / grid_world_size for obstacle generation

            # Optional
            is_obs_prestige: Whether the prestige of all agents are observable.
            is_obs_last_step_r: Whether the agent's observation includes the reward received in the previous step.
            is_obs_all_agent_pos: Whether each agent's observation includes the positions of all agents (otherwise only self-position is visible).
        """
        self.n_agent = num_agents
        self.n_goal = num_goals
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.distance_penalty = distance_penalty
        self.goal_reward = goal_reward
        self.goal_penalty = goal_penalty
        self.n_obstacle = int(obstacle_density * (grid_size - 2) * (grid_size - 2))

        self.is_obs_prestige = is_obs_prestige
        self.is_obs_last_step_r = is_obs_last_step_r
        self.is_obs_all_agent_pos = is_obs_all_agent_pos

        h, w = self.grid_size, self.grid_size
        self.available_positions = jnp.array([
            (x, y) for x in range(1, h - 1) for y in range(1, w - 1)
        ])
        # Currently the goal order is 1...n by default; Customized orders need further development
    
    @property
    def default_const(self) -> GoalCycleConst:
        return GoalCycleConst(max_steps=self.max_steps, goal_reward=self.goal_reward, goal_penalty=self.goal_penalty, distance_penalty=self.distance_penalty)
    
    @property
    def env_observation_space(self) -> tuple[ObservationSpace, dict[str, tuple[int]]]:
        grid_num_channels = {
            'wall': 1,
            'my_pos': 1,
            'goal_pos': self.n_goal,
            **({'all_agent_pos': self.n_agent} if self.is_obs_all_agent_pos else {})
        }
        agent_info_num_features = {
            'time_duration': 1,
            **({'onehot_id': self.n_agent} if self.is_obs_prestige or self.is_obs_all_agent_pos else {}),
            **({'prestige': self.n_agent} if self.is_obs_prestige else {}),
            **({'last_reward': 1} if self.is_obs_last_step_r else {}),
        }

        grid_obs_space = (self.grid_size, self.grid_size, sum(grid_num_channels.values()))
        agent_info_obs_space = (sum(agent_info_num_features.values()),)
        return ObservationSpace({'grid': grid_obs_space, 'agent_info': agent_info_obs_space}), {'grid': (self.n_agent,), 'agent_info': (self.n_agent,)}
    
    @property
    def env_action_space(self) -> tuple[ActionSpace, dict[str, tuple[int]]]:
        num_actions = len(Actions)
        return ActionSpace({'all': num_actions}), {'all': (self.n_agent,)}
    
    def _reset_positions(self, rng: jax.Array, const: GoalCycleConst) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Reset the map, and the positions of all agents and goals.
        """
        h, w = self.grid_size, self.grid_size
        n_agent, n_goal, n_obstacle = self.n_agent, self.n_goal, self.n_obstacle
        available_positions = self.available_positions

        rng, rng_index = jax.random.split(rng)
        indices = jax.random.choice(rng_index, len(available_positions), shape=(n_agent + n_goal + n_obstacle,), replace=False)
        agent_indicies, goal_indicies, obstacle_indicies = indices[:n_agent], indices[n_agent:n_agent+n_goal], indices[n_agent+n_goal:]
        
        agent_pos = available_positions[agent_indicies]
        goal_pos = available_positions[goal_indicies]
        obstacle_pos = available_positions[obstacle_indicies]
        return agent_pos, goal_pos, obstacle_pos
    
    def env_reset(
        self,
        rng: jax.Array,
        const: GoalCycleConst
    ) -> tuple[GoalCycleState, Observation]:
        """Resets the environment to an initial state."""
        rng, rng_reset_pos = jax.random.split(rng)
        agent_pos, goal_pos, obstacle_pos = self._reset_positions(rng_reset_pos, const)

        h, w = self.grid_size, self.grid_size

        wall_map = jnp.zeros((h, w), dtype=jnp.bool_)
        wall_map = wall_map.at[0, :].set(True)
        wall_map = wall_map.at[-1, :].set(True)
        wall_map = wall_map.at[:, 0].set(True)
        wall_map = wall_map.at[:, -1].set(True)
        wall_map = wall_map.at[obstacle_pos[:, 0], obstacle_pos[:, 1]].set(True)

        goal_channel_assignment = jnp.arange(self.n_goal) # Currently assume the order of goal is 1...n

        state = GoalCycleState(
            _step=0,
            agent_state=AgentState(
                pos=agent_pos,
                last_visited_goal=-jnp.ones((self.n_agent,), dtype=jnp.int32),
                last_step_reward=jnp.zeros((self.n_agent,), dtype=jnp.float32),
                prestige=jnp.zeros((self.n_agent,), dtype=jnp.float32)
            ),
            goal_pos=goal_pos,
            goal_channel_assignment=goal_channel_assignment,
            wall_map=wall_map
        )
        obs = self.get_obs(state)
        return stop_gradient(state), stop_gradient(obs)
    
    def get_agent_obs(self, state: GoalCycleState, agent_state: AgentState, agent_id: int) -> Observation:
        # grid_num_channels = {
        #     'wall': 1,
        #     'my_pos': 1,
        #     'goal_pos': self.n_goal,
        #     **({'all_agent_pos': self.n_agent} if self.is_obs_all_agent_pos else {})
        # }
        # agent_info_num_features = {
        #     'time_duration': 1,
        #     **({'onehot_id': self.n_agent} if self.is_obs_prestige or self.is_obs_all_agent_pos else {}),
        #     **({'prestige': self.n_agent} if self.is_obs_prestige else {}),
        #     **({'last_reward': 1} if self.is_obs_last_step_r else {}),
        # }
        h, w = self.grid_size, self.grid_size
        agent_pos = agent_state.pos
        
        # 2D observation: [H, W, Channel]
        grid_channels = {
            'wall': state.wall_map.reshape(h, w, 1).astype(jnp.float32),
            'my_pos': jnp.zeros((h, w, 1), dtype=jnp.float32).at[agent_pos[0], agent_pos[1], 0].set(1),
            'goal_pos': jnp.zeros((h, w, self.n_goal), dtype=jnp.float32).at[state.goal_pos[:, 0], state.goal_pos[:, 1], state.goal_channel_assignment].set(1),
            **({
                'all_agent_pos': jnp.zeros((h, w, self.n_agent), dtype=jnp.float32).at[state.agent_state.pos[:, 0], state.agent_state.pos[:, 1], jnp.arange(self.n_agent)].set(1)
            } if self.is_obs_all_agent_pos else {})
        }
        grid_obs = jnp.concatenate([v for k, v in sorted(grid_channels.items())], axis=-1)

        agent_info_features = {
            'time_duration': jnp.array([state._step / self.max_steps], dtype=jnp.float32),
            **({
                'onehot_id': jnp.eye(self.n_agent, dtype=jnp.float32)[agent_id]
            } if self.is_obs_prestige or self.is_obs_all_agent_pos else {}),
            **({
                'prestige': state.agent_state.prestige
            } if self.is_obs_prestige else {}),
            **({
                'last_reward': jnp.array([agent_state.last_step_reward])
            } if self.is_obs_last_step_r else {})
        }
        agent_info_obs = jnp.concatenate([v for k, v in sorted(agent_info_features.items())], axis=-1)
        return Observation({'grid': grid_obs, 'agent_info': agent_info_obs})

    def get_obs(self, state: GoalCycleState) -> Observation:
        return jax.vmap(self.get_agent_obs, in_axes=(None, 0, 0))(state, state.agent_state, jnp.arange(self.n_agent))

    def agent_step(self, rng: jax.Array, const: GoalCycleConst, state: GoalCycleState, agent_state: AgentState, action: int) -> dict[str, jax.Array]:
        agent_pos = agent_state.pos
        agent_last_visited_goal = agent_state.last_visited_goal
        agent_prestige = agent_state.prestige

        # Process agent movements
        move = jnp.array([0, 0])
        move = jax.lax.cond(action == Actions.UP, lambda _: jnp.array([-1, 0]), lambda _: move, None)
        move = jax.lax.cond(action == Actions.DOWN, lambda _: jnp.array([1, 0]), lambda _: move, None)
        move = jax.lax.cond(action == Actions.LEFT, lambda _: jnp.array([0, -1]), lambda _: move, None)
        move = jax.lax.cond(action == Actions.RIGHT, lambda _: jnp.array([0, 1]), lambda _: move, None)

        new_pos = agent_pos + move
        collides = state.wall_map[new_pos[0], new_pos[1]]
        new_pos = jax.lax.cond(collides, lambda _: agent_pos, lambda _: new_pos, None)

        is_moving = jnp.any(agent_pos != new_pos)
        is_goal_hit = jax.vmap(lambda goal_pos: jnp.logical_and(jnp.all(goal_pos == new_pos), is_moving))(state.goal_pos)
        goal_id = jnp.select(is_goal_hit, jnp.arange(self.n_goal), -1) # -1 if not hit/move
        is_correct_goal_hit = jnp.logical_and(
            goal_id >= 0,
            jnp.logical_or(
                agent_last_visited_goal == -1,
                goal_id == (agent_last_visited_goal + 1) % self.n_goal
            )
        )
        is_incorrect_goal_hit = jnp.logical_and(
            goal_id >= 0,
            jnp.logical_and(
                agent_last_visited_goal != -1,
                goal_id != (agent_last_visited_goal + 1) % self.n_goal
            )
        )

        new_agent_last_visited_goal = jax.lax.cond(is_correct_goal_hit, lambda _: goal_id, lambda _: agent_last_visited_goal, None)
        reward = const.goal_reward * is_correct_goal_hit + const.goal_penalty * is_incorrect_goal_hit
        agent_prestige = agent_prestige * 0.99 + reward
        return {
            'new_pos': new_pos,
            'new_agent_last_visited_goal': new_agent_last_visited_goal,
            'reward': reward,
            'prestige': agent_prestige
        }

    def env_step(
        self,
        rng: jax.Array,
        const: GoalCycleConst,
        state: GoalCycleState,
        action: Action
    ) -> tuple[GoalCycleState, Observation, jax.Array, jax.Array, dict[Any, Any]]:
        env_action = action['all']
        rng, rng_agent_step_batch = jax.random.split(rng)
        rng_agent_step_batch = jax.random.split(rng_agent_step_batch, self.n_agent)
        data = jax.vmap(self.agent_step, in_axes=(0, None, None, 0, 0))(rng_agent_step_batch, const, state, state.agent_state, env_action)
        
        num_steps = state._step + 1
        done = num_steps >= const.max_steps
        new_state = state.replace(
            _step=state._step + 1,
            agent_state=AgentState(
                pos=data['new_pos'],
                last_visited_goal=data['new_agent_last_visited_goal'],
                last_step_reward=data['reward'],
                prestige=data['prestige']
            )
        )
        reward = data['reward']

        rng, rng_reset = jax.random.split(rng)
        state_reset, obs_reset = self.env_reset(rng_reset, const)

        s, o, r, d, info = jax.lax.cond(
            done,
            lambda _: (state_reset, obs_reset, reward, jnp.repeat(done, self.n_agent), {}),
            lambda _: (new_state, self.get_obs(new_state), reward, jnp.repeat(done, self.n_agent), {}),
            None
        )
        return stop_gradient(s), stop_gradient(o), stop_gradient(r), stop_gradient(d), stop_gradient(info)
    
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
        const: GoalCycleConst
    ) -> tuple[GoalCycleState, list[Observation]]:
        state, env_obs = self.env_reset(rng, const)
        obs_lst = [env_obs[i:i+1] for i in range(self.n_agent)]
        return state, obs_lst
    
    def step(
        self,
        rng: jax.Array,
        const: GoalCycleConst,
        state: GoalCycleState,
        action: Sequence[Action]
    ) -> tuple[GoalCycleState, list[Observation], list[jax.Array], list[jax.Array], dict[Any, Any]]:
        env_action = Action({
            'all': jnp.concatenate([agent_action['all'] for agent_action in action])
        })
        state, env_obs, env_reward, env_done, info = self.env_step(rng, const, state, env_action)
        obs_lst = [env_obs[i:i+1] for i in range(self.n_agent)]
        reward_lst = [env_reward[i:i+1] for i in range(self.n_agent)]
        done_lst = [env_done[i:i+1] for i in range(self.n_agent)]
        return state, obs_lst, reward_lst, done_lst, info

    def export_gif(
        self,
        states: list[GoalCycleState],
        agent_id: int = 0,
        filename: str = "goal_cycle.gif",
    ):
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        from tqdm import tqdm
        import jax.numpy as jnp

        TILE_SIZE = 80
        W = self.grid_size * TILE_SIZE
        font_size = 52
        try:
            font = ImageFont.truetype("LiberationSerif-Regular.ttf", font_size)
        except:
            font = ImageFont.load_default(size=font_size)

        COLOR_GRAY, COLOR_BLACK, COLOR_GREEN, COLOR_RED = (60, 60, 60), (20, 20, 20), (0, 255, 0), (255, 0, 0)

        wall_mask = np.array(states[0].wall_map)
        base_layer = Image.new("RGBA", (W, W), (0, 0, 0, 0))
        d_base = ImageDraw.Draw(base_layer)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                rect = [j*TILE_SIZE, i*TILE_SIZE, (j+1)*TILE_SIZE, (i+1)*TILE_SIZE]
                if wall_mask[i, j]:
                    d_base.rectangle(rect, fill=(*COLOR_GRAY, 255)) # Wall
                d_base.rectangle(rect, outline=(0, 0, 0, 255), width=1) # Boundary

        frames = []
        for env_step, state in enumerate(states):
            bg_map = np.full((self.grid_size, self.grid_size), 255.0)

            pixel_bg = np.repeat(np.repeat(bg_map, TILE_SIZE, axis=0), TILE_SIZE, axis=1).astype(np.uint8)
            img = Image.fromarray(np.stack([pixel_bg] * 3, axis=-1), mode="RGB")

            img.paste(base_layer, (0, 0), base_layer)
            draw = ImageDraw.Draw(img)

            # -------- Goals (A, B, C, ...)
            for k in range(self.n_goal):
                i, j = int(state.goal_pos[k, 0]), int(state.goal_pos[k, 1])
                draw.rectangle(
                    [j * TILE_SIZE + 10, i * TILE_SIZE + 10, (j + 1) * TILE_SIZE - 10, (i + 1) * TILE_SIZE - 10]
                )

                text = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdedfhijklmnopqrstuvwxyz"[k]
                bbox = draw.textbbox((0, 0), text, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                draw.text(
                    (j * TILE_SIZE + (TILE_SIZE - tw) // 2, i * TILE_SIZE + (TILE_SIZE - th) // 2 - 10),
                    text,
                    fill=(0, 0, 0),
                    font=font,
                )

            # -------- Agents
            for i in range(self.n_agent):
                black, green, red = jnp.array(COLOR_BLACK), jnp.array(COLOR_GREEN), jnp.array(COLOR_RED)

                pos_i, pos_j = int(state.agent_state.pos[i][0]), int(state.agent_state.pos[i][1])
                rect = [pos_j * TILE_SIZE, pos_i * TILE_SIZE, (pos_j + 1) * TILE_SIZE, (pos_i + 1) * TILE_SIZE]
                draw.rectangle(rect, fill=COLOR_BLACK, outline=(0, 0, 0), width=2)

                prestige = state.agent_state.prestige[i]
                fill_ratio = prestige / 10
                filled_h = int(TILE_SIZE * fill_ratio)
                if filled_h > 0:
                    prestige_color = black + (green - black) * jnp.clip(prestige, 0, 1) + + (red - black) * jnp.clip(-prestige, 0, 1).clip(0, 255)
                    prestige_fill = tuple(np.array(prestige_color).astype(np.uint8).tolist())
                    draw.rectangle([rect[0], rect[3] - filled_h, rect[2], rect[3]], fill=prestige_fill)

                text = str(i)
                bbox = draw.textbbox((0, 0), text, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                draw.text((rect[0] + (TILE_SIZE - tw) // 2, rect[1] + (TILE_SIZE - th) // 2 - 10), text, fill=(255, 255, 255), font=font)

                if i != agent_id and not self.is_obs_all_agent_pos:
                    draw.line([rect[0] + 15, rect[1] + 15, rect[2] - 15, rect[3] - 15], fill=(255, 255, 255), width=6)
                    draw.line([rect[2] - 15, rect[1] + 15, rect[0] + 15, rect[3] - 15], fill=(255, 255, 255), width=6)

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
