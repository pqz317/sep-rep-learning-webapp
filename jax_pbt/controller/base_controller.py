from abc import ABC
import os
from typing import Any, Sequence

import jax
import jax.numpy as jnp
from flax.struct import PyTreeNode
import numpy as np
import wandb

from ..env import Observation, Action, Env, EnvState, EnvConst
from ..policy.agent_wrapper import RecurrentAgent
from ..utils import split_rng_to_list

class BaseController(ABC):
    def __init__(self) -> None:
        self.init_wandb()

    @staticmethod
    def agent_state_inference_step(
        agent_lst: Sequence[RecurrentAgent],
        rng: jax.Array,
        model_state_lst: Sequence[PyTreeNode],
        agent_state_lst: Sequence[PyTreeNode],
        obs_lst: Sequence[Observation],
        done_lst: Sequence[jax.Array]
    ) -> tuple[
        jax.Array,
        list[PyTreeNode],
    ]:
        """
        Inference on the trajectory to get a sequence of agent states.

        [Input]
            - agent_lst: Sequence of agents interacting with the environment.
            - rng: Random number generator state.
            - model_state_lst: Sequence of model states for each agent.
            - agent_state_lst: Sequence of internal agent states.
            - obs_lst: Sequence of observations for each agent.
            - done_lst: Sequence of done-flags for each agent.

        [Output]
            - rng: Random number generator state after one iteration.
            - next_agent_state_lst: The updated agent states after the step.
        """
        rng, rng_act_lst = split_rng_to_list(rng, len(agent_lst))
        next_agent_state_lst = [
            agent_fn.step(rng_act, model_state, agent_state, obs)[0]
            for rng_act, agent_fn, model_state, agent_state, obs in zip(rng_act_lst, agent_lst, model_state_lst, agent_state_lst, obs_lst)
        ]
        rng, rng_reset_lst = split_rng_to_list(rng, len(agent_lst))
        next_agent_state_lst = [
            agent_fn.reset_agent_state(rng_reset, next_agent_state, done)
            for rng_reset, agent_fn, next_agent_state, done in zip(
                rng_reset_lst, agent_lst, next_agent_state_lst, done_lst
            )
        ]
        return rng, next_agent_state_lst

    @staticmethod
    def rollout_step(
        env_fn: Env,
        agent_lst: Sequence[RecurrentAgent],
        rng: jax.Array,
        model_state_lst: Sequence[PyTreeNode],
        agent_state_lst: Sequence[PyTreeNode],
        env_const: EnvConst,
        env_state: EnvState,
        obs_lst: Sequence[Observation],
        is_collect_extra_data: bool = False
    ) -> tuple[
        jax.Array,
        EnvState,
        list[PyTreeNode],
        list[Observation],
        list[Action],
        list[jax.Array],
        list[jax.Array],
        dict
    ]:
        """
        Interaction with the environment for one step.

        [Input]
            - env_fn: Environment function that handles state transitions.
            - agent_lst: Sequence of agents interacting with the environment.
            - rng: Random number generator state.
            - model_state_lst: Sequence of model states for each agent.
            - agent_state_lst: Sequence of internal agent states.
            - env_const: Environment constants/configurations.
            - env_state: Current state of the environment.
            - obs_lst: Sequence of observations for each agent.

        [Output]
            - rng: Random number generator state after one iteration.
            - next_agent_state_lst: The updated agent states after the step.
            - next_env_state: The updated environment state after the step.
            - next_obs_lst: New observations received by agents after the step.
            - action_lst: Actions taken by each agent.
            - reward_lst: Rewards received by each agent.
            - done_lst: Boolean flags indicating whether each agent's episode has ended.
            - info: Additional environment-specific information.
        """
        rng, rng_act_lst = split_rng_to_list(rng, len(agent_lst))
        next_agent_state_lst, action_lst = [], []
        extra_data_info = {}
        for i, (rng_act, agent_fn, model_state, agent_state, obs) in enumerate(zip(rng_act_lst, agent_lst, model_state_lst, agent_state_lst, obs_lst)):
            next_agent_state, action, extra_data = agent_fn.step(rng_act, model_state, agent_state, obs)
            next_agent_state_lst.append(next_agent_state)
            action_lst.append(action)
            if is_collect_extra_data:
                extra_data_info[f'agent-{i}-extra-data'] = extra_data
        
        rng, rng_env_step = jax.random.split(rng)
        next_env_state, next_obs_lst, reward_lst, done_lst, info = env_fn.step(rng_env_step, env_const, env_state, action_lst)
        rng, rng_reset_lst = split_rng_to_list(rng, len(agent_lst))
        next_agent_state_lst = [
            agent_fn.reset_agent_state(rng_reset, next_agent_state, done)
            for rng_reset, agent_fn, next_agent_state, done in zip(
                rng_reset_lst, agent_lst, next_agent_state_lst, done_lst
            )
        ]
        return rng, next_agent_state_lst, next_env_state, next_obs_lst, action_lst, reward_lst, done_lst, info | extra_data_info

    def init_wandb(
        self,
        name: str = 'default',
        entity: str = None,
        project: str = None,
        group: str = None,
        job_type: str = None,
        config: dict = None,
        resume_from: str = None
    ) -> None:
        if entity is None:
            os.environ['WANDB_MODE'] = 'offline'

        slurm_job_id = os.getenv('SLURM_JOB_ID')
        run_config = dict(config) if config is not None else {}
        if slurm_job_id is not None and slurm_job_id.strip() != '':
            run_config['slurm_job_id'] = slurm_job_id

        os.makedirs(f"./results/{project}/{group}/{job_type}", exist_ok=True)
        wandb.init(
            name=name,
            entity=entity,
            project=project,
            group=group,
            job_type=job_type,
            config=run_config,
            dir=f"./results/{project}/{group}/{job_type}"
        )
        if slurm_job_id is not None and slurm_job_id.strip() != '':
            wandb.run.summary['slurm_job_id'] = slurm_job_id
        self.run_dir = os.path.abspath(str(wandb.run.dir))
    
    @staticmethod
    def itemize_dict(data: dict[str, Any]) -> dict[str, int | float]:
        res = {}
        for k, v in data.items():
            if isinstance(v, int) or isinstance(v, float):
                pass
            elif isinstance(v, (np.int32, np.int64, np.float32, np.float64)):
                v = v.item()
            elif isinstance(v, np.ndarray):
                v = np.mean(v).item() if v.size > 1 else v.item()
            elif isinstance(v, jax.Array):
                v = jnp.mean(v).item() if v.size > 1 else v.item()
            elif isinstance(v, list):
                v = np.mean(v).item()
            else:
                raise TypeError(f"log type {type(v)} of {v} not supported")
            res[k] = v
        return res

    def log(self, step: int, info: dict[str, Any], log_prefix: str = None):
        info = self.itemize_dict(info)
        if log_prefix is not None:
            info = {f'{log_prefix}/{k}': v for k, v in info.items()}
        wandb.log(info, step)
    