import csv
import dataclasses
import glob
import os
import pickle
import re
import time
from typing import Any, Callable, Literal, Sequence, TypeAlias

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.serialization import from_state_dict
from flax.struct import PyTreeNode

from jax_pbt.controller.base_controller import BaseController
from jax_pbt.env.batched_env import FrozenAssignmentEnv
from jax_pbt.env.gridworld.coop_foraging_env import CoopForagingConst, CoopForagingState, CoopForagingEnv
from jax_pbt.env.overcooked.overcooked_env import OvercookedConst, OvercookedState, OvercookedEnv
from jax_pbt.policy.actor_critic import ActorWrapperAgent as InferenceAgent
from jax_pbt.utils import split_rng_to_list

EnvConst: TypeAlias = CoopForagingConst | OvercookedConst
EnvState: TypeAlias = CoopForagingState | OvercookedState
EvalEnvMode: TypeAlias = Literal['coop_foraging', 'overcooked']


def extract_agent_id_from_model_dir(model_dir: str) -> int:
    name = os.path.basename(os.path.normpath(model_dir))
    match = re.match(r'model_.*-(\d+)-best$', name)
    if match is None:
        raise ValueError(
            f"Could not parse agent id from checkpoint directory '{model_dir}'. "
            'Expected directory name to match pattern: model_<name>-<id>-best'
        )
    return int(match.group(1))


def discover_model_dirs(checkpoint_glob: str, run_id: str) -> list[str]:
    candidate_paths = glob.glob(checkpoint_glob, recursive=True)
    model_dirs = []
    for path in candidate_paths:
        if not os.path.isdir(path):
            continue
        normalized_path_parts = os.path.normpath(path).split(os.sep)
        run_id_match = any(part == run_id or part.endswith(f'-{run_id}') for part in normalized_path_parts)
        if not run_id_match:
            continue
        model_fn_path = os.path.join(path, 'model_fn.pkl')
        model_state_path = os.path.join(path, 'model_state_dict.pkl')
        if os.path.isfile(model_fn_path) and os.path.isfile(model_state_path):
            model_dirs.append(path)

    if not model_dirs:
        raise FileNotFoundError(
            f"No valid model checkpoint directories found for run_id '{run_id}' with glob '{checkpoint_glob}'. "
            'Expected each directory to contain model_fn.pkl and model_state_dict.pkl.'
        )

    return sorted(model_dirs, key=extract_agent_id_from_model_dir)


def _backfill_nn_module_defaults(obj: object, _visited: set | None = None) -> None:
    """Patch nn.Module instances loaded from old pickles by filling in fields added since pickling."""
    if _visited is None:
        _visited = set()
    if id(obj) in _visited:
        return
    _visited.add(id(obj))

    if isinstance(obj, nn.Module) and dataclasses.is_dataclass(obj):
        for field in dataclasses.fields(obj):
            if not hasattr(obj, field.name):
                if field.default is not dataclasses.MISSING:
                    object.__setattr__(obj, field.name, field.default)
                elif field.default_factory is not dataclasses.MISSING:
                    object.__setattr__(obj, field.name, field.default_factory())
        for val in vars(obj).values():
            _backfill_nn_module_defaults(val, _visited)
    elif hasattr(obj, '__dict__') and not isinstance(obj, type):
        for val in vars(obj).values():
            _backfill_nn_module_defaults(val, _visited)


def load_model_checkpoint(model_dir: str) -> tuple[object, PyTreeNode]:
    with open(os.path.join(model_dir, 'model_fn.pkl'), 'rb') as f:
        model_fn = pickle.load(f)
    _backfill_nn_module_defaults(model_fn)
    with open(os.path.join(model_dir, 'model_state_dict.pkl'), 'rb') as f:
        model_state_dict = pickle.load(f)

    model_state_template = model_fn.init_model_state(jax.random.key(0))
    model_state = from_state_dict(model_state_template, model_state_dict)
    return model_fn, model_state


def compute_stats(values: np.ndarray) -> dict[str, float]:
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'p05': float(np.percentile(values, 5)),
        'p25': float(np.percentile(values, 25)),
        'p50': float(np.percentile(values, 50)),
        'p75': float(np.percentile(values, 75)),
        'p95': float(np.percentile(values, 95)),
    }


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_square_matrix_csv(path: str, matrix: np.ndarray, agent_ids: Sequence[int]) -> None:
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['agent_id', *agent_ids])
        for row_idx, agent_id in enumerate(agent_ids):
            writer.writerow([agent_id, *matrix[row_idx].tolist()])


def write_rect_matrix_csv(path: str, matrix: np.ndarray, row_agent_ids: Sequence[int], col_agent_ids: Sequence[int]) -> None:
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['agent_i', *col_agent_ids])
        for row_idx, row_agent_id in enumerate(row_agent_ids):
            writer.writerow([row_agent_id, *matrix[row_idx].tolist()])


def evaluate_pair_returns(
    compiled_eval_fn: Callable[[jax.Array, PyTreeNode, PyTreeNode], tuple[jax.Array, jax.Array]],
    rng: jax.Array,
    model_state_i: PyTreeNode,
    model_state_j: PyTreeNode,
) -> tuple[jax.Array, jax.Array]:
    return compiled_eval_fn(rng, model_state_i, model_state_j)


def assert_uniform_model_state_structure(model_states: dict[int, PyTreeNode], context: str) -> object:
    if not model_states:
        raise ValueError(f'No model states found for {context}.')

    agent_ids = sorted(model_states.keys())
    ref_agent_id = agent_ids[0]
    ref_structure = jax.tree_util.tree_structure(model_states[ref_agent_id])
    for agent_id in agent_ids[1:]:
        structure = jax.tree_util.tree_structure(model_states[agent_id])
        if structure != ref_structure:
            raise ValueError(
                f'Model state tree structure mismatch for {context} at agent {agent_id}. '
                'All checkpoints must share architecture/parameter structure for fast compiled evaluation.'
            )
    return ref_structure


def build_eval_environment(train_config: Any, force_eval_on_reserved_envs: bool | None = None) -> tuple[object, EvalEnvMode, bool]:
    env_cfg = train_config.env
    env_mode = str(env_cfg.name)

    if env_mode == 'coop_foraging':
        env = CoopForagingEnv(
            num_agents=train_config.general.num_players,
            max_steps=train_config.general.game_length,
            obstacle_density=float(env_cfg.obstacle_density),
            heldout_goal_ratio=float(env_cfg.heldout_goal_ratio),
            heldout_goal_seed=int(env_cfg.heldout_goal_seed),
            train_layout_mode=str(env_cfg.train_layout_mode),
            train_procgen_seed=int(env_cfg.train_procgen_seed),
        )
        eval_on_reserved_envs = bool(env_cfg.eval_on_reserved_goals)
        if force_eval_on_reserved_envs is not None:
            eval_on_reserved_envs = force_eval_on_reserved_envs
        if eval_on_reserved_envs and env.num_heldout_goal_positions < env.n_goal:
            raise ValueError(
                f"env.eval_on_reserved_goals requires at least n_goal={env.n_goal} reserved positions, "
                f"but got {env.num_heldout_goal_positions}. Increase env.heldout_goal_ratio."
            )
        return env, 'coop_foraging', eval_on_reserved_envs

    if env_mode == 'overcooked':
        env = OvercookedEnv(
            layout=str(env_cfg.layout),
            max_steps=train_config.general.game_length,
            pad_obs_shape_to=getattr(env_cfg, 'pad_obs_shape_to', None),
            heldout_layout_seed=int(env_cfg.heldout_layout_seed),
            heldout_layout_num_samples=int(env_cfg.heldout_layout_num_samples),
        )
        eval_on_reserved_envs = bool(env_cfg.eval_on_reserved_layouts)
        if force_eval_on_reserved_envs is not None:
            eval_on_reserved_envs = force_eval_on_reserved_envs
        if eval_on_reserved_envs and env.num_heldout_layouts < 1:
            raise ValueError(
                'env.eval_on_reserved_layouts requires heldout layouts, but none were generated. '
                'Increase env.heldout_layout_num_samples.'
            )
        return env, 'overcooked', eval_on_reserved_envs

    raise ValueError(f"env.name must be one of ('coop_foraging', 'overcooked'), got {env_mode}")


def build_eval_env_summary_fields(train_config: Any, env_mode: EvalEnvMode, eval_on_reserved_envs: bool) -> dict[str, Any]:
    if env_mode == 'coop_foraging':
        return {
            'env_name': env_mode,
            'eval_on_reserved_envs': eval_on_reserved_envs,
            'eval_on_reserved_goals': eval_on_reserved_envs,
            'heldout_goal_ratio': float(train_config.env.heldout_goal_ratio),
            'heldout_goal_seed': int(train_config.env.heldout_goal_seed),
        }

    return {
        'env_name': env_mode,
        'eval_on_reserved_envs': eval_on_reserved_envs,
        'eval_on_reserved_layouts': eval_on_reserved_envs,
        'layout': str(train_config.env.layout),
        'heldout_layout_seed': int(train_config.env.heldout_layout_seed),
        'heldout_layout_num_samples': int(train_config.env.heldout_layout_num_samples),
        'pad_obs_shape_to': list(train_config.env.pad_obs_shape_to) if getattr(train_config.env, 'pad_obs_shape_to', None) is not None else None,
    }


def build_eval_env_const(eval_env: FrozenAssignmentEnv[EnvConst, EnvState], env_mode: EvalEnvMode, eval_on_reserved_envs: bool) -> EnvConst:
    if env_mode == 'coop_foraging':
        return eval_env.default_const.replace(
            use_reserved_goal_positions=jnp.full((eval_env.num_envs,), eval_on_reserved_envs, dtype=jnp.bool_)
        )
    return eval_env.default_const.replace(
        use_reserved_layouts=jnp.full((eval_env.num_envs,), eval_on_reserved_envs, dtype=jnp.bool_)
    )


def build_compiled_pair_evaluator(
    eval_env: FrozenAssignmentEnv[EnvConst, EnvState],
    eval_episode_length: int,
    env_mode: EvalEnvMode,
    eval_on_reserved_envs: bool,
    model_fn_i: object,
    model_fn_j: object,
    model_state_i_example: PyTreeNode,
    model_state_j_example: PyTreeNode,
) -> Callable[[jax.Array, PyTreeNode, PyTreeNode], tuple[jax.Array, jax.Array]]:
    eval_agent_lst = [InferenceAgent(model_fn_i), InferenceAgent(model_fn_j)]
    agent_batch_sizes = eval_env.get_agent_batch_size()
    env_const = build_eval_env_const(
        eval_env=eval_env,
        env_mode=env_mode,
        eval_on_reserved_envs=eval_on_reserved_envs,
    )

    def eval_pair(rng: jax.Array, model_state_i: PyTreeNode, model_state_j: PyTreeNode) -> tuple[jax.Array, jax.Array]:
        model_state_lst = [model_state_i, model_state_j]
        rng, rng_init_agent_lst = split_rng_to_list(rng, len(eval_agent_lst))
        agent_state_lst = [
            agent_fn.init_agent_state(rng_init_agent, batch_size)
            for rng_init_agent, agent_fn, batch_size in zip(rng_init_agent_lst, eval_agent_lst, agent_batch_sizes)
        ]
        rng, rng_reset = jax.random.split(rng)
        env_state, obs_lst = eval_env.reset(rng_reset, env_const)

        Carry: TypeAlias = tuple[jax.Array, list[PyTreeNode], EnvState, list, jax.Array, jax.Array]
        returns0_init = jnp.zeros((agent_batch_sizes[0],), dtype=jnp.float32)
        returns1_init = jnp.zeros((agent_batch_sizes[1],), dtype=jnp.float32)

        def step(carry: Carry, _) -> tuple[Carry, None]:
            rng, agent_state_lst, env_state, obs_lst, returns0, returns1 = carry
            rng, agent_state_lst, env_state, obs_lst, _, reward_lst, _, _ = BaseController.rollout_step(
                env_fn=eval_env,
                agent_lst=eval_agent_lst,
                rng=rng,
                model_state_lst=model_state_lst,
                agent_state_lst=agent_state_lst,
                env_const=env_const,
                env_state=env_state,
                obs_lst=obs_lst,
            )
            returns0 = returns0 + reward_lst[0]
            returns1 = returns1 + reward_lst[1]
            return (rng, agent_state_lst, env_state, obs_lst, returns0, returns1), None

        carry_init: Carry = (rng, agent_state_lst, env_state, obs_lst, returns0_init, returns1_init)
        carry_final, _ = jax.lax.scan(step, carry_init, None, length=eval_episode_length)
        _, _, _, _, returns0, returns1 = carry_final
        return returns0, returns1

    print('Compiling JAX pair evaluator...')
    compile_start = time.perf_counter()
    compiled_eval_fn = jax.jit(eval_pair).lower(jax.random.key(0), model_state_i_example, model_state_j_example).compile()
    compile_end = time.perf_counter()
    print(f'Compiled in {compile_end - compile_start:.2f}s')
    return compiled_eval_fn
