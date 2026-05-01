"""Environment setup and nicewebrl stage/experiment construction.

Creates the jaxmarl Overcooked environment, wraps it for nicewebrl,
builds the adapter model, and assembles the experiment stages.
"""

import uuid
import jax
import jax.numpy as jnp
from flax import struct, serialization
from jaxmarl.environments.overcooked import Actions as OvercookedActions
from jaxmarl.environments.overcooked.overcooked import Overcooked
from jaxmarl.environments.overcooked.layouts import overcooked_layouts
from jaxmarl.viz.overcooked_jitted_visualizer import render_fn as overcooked_render_fn

try:
    from jaxmarl.environments.overcooked_v2.common import (
        Actions as OvercookedV2Actions,
        Agent as OvercookedV2Agent,
        Position as OvercookedV2Position,
    )
    from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2, State as OvercookedV2State
    from jaxmarl.environments.overcooked_v2.layouts import overcooked_v2_layouts
    from jaxmarl.viz.overcooked_v2_visualizer import OvercookedV2Visualizer
    OVERCOOKED_V2_AVAILABLE = True
    OVERCOOKED_V2_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover - runtime optional dependency check
    OvercookedV2Actions = None
    OvercookedV2Agent = None
    OvercookedV2Position = None
    OvercookedV2State = None
    OvercookedV2 = None
    overcooked_v2_layouts = {}
    OvercookedV2Visualizer = None
    OVERCOOKED_V2_AVAILABLE = False
    OVERCOOKED_V2_IMPORT_ERROR = e


def _register_overcooked_v2_serialization() -> None:
    if not OVERCOOKED_V2_AVAILABLE:
        return

    def _position_to_state_dict(pos: OvercookedV2Position):
        return {
            "x": serialization.to_state_dict(pos.x),
            "y": serialization.to_state_dict(pos.y),
        }

    def _position_from_state_dict(pos: OvercookedV2Position, state_dict):
        return OvercookedV2Position(
            x=serialization.from_state_dict(pos.x, state_dict["x"]),
            y=serialization.from_state_dict(pos.y, state_dict["y"]),
        )

    def _agent_to_state_dict(agent: OvercookedV2Agent):
        return {
            "pos": serialization.to_state_dict(agent.pos),
            "dir": serialization.to_state_dict(agent.dir),
            "inventory": serialization.to_state_dict(agent.inventory),
        }

    def _agent_from_state_dict(agent: OvercookedV2Agent, state_dict):
        return OvercookedV2Agent(
            pos=serialization.from_state_dict(agent.pos, state_dict["pos"]),
            dir=serialization.from_state_dict(agent.dir, state_dict["dir"]),
            inventory=serialization.from_state_dict(agent.inventory, state_dict["inventory"]),
        )

    def _state_to_state_dict(state: OvercookedV2State):
        return {
            "agents": serialization.to_state_dict(state.agents),
            "grid": serialization.to_state_dict(state.grid),
            "time": serialization.to_state_dict(state.time),
            "terminal": serialization.to_state_dict(state.terminal),
            "recipe": serialization.to_state_dict(state.recipe),
            "new_correct_delivery": serialization.to_state_dict(state.new_correct_delivery),
            "ingredient_permutations": serialization.to_state_dict(state.ingredient_permutations),
        }

    def _state_from_state_dict(state: OvercookedV2State, state_dict):
        ingredient_permutations = state_dict.get("ingredient_permutations", None)
        if state.ingredient_permutations is None:
            restored_ingredient_permutations = ingredient_permutations
        else:
            restored_ingredient_permutations = serialization.from_state_dict(
                state.ingredient_permutations,
                ingredient_permutations,
            )

        return OvercookedV2State(
            agents=serialization.from_state_dict(state.agents, state_dict["agents"]),
            grid=serialization.from_state_dict(state.grid, state_dict["grid"]),
            time=serialization.from_state_dict(state.time, state_dict["time"]),
            terminal=serialization.from_state_dict(state.terminal, state_dict["terminal"]),
            recipe=serialization.from_state_dict(state.recipe, state_dict["recipe"]),
            new_correct_delivery=serialization.from_state_dict(
                state.new_correct_delivery,
                state_dict["new_correct_delivery"],
            ),
            ingredient_permutations=restored_ingredient_permutations,
        )

    try:
        serialization.register_serialization_state(
            OvercookedV2Position,
            _position_to_state_dict,
            _position_from_state_dict,
        )
    except ValueError:
        pass

    try:
        serialization.register_serialization_state(
            OvercookedV2Agent,
            _agent_to_state_dict,
            _agent_from_state_dict,
        )
    except ValueError:
        pass

    try:
        serialization.register_serialization_state(
            OvercookedV2State,
            _state_to_state_dict,
            _state_from_state_dict,
        )
    except ValueError:
        pass

    if not hasattr(OvercookedV2State, "agent_pos"):
        OvercookedV2State.agent_pos = property(
            lambda state: jnp.stack([state.agents.pos.x, state.agents.pos.y], axis=-1)
        )


_register_overcooked_v2_serialization()

import os

from nicegui import app, ui
import nicewebrl

DATA_DIR = os.environ.get("DATA_DIR", "data")
from nicewebrl import (
    MultiAgentJaxWebEnv,
    TimestepWrapper,
    base64_npimage,
    MultiAgentEnvStage,
    Stage,
    Block,
    Experiment,
)

from web_app.adapter import SepRepModelAdapter
from web_app.model_loader import load_models_for_tag, load_models_for_run_id

########################################
# Actions and key mappings (same as CEC example)
########################################
action_keys = ["ArrowLeft", "ArrowDown", "ArrowRight", "ArrowUp", "s", " "]


def get_action_config(env_name: str):
    if env_name == "overcooked":
        # Order matches action_keys = [ArrowLeft, ArrowDown, ArrowRight, ArrowUp, s, space].
        # The jaxmarl coordinate system is rotated 90°: the "right" action moves the agent
        # visually upward (delta [0,-1]), so keys must be remapped accordingly.
        actions = [
            OvercookedActions.up,       # ArrowLeft  → visually left
            OvercookedActions.down,     # ArrowDown  → visually down
            OvercookedActions.left,     # ArrowRight → visually right
            OvercookedActions.right,    # ArrowUp    → visually up
            OvercookedActions.stay,
            OvercookedActions.interact,
        ]
    elif env_name == "overcooked_v2":
        if not OVERCOOKED_V2_AVAILABLE:
            raise RuntimeError(
                f"overcooked_v2 is unavailable in this environment: {OVERCOOKED_V2_IMPORT_ERROR}"
            )
        actions = [
            OvercookedV2Actions.up,
            OvercookedV2Actions.down,
            OvercookedV2Actions.left,
            OvercookedV2Actions.right,
            OvercookedV2Actions.stay,
            OvercookedV2Actions.interact,
        ]
    else:
        raise ValueError(f"Unsupported env_name: {env_name}")

    return jnp.array([a.value for a in actions]), [a.name for a in actions]

MAX_EPISODE_TIMESTEPS = 256
MAX_STAGE_EPISODES = 1
MIN_SUCCESS_EPISODES = 100  # intentionally unreachable; episode ends via max_episodes
DEFAULT_ENV_PARAMS = {"random_reset_fn": 0}

########################################
# Available layouts (9x9 variants)
########################################
AVAILABLE_ENVS = ["overcooked"] + (["overcooked_v2"] if OVERCOOKED_V2_AVAILABLE else [])
DEFAULT_ENV_NAME = AVAILABLE_ENVS[0]

AVAILABLE_LAYOUTS_BY_ENV = {
    "overcooked": sorted(overcooked_layouts.keys()),
}
if OVERCOOKED_V2_AVAILABLE:
    AVAILABLE_LAYOUTS_BY_ENV["overcooked_v2"] = sorted(overcooked_v2_layouts.keys())


def get_available_layouts(env_name: str) -> list[str]:
    if env_name not in AVAILABLE_LAYOUTS_BY_ENV:
        raise ValueError(f"Unsupported env_name: {env_name}")
    return AVAILABLE_LAYOUTS_BY_ENV[env_name]


def create_environment(layout_name: str, env_name: str):
    """Create and configure a jaxmarl Overcooked environment for a given layout/env."""
    if env_name == "overcooked":
        env = Overcooked(
            layout=overcooked_layouts[layout_name],
            max_steps=MAX_EPISODE_TIMESTEPS - 1,
            check_held_out=False,
            shuffle_inv_and_pot=False,
            random_reset=False,
        )

        # Initialize held-out layout pools (needed by env.reset tracing).
        # Collect handcrafted 9x9 layouts as held-out set.
        ho_goal, ho_wall, ho_pot = [], [], []
        for ln, ld in overcooked_layouts.items():
            if "9" in ln:
                _, ho_state = env.custom_reset(
                    jax.random.PRNGKey(0),
                    random_reset=False,
                    shuffle_inv_and_pot=False,
                    layout=ld,
                )
                ho_goal.append(ho_state.goal_pos)
                ho_wall.append(ho_state.wall_map)
                ho_pot.append(ho_state.pot_pos)
        env.held_out_goal = jnp.stack(ho_goal, axis=0)
        env.held_out_wall = jnp.stack(ho_wall, axis=0)
        env.held_out_pot = jnp.stack(ho_pot, axis=0)
    elif env_name == "overcooked_v2":
        if not OVERCOOKED_V2_AVAILABLE:
            raise RuntimeError(
                f"overcooked_v2 is unavailable in this environment: {OVERCOOKED_V2_IMPORT_ERROR}"
            )
        env = OvercookedV2(
            layout=layout_name,
            max_steps=MAX_EPISODE_TIMESTEPS - 1,
            random_reset=False,
        )
    else:
        raise ValueError(f"Unsupported env_name: {env_name}")

    if env_name == "overcooked_v2":
        sample_obs, _ = env.reset(jax.random.PRNGKey(0))
        obs_dim = sample_obs[env.agents[0]].shape
    else:
        obs_dim = env.observation_space(env.agents[0]).shape
    return env, obs_dim


def wrap_environment(env, action_array: jnp.ndarray):
    """Wrap a jaxmarl env for nicewebrl and precompile JAX functions."""
    jax_env = TimestepWrapper(
        env, autoreset=True, reset_w_batch_dim=False, use_params=False
    )
    jax_web_env = MultiAgentJaxWebEnv(env=jax_env, actions=action_array)
    jax_web_env.precompile(dummy_env_params=DEFAULT_ENV_PARAMS)
    return jax_web_env


def compile_render_fns(jax_web_env, env_name: str):
    """Create and precompile render functions."""
    if env_name == "overcooked":
        def render_fn(timestep: nicewebrl.Timestep):
            image = overcooked_render_fn(timestep.state)
            return image.astype(jnp.uint8)
    elif env_name == "overcooked_v2":
        if not OVERCOOKED_V2_AVAILABLE:
            raise RuntimeError(
                f"overcooked_v2 is unavailable in this environment: {OVERCOOKED_V2_IMPORT_ERROR}"
            )
        visualizer = OvercookedV2Visualizer()

        def render_fn(timestep: nicewebrl.Timestep):
            image = visualizer._render_state(timestep.state)
            return image.astype(jnp.uint8)
    else:
        raise ValueError(f"Unsupported env_name: {env_name}")

    vmap_render_fn = jax_web_env.precompile_vmap_render_fn(render_fn, DEFAULT_ENV_PARAMS)
    render_fn_compiled = (
        jax.jit(render_fn)
        .lower(jax_web_env.reset(jax.random.PRNGKey(0), DEFAULT_ENV_PARAMS))
        .compile()
    )
    return render_fn_compiled, vmap_render_fn


########################################
# Display functions for nicewebrl stages
########################################

def make_image_html(src):
    return f'''
    <div id="stateImageContainer" style="display: flex; justify-content: center; align-items: center;">
        <img id="stateImage" src="{src}" style="width: 100%; height: 100%; object-fit: contain;">
    </div>
    '''


async def env_stage_display_fn(
    stage: MultiAgentEnvStage, container: ui.element, timestep: nicewebrl.Timestep
):
    state_image = stage.render_fn(timestep)
    state_image = base64_npimage(state_image)
    stage_state = stage.get_user_data("stage_state")
    human_color = stage.get_user_data("human_color")

    with container.style("align-items: center;"):
        nicewebrl.clear_element(container)
        with ui.row().classes("gap-4"):
            with ui.element("div").classes("p-2 bg-blue-100 rounded"):
                ui.label().bind_text_from(
                    app.storage.user,
                    "current_step",
                    lambda n: f"Step: {n}/{MAX_EPISODE_TIMESTEPS}",
                )
            with ui.element("div").classes("p-2 bg-green-100 rounded"):
                ui.label().bind_text_from(
                    app.storage.user,
                    "cumulative_reward",
                    lambda r: f"Total Reward: {r:.0f}",
                )
            with ui.element("div").classes("p-2 bg-yellow-100 rounded"):
                ui.label(f"You control the {human_color} agent.")
        ui.html(make_image_html(src=state_image))
        ui.markdown(
            "**Controls:** Arrow keys to move | Space to interact | S to stay"
        )


def make_check_finished_fn():
    """Create a check_finished callback that tracks step count and cumulative reward.

    Called every step. Updates app.storage.user with current_step and
    cumulative_reward (both reactive for nicegui bind_text_from).
    Always returns False — episode termination is handled by max_episodes.
    """
    def check_finished(timestep):
        app.storage.user["current_step"] = (
            app.storage.user.get("current_step", 0) + 1
        )
        reward = float(timestep.reward)
        app.storage.user["cumulative_reward"] = (
            app.storage.user.get("cumulative_reward", 0.0) + reward
        )
        return False
    return check_finished


def evaluate_success_fn(timestep: nicewebrl.Timestep, env_params: struct.PyTreeNode):
    success = int(timestep.state.terminal)
    return success


########################################
# Main setup function
########################################

def setup_experiment(
    tag: str | None = None,
    run_id: str | None = None,
    layout_name: str | None = None,
    agent_id: int = 0,
    env_name: str | None = None,
) -> dict:
    """Set up everything needed for the web game.

    Args:
        tag: W&B tag to resolve. Mutually exclusive with run_id.
        run_id: W&B run ID. Mutually exclusive with tag.
        layout_name: Override the training layout. If None, uses the layout from training.
        agent_id: Which agent model to load (0 or 1).
        env_name: Environment name ("overcooked" or "overcooked_v2").

    Returns:
        Dictionary with 'experiment' (nicewebrl.Experiment) and metadata.
    """
    # Load model
    print("Loading model...")
    if tag is not None:
        model_info = load_models_for_tag(tag, agent_id=agent_id)
    elif run_id is not None:
        model_info = load_models_for_run_id(run_id, agent_id=agent_id)
    else:
        raise ValueError("Either tag or run_id must be provided.")

    model_env_name = model_info.get("env_name", "overcooked")
    if env_name is None:
        env_name = model_env_name
    if env_name != model_env_name:
        raise ValueError(
            f"Selected env '{env_name}' does not match model env '{model_env_name}'."
        )

    # Determine layout
    if layout_name is None:
        layout_name = model_info['layout']
    available_layouts = get_available_layouts(env_name)
    if layout_name not in available_layouts:
        raise ValueError(
            f"Layout '{layout_name}' is not available for env '{env_name}'. "
            f"Available: {available_layouts}"
        )
    print(f"Using env/layout: {env_name}/{layout_name}")

    action_array, action_to_name = get_action_config(env_name)

    # Create environment
    print("Creating environment...")
    env, obs_dim = create_environment(layout_name, env_name)

    # Wrap for nicewebrl
    print("Wrapping environment for web...")
    jax_web_env = wrap_environment(env, action_array)

    # Compile render functions
    print("Compiling render functions...")
    render_fn, vmap_render_fn = compile_render_fns(jax_web_env, env_name)

    # Build adapter model
    print("Building model adapter...")
    actor_critic_fn = model_info['actor_critic_fn']
    model_state = model_info['model_state']

    # The adapter reshapes the flat obs to obs_dim (env's native shape),
    # then pads to the model's expected shape if they differ.
    model_obs_shape = model_info['obs_shape']
    pad_target = None
    if model_obs_shape != obs_dim:
        # Model was trained with a different obs shape (e.g., padded or different layout)
        pad_target = model_obs_shape
        print(f"  Obs shape mismatch: env={obs_dim}, model={model_obs_shape}. Will pad.")
    elif model_info['pad_obs_shape_to'] is not None:
        pad_target = model_info['pad_obs_shape_to']
        print(f"  Model uses pad_obs_shape_to={pad_target}.")

    adapter = SepRepModelAdapter(
        inner_model=actor_critic_fn,
        obs_shape=obs_dim,
        pad_obs_shape_to=pad_target,
    )

    # Construct adapter params: nest the loaded checkpoint params under 'inner_model'
    adapter_params = {'params': {'inner_model': model_state['actor_critic_params']['params']}}

    # Hidden state initialization
    def init_hidden_state_fn():
        return actor_critic_fn.init_rnn_state(jax.random.key(0), batch_size=1)

    # Build the game stage
    print("Building game stage...")
    game_label = tag or run_id or "unknown"
    game_stage = MultiAgentEnvStage(
        name=f"play_{game_label}_{layout_name}_{uuid.uuid4().hex[:8]}",
        web_env=jax_web_env,
        action_keys=action_keys,
        action_to_name=action_to_name,
        env_params=DEFAULT_ENV_PARAMS,
        render_fn=render_fn,
        vmap_render_fn=vmap_render_fn,
        display_fn=env_stage_display_fn,
        evaluate_success_fn=evaluate_success_fn,
        check_finished=make_check_finished_fn(),
        notify_success=False,
        min_success=MIN_SUCCESS_EPISODES,
        max_episodes=MAX_STAGE_EPISODES,
        verbosity=0,
        user_save_file_fn=lambda: (
            f"{DATA_DIR}/gameplay_user={app.storage.user.get('seed', 'unknown')}"
            f"_model={game_label}_{layout_name}.json"
        ),
        model=adapter,
        model_params=adapter_params,
        num_seeds=1,
        using_param_stack=False,
        init_hidden_state_fn=init_hidden_state_fn,
        max_timesteps=MAX_EPISODE_TIMESTEPS,
        human_id=None,  # randomly assign
    )

    game_block = Block(
        stages=[game_stage],
        metadata={"desc": f"Play {layout_name} with {game_label}"},
        randomize=False,
    )

    experiment = Experiment(
        blocks=[game_block],
        randomize=[False],
        name=f"play_{game_label}",
    )

    print("Setup complete.")
    return {
        'experiment': experiment,
        'layout_name': layout_name,
        'env_name': env_name,
        'model_info': model_info,
        'jax_web_env': jax_web_env,
    }
