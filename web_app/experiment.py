"""Environment setup and nicewebrl stage/experiment construction.

Creates the jaxmarl Overcooked environment, wraps it for nicewebrl,
builds the adapter model, and assembles the experiment stages.
"""

import asyncio
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
    FeedbackStage,
    Stage,
    Block,
    Experiment,
)

from web_app.adapter import SepRepModelAdapter
from web_app.model_loader import load_models_for_tag

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

MAX_EPISODE_TIMESTEPS = int(os.environ.get("EPISODE_TIMESTEPS", 200))
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
    if timestep.first():
        app.storage.user["current_step"] = 0
        app.storage.user["cumulative_reward"] = 0.0

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
    """Create a check_finished callback that tracks per-episode step count and reward.

    Uses closure-local counters so each EnvStage starts fresh at 0.
    Always returns False — episode termination is handled by max_episodes.
    """
    episode_step = 0
    episode_reward = 0.0

    def check_finished(timestep):
        nonlocal episode_step, episode_reward
        episode_step += 1
        episode_reward += float(timestep.reward)
        app.storage.user["current_step"] = episode_step
        app.storage.user["cumulative_reward"] = episode_reward
        return False
    return check_finished


def evaluate_success_fn(timestep: nicewebrl.Timestep, env_params: struct.PyTreeNode):
    success = int(timestep.state.terminal)
    return success


########################################
# Instruction / tutorial display functions
########################################

async def instruction_display_fn(stage: Stage, container: ui.element):
    with container.style("align-items: center;"):
        nicewebrl.clear_element(container)
        ui.markdown(f"## {stage.name}")
        ui.markdown(
            "You'll be playing a cooperative cooking game (Overcooked) with an AI agent partner."
        )
        ui.markdown(
            "Your goal is to work together to prepare and deliver as many dishes as possible."
        )
        ui.markdown(
            "To deliver a dish, place 3 onions from the yellow pile into the black pot, and wait for them to cook. Then, use a white plate to pick up the cooked dish and deliver it to the green delivery area."
        )
        ui.markdown("**Controls:**")
        ui.markdown("- **Arrow keys** — move up / down / left / right")
        ui.markdown("- **Space bar** — interact with the environment (pick up / put down items)")
        ui.markdown("- **S** — stay in place / wait")


async def tutorial_display_fn(stage: Stage, container: ui.element):
    with container.style("align-items: center;"):
        nicewebrl.clear_element(container)
        ui.markdown(f"## {stage.name}")
        ui.markdown(
            "You will now play a **tutorial round** to get used to the controls."
        )
        ui.markdown(
            "> Please **do not close or leave this page** until the experiment is complete, "
            "as you will not be able to return."
        )


async def post_tutorial_display_fn(stage: Stage, container: ui.element):
    with container.style("align-items: center;"):
        nicewebrl.clear_element(container)
        ui.markdown(f"## {stage.name}")
        ui.markdown(
            "Great job! Now that you have practised the controls, the actual experiment will begin."
        )
        ui.markdown(
            f"You will play **{len(EXPERIMENT_TAGS)} rounds** with different AI partners. "
            "After each round you will fill out a short survey."
        )


########################################
# Survey stage
########################################

_SURVEY_QUESTIONS = [
    "The agent adapted to me when making decisions.",
    "The agent was consistent in its actions.",
    "The agent's actions were human-like.",
    "The agent frequently got in my way.",
    "The agent's behavior was frustrating.",
    "Overall, I enjoyed playing with the agent.",
    "Overall, I felt that the agent's ability to coordinate with me was:",
]

_LIKERT_OPTIONS = {
    "Strongly disagree": "Strongly disagree",
    "Disagree": "Disagree",
    "Neutral": "Neutral",
    "Agree": "Agree",
    "Strongly agree": "Strongly agree",
}

_COORD_OPTIONS = {
    "Very poor": "Very poor",
    "Poor": "Poor",
    "Neutral": "Neutral",
    "Good": "Good",
    "Very good": "Very good",
}

async def user_survey_display_fn(stage, container):
    nicewebrl.clear_element(container)
    with container.style("align-items: center;"):
        ui.markdown("## Survey")
        ui.markdown("Please answer the following questions about your experience.")

        responses = {}
        completed = {}
        completed_all = asyncio.Event()

        def make_on_change(q_idx):
            def on_change(val):
                completed[q_idx] = True
                if len(completed) == len(_SURVEY_QUESTIONS):
                    completed_all.set()
            return on_change

        for i, question in enumerate(_SURVEY_QUESTIONS):
            ui.markdown(question)
            options = _LIKERT_OPTIONS if i < len(_SURVEY_QUESTIONS) - 1 else _COORD_OPTIONS
            dropdown = ui.select(options, on_change=make_on_change(i))
            responses[question] = dropdown

        await completed_all.wait()
        return {k: v.value for k, v in responses.items()}


def make_survey_stage(name: str, tag: str, layout: str) -> FeedbackStage:
    return FeedbackStage(
        name=name,
        body="",
        display_fn=user_survey_display_fn,
        user_save_file_fn=lambda: (
            f"{DATA_DIR}/survey_user={app.storage.user.get('seed', 'unknown')}"
            f"_tag={tag}_{layout}.json"
        ),
        next_button=True,
    )


########################################
# Per-stage env setup (no Block/Experiment wrapping)
########################################

_env_build_cache: dict[tuple[str, str], dict] = {}


def build_env(layout_name: str, env_name: str) -> dict:
    """Compile the web env and render fns for a layout/env pair.

    Results are cached process-wide by (layout_name, env_name). Repeated calls
    return the same compiled objects so JAX's trace cache is always a hit and
    LLVM does not recompile — preventing mmap region accumulation across users.
    """
    key = (layout_name, env_name)
    if key in _env_build_cache:
        print(f"  Using cached environment for {env_name}/{layout_name}...")
        return _env_build_cache[key]

    action_array, action_to_name = get_action_config(env_name)
    print(f"  Creating environment {env_name}/{layout_name}...")
    env, obs_dim = create_environment(layout_name, env_name)
    print("  Wrapping environment for web...")
    jax_web_env = wrap_environment(env, action_array)
    print("  Compiling render functions...")
    render_fn, vmap_render_fn = compile_render_fns(jax_web_env, env_name)
    result = {
        "jax_web_env": jax_web_env,
        "render_fn": render_fn,
        "vmap_render_fn": vmap_render_fn,
        "obs_dim": obs_dim,
        "action_to_name": action_to_name,
    }
    _env_build_cache[key] = result
    print(f"  Environment for {env_name}/{layout_name} ready, env cache now contains entries: {list(_env_build_cache.keys())}")
    return result


def setup_env_stage(
    tag: str,
    layout_name: str,
    agent_id: int,
    env_name: str = "overcooked",
    stage_name_prefix: str = "play",
    model_tag: str | None = None,
    env_build: dict | None = None,
    save_data: bool = True,
) -> MultiAgentEnvStage:
    """Load a model and create a MultiAgentEnvStage without wrapping it in Block/Experiment.

    Args:
        tag: Tracking label (used in stage names and save filenames).
        model_tag: Directory name to load from (defaults to tag). Use when the model
                   lives in a layout-specific directory, e.g. fcp_coord_ring_9.
    """
    load_tag = model_tag if model_tag is not None else tag
    print(f"Loading model: dir={load_tag}, tracking as tag={tag}, layout={layout_name}, agent_id={agent_id}...")
    model_info = load_models_for_tag(load_tag, agent_id=agent_id)

    model_env_name = model_info.get("env_name", "overcooked")
    if env_name != model_env_name:
        raise ValueError(
            f"Selected env '{env_name}' does not match model env '{model_env_name}'."
        )

    available_layouts = get_available_layouts(env_name)
    if layout_name not in available_layouts:
        raise ValueError(
            f"Layout '{layout_name}' not available for env '{env_name}'. "
            f"Available: {available_layouts}"
        )

    if env_build is None:
        env_build = build_env(layout_name, env_name)
    jax_web_env = env_build["jax_web_env"]
    render_fn = env_build["render_fn"]
    vmap_render_fn = env_build["vmap_render_fn"]
    obs_dim = env_build["obs_dim"]
    action_to_name = env_build["action_to_name"]

    print("  Building model adapter...")
    actor_critic_fn = model_info['actor_critic_fn']
    model_state = model_info['model_state']

    model_obs_shape = model_info['obs_shape']
    pad_target = None
    if model_obs_shape != obs_dim:
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
    adapter_params = {'params': {'inner_model': model_state['actor_critic_params']['params']}}

    def init_hidden_state_fn():
        return actor_critic_fn.init_rnn_state(jax.random.key(0), batch_size=1)

    game_label = f"{tag}_{layout_name}_agent{agent_id}"
    if save_data:
        save_file_fn = lambda: (
            f"{DATA_DIR}/gameplay_user={app.storage.user.get('seed', 'unknown')}"
            f"_{game_label}.json"
        )
    else:
        save_file_fn = lambda: os.devnull

    stage = MultiAgentEnvStage(
        name=f"{stage_name_prefix}_{game_label}",
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
        user_save_file_fn=save_file_fn,
        model=adapter,
        model_params=adapter_params,
        num_seeds=1,
        using_param_stack=False,
        init_hidden_state_fn=init_hidden_state_fn,
        max_timesteps=MAX_EPISODE_TIMESTEPS,
        human_id=None,
    )
    print(f"  Stage '{stage.name}' ready.")
    return stage


########################################
# Full experiment builder
########################################

from web_app.constants import (
    TUTORIAL_TAG,
    EXPERIMENT_TAGS,
    ORIGINAL_5_TAGS,
)


def build_instruction_block(
    session_layout: str,
    tutorial_agent_id: int = 0,
    env_build: dict | None = None,
) -> Block:
    """Build the instruction + tutorial block."""
    instruction_stage = Stage(name="Instructions", display_fn=instruction_display_fn)
    tutorial_intro_stage = Stage(name="Tutorial", display_fn=tutorial_display_fn)
    post_tutorial_stage = Stage(name="Post-Tutorial", display_fn=post_tutorial_display_fn)

    print("=== Loading tutorial model ===")
    tutorial_env_stage = setup_env_stage(
        tag=TUTORIAL_TAG,
        layout_name=session_layout,
        agent_id=tutorial_agent_id,
        env_name="overcooked",
        stage_name_prefix="tutorial",
        env_build=env_build,
        save_data=False,
    )

    return Block(
        stages=[
            instruction_stage,
            tutorial_intro_stage,
            tutorial_env_stage,
            post_tutorial_stage,
        ],
        metadata={"desc": "Instructions & Tutorial"},
        randomize=False,
    )


def build_experiment_block(
    i: int,
    tag: str,
    agent_id: int,
    session_layout: str,
    env_build: dict | None = None,
) -> Block:
    """Build a single experiment block (env stage + survey) for one tag."""
    print(f"=== Loading experiment model {i + 1}: tag={tag} ===")
    model_tag = f"{tag}_{session_layout}" if tag in ORIGINAL_5_TAGS else None
    env_stage = setup_env_stage(
        tag=tag,
        layout_name=session_layout,
        agent_id=agent_id,
        env_name="overcooked",
        stage_name_prefix=f"exp{i}",
        model_tag=model_tag,
        env_build=env_build,
    )
    survey_stage = make_survey_stage(
        name=f"{tag} Survey",
        tag=tag,
        layout=session_layout,
    )
    return Block(
        stages=[env_stage, survey_stage],
        metadata={"desc": f"{tag} on {session_layout}", "tag": tag, "agent_id": agent_id},
        randomize=False,
    )


def build_full_experiment(
    session_layout: str,
    tag_agent_pairs: list,
    tutorial_agent_id: int = 0,
) -> dict:
    """Build the complete experiment upfront (tutorial + all experiment blocks)."""
    all_blocks = [build_instruction_block(session_layout, tutorial_agent_id)]
    for i, (tag, agent_id) in enumerate(tag_agent_pairs):
        all_blocks.append(build_experiment_block(i, tag, agent_id, session_layout))

    experiment = Experiment(
        blocks=all_blocks,
        randomize=[False] * len(all_blocks),
        name="sep_rep_experiment",
    )
    print("=== Full experiment built ===")
    return {
        "experiment": experiment,
        "session_layout": session_layout,
        "tag_agent_pairs": tag_agent_pairs,
    }
