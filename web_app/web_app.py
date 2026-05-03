"""Main entry point for the Overcooked web app.

Fixed human-subjects experiment flow:
  consent → demographics → instructions → tutorial → (env + survey) × 2 → finish

Usage:
    python -m web_app.web_app
    # or
    python web_app/web_app.py
"""

import asyncio
import concurrent.futures
import os
import random
import sys

# JAX 0.6.0 removed jax.tree_map; restore it for libraries that haven't migrated yet.
import jax
if not hasattr(jax, "tree_map"):
    jax.tree_map = jax.tree_util.tree_map

# Enable JAX persistent compilation cache when env var is set.
# Must be configured before any jax.jit/compile calls happen.
_jax_cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR")
if _jax_cache_dir:
    jax.config.update("jax_compilation_cache_dir", _jax_cache_dir)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    if os.environ.get("JAX_EXPLAIN_CACHE_MISSES"):
        jax.config.update("jax_explain_cache_misses", True)

# Single-threaded executor for all JAX compilation work.
# JAX's XLA backend is not thread-safe: concurrent jit/lower/compile calls
# from multiple threads can race on shared global state.  Serialising every
# run_in_executor call that touches JAX (build_env, load_model_checkpoint →
# init_model_state → Flax init) through this executor prevents that.
_compile_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

import aiofiles

# Ensure the project root is on the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from nicegui import app, ui, Client
from tortoise import Tortoise

import nicewebrl
from nicewebrl import stages as nicewebrl_stages
from nicewebrl.logging import setup_logging, get_logger
from nicewebrl.utils import get_user_lock, write_msgpack_record

from web_app.experiment import (
    build_full_experiment,
    build_instruction_block,
    build_experiment_block,
    build_env,
)
from web_app.constants import EXPERIMENT_LAYOUTS, EXPERIMENT_TAGS

DATA_DIR = os.environ.get("DATA_DIR", "data")
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 8080))
# Number of timesteps per episode. Read by web_app/experiment.py at import time.
EPISODE_TIMESTEPS = int(os.environ.get("EPISODE_TIMESTEPS", 200))

logger = None


########################################
# Database lifecycle (required by nicewebrl stages)
########################################

db_ready = asyncio.Event()


async def init_db():
    db_path = os.path.abspath(os.path.join(DATA_DIR, "web_app.sqlite"))
    await Tortoise.init(
        db_url=f"sqlite:///{db_path}",
        modules={"models": ["nicewebrl.stages"]},
    )
    await Tortoise.generate_schemas()
    db_ready.set()


async def close_db():
    await Tortoise.close_connections()


########################################
# Stage runner (adapted from nicewebrl.run_experiment)
########################################

async def run_stage(stage: nicewebrl_stages.Stage, container: ui.element):
    event = asyncio.Event()

    async def signal():
        async with get_user_lock():
            if stage.get_user_data("finished", False):
                event.set()

    await stage.set_user_data(stage_completion_signal_from_event=signal, finished=False)
    await stage.activate(container)

    if stage.get_user_data("finished", False):
        event.set()

    if stage.next_button:
        with container:
            ui.button(
                "Next",
                on_click=lambda: asyncio.create_task(stage.handle_button_press(container)),
            ).on("click", signal)

    await event.wait()
    await stage.set_user_data(stage_completion_signal_from_event=None)


async def run_experiment_loop(experiment, stage_container: ui.element, episode_metadata=None):
    """Run all stages of the experiment sequentially."""
    await experiment.initialize()

    ui.on(
        "key_pressed",
        lambda e: handle_key_press(e, experiment, stage_container),
    )

    gameplay_file = None
    while experiment.not_finished():
        stage = await experiment.get_stage()
        nicewebrl.clear_element(stage_container)
        await run_stage(stage, stage_container)
        if isinstance(stage, nicewebrl_stages.EnvStage):
            await stage.finish_saving_user_data()
            gameplay_file = stage.user_save_file_fn()
        await experiment.advance()

    # Game over — collect feedback and save data
    nicewebrl.clear_element(stage_container)
    await finish_experiment(stage_container, episode_metadata=episode_metadata, gameplay_file=gameplay_file)


async def handle_key_press(e, experiment, container):
    if experiment.finished():
        return
    stage = await experiment.get_stage()
    if stage.get_user_data("finished", False):
        return
    await stage.handle_key_press(e, container)
    fn = stage.get_user_data("stage_completion_signal_from_event")
    if fn:
        await fn()


async def _handle_active_key(e, active_stage_ref: list, container):
    """Key-press handler that forwards to whatever stage is currently active."""
    stage = active_stage_ref[0]
    if stage is None or stage.get_user_data("finished", False):
        return
    await stage.handle_key_press(e, container)
    fn = stage.get_user_data("stage_completion_signal_from_event")
    if fn:
        await fn()


async def _run_block_stages(
    block, active_stage_ref: list, container, block_idx: int
):
    """Run all stages in a block sequentially. Returns the last gameplay_file (if any)."""
    app.storage.user["block_name"] = block.name
    app.storage.user["block_idx"] = block_idx
    gameplay_file = None
    while await block.not_finished():
        stage = await block.get_stage()
        active_stage_ref[0] = stage
        nicewebrl.clear_element(container)
        await run_stage(stage, container)
        if isinstance(stage, nicewebrl_stages.EnvStage):
            await stage.finish_saving_user_data()
            gameplay_file = stage.user_save_file_fn()
        await block.advance_stage()
        app.storage.user["stage_idx"] = app.storage.user.get("stage_idx", -1) + 1
    active_stage_ref[0] = None
    return gameplay_file


########################################
# Pre-experiment screens
########################################

async def make_consent_form(container):
    consent_given = asyncio.Event()
    nicewebrl.clear_element(container)
    with container:
        ui.markdown("## Consent Form")
        consent_path = os.path.join(project_root, "consent.md")
        with open(consent_path, "r") as f:
            ui.markdown(f.read())
        ui.checkbox(
            "I agree to participate.",
            on_change=lambda: consent_given.set(),
        )
    await consent_given.wait()


async def collect_demographic_info(container):
    nicewebrl.clear_element(container)
    with container:
        ui.markdown("## About You")
        ui.markdown("Please fill out the following before we begin.")

        with ui.column().classes("w-full gap-4"):
            prolific_input = ui.input("Prolific ID").classes("w-full")

            ui.markdown(
                '**"I have experience playing the game Overcooked."**'
            )
            experience_input = ui.radio(
                ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"],
                value="Neutral",
            ).props("inline")

        submitted = asyncio.Event()

        async def submit():
            prolific_id = prolific_input.value.strip()

            if not prolific_id:
                ui.notify("Please enter your Prolific ID.", type="warning")
                return

            app.storage.user["prolific_id"] = prolific_id
            app.storage.user["overcooked_experience"] = experience_input.value
            submitted.set()

        ui.button("Continue", on_click=submit)
        await submitted.wait()


########################################
# Data collection helpers
########################################

async def save_data(final_save=True, feedback=None, episode_metadata=None, gameplay_file=None, **kwargs):
    user_data_file = nicewebrl.user_data_file()
    if final_save:
        user_storage = nicewebrl.make_serializable(dict(app.storage.user))
        last_line = dict(
            finished=True,
            feedback=feedback,
            episode_metadata=episode_metadata,
            user_storage=user_storage,
            **kwargs,
        )
        async with aiofiles.open(user_data_file, "ab") as f:
            await write_msgpack_record(f, last_line)
        if gameplay_file:
            async with aiofiles.open(gameplay_file, "ab") as f:
                await write_msgpack_record(f, last_line)


async def finish_experiment(container, episode_metadata=None, gameplay_file=None):
    nicewebrl.clear_element(container)

    if app.storage.user.get("experiment_finished", False):
        with container:
            ui.markdown("## Data saved")
            ui.markdown("### Your completion code:")
            ui.markdown("### `TODO_COMPLETION_CODE`")
            ui.markdown("#### You may now close this tab.")
            ui.button("Play Again", on_click=lambda: ui.navigate.to("/"))
        return

    async def submit(feedback):
        app.storage.user["experiment_finished"] = True
        nicewebrl.clear_element(container)
        with container:
            ui.markdown("## Saving data. Please wait...")
        await save_data(
            final_save=True,
            feedback=feedback,
            episode_metadata=episode_metadata,
            gameplay_file=gameplay_file,
        )
        app.storage.user["data_saved"] = True
        nicewebrl.clear_element(container)
        with container:
            ui.markdown("## Experiment Complete")
            ui.markdown("Thank you for participating!")
            ui.markdown("### Your completion code:")
            ui.markdown("### CWWOOUES")
            ui.markdown("#### You may now close this tab.")

    app.storage.user["data_saved"] = app.storage.user.get("data_saved", False)
    if not app.storage.user["data_saved"]:
        with container:
            ui.markdown("## Session complete!")
            ui.markdown(
                "Please provide any feedback on this session "
                "(e.g., issues, observations, suggestions)."
            )
            text = ui.textarea().style("width: 80%;")
            button = ui.button("Submit")
            await button.clicked()
            await submit(text.value)


########################################
# Page handler
########################################

@ui.page("/")
async def index(client: Client):
    # ui.add_body_html is embedded in the initial HTTP response template and is
    # NOT sent via socket.io.  It must therefore be called before any await that
    # causes NiceGUI to build and return the HTTP response (i.e. before
    # client.connected(), which sets _waiting_for_connection and triggers the
    # response to be sent).
    basic_js_file = nicewebrl.basic_javascript_file()
    with open(basic_js_file) as f:
        ui.add_body_html("<script>" + f.read() + "</script>")

    # Let NiceGUI complete its socket.io handshake before any heavy async work.
    # Without this, awaiting db_ready before rendering any UI causes
    # "implicit handshake failed" and a reload loop.
    await client.connected()

    # Ensure the database is ready before doing anything
    try:
        await asyncio.wait_for(db_ready.wait(), timeout=10.0)
    except asyncio.TimeoutError:
        ui.notify("Database failed to initialize.", type="negative")
        return

    nicewebrl.initialize_user()

    # Main card container
    card = (
        ui.card(align_items=["center"])
        .classes("fixed-center")
        .style(
            "max-width: 90vw;"
            "max-height: 90vh;"
            "overflow: auto;"
            "display: flex;"
            "flex-direction: column;"
            "justify-content: flex-start;"
            "align-items: center;"
            "min-width: 600px;"
        )
        .props("tabindex=0")
    )

    # ── Step 1: consent + demographics (once per user) ──────────────────────
    if not app.storage.user.get("experiment_started"):
        await make_consent_form(card)
        await collect_demographic_info(card)
        app.storage.user["experiment_started"] = True

    # ── Step 2: session config (once per user, persists across page reloads) ─
    if not app.storage.user.get("session_config_set"):
        session_layout = random.choice(EXPERIMENT_LAYOUTS)
        tags = list(EXPERIMENT_TAGS)
        random.shuffle(tags)
        tag_agent_pairs = [[t, random.randint(0, 7)] for t in tags]
        app.storage.user["session_layout"] = session_layout
        app.storage.user["session_tag_agent_pairs"] = tag_agent_pairs
        app.storage.user["session_config_set"] = True
    else:
        session_layout = app.storage.user["session_layout"]
        tag_agent_pairs = app.storage.user["session_tag_agent_pairs"]

    # ── Step 3: build tutorial block + compile session layout (one loading screen)
    nicewebrl.clear_element(card)
    with card:
        ui.markdown("## Loading…")
        ui.markdown(
            "Setting up the environments and loading AI models. "
            "This may take a few minutes, please don't close this tab."
        )
        ui.spinner(size="lg")

    loop = asyncio.get_running_loop()

    def _build_tutorial_and_env():
        env = build_env(session_layout, "overcooked")
        block = build_instruction_block(session_layout, tutorial_agent_id=0, env_build=env)
        return block, env

    try:
        instruction_block, session_env_build = await loop.run_in_executor(
            _compile_executor, _build_tutorial_and_env
        )
    except Exception as e:
        if client.id not in Client.instances:
            return
        nicewebrl.clear_element(card)
        with card:
            ui.markdown("## Error loading models")
            ui.markdown(f"`{e}`")
            ui.button("Retry", on_click=lambda: ui.navigate.to("/"))
        return

    if client.id not in Client.instances:
        return

    # ── Step 4: reset per-session state and switch to game UI ────────────────
    app.storage.user["current_step"] = 0
    app.storage.user["cumulative_reward"] = 0.0
    app.storage.user["experiment_finished"] = False
    app.storage.user["data_saved"] = False
    app.storage.user["stage_idx"] = 0
    app.storage.user["block_idx"] = 0

    nicewebrl.clear_element(card)
    with card:
        stage_container = ui.column()

    episode_metadata = {
        "session_layout": session_layout,
        "tag_agent_pairs": tag_agent_pairs,
        "prolific_id": app.storage.user.get("prolific_id"),
    }

    active_stage = [None]
    ui.on(
        "key_pressed",
        lambda e: _handle_active_key(e, active_stage, stage_container),
    )

    # ── Step 5: run instruction/tutorial block ────────────────────────────────
    gameplay_file = await _run_block_stages(
        instruction_block, active_stage, stage_container, block_idx=0
    )

    # ── Step 6: build and run each experiment block with a per-block loading screen
    for i, (tag, agent_id) in enumerate(tag_agent_pairs):
        if client.id not in Client.instances:
            return

        nicewebrl.clear_element(stage_container)
        with stage_container:
            ui.markdown(f"## Loading round {i + 1}/{len(tag_agent_pairs)}…")
            ui.markdown("Please don't close this tab.")
            ui.spinner(size="lg")

        try:
            block = await loop.run_in_executor(
                _compile_executor,
                lambda t=tag, a=agent_id, idx=i: build_experiment_block(
                    idx, t, a, session_layout, env_build=session_env_build
                ),
            )
        except Exception as e:
            if client.id not in Client.instances:
                return
            nicewebrl.clear_element(stage_container)
            with stage_container:
                ui.markdown("## Error loading models")
                ui.markdown(f"`{e}`")
                ui.button("Retry", on_click=lambda: ui.navigate.to("/"))
            return

        if client.id not in Client.instances:
            return

        file = await _run_block_stages(
            block, active_stage, stage_container, block_idx=i + 1
        )
        if file:
            gameplay_file = file

    nicewebrl.clear_element(stage_container)
    await finish_experiment(
        stage_container, episode_metadata=episode_metadata, gameplay_file=gameplay_file
    )


########################################
# App startup
########################################

def main():
    global logger
    os.makedirs(DATA_DIR, exist_ok=True)
    setup_logging(DATA_DIR, nicegui_storage_user_key="seed")
    logger = get_logger("web_app")

    app.on_startup(init_db)
    app.on_shutdown(close_db)

    Client.disconnect_timeout = 30.0
    ui.run(
        storage_secret="sep_rep_web_app_secret_key_12345",
        host=HOST,
        port=PORT,
        reload=False,
        title="Overcooked - Sep-Rep Learning",
    )


if __name__ == "__main__":
    main()
