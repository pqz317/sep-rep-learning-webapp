"""Main entry point for the Overcooked web app.

Bypasses nicewebrl.run() to support dynamic model loading from the browser UI.
The user selects a W&B tag (or run ID), layout, and agent ID, then plays
the Overcooked game against the trained AI model.

Usage:
    python -m web_app.web_app
    # or
    python web_app/web_app.py
"""

import asyncio
import os
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
    setup_experiment,
    AVAILABLE_ENVS,
    DEFAULT_ENV_NAME,
    get_available_layouts,
)
from web_app.model_loader import discover_agents_for_tag, discover_agents_for_run_id

DATA_DIR = os.environ.get("DATA_DIR", "data")
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 8080))

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
    total_reward = app.storage.user.get("cumulative_reward", 0.0)
    with stage_container:
        ui.markdown(f"**Total Reward: {total_reward:.0f}**")
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


########################################
# Data collection helpers
########################################

async def collect_user_info(container):
    nicewebrl.clear_element(container)
    with container:
        ui.markdown("## Welcome")
        ui.markdown("Please enter your name before starting.")
        name_input = ui.input("Name").classes("w-full")

        async def submit():
            name = name_input.value.strip()
            if not name:
                ui.notify("Please enter your name.", type="warning")
                return
            app.storage.user["name"] = name

        button = ui.button("Continue", on_click=submit)
        await button.clicked()


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
        # Also append the completion record to the per-episode gameplay file
        # so feedback is directly associated with the episode it was given for.
        if gameplay_file:
            async with aiofiles.open(gameplay_file, "ab") as f:
                await write_msgpack_record(f, last_line)


async def finish_experiment(container, episode_metadata=None, gameplay_file=None):
    nicewebrl.clear_element(container)

    if app.storage.user.get("experiment_finished", False):
        with container:
            ui.markdown("## Data saved")
            ui.button("Play Again", on_click=lambda: ui.navigate.to("/"))
        return

    async def submit(feedback):
        app.storage.user["experiment_finished"] = True
        nicewebrl.clear_element(container)
        with container:
            ui.markdown("## Saving data. Please wait...")
        await save_data(final_save=True, feedback=feedback, episode_metadata=episode_metadata, gameplay_file=gameplay_file)
        app.storage.user["data_saved"] = True
        nicewebrl.clear_element(container)
        with container:
            ui.markdown("## Data saved")
            ui.button("Play Again", on_click=lambda: ui.navigate.to("/"))

    app.storage.user["data_saved"] = app.storage.user.get("data_saved", False)
    if not app.storage.user["data_saved"]:
        with container:
            ui.markdown("## Session complete!")
            if episode_metadata:
                partner = episode_metadata.get("partner", "unknown")
                layout = episode_metadata.get("layout", "unknown")
                agent_id = episode_metadata.get("agent_id", "unknown")
                total_reward = app.storage.user.get("cumulative_reward", 0.0)
                ui.markdown(
                    f"**Partner:** {partner} | **Layout:** {layout} | "
                    f"**Agent:** {agent_id} | **Reward:** {total_reward:.0f}"
                )
            ui.markdown(
                "Please provide any feedback on this episode (e.g., issues, observations, suggestions)."
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
    # Ensure the database is ready before doing anything
    try:
        await asyncio.wait_for(db_ready.wait(), timeout=10.0)
    except asyncio.TimeoutError:
        ui.notify("Database failed to initialize.", type="negative")
        return

    nicewebrl.initialize_user()

    # Inject nicewebrl's JavaScript for keyboard handling
    basic_js_file = nicewebrl.basic_javascript_file()
    with open(basic_js_file) as f:
        ui.add_body_html("<script>" + f.read() + "</script>")

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

    # Collect user name on every page load
    await collect_user_info(card)
    nicewebrl.clear_element(card)

    with card:
        ui.markdown("## Overcooked: Play with Trained Models")

        # Configuration form
        with ui.column().classes("w-full gap-4"):
            env_select = ui.select(
                label="Environment",
                options=AVAILABLE_ENVS,
                value=DEFAULT_ENV_NAME,
            ).classes("w-full")

            use_tag = ui.switch("Use W&B Tag (vs Run ID)", value=True)

            tag_input = ui.input(
                "W&B Tag",
                placeholder="e.g., cec_pred",
            ).classes("w-full")

            run_id_input = ui.input(
                "W&B Run ID",
                placeholder="e.g., abc123xy",
            ).classes("w-full")
            run_id_input.set_visibility(False)

            def toggle_input_mode():
                tag_input.set_visibility(use_tag.value)
                run_id_input.set_visibility(not use_tag.value)

            use_tag.on_value_change(lambda _: toggle_input_mode())

            def update_layout_options():
                env_name = env_select.value
                layouts = get_available_layouts(env_name)
                layout_select.options = layouts
                if layout_select.value not in layouts:
                    layout_select.value = layouts[0] if layouts else None
                layout_select.update()

            layout_select = ui.select(
                label="Layout",
                options=get_available_layouts(DEFAULT_ENV_NAME),
                value=(get_available_layouts(DEFAULT_ENV_NAME)[0] if get_available_layouts(DEFAULT_ENV_NAME) else None),
            ).classes("w-full")
            env_select.on_value_change(lambda _: update_layout_options())

            agent_select = ui.select(
                label="AI Agent ID",
                options=[0, 1],
                value=0,
            ).classes("w-full")

            discover_status = ui.label("")
            discover_button = ui.button("Discover Available Agents").classes("w-full")

            async def on_discover():
                """Resolve tag/run_id and populate agent dropdown with actual available agents."""
                discover_button.disable()
                discover_status.text = "Querying W&B and scanning checkpoints..."
                try:
                    loop = asyncio.get_running_loop()
                    if use_tag.value:
                        tag_val = tag_input.value.strip() if tag_input.value else ""
                        if not tag_val:
                            ui.notify("Enter a tag first.", type="warning")
                            return
                        _, agent_ids = await loop.run_in_executor(
                            None, lambda: discover_agents_for_tag(tag_val)
                        )
                    else:
                        rid_val = run_id_input.value.strip() if run_id_input.value else ""
                        if not rid_val:
                            ui.notify("Enter a run ID first.", type="warning")
                            return
                        _, agent_ids = await loop.run_in_executor(
                            None, lambda: discover_agents_for_run_id(rid_val)
                        )
                    agent_select.options = agent_ids
                    agent_select.value = agent_ids[0] if agent_ids else 0
                    agent_select.update()
                    discover_status.text = f"Found {len(agent_ids)} agents: {agent_ids}"
                except Exception as e:
                    discover_status.text = f"Error: {e}"
                    ui.notify(f"Discovery failed: {e}", type="negative")
                finally:
                    discover_button.enable()

            discover_button.on_click(lambda: asyncio.create_task(on_discover()))

            status_label = ui.label("")
            start_button = ui.button("Start Game").classes("w-full")

        await start_button.clicked()

        # Validate inputs
        if use_tag.value:
            tag_val = tag_input.value.strip() if tag_input.value else ""
            run_id_val = None
            if not tag_val:
                ui.notify("Please enter a W&B tag.", type="warning")
                return
        else:
            tag_val = None
            run_id_val = run_id_input.value.strip() if run_id_input.value else ""
            if not run_id_val:
                ui.notify("Please enter a W&B run ID.", type="warning")
                return

        layout_val = layout_select.value
        agent_id_val = agent_select.value
        env_val = env_select.value

        # Clear form and show loading
        nicewebrl.clear_element(card)
        with card:
            ui.markdown("## Loading...")
            ui.markdown(
                "Setting up environment and loading model. "
                "This may take a few minutes on first run (JAX compilation)."
            )
            spinner = ui.spinner(size="lg")

        # Load model and set up experiment in a background thread
        loop = asyncio.get_running_loop()
        try:
            setup_result = await loop.run_in_executor(
                None,
                lambda: setup_experiment(
                    tag=tag_val,
                    run_id=run_id_val,
                    layout_name=layout_val,
                    agent_id=agent_id_val,
                    env_name=env_val,
                ),
            )
        except Exception as e:
            nicewebrl.clear_element(card)
            with card:
                ui.markdown("## Error")
                ui.markdown(f"Failed to load model: `{e}`")
                ui.button("Retry", on_click=lambda: ui.navigate.to("/"))
            return

        experiment = setup_result['experiment']

        # Reset per-game state
        app.storage.user["current_step"] = 0
        app.storage.user["cumulative_reward"] = 0.0
        app.storage.user["experiment_finished"] = False
        app.storage.user["data_saved"] = False
        # Reset nicewebrl experiment progress indices (global keys, not namespaced)
        app.storage.user["stage_idx"] = 0
        app.storage.user["block_idx"] = 0

        # Switch to game UI
        nicewebrl.clear_element(card)
        with card:
            meta_container = ui.column().style("align-items: center;")
            with meta_container:
                label_text = tag_val or run_id_val
                setup_env = setup_result.get("env_name", env_val)
                setup_layout = setup_result.get("layout_name", layout_val)
                ui.markdown(
                    f"**Env:** {setup_env} | **Playing:** {setup_layout} | **Model:** {label_text} | **AI Agent:** {agent_id_val}"
                )
                stage_container = ui.column()

        episode_metadata = {
            "partner": tag_val or run_id_val,
            "layout": setup_result.get("layout_name", layout_val),
            "agent_id": agent_id_val,
            "env": setup_result.get("env_name", env_val),
        }
        await run_experiment_loop(experiment, stage_container, episode_metadata=episode_metadata)


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
