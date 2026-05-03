#!/usr/bin/env python3
"""Precompile JAX XLA artifacts for all unique (layout, env) combinations.

JAX compilation depends only on layout and env type, not on model weights or
agent ID. Running this at Docker build time populates the persistent cache so
that runtime JAX JIT calls are cache hits rather than full LLVM compilations.

Run this inside the Docker build environment (same hardware as deployment)
so the compiled artifacts are cache-valid at runtime.

Usage:
    JAX_COMPILATION_CACHE_DIR=.jax_cache python scripts/precompile.py
"""

import os
import sys

_cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR", ".jax_cache")
os.makedirs(_cache_dir, exist_ok=True)

import jax
if not hasattr(jax, "tree_map"):
    jax.tree_map = jax.tree_util.tree_map
jax.config.update("jax_compilation_cache_dir", _cache_dir)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from web_app.constants import TUTORIAL_LAYOUT, EXPERIMENT_LAYOUTS
from web_app.experiment import (
    create_environment,
    wrap_environment,
    compile_render_fns,
    get_action_config,
)

LAYOUTS = sorted({TUTORIAL_LAYOUT} | set(EXPERIMENT_LAYOUTS))
ENV_NAME = "overcooked"


def main():
    action_array, _ = get_action_config(ENV_NAME)

    print(f"JAX cache dir: {os.path.abspath(_cache_dir)}")
    print(f"Compiling {len(LAYOUTS)} unique layout(s): {LAYOUTS}\n")

    failed = []
    for i, layout in enumerate(LAYOUTS, 1):
        print(f"[{i}/{len(LAYOUTS)}] layout={layout}")
        try:
            env, _ = create_environment(layout, ENV_NAME)
            jax_web_env = wrap_environment(env, action_array)
            compile_render_fns(jax_web_env, ENV_NAME)
            print("  OK\n")
        except Exception as e:
            print(f"  FAILED: {e}\n")
            failed.append((layout, e))

    print("=" * 60)
    print(f"Finished: {len(LAYOUTS) - len(failed)}/{len(LAYOUTS)} succeeded.")
    if failed:
        for layout, e in failed:
            print(f"  FAILED: layout={layout}: {e}")
        sys.exit(1)
    print(f"Cache written to: {os.path.abspath(_cache_dir)}")


if __name__ == "__main__":
    main()
