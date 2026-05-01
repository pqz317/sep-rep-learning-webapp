#!/usr/bin/env python3
"""Precompile JAX XLA artifacts for known run/layout/agent combinations.

Run this inside the Docker build environment (same hardware as deployment)
so the compiled artifacts are cache-valid at runtime.

Usage:
    JAX_COMPILATION_CACHE_DIR=.jax_cache python scripts/precompile.py

The cache dir is read from JAX_COMPILATION_CACHE_DIR (defaults to .jax_cache).
WANDB_API_KEY must be set in the environment to resolve run configs.
"""

import os
import sys

# Must be set before any JAX compilation — do it before importing web_app modules.
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

from web_app.experiment import setup_experiment

# ── Define combinations to precompile ────────────────────────────────────────
# Each entry: (tag, layout_name, agent_id, env_name)
# Set layout_name=None to use the layout recorded in the W&B run config.
COMBINATIONS = [
    ("oc_cec_v3", "coord_ring_9", 0, "overcooked"),
    ("oc_cec_v3", "counter_circuit_9", 0, "overcooked"),
    ("oc_cecp_pred_1000", "coord_ring_9", 0, "overcooked"),
    ("oc_cecp_pred_1000", "counter_circuit_9", 0, "overcooked"),
]
# ─────────────────────────────────────────────────────────────────────────────


def main():
    if not COMBINATIONS:
        print("No combinations defined — skipping precompile. Edit scripts/precompile.py to add them.")
        return

    print(f"JAX cache dir: {os.path.abspath(_cache_dir)}")
    print(f"Precompiling {len(COMBINATIONS)} combination(s)...\n")

    failed = []
    for i, (tag, layout, agent_id, env_name) in enumerate(COMBINATIONS, 1):
        label = f"tag={tag} layout={layout} agent={agent_id} env={env_name}"
        print(f"[{i}/{len(COMBINATIONS)}] {label}")
        try:
            setup_experiment(
                tag=tag,
                layout_name=layout,
                agent_id=agent_id,
                env_name=env_name,
            )
            print(f"  OK\n")
        except Exception as e:
            print(f"  FAILED: {e}\n")
            failed.append((label, e))

    print("=" * 60)
    print(f"Finished: {len(COMBINATIONS) - len(failed)}/{len(COMBINATIONS)} succeeded.")
    if failed:
        for label, e in failed:
            print(f"  FAILED: {label}: {e}")
        sys.exit(1)
    print(f"Cache written to: {os.path.abspath(_cache_dir)}")


if __name__ == "__main__":
    main()
