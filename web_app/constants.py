"""Shared experiment constants used by both the web app and offline scripts."""

TUTORIAL_TAG = "oc_cecp_pred_1000"

# Tags that appear in the experiment proper
EXPERIMENT_TAGS = ["oc_cec_v3", "oc_cecp_pred_1000", "fcp", "mep", "comedi_br", "pace_br"]
EXPERIMENT_LAYOUTS = ["coord_ring_9", "counter_circuit_9"]

# Tags whose models are stored per-layout (e.g. fcp_coord_ring_9 instead of fcp)
ORIGINAL_5_TAGS = ["fcp", "mep", "comedi_br", "pace_br"]

# Agent slot assignment. Since 2026-09-20 the human always controls agent 0
# (rendered red) and the model always controls agent 1 (rendered blue), and the
# slot is written to every gameplay record's metadata["human_id"] / ["model_id"].
# Data collected before that date had human_id drawn randomly per stage by
# nicewebrl's MultiAgentEnvStage and was NOT logged; see the slot-inference code in
# scripts/evaluate_human_action_prediction.py for how it is recovered there.
HUMAN_AGENT_ID = 0
MODEL_AGENT_ID = 1
# Matches jaxmarl's overcooked visualizer: agent 0 is red, agent 1 is blue.
AGENT_COLORS = {0: "Red", 1: "Blue"}
