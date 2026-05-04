# Map from W&B tag (or run ID) to the display name used in figures.
# File and folder names are unaffected — only axis labels and titles change.
DISPLAY_NAMES: dict[str, str] = {
    "sp": "SP",
    "fcp": "FCP",
    "mep": "MEP",
    "oc_cec": "CEC",
    "oc_cecp": "CECP no pred",
    "oc_cecp_pred": "CECP Pred",
    "oc_cecp_pred_vib": "CECP+Pred+VIB",
    # "oc_cecp_pred_1000": "CECP+Pred+1000",
    "oc_cecp_pred_1000": "CECP",
    "cf_cec": "CEC",
    "cf_cecp": "CECP",
    "cf_cecp_pred": "CECP+Pred",
    "cf_cecp_pred_vib": "CECP+Pred+VIB",
    "oc_cec_v3": "CEC",
    "oc_fcp_pred_v3_partners_3_ckpts": "CECP+Pred",
}

OC_ORIGINAL_LAYOUT_NAMES: dict[str, str] = {
    'cramped_room_9': 'Cramped Room',
    'asymm_advantages_9': 'Asymmetric Advantages',
    'coord_ring_9': 'Coordination Ring',
    'counter_circuit_9': 'Counter Circuit',
    'forced_coord_9': 'Forced Coordination',
}

DISPLAY_ORDERING: list[str] = ["SP", "FCP", "MEP","CEC", "CECP"]