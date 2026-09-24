# Map from W&B tag (or run ID) to the display name used in figures.
# File and folder names are unaffected — only axis labels and titles change.
DISPLAY_NAMES: dict[str, str] = {
    "sp": "SP",
    "fcp": "FCP",
    "mep": "MEP",
    "comedi_br": "CoMeDi",
    "pace_br": "PACE",
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

DISPLAY_ORDERING: list[str] = ["SP", "FCP", "MEP", "CoMeDi", "PACE", "CEC", "CECP"]

DISPLAY_COLORS: dict[str, str] = {
    "SP": "tab:purple",
    "FCP": "tab:blue",
    "MEP": "tab:orange",
    "PACE": "tab:brown",
    "CoMeDi": "tab:pink",
    "CEC": "tab:green",
    "CEC+LIPO": "tab:olive",
    "CECP no pred": "tab:gray",
    "CECP no pred + PACE": "tab:cyan",
    "CECP": "tab:red",
}
