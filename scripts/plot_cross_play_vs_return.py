"""Side-by-side comparison: cross-play returns (left) and human-study returns (right), shared y-axis."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SEP_REP_ROOT = "/sep-rep-learning"

CROSS_PLAY_CSV = os.path.join(
    _SEP_REP_ROOT,
    "results/evaluate_cross_play/overcooked/cross_play_barplot_original_layouts_by_model_summary.csv",
)
RETURN_DATA_CSV = os.path.join(_PROJ_ROOT, "results/analyze_data/return_data.csv")
RETURN_SIG_CSV = os.path.join(_PROJ_ROOT, "results/analyze_data/return_significance.csv")
OUT_PATH = os.path.join(_PROJ_ROOT, "figures/cross_play_vs_return.png")

DISPLAY_COLORS = {
    "SP": "tab:purple",
    "FCP": "tab:blue",
    "MEP": "tab:orange",
    "CEC": "tab:green",
    "CECP": "tab:red",
}
DISPLAY_ORDERING = ["SP", "FCP", "MEP", "CEC", "CECP"]

CROSS_PLAY_LAYOUT_ORDER = [
    "Asymmetric Advantages",
    "Coordination Ring",
    "Counter Circuit",
    "Cramped Room",
    "Forced Coordination",
    "All 5",
]
RETURN_LAYOUT_ORDER = ["Coordination Ring", "Counter Circuit", "Both Layouts"]


def _draw_brackets(ax, sig_df, bar_positions, x_display_order, hue_order,
                   group_col, tick_frac=0.04):
    """Draw significance brackets for CECP vs others."""
    cecp = "CECP"
    y_top = ax.get_ylim()[1]
    step = 0.09 * y_top
    tick_len = tick_frac * y_top
    max_y = y_top

    for x_idx, x_label in enumerate(x_display_order):
        grp = sig_df[sig_df[group_col] == x_label].sort_values("p_raw")
        if grp.empty:
            continue
        cecp_x = bar_positions.get((x_idx, cecp), (None, None))[0]
        if cecp_x is None:
            continue

        bar_heights = [bar_positions.get((x_idx, h), (0, 0))[1] for h in hue_order]
        base_y = max(bar_heights) + 0.5 * step

        for b_idx, (_, row) in enumerate(grp.iterrows()):
            other = row["other_display"]
            label = row["label"]
            other_x = bar_positions.get((x_idx, other), (None, None))[0]
            if other_x is None:
                continue
            y = base_y + b_idx * step
            max_y = max(max_y, y)
            color = "black" if label != "ns" else "#888888"
            ls = "-" if label != "ns" else "--"
            ax.plot([cecp_x, cecp_x], [y - tick_len, y], color=color, lw=1.2, ls=ls, clip_on=False)
            ax.plot([other_x, other_x], [y - tick_len, y], color=color, lw=1.2, ls=ls, clip_on=False)
            ax.plot([cecp_x, other_x], [y, y], color=color, lw=1.2, ls=ls, clip_on=False)
            ax.text((cecp_x + other_x) / 2, y, label, ha="center", va="bottom",
                    fontsize=9 if label == "ns" else 11, color=color)

    ax.set_ylim(top=max_y + 1.5 * step)


def _draw_grouped_bars(ax, means_pivot, stds_pivot, layout_order, hue_order, bar_width):
    """Draw grouped bars from pivots (layout x hue). Returns bar_positions dict."""
    x_centers = np.arange(len(layout_order))
    n_hues = len(hue_order)
    bar_positions = {}

    for h_idx, label in enumerate(hue_order):
        offset = (h_idx - (n_hues - 1) / 2) * bar_width
        xs = x_centers + offset
        means = means_pivot[label].values if label in means_pivot.columns else np.full(len(layout_order), np.nan)
        stds = stds_pivot[label].values if label in stds_pivot.columns else np.zeros(len(layout_order))
        ax.bar(xs, means, width=bar_width * 0.95,
               color=DISPLAY_COLORS.get(label), label=label, alpha=0.85)
        ax.errorbar(xs, means, yerr=stds, fmt="none", ecolor="black",
                    elinewidth=1.2, capsize=0, zorder=4)
        for l_idx, (x, h) in enumerate(zip(xs, means)):
            bar_positions[(l_idx, label)] = (float(x), 0.0 if np.isnan(h) else float(h))

    ax.set_xticks(x_centers)
    ax.set_xticklabels(layout_order, rotation=30, ha="right")
    return bar_positions


def _plot_cross_play(ax, df, layout_order, bar_width):
    df = df[df["layout"].isin(layout_order)].copy()
    hue_order = [h for h in DISPLAY_ORDERING if h in df["label"].unique()]
    means_pivot = df.pivot(index="layout", columns="label", values="mean").reindex(layout_order)
    stds_pivot = df.pivot(index="layout", columns="label", values="std").reindex(layout_order)
    _draw_grouped_bars(ax, means_pivot, stds_pivot, layout_order, hue_order, bar_width)
    ax.set_xlabel("Layout")
    ax.set_ylabel("Average Return")
    ax.set_title("Cross-play Returns")
    return hue_order


def _plot_return(ax, df, sig_df, layout_order, bar_width):
    df = df[df["layout"].isin(layout_order)].copy()
    hue_order = [h for h in DISPLAY_ORDERING if h in df["tag"].unique()]
    agg = df.groupby(["layout", "tag"])["total_return"].agg(["mean", "std"]).reset_index()
    means_pivot = agg.pivot(index="layout", columns="tag", values="mean").reindex(layout_order)
    stds_pivot = agg.pivot(index="layout", columns="tag", values="std").reindex(layout_order)
    bar_positions = _draw_grouped_bars(ax, means_pivot, stds_pivot, layout_order, hue_order, bar_width)

    if sig_df is not None and not sig_df.empty:
        _draw_brackets(ax, sig_df, bar_positions, layout_order, hue_order, group_col="layout_display")

    ax.set_xlabel("Layout")
    ax.set_ylabel("")
    ax.set_title("Human-study Returns")
    return hue_order


def main():
    cross_df = pd.read_csv(CROSS_PLAY_CSV)
    return_df = pd.read_csv(RETURN_DATA_CSV)
    sig_df = pd.read_csv(RETURN_SIG_CSV) if os.path.exists(RETURN_SIG_CSV) else None

    cross_layout_order = [l for l in CROSS_PLAY_LAYOUT_ORDER if l in cross_df["layout"].values]
    return_layout_order = [l for l in RETURN_LAYOUT_ORDER if l in return_df["layout"].values]

    n_left = len(cross_layout_order)
    n_right = len(return_layout_order)

    hue_order_left = [h for h in DISPLAY_ORDERING if h in cross_df["label"].unique()]
    hue_order_right = [h for h in DISPLAY_ORDERING if h in return_df["tag"].unique()]
    # Same bar_width for both panels; width_ratios=[n_left, n_right] ensures equal
    # pixels-per-data-unit in both axes, so identical physical bar sizes.
    bar_width = 0.8 / max(len(hue_order_left), len(hue_order_right))

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2,
        figsize=(1.5 + (n_left + n_right) * 1.1, 5),
        sharey=True,
        gridspec_kw={"width_ratios": [n_left, n_right], "wspace": 0.05},
        layout="constrained",
    )

    hue_order = _plot_cross_play(ax_left, cross_df, cross_layout_order, bar_width)
    _plot_return(ax_right, return_df, sig_df, return_layout_order, bar_width)

    handles = [mpatches.Patch(color=DISPLAY_COLORS[h], label=h) for h in hue_order]
    fig.legend(handles=handles, title="Model", loc="lower center",
               ncol=len(hue_order), bbox_to_anchor=(0.5, -0.02))

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    fig.savefig(OUT_PATH.replace(".png", ".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
