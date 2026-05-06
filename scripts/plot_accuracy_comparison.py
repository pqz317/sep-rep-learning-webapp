"""Plot action-prediction accuracy for AI partners vs human subjects as a
combined bar chart with mean ± SD error bars.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

AI_SUMMARY = os.path.join(
    os.path.dirname(__file__),
    "../../sep-rep-learning/results/evaluate_action_prediction/hksyr2i5/summary.json",
)
HUMAN_SUMMARY = os.path.join(
    os.path.dirname(__file__),
    "../results/evaluate_human_action_prediction/oc_cecp_pred_1000/summary.json",
)


def load_data(ai_path, human_path):
    with open(ai_path) as f:
        ai = json.load(f)
    with open(human_path) as f:
        human = json.load(f)

    ai_accs = list(ai["per_partner_accuracy"].values())
    human_accs = list(human["per_user_accuracy"].values())
    chance_level = ai["chance_level"]

    rows = (
        [{"group": "Best AI Partners", "accuracy": v} for v in ai_accs]
        + [{"group": "Human Subjects", "accuracy": v} for v in human_accs]
    )
    return pd.DataFrame(rows), chance_level


def plot(df, chance_level, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.barplot(
        data=df,
        x="group",
        y="accuracy",
        order=["Best AI Partners", "Human Subjects"],
        errorbar="sd",
        capsize=0.15,
        ax=ax,
    )

    ax.axhline(chance_level, linestyle="--", color="gray", linewidth=1,
               label=f"Chance ({chance_level})")

    for group, container in zip(["Best AI Partners", "Human Subjects"], ax.containers):
        sub = df[df["group"] == group]["accuracy"]
        mean, std = sub.mean(), sub.std()
        bar = container[0]
        # ax.text(
        #     bar.get_x() + bar.get_width() / 2,
        #     mean + std + 0.012,
        #     f"{mean:.3f}±{std:.3f}",
        #     ha="center", va="bottom", fontsize=9,
        # )

    ax.set_title("Partner movement prediction")
    ax.set_xlabel("")
    ax.set_ylabel("Prediction Accuracy")
    ax.set_ylim(0, 0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.replace(".png", ".svg"))
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ai-summary",
        default=AI_SUMMARY,
        help="Path to AI partners summary.json.",
    )
    parser.add_argument(
        "--human-summary",
        default=HUMAN_SUMMARY,
        help="Path to human subjects summary.json.",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(
            os.path.dirname(__file__),
            "../results/evaluate_human_action_prediction/oc_cecp_pred_1000/accuracy_comparison.png",
        ),
        help="Output path for the figure.",
    )
    args = parser.parse_args()

    df, chance_level = load_data(args.ai_summary, args.human_summary)
    plot(df, chance_level, args.out)
