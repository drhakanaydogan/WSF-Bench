from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt

from config import INTERPRETABILITY_FILE, FIGURE_DIR
from figure_utils import save_figure

ORDER = [
    "Target lags",
    "Lagged differences",
    "Cyclic calendar terms",
    "Calendar year",
    "Country indicators",
    "Lagged annual FAOSTAT anchors",
]
COLOURS = {
    "Target lags": "#4C78A8",
    "Lagged differences": "#F58518",
    "Cyclic calendar terms": "#54A24B",
    "Calendar year": "#B279A2",
    "Country indicators": "#E45756",
    "Lagged annual FAOSTAT anchors": "#72B7B2",
}


def main() -> None:
    plt.rcParams.update({
        "figure.dpi": 180,
        "savefig.dpi": 600,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    data = pd.read_excel(INTERPRETABILITY_FILE, sheet_name="FigureS2_plot_data")
    labels = list(data["plot_label"].drop_duplicates())
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.0), sharex=True)
    axes = axes.ravel()
    for ax, label in zip(axes, labels):
        sub = data[data["plot_label"] == label].copy()
        sub["feature_group"] = pd.Categorical(sub["feature_group"], ORDER, ordered=True)
        sub = sub.sort_values("feature_group", ascending=False)
        colours = [COLOURS.get(str(v), "#777777") for v in sub["feature_group"]]
        ax.barh(sub["feature_group"].astype(str), sub["importance_percent"], color=colours)
        for y, val in enumerate(sub["importance_percent"]):
            ax.text(val + 0.8, y, f"{val:.1f}", va="center", fontsize=8.8)
        ax.set_title(label, loc="left", fontweight="bold")
        ax.set_xlabel("Grouped SHAP contribution (%)")
        ax.set_xlim(0, 100)
        ax.grid(axis="x", linestyle="--", alpha=0.30)
    fig.tight_layout()
    save_figure(fig, FIGURE_DIR / "FigureS2_grouped_lgbm_interpretability.png")


if __name__ == "__main__":
    main()
