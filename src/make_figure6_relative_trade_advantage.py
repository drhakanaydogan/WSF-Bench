from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import TRADE_RECURRENT_RESULTS_FILE, FIGURE_DIR
from figure_utils import save_figure

COLOUR_SEASONAL = "#4C78A8"
COLOUR_ETS = "#F58518"
GRID_COLOUR = "#D9D9D9"


def _prepare_data() -> pd.DataFrame:
    summary = pd.read_excel(TRADE_RECURRENT_RESULTS_FILE, sheet_name="summary_by_split_target")
    trade = summary[
        (summary["panel"] == "Trade_8c")
        & (summary["target"].isin(["Trade export", "Trade import"]))
        & (summary["model"].isin(["Seasonal Naive", "ETS", "LightGBM"]))
    ].copy()
    pivot = (
        trade.pivot_table(index=["target", "split_id"], columns="model", values="mean_mase", aggfunc="mean")
        .reset_index()
    )
    pivot["Against Seasonal Naive"] = 100 * (pivot["LightGBM"] / pivot["Seasonal Naive"] - 1)
    pivot["Against ETS"] = 100 * (pivot["LightGBM"] / pivot["ETS"] - 1)
    split_order = {"T1": 0, "T2": 1, "T3": 2}
    pivot["split_order"] = pivot["split_id"].map(split_order)
    return pivot.sort_values(["target", "split_order"])


def _add_bar_label(ax, value: float, y: float, colour: str) -> None:
    label = f"{value:.1f}"
    if value <= -8:
        ax.text(value + 1.2, y, label, ha="left", va="center", fontsize=8.7, color="white", fontweight="bold" if abs(value) >= 30 else "normal")
    else:
        ax.text(value - 1.4, y, label, ha="right", va="center", fontsize=8.7, color=colour)


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
        "legend.fontsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
    })

    data = _prepare_data()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), sharex=True)
    panel_specs = [("Trade export", "A. Trade: exports"), ("Trade import", "B. Trade: imports")]
    bar_height = 0.28
    offset = 0.17

    for ax, (target, title) in zip(axes, panel_specs):
        sub = data[data["target"] == target].copy().sort_values("split_order")
        y_base = np.arange(len(sub))
        seasonal_vals = sub["Against Seasonal Naive"].to_numpy(dtype=float)
        ets_vals = sub["Against ETS"].to_numpy(dtype=float)

        ax.axvline(0, color="#333333", linewidth=0.9, zorder=1)
        ax.barh(y_base - offset, seasonal_vals, height=bar_height, color=COLOUR_SEASONAL, edgecolor="none", label="Against Seasonal Naive", zorder=3)
        ax.barh(y_base + offset, ets_vals, height=bar_height, color=COLOUR_ETS, edgecolor="none", label="Against ETS", zorder=3)

        for y, value in zip(y_base - offset, seasonal_vals):
            _add_bar_label(ax, value, y, COLOUR_SEASONAL)
        for y, value in zip(y_base + offset, ets_vals):
            _add_bar_label(ax, value, y, COLOUR_ETS)

        ax.set_yticks(y_base)
        ax.set_yticklabels(sub["split_id"].tolist())
        ax.invert_yaxis()
        ax.set_xlim(-65, 2)
        ax.set_xticks(np.arange(-60, 1, 10))
        ax.set_xlabel("Percentage difference in mean MASE")
        ax.set_title(title, loc="left", fontweight="bold", pad=10)
        ax.grid(axis="x", color=GRID_COLOUR, linestyle="--", linewidth=0.7, alpha=0.9)
        ax.grid(axis="y", visible=False)

    axes[0].set_ylabel("Evaluation split")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save_figure(fig, FIGURE_DIR / "Figure6_trade_percentage_MASE_difference.png")


if __name__ == "__main__":
    main()
