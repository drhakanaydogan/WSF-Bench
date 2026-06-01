from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from config import MAIN_RESULTS_FILE, TRADE_RECURRENT_RESULTS_FILE, FIGURE_DIR
from figure_utils import save_figure

COLOURS = {
    "Seasonal Naive": "#8A8A8A",
    "ETS": "#4C78A8",
    "LightGBM": "#F58518",
    "LightGBM path*": "#F58518",
    "LSTM": "#D62728",
    "Grid": "#D9D9D9",
    "Text": "#1F1F1F",
}

PANEL_SPECS = [
    ("Production C16", "A. Production: C16", ["Seasonal Naive", "ETS", "LightGBM path*"]),
    ("Production C31", "B. Production: C31", ["Seasonal Naive", "ETS", "LightGBM path*"]),
    ("Trade export", "C. Trade: exports", ["Seasonal Naive", "ETS", "LightGBM", "LSTM"]),
    ("Trade import", "D. Trade: imports", ["Seasonal Naive", "ETS", "LightGBM", "LSTM"]),
]


def _prepare_data() -> pd.DataFrame:
    main = pd.read_excel(MAIN_RESULTS_FILE, sheet_name="summary_by_split_target")
    recurrent = pd.read_excel(TRADE_RECURRENT_RESULTS_FILE, sheet_name="summary_by_split_target")
    rows = []

    production = main[
        (main["panel"] == "Production_12c")
        & (main["target"].isin(["Production C16", "Production C31"]))
    ].copy()
    production["model_plot"] = production["model"].replace({"LightGBM": "LightGBM path*", "LightGBM_fallback": "LightGBM path*"})
    production = production[production["model_plot"].isin(["Seasonal Naive", "ETS", "LightGBM path*"])]
    for (target, model), g in production.groupby(["target", "model_plot"]):
        rows.append({"target": target, "model": model, "mean_mase": g["mean_mase"].mean(), "min_mase": g["mean_mase"].min(), "max_mase": g["mean_mase"].max()})

    trade = recurrent[
        (recurrent["panel"] == "Trade_8c")
        & (recurrent["target"].isin(["Trade export", "Trade import"]))
        & (recurrent["model"].isin(["Seasonal Naive", "ETS", "LightGBM", "LSTM"]))
    ].copy()
    for (target, model), g in trade.groupby(["target", "model"]):
        rows.append({"target": target, "model": model, "mean_mase": g["mean_mase"].mean(), "min_mase": g["mean_mase"].min(), "max_mase": g["mean_mase"].max()})
    return pd.DataFrame(rows)


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
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    data = _prepare_data()
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.0))
    axes = axes.ravel()

    for ax, (target, title, model_order) in zip(axes, PANEL_SPECS):
        sub = data[data["target"] == target].copy()
        sub["model"] = pd.Categorical(sub["model"], categories=model_order, ordered=True)
        sub = sub.sort_values("model")
        y_positions = list(range(len(sub)))[::-1]
        for y, (_, row) in zip(y_positions, sub.iterrows()):
            model = str(row["model"])
            colour = COLOURS.get(model, "#333333")
            ax.hlines(y=y, xmin=row["min_mase"], xmax=row["max_mase"], color=colour, linewidth=1.6, alpha=0.85)
            ax.scatter(row["mean_mase"], y, s=38, color=colour, edgecolor="white", linewidth=0.7, zorder=3)
            ax.text(row["mean_mase"] + 0.035, y, f"{row['mean_mase']:.2f}", va="center", ha="left", fontsize=8.5, color=COLOURS["Text"])
        ax.set_yticks(y_positions)
        ax.set_yticklabels(sub["model"].astype(str).tolist())
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("Mean MASE")
        ax.grid(axis="x", color=COLOURS["Grid"], linewidth=0.6, alpha=0.8)
        ax.grid(axis="y", visible=False)
        xmax = max(sub["max_mase"].max(), sub["mean_mase"].max())
        ax.set_xlim(left=0, right=xmax * 1.23)

    legend_items = [
        Line2D([0], [0], marker="o", color=COLOURS["Seasonal Naive"], label="Seasonal Naive", markerfacecolor=COLOURS["Seasonal Naive"], linewidth=1.4),
        Line2D([0], [0], marker="o", color=COLOURS["ETS"], label="ETS", markerfacecolor=COLOURS["ETS"], linewidth=1.4),
        Line2D([0], [0], marker="o", color=COLOURS["LightGBM"], label="LightGBM / LightGBM path*", markerfacecolor=COLOURS["LightGBM"], linewidth=1.4),
        Line2D([0], [0], marker="o", color=COLOURS["LSTM"], label="LSTM", markerfacecolor=COLOURS["LSTM"], linewidth=1.4),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.00))
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    save_figure(fig, FIGURE_DIR / "Figure4_mean_MASE_profiles.png")


if __name__ == "__main__":
    main()
