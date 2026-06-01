from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

from config import MAIN_RESULTS_FILE, TRADE_RECURRENT_RESULTS_FILE, FIGURE_DIR
from figure_utils import save_figure

COLOURS = {
    "Seasonal Naive": "#8A8A8A",
    "ETS": "#4C78A8",
    "LightGBM (valid)": "#F58518",
    "Fallback-contingent": "#B8B8B8",
    "LSTM": "#D62728",
}

PRODUCTION_ROWS = [
    ("Production_12c", "Production C16", "Production (12c): C16"),
    ("Production_12c", "Production C31", "Production (12c): C31"),
    ("Production_8c", "Production C16", "Production sensitivity (8c): C16"),
    ("Production_8c", "Production C31", "Production sensitivity (8c): C31"),
]

TRADE_ROWS = [
    ("Trade_8c", "Trade export", "Trade (8c): exports"),
    ("Trade_8c", "Trade import", "Trade (8c): imports"),
]


def _winner_label(model: str) -> str:
    if model == "LightGBM":
        return "LightGBM (valid)"
    if model == "LightGBM_fallback":
        return "Fallback-contingent"
    return str(model)


def _load_winners() -> tuple[pd.DataFrame, pd.DataFrame]:
    main = pd.read_excel(MAIN_RESULTS_FILE, sheet_name="summary_by_split_target")
    recurrent = pd.read_excel(TRADE_RECURRENT_RESULTS_FILE, sheet_name="summary_by_split_target")
    prod = main[main["panel"].isin(["Production_12c", "Production_8c"])].copy()
    trade = recurrent[recurrent["panel"] == "Trade_8c"].copy()
    prod_winners = prod.sort_values(["panel", "target", "split_id", "mean_mase"]).groupby(["panel", "target", "split_id"], as_index=False).first()
    trade_winners = trade.sort_values(["panel", "target", "split_id", "mean_mase"]).groupby(["panel", "target", "split_id"], as_index=False).first()
    prod_winners["winner_plot"] = prod_winners["model"].apply(_winner_label)
    trade_winners["winner_plot"] = trade_winners["model"].apply(_winner_label)
    return prod_winners, trade_winners


def _draw_map(ax, winners: pd.DataFrame, rows: list[tuple[str, str, str]], splits: list[str], title: str) -> None:
    ax.set_title(title, loc="left", fontweight="bold")
    for i, (panel, target, row_label) in enumerate(rows):
        for j, split in enumerate(splits):
            hit = winners[(winners["panel"] == panel) & (winners["target"] == target) & (winners["split_id"] == split)]
            label = str(hit["winner_plot"].iloc[0]) if not hit.empty else ""
            face = COLOURS.get(label, "white")
            hatch = "///" if label == "Fallback-contingent" else None
            rect = Rectangle((j, len(rows) - i - 1), 1, 1, facecolor=face, edgecolor="white", linewidth=1.0, hatch=hatch)
            ax.add_patch(rect)
            text_colour = "white" if label == "LightGBM (valid)" else "black"
            display = "LightGBM\n(valid)" if label == "LightGBM (valid)" else "Fallback\ncontingent" if label == "Fallback-contingent" else label
            ax.text(j + 0.5, len(rows) - i - 0.5, display, ha="center", va="center", fontsize=9, color=text_colour)
    ax.set_xlim(0, len(splits))
    ax.set_ylim(0, len(rows))
    ax.set_xticks(np.arange(len(splits)) + 0.5)
    ax.set_xticklabels(splits)
    ax.set_yticks(np.arange(len(rows)) + 0.5)
    ax.set_yticklabels([r[2] for r in rows][::-1])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def main() -> None:
    plt.rcParams.update({
        "figure.dpi": 180,
        "savefig.dpi": 600,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
    })
    prod_winners, trade_winners = _load_winners()
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 6.2), gridspec_kw={"height_ratios": [2.1, 1.1]})
    _draw_map(axes[0], prod_winners, PRODUCTION_ROWS, ["S1", "S2", "S3"], "A. Production benchmark blocks")
    _draw_map(axes[1], trade_winners, TRADE_ROWS, ["T1", "T2", "T3"], "B. Trade benchmark blocks")
    handles = [Patch(facecolor=COLOURS[k], label=k, hatch="///" if k == "Fallback-contingent" else None) for k in COLOURS]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.00))
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    save_figure(fig, FIGURE_DIR / "Figure5_execution_validity_aware_winners.png")


if __name__ == "__main__":
    main()
