from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

from config import DIAGNOSTICS_FILE, FIGURE_DIR
from figure_utils import save_figure

ROWS = [
    ("Production_12c", "ln_sts_c16_sa_idx2021_100", "Production (12c): C16"),
    ("Production_12c", "ln_sts_c31_sa_idx2021_100", "Production (12c): C31"),
    ("Production_8c", "ln_sts_c16_sa_idx2021_100", "Production sensitivity (8c): C16"),
    ("Production_8c", "ln_sts_c31_sa_idx2021_100", "Production sensitivity (8c): C31"),
    ("Trade_8c", "ln_trade_world_export_eur_hs4sum", "Trade (8c): exports"),
    ("Trade_8c", "ln_trade_world_import_eur_hs4sum", "Trade (8c): imports"),
]
SPLITS = ["Pre-shock", "Shock", "Post-shock"]
SPLIT_MAP = {"S1": "Pre-shock", "S2": "Shock", "S3": "Post-shock", "T1": "Pre-shock", "T2": "Shock", "T3": "Post-shock"}
COLOURS = {"V0": "#4C78A8", "E1": "#B8B8B8", "E2": "#ECA82C", "E3": "#D62728"}


def main() -> None:
    plt.rcParams.update({
        "figure.dpi": 180,
        "savefig.dpi": 600,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
        "font.size": 10,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
    })
    data = pd.read_excel(DIAGNOSTICS_FILE, sheet_name="Table_A1_execution_validity")
    data["regime"] = data["split_id"].map(SPLIT_MAP)
    fig, ax = plt.subplots(figsize=(10.4, 5.4))
    for i, (panel, target, label) in enumerate(ROWS):
        for j, regime in enumerate(SPLITS):
            hit = data[(data["panel"] == panel) & (data["target"] == target) & (data["regime"] == regime)]
            code = str(hit["failure_code"].iloc[0]) if not hit.empty else ""
            face = COLOURS.get(code, "white")
            hatch = "///" if code == "E1" else None
            ax.add_patch(Rectangle((j, len(ROWS) - i - 1), 1, 1, facecolor=face, edgecolor="white", linewidth=1.0, hatch=hatch))
            text = "V0\nvalid" if code == "V0" else f"{code}\nfallback" if code else ""
            ax.text(j + 0.5, len(ROWS) - i - 0.5, text, ha="center", va="center", fontsize=9.5, color="white" if code == "V0" else "black")
    ax.set_xlim(0, len(SPLITS))
    ax.set_ylim(0, len(ROWS))
    ax.set_xticks([0.5, 1.5, 2.5])
    ax.set_xticklabels(SPLITS)
    ax.set_yticks([i + 0.5 for i in range(len(ROWS))])
    ax.set_yticklabels([r[2] for r in ROWS][::-1])
    ax.tick_params(length=0)
    ax.set_title("Execution-validity map for pooled LightGBM", loc="left", fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    handles = [
        Patch(facecolor=COLOURS["V0"], label="V0: execution-valid LightGBM"),
        Patch(facecolor=COLOURS["E1"], hatch="///", label="E1: empty effective train/test design"),
        Patch(facecolor=COLOURS["E2"], label="E2: empty feature matrix"),
        Patch(facecolor=COLOURS["E3"], label="E3: fitting or prediction exception"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.00))
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    save_figure(fig, FIGURE_DIR / "FigureS1_execution_validity_map.png")


if __name__ == "__main__":
    main()
