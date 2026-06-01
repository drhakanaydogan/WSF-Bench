from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import PROTOCOL_FILE, FIGURE_DIR
from figure_utils import save_figure

TARGET_SPECS = [
    ("production_panel_12c", "ln_sts_c16_sa_idx2021_100", "A. Production: C16"),
    ("production_panel_12c", "ln_sts_c31_sa_idx2021_100", "B. Production: C31"),
    ("trade_panel_8c", "ln_trade_world_export_eur_hs4sum", "C. Trade: exports"),
    ("trade_panel_8c", "ln_trade_world_import_eur_hs4sum", "D. Trade: imports"),
]

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _monthly_profile(sheet_name: str, target: str) -> pd.DataFrame:
    df = pd.read_excel(PROTOCOL_FILE, sheet_name=sheet_name)
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    g = (
        df[["month", target]]
        .dropna()
        .groupby("month")[target]
        .agg(
            mean="mean",
            q25=lambda x: np.nanpercentile(x, 25),
            q75=lambda x: np.nanpercentile(x, 75),
        )
        .reindex(range(1, 13))
        .reset_index()
    )
    g["month_name"] = MONTH_LABELS
    return g


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

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.9), sharex=True)
    axes = axes.ravel()

    for ax, (sheet, target, title) in zip(axes, TARGET_SPECS):
        profile = _monthly_profile(sheet, target)
        x = np.arange(1, 13)
        y = profile["mean"].to_numpy(dtype=float)
        q25 = profile["q25"].to_numpy(dtype=float)
        q75 = profile["q75"].to_numpy(dtype=float)

        ax.fill_between(x, q25, q75, alpha=0.12, linewidth=0)
        ax.plot(x, y, marker="o", linewidth=1.5, markersize=3.2)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel("Log-transformed value")
        ax.set_xticks(x)
        ax.set_xticklabels(MONTH_LABELS)
        ax.grid(axis="y", linestyle="--", alpha=0.28)

    axes[2].set_xlabel("Month")
    axes[3].set_xlabel("Month")
    fig.tight_layout()
    save_figure(fig, FIGURE_DIR / "Figure3_monthly_profiles.png")


if __name__ == "__main__":
    main()
