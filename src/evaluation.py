from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def paired_wilcoxon_by_country(df: pd.DataFrame, model_a: str, model_b: str) -> dict:
    pivot = df.pivot_table(index="country", columns="model", values="MASE", aggfunc="mean")
    if model_a not in pivot or model_b not in pivot:
        return {"n": 0, "statistic": np.nan, "p_value": np.nan, "mean_difference": np.nan}
    paired = pivot[[model_a, model_b]].dropna()
    if len(paired) < 2:
        return {"n": len(paired), "statistic": np.nan, "p_value": np.nan, "mean_difference": np.nan}
    diff = paired[model_a] - paired[model_b]
    if np.allclose(diff, 0):
        return {"n": len(paired), "statistic": 0.0, "p_value": 1.0, "mean_difference": 0.0}
    stat, p_value = wilcoxon(diff)
    return {
        "n": int(len(paired)),
        "statistic": float(stat),
        "p_value": float(p_value),
        "mean_difference": float(diff.mean()),
    }


def bootstrap_ci(values, n_boot: int = 5000, alpha: float = 0.05, seed: int = 42) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(samples, alpha / 2)), float(np.quantile(samples, 1 - alpha / 2))


def regime_degradation_ratio(pre_value: float, comparison_value: float) -> float:
    if pre_value is None or not np.isfinite(pre_value) or pre_value == 0:
        return np.nan
    return float(comparison_value / pre_value)


def regime_stability_index(values) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0 or np.nanmean(values) == 0:
        return np.nan
    return float(np.nanstd(values, ddof=0) / np.nanmean(values))
