from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from config import MAIN_RESULTS_FILE, DIAGNOSTICS_FILE


def bootstrap_ci_diff(x, y, n_boot: int = 10000, seed: int = 42) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    diff = x - y
    if len(diff) == 0:
        return np.nan, np.nan, np.nan
    boot_means = np.asarray([np.nanmean(rng.choice(diff, size=len(diff), replace=True)) for _ in range(n_boot)])
    return float(np.nanmean(diff)), float(np.nanpercentile(boot_means, 2.5)), float(np.nanpercentile(boot_means, 97.5))


def paired_tests(metrics: pd.DataFrame, model_a: str, model_b: str) -> pd.DataFrame:
    rows = []
    for (panel, target, split_id), group in metrics.groupby(["panel", "target", "split_id"]):
        a = group[group["model"] == model_a][["country", "mase"]].rename(columns={"mase": "mase_a"})
        b = group[group["model"] == model_b][["country", "mase"]].rename(columns={"mase": "mase_b"})
        merged = a.merge(b, on="country", how="inner").dropna()
        if merged.shape[0] < 3:
            continue
        x = merged["mase_a"].to_numpy(dtype=float)
        y = merged["mase_b"].to_numpy(dtype=float)
        try:
            stat, p_value = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
        except Exception:
            stat, p_value = np.nan, np.nan
        mean_diff, ci_low, ci_high = bootstrap_ci_diff(x, y)
        rows.append({
            "panel": panel,
            "target": target,
            "split_id": split_id,
            "model_a": model_a,
            "model_b": model_b,
            "n_countries": merged.shape[0],
            "mean_mase_a": np.nanmean(x),
            "mean_mase_b": np.nanmean(y),
            "mean_difference_a_minus_b": mean_diff,
            "bootstrap_ci_low": ci_low,
            "bootstrap_ci_high": ci_high,
            "wilcoxon_stat": stat,
            "wilcoxon_p_value": p_value,
            "interpretation": "Negative difference favours model_a; positive difference favours model_b",
        })
    return pd.DataFrame(rows)


def build_diagnostics() -> dict[str, pd.DataFrame]:
    summary = pd.read_excel(MAIN_RESULTS_FILE, sheet_name="summary_by_split_target")
    metrics = pd.read_excel(MAIN_RESULTS_FILE, sheet_name="country_level_metrics")
    execution_log = pd.read_excel(MAIN_RESULTS_FILE, sheet_name="execution_validity_log")

    valid_summary = summary[summary["model"] != "LightGBM_fallback"].copy()
    no_fallback_winners = (
        valid_summary.sort_values(["panel", "target", "split_id", "mean_mase"])
        .groupby(["panel", "target", "split_id"], as_index=False)
        .first()
    )
    no_fallback_means = (
        valid_summary.groupby(["panel", "model"], as_index=False)["mean_mase"]
        .mean()
        .sort_values(["panel", "mean_mase"])
    )

    test_pairs = [
        ("LightGBM", "Seasonal Naive"),
        ("LightGBM", "ETS"),
        ("ETS", "Seasonal Naive"),
        ("LightGBM_fallback", "ETS"),
        ("LightGBM_fallback", "Seasonal Naive"),
    ]
    stat_tests = pd.concat([paired_tests(metrics, a, b) for a, b in test_pairs], ignore_index=True)

    regime_map = {"S1": "pre", "S2": "shock", "S3": "post", "T1": "pre", "T2": "shock", "T3": "post"}
    regime_input = summary.copy()
    regime_input["regime"] = regime_input["split_id"].map(regime_map)
    regime_input["model_path"] = regime_input["model"].replace({"LightGBM_fallback": "LightGBM_path", "LightGBM": "LightGBM_path"})
    regime = (
        regime_input.pivot_table(index=["panel", "target", "model_path"], columns="regime", values="mean_mase", aggfunc="mean")
        .reset_index()
    )
    for col in ["pre", "shock", "post"]:
        if col not in regime.columns:
            regime[col] = np.nan
    regime["RDR_shock_over_pre"] = regime["shock"] / regime["pre"]
    regime["RDR_post_over_pre"] = regime["post"] / regime["pre"]
    regime["RSI_cv_across_regimes"] = regime[["pre", "shock", "post"]].std(axis=1) / regime[["pre", "shock", "post"]].mean(axis=1)

    flags = []
    for (panel, target), group in summary[summary["model"].isin(["LightGBM", "LightGBM_fallback"])].groupby(["panel", "target"]):
        flags.append({"panel": panel, "target": target, "model_path": "LightGBM_path", "has_fallback_block": (group["model"] == "LightGBM_fallback").any()})
    flags = pd.DataFrame(flags)
    regime = regime.merge(flags, on=["panel", "target", "model_path"], how="left")
    regime["has_fallback_block"] = regime["has_fallback_block"].fillna(False)
    regime["interpretation_note"] = np.where(
        (regime["model_path"] == "LightGBM_path") & (regime["has_fallback_block"]),
        "LightGBM path includes fallback-contingent block(s); do not interpret as fully valid ML robustness.",
        "No fallback flag for this model path.",
    )

    robust_metrics = (
        metrics.groupby(["panel", "target", "split_id", "model"], as_index=False)
        .agg(
            mean_mae=("mae", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_smape=("smape", "mean"),
            mean_mase=("mase", "mean"),
            n_countries=("country", "nunique"),
        )
    )

    return {
        "Table_A1_execution_validity": execution_log,
        "Table_A2_robust_metrics": robust_metrics,
        "Table_A3_no_fallback_winners": no_fallback_winners,
        "Table_A4_no_fallback_means": no_fallback_means,
        "Table_A5_stat_tests": stat_tests,
        "Table_A6_regime_robustness": regime,
    }


def main() -> None:
    DIAGNOSTICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tables = build_diagnostics()
    with pd.ExcelWriter(DIAGNOSTICS_FILE, engine="xlsxwriter") as writer:
        for sheet, table in tables.items():
            table.to_excel(writer, sheet_name=sheet, index=False)


if __name__ == "__main__":
    main()
