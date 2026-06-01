from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from config import (
    INTERPRETABILITY_FILE,
    LIGHTGBM_PARAMS,
    PRODUCTION_SPLITS,
    PRODUCTION_TARGETS,
    PROTOCOL_FILE,
    TRADE_SPLITS,
    TRADE_TARGETS,
)
from run_leaksafe_benchmark import add_leaksafe_lgbm_features


def group_feature(feature_name: str) -> str:
    feature = str(feature_name)
    if feature.startswith("country_"):
        return "Country indicators"
    if feature in {"lag1", "lag3", "lag6", "lag12"}:
        return "Target lags"
    if feature in {"diff1_lagged", "diff12_lagged"}:
        return "Lagged differences"
    if feature in {"month_sin", "month_cos"}:
        return "Cyclic calendar terms"
    if feature == "year":
        return "Calendar year"
    if feature.startswith("fao_") and feature.endswith("_lag12m"):
        return "Lagged annual FAOSTAT anchors"
    return "Other"


def train_lgbm_for_interpretability(df, target, target_label, fao_cols, split_id, split_start, split_end, panel_name):
    data, feature_cols = add_leaksafe_lgbm_features(df, target, fao_cols)
    start = pd.Timestamp(split_start)
    end = pd.Timestamp(split_end)
    train = data[data["date"] < start].copy()
    test = data[(data["date"] >= start) & (data["date"] <= end)].copy()
    train = train[train[target].notna()].copy()
    test = test[test[target].notna()].copy()
    usable = [col for col in feature_cols if col == "country" or (col in train.columns and train[col].notna().sum() > 0)]
    train = train.dropna(subset=usable + [target]).copy()
    test = test.dropna(subset=usable + [target]).copy()
    base_log = {
        "panel": panel_name,
        "target": target_label,
        "target_variable": target,
        "split_id": split_id,
        "split_start": split_start,
        "split_end": split_end,
        "candidate_features": len(feature_cols),
        "usable_features_before_dummies": len(usable),
    }
    if train.empty or test.empty:
        return None, {**base_log, "status": "fallback_empty_train_or_test", "failure_code": "E1", "train_rows": train.shape[0], "test_rows": test.shape[0], "n_features": np.nan}
    train_x = pd.get_dummies(train[usable], columns=["country"], drop_first=False)
    test_x = pd.get_dummies(test[usable], columns=["country"], drop_first=False)
    train_x, test_x = train_x.align(test_x, join="left", axis=1, fill_value=0)
    if train_x.empty or test_x.empty or train_x.shape[1] == 0:
        return None, {**base_log, "status": "fallback_empty_matrix", "failure_code": "E2", "train_rows": train_x.shape[0], "test_rows": test_x.shape[0], "n_features": train_x.shape[1] if train_x.ndim == 2 else 0}
    try:
        model = LGBMRegressor(**LIGHTGBM_PARAMS)
        model.fit(train_x, train[target])
        log = {**base_log, "status": "success", "failure_code": "V0", "train_rows": train_x.shape[0], "test_rows": test_x.shape[0], "n_features": train_x.shape[1]}
        return {"model": model, "X_train": train_x, "X_test": test_x, "metadata": log}, log
    except Exception as exc:
        return None, {**base_log, "status": f"fallback_exception: {exc}", "failure_code": "E3", "train_rows": train_x.shape[0], "test_rows": test_x.shape[0], "n_features": train_x.shape[1]}


def build_interpretability_outputs() -> dict[str, pd.DataFrame]:
    import shap

    prod12 = pd.read_excel(PROTOCOL_FILE, sheet_name="production_panel_12c")
    prod8 = pd.read_excel(PROTOCOL_FILE, sheet_name="production_panel_8c")
    trade8 = pd.read_excel(PROTOCOL_FILE, sheet_name="trade_panel_8c")
    for df in [prod12, prod8, trade8]:
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month

    panel_specs = [
        ("Production_12c", prod12, PRODUCTION_TARGETS, PRODUCTION_SPLITS),
        ("Production_8c", prod8, PRODUCTION_TARGETS, PRODUCTION_SPLITS),
        ("Trade_8c", trade8, TRADE_TARGETS, TRADE_SPLITS),
    ]
    runs = []
    logs = []
    for panel_name, df, targets, splits in panel_specs:
        fao_cols = [col for col in df.columns if col.startswith("fao_")]
        for target, label in targets.items():
            for split_id, start, end in splits:
                run, log = train_lgbm_for_interpretability(df, target, label, fao_cols, split_id, start, end, panel_name)
                logs.append(log)
                if run is not None:
                    runs.append(run)

    gain_rows = []
    shap_rows = []
    shap_errors = []
    for run in runs:
        model = run["model"]
        meta = run["metadata"]
        features = model.booster_.feature_name()
        gains = model.booster_.feature_importance(importance_type="gain")
        total_gain = gains.sum()
        gain_pct = np.zeros_like(gains, dtype=float) if total_gain == 0 else 100 * gains / total_gain
        for feature, gain, pct in zip(features, gains, gain_pct):
            gain_rows.append({
                "panel": meta["panel"],
                "target": meta["target"],
                "split_id": meta["split_id"],
                "feature": feature,
                "feature_group": group_feature(feature),
                "gain": gain,
                "gain_percent_within_run": pct,
                "train_rows": meta["train_rows"],
                "test_rows": meta["test_rows"],
                "n_features": meta["n_features"],
            })
        try:
            explainer = shap.TreeExplainer(model)
            values = explainer.shap_values(run["X_test"])
            if isinstance(values, list):
                values = values[0]
            mean_abs = np.abs(values).mean(axis=0)
            total = mean_abs.sum()
            shap_pct = np.zeros_like(mean_abs, dtype=float) if total == 0 else 100 * mean_abs / total
            for feature, val, pct in zip(run["X_test"].columns, mean_abs, shap_pct):
                shap_rows.append({
                    "panel": meta["panel"],
                    "target": meta["target"],
                    "split_id": meta["split_id"],
                    "feature": feature,
                    "feature_group": group_feature(feature),
                    "mean_abs_shap": val,
                    "shap_percent_within_run": pct,
                    "train_rows": meta["train_rows"],
                    "test_rows": meta["test_rows"],
                    "n_features": meta["n_features"],
                })
        except Exception as exc:
            shap_errors.append({"panel": meta["panel"], "target": meta["target"], "split_id": meta["split_id"], "error": str(exc)})

    logs_df = pd.DataFrame(logs)
    gain_individual = pd.DataFrame(gain_rows)
    if gain_individual.empty:
        gain_grouped_by_run = pd.DataFrame()
        gain_grouped_summary = pd.DataFrame()
    else:
        gain_grouped_by_run = gain_individual.groupby(["panel", "target", "split_id", "feature_group"], as_index=False).agg(gain=("gain", "sum"), gain_percent_within_run=("gain_percent_within_run", "sum"), train_rows=("train_rows", "first"), test_rows=("test_rows", "first"), n_features=("n_features", "first"))
        gain_grouped_summary = gain_grouped_by_run.groupby(["panel", "target", "feature_group"], as_index=False).agg(mean_gain_percent=("gain_percent_within_run", "mean"), min_gain_percent=("gain_percent_within_run", "min"), max_gain_percent=("gain_percent_within_run", "max"), n_valid_runs=("split_id", "nunique"))

    shap_individual = pd.DataFrame(shap_rows)
    if shap_individual.empty:
        shap_grouped_by_run = pd.DataFrame()
        shap_grouped_summary = pd.DataFrame()
        plot_data = pd.DataFrame()
    else:
        shap_grouped_by_run = shap_individual.groupby(["panel", "target", "split_id", "feature_group"], as_index=False).agg(shap_percent_within_run=("shap_percent_within_run", "sum"), train_rows=("train_rows", "first"), test_rows=("test_rows", "first"), n_features=("n_features", "first"))
        shap_grouped_summary = shap_grouped_by_run.groupby(["panel", "target", "feature_group"], as_index=False).agg(importance_percent=("shap_percent_within_run", "mean"), min_importance_percent=("shap_percent_within_run", "min"), max_importance_percent=("shap_percent_within_run", "max"), n_valid_runs=("split_id", "nunique"))
        plot_specs = [
            ("Production_12c", "Production C16", ["S1"], "A. Production C16 (S1)"),
            ("Production_12c", "Production C31", ["S1"], "B. Production C31 (S1)"),
            ("Trade_8c", "Trade export", ["T1", "T2", "T3"], "C. Trade exports (T1–T3)"),
            ("Trade_8c", "Trade import", ["T1", "T2", "T3"], "D. Trade imports (T1–T3)"),
        ]
        rows = []
        for panel, target, splits, label in plot_specs:
            subset = shap_grouped_by_run[(shap_grouped_by_run["panel"] == panel) & (shap_grouped_by_run["target"] == target) & (shap_grouped_by_run["split_id"].isin(splits))]
            grouped = subset.groupby("feature_group", as_index=False)["shap_percent_within_run"].mean().rename(columns={"shap_percent_within_run": "importance_percent"})
            grouped["plot_label"] = label
            rows.append(grouped)
        plot_data = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    return {
        "execution_logs": logs_df,
        "individual_gain_importance": gain_individual,
        "grouped_gain_by_run": gain_grouped_by_run,
        "grouped_gain_summary": gain_grouped_summary,
        "individual_shap_importance": shap_individual,
        "grouped_shap_by_run": shap_grouped_by_run,
        "grouped_shap_summary": shap_grouped_summary,
        "shap_errors": pd.DataFrame(shap_errors),
        "FigureS2_plot_data": plot_data,
    }


def main() -> None:
    outputs = build_interpretability_outputs()
    INTERPRETABILITY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(INTERPRETABILITY_FILE, engine="xlsxwriter") as writer:
        for sheet, data in outputs.items():
            data.to_excel(writer, sheet_name=sheet, index=False)


if __name__ == "__main__":
    main()
