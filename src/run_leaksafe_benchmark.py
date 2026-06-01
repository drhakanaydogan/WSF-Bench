from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from config import (
    LIGHTGBM_PARAMS,
    MAIN_RESULTS_FILE,
    PRODUCTION_SPLITS,
    PRODUCTION_TARGETS,
    PROTOCOL_FILE,
    TRADE_SPLITS,
    TRADE_TARGETS,
)


def mase(y_true, y_pred, y_train, seasonality: int = 12) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_train = pd.Series(y_train).dropna().astype(float).to_numpy()
    if len(y_train) <= seasonality:
        if len(y_train) <= 1:
            return np.nan
        denom = np.mean(np.abs(np.diff(y_train)))
    else:
        denom = np.mean(np.abs(y_train[seasonality:] - y_train[:-seasonality]))
    if denom == 0 or np.isnan(denom):
        return np.nan
    return float(np.mean(np.abs(y_true - y_pred)) / denom)


def smape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    valid = denom != 0
    if valid.sum() == 0:
        return np.nan
    return float(np.mean(200 * np.abs(y_true[valid] - y_pred[valid]) / denom[valid]))


def seasonal_naive_forecast(train_series, horizon: int, season: int = 12) -> np.ndarray:
    history = list(pd.Series(train_series).dropna().astype(float))
    if not history:
        return np.full(horizon, np.nan)
    predictions = []
    for _ in range(horizon):
        pred = history[-season] if len(history) >= season else history[-1]
        predictions.append(pred)
        history.append(pred)
    return np.asarray(predictions, dtype=float)


def ets_forecast(train_series, horizon: int) -> np.ndarray:
    y = pd.Series(train_series).dropna().astype(float)
    if y.empty:
        return np.full(horizon, np.nan)
    if len(y) < 24:
        return np.repeat(y.iloc[-1], horizon)
    try:
        fit = ExponentialSmoothing(
            y,
            trend="add",
            seasonal="add",
            seasonal_periods=12,
            damped_trend=True,
        ).fit(optimized=True, use_brute=False)
        return np.asarray(fit.forecast(horizon), dtype=float)
    except Exception:
        return np.repeat(y.iloc[-1], horizon)


def country_baseline_eval(df: pd.DataFrame, target: str, split_start: str, split_end: str, model_name: str) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    start = pd.Timestamp(split_start)
    end = pd.Timestamp(split_end)
    rows = []
    train_cache = {}
    for country, group in df[["country", "date", target]].dropna().sort_values("date").groupby("country"):
        train = group.loc[group["date"] < start, target].to_numpy(dtype=float)
        test_group = group[(group["date"] >= start) & (group["date"] <= end)]
        test = test_group[target].to_numpy(dtype=float)
        if len(train) == 0 or len(test) == 0:
            continue
        if model_name == "Seasonal Naive":
            pred = seasonal_naive_forecast(train, len(test), season=12)
        elif model_name == "ETS":
            pred = ets_forecast(train, len(test))
        else:
            raise ValueError(f"Unknown baseline model: {model_name}")
        rows.append(pd.DataFrame({
            "country": country,
            "date": pd.to_datetime(test_group["date"].to_numpy()),
            "y_true": test,
            "y_pred": pred,
            "model": model_name,
        }))
        train_cache[country] = train
    if not rows:
        return pd.DataFrame(columns=["country", "date", "y_true", "y_pred", "model"]), {}
    return pd.concat(rows, ignore_index=True), train_cache


def add_leaksafe_lgbm_features(df: pd.DataFrame, target: str, fao_cols: Iterable[str]) -> tuple[pd.DataFrame, list[str]]:
    data = df.copy().sort_values(["country", "date"])
    for lag in [1, 2, 3, 6, 12, 13]:
        data[f"lag{lag}"] = data.groupby("country")[target].shift(lag)
    data["diff1_lagged"] = data["lag1"] - data["lag2"]
    data["diff12_lagged"] = data["lag1"] - data["lag13"]
    data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12)
    data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12)
    lagged_anchors = []
    for col in fao_cols:
        if col in data.columns and data[col].notna().sum() > 0:
            name = f"{col}_lag12m"
            data[name] = data.groupby("country")[col].shift(12)
            if data[name].notna().sum() > 0:
                lagged_anchors.append(name)
    features = [
        "country",
        "year",
        "month_sin",
        "month_cos",
        "lag1",
        "lag3",
        "lag6",
        "lag12",
        "diff1_lagged",
        "diff12_lagged",
    ] + lagged_anchors
    return data, features


def make_train_dict(df: pd.DataFrame, target: str, split_start: str) -> dict[str, np.ndarray]:
    start = pd.Timestamp(split_start)
    return {
        country: group.loc[group["date"] < start, target].dropna().to_numpy(dtype=float)
        for country, group in df[["country", "date", target]].groupby("country")
    }


def fallback_predictions(df: pd.DataFrame, target: str, split_start: str, split_end: str) -> pd.DataFrame:
    start = pd.Timestamp(split_start)
    end = pd.Timestamp(split_end)
    rows = []
    for country, group in df[["country", "date", target]].dropna().sort_values("date").groupby("country"):
        train = group.loc[group["date"] < start, target].to_numpy(dtype=float)
        test = group[(group["date"] >= start) & (group["date"] <= end)]
        if len(train) == 0 or test.empty:
            continue
        pred = np.repeat(pd.Series(train).dropna().iloc[-1], test.shape[0])
        rows.append(pd.DataFrame({
            "country": country,
            "date": pd.to_datetime(test["date"].to_numpy()),
            "y_true": test[target].to_numpy(dtype=float),
            "y_pred": pred,
            "model": "LightGBM_fallback",
        }))
    if not rows:
        return pd.DataFrame(columns=["country", "date", "y_true", "y_pred", "model"])
    return pd.concat(rows, ignore_index=True)


def _failure_code(status: str) -> str:
    if status == "success":
        return "V0"
    if "empty_train_or_test" in status:
        return "E1"
    if "empty_matrix" in status:
        return "E2"
    return "E3"


def pooled_lgbm_eval_leaksafe(df: pd.DataFrame, target: str, fao_cols: list[str], split_start: str, split_end: str, panel_name: str, split_id: str, target_label: str, logs: list[dict]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
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

    def log_and_fallback(status: str, n_train: int, n_test: int, n_features: int) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
        logs.append({
            "panel": panel_name,
            "target": target,
            "target_label": target_label,
            "split_id": split_id,
            "model": "LightGBM",
            "status": status,
            "failure_code": _failure_code(status),
            "train_rows": n_train,
            "test_rows": n_test,
            "n_features": n_features,
        })
        return fallback_predictions(df, target, split_start, split_end), make_train_dict(df, target, split_start)

    if train.shape[0] == 0 or test.shape[0] == 0:
        return log_and_fallback("fallback_empty_train_or_test", train.shape[0], test.shape[0], len(usable))

    train_x = pd.get_dummies(train[usable], columns=["country"], drop_first=False)
    test_x = pd.get_dummies(test[usable], columns=["country"], drop_first=False)
    train_x, test_x = train_x.align(test_x, join="left", axis=1, fill_value=0)
    if train_x.shape[0] == 0 or train_x.shape[1] == 0 or test_x.shape[0] == 0:
        return log_and_fallback("fallback_empty_matrix", train_x.shape[0], test_x.shape[0], train_x.shape[1] if train_x.ndim == 2 else 0)

    try:
        model = LGBMRegressor(**LIGHTGBM_PARAMS)
        model.fit(train_x, train[target])
        pred = model.predict(test_x)
        pred_df = test[["country", "date"]].copy()
        pred_df["y_true"] = test[target].to_numpy(dtype=float)
        pred_df["y_pred"] = pred
        pred_df["model"] = "LightGBM"
        logs.append({
            "panel": panel_name,
            "target": target,
            "target_label": target_label,
            "split_id": split_id,
            "model": "LightGBM",
            "status": "success",
            "failure_code": "V0",
            "train_rows": train_x.shape[0],
            "test_rows": test_x.shape[0],
            "n_features": train_x.shape[1],
        })
        return pred_df, make_train_dict(df, target, split_start)
    except Exception as exc:
        return log_and_fallback(f"fallback_exception: {exc}", train_x.shape[0], test_x.shape[0], train_x.shape[1])


def compute_metrics(pred_df: pd.DataFrame, train_source: dict[str, np.ndarray], split_id: str, panel_name: str, target_label: str) -> pd.DataFrame:
    rows = []
    for country, group in pred_df.groupby("country"):
        if country not in train_source or len(train_source[country]) == 0:
            continue
        y_true = group["y_true"].to_numpy(dtype=float)
        y_pred = group["y_pred"].to_numpy(dtype=float)
        rows.append({
            "panel": panel_name,
            "split_id": split_id,
            "target": target_label,
            "model": group["model"].iloc[0],
            "country": country,
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": math.sqrt(mean_squared_error(y_true, y_pred)),
            "smape": smape(y_true, y_pred),
            "mase": mase(y_true, y_pred, train_source[country], seasonality=12),
            "n_test": len(y_true),
        })
    return pd.DataFrame(rows)


def summarise_outputs(country_metrics: pd.DataFrame, predictions: pd.DataFrame, execution_log: pd.DataFrame) -> dict[str, pd.DataFrame]:
    summary = (
        country_metrics.groupby(["panel", "target", "split_id", "model"], as_index=False)
        .agg(mean_mae=("mae", "mean"), mean_rmse=("rmse", "mean"), mean_smape=("smape", "mean"), mean_mase=("mase", "mean"), n_countries=("country", "nunique"), n_test_total=("n_test", "sum"))
    )
    winners = summary.sort_values(["panel", "target", "split_id", "mean_mase"]).groupby(["panel", "target", "split_id"], as_index=False).first()
    win_counts = winners.groupby(["panel", "model"], as_index=False).size().rename(columns={"size": "n_wins"})
    global_means = summary.groupby(["panel", "model"], as_index=False)["mean_mase"].mean().sort_values(["panel", "mean_mase"])
    relative_rows = []
    for (panel, target, split), group in summary.groupby(["panel", "target", "split_id"]):
        values = dict(zip(group["model"], group["mean_mase"]))
        if "LightGBM" in values:
            for comparator in ["Seasonal Naive", "ETS"]:
                if comparator in values and values[comparator] != 0:
                    relative_rows.append({
                        "panel": panel,
                        "target": target,
                        "split_id": split,
                        "model": "LightGBM",
                        "comparator": comparator,
                        "percent_difference": 100 * (values["LightGBM"] / values[comparator] - 1),
                    })
    relative_perf = pd.DataFrame(relative_rows)
    return {
        "summary_by_split_target": summary,
        "block_winners": winners,
        "model_win_counts": win_counts,
        "relative_perf": relative_perf,
        "country_level_metrics": country_metrics,
        "all_predictions": predictions,
        "global_means": global_means,
        "execution_validity_log": execution_log,
    }


def run_benchmark() -> dict[str, pd.DataFrame]:
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
    metrics = []
    predictions = []
    logs = []
    for panel_name, df, targets, splits in panel_specs:
        fao_cols = [col for col in df.columns if col.startswith("fao_")]
        for target_col, target_label in targets.items():
            for split_id, split_start, split_end in splits:
                for model_name in ["Seasonal Naive", "ETS"]:
                    pred_df, train_source = country_baseline_eval(df, target_col, split_start, split_end, model_name)
                    metrics.append(compute_metrics(pred_df, train_source, split_id, panel_name, target_label))
                    pred_df = pred_df.assign(panel=panel_name, split_id=split_id, target=target_label)
                    predictions.append(pred_df)
                pred_df, train_source = pooled_lgbm_eval_leaksafe(df, target_col, fao_cols, split_start, split_end, panel_name, split_id, target_label, logs)
                metrics.append(compute_metrics(pred_df, train_source, split_id, panel_name, target_label))
                pred_df = pred_df.assign(panel=panel_name, split_id=split_id, target=target_label)
                predictions.append(pred_df)
    country_metrics = pd.concat(metrics, ignore_index=True)
    all_predictions = pd.concat(predictions, ignore_index=True)
    execution_log = pd.DataFrame(logs)
    return summarise_outputs(country_metrics, all_predictions, execution_log)


def main() -> None:
    outputs = run_benchmark()
    MAIN_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(MAIN_RESULTS_FILE, engine="xlsxwriter") as writer:
        for sheet_name, table in outputs.items():
            table.to_excel(writer, sheet_name=sheet_name, index=False)


if __name__ == "__main__":
    main()
