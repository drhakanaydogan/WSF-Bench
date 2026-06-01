from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.nanmean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.nanmean((y_true - y_pred) ** 2)))


def smape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    valid = denom > 0
    if not valid.any():
        return np.nan
    return float(np.nanmean(np.abs(y_true[valid] - y_pred[valid]) / denom[valid]) * 100.0)


def mase(y_true, y_pred, train_series, seasonal_period: int = 12) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    train_series = pd.Series(train_series).dropna().astype(float).to_numpy()
    if len(train_series) <= seasonal_period:
        return np.nan
    scale = np.nanmean(np.abs(train_series[seasonal_period:] - train_series[:-seasonal_period]))
    if not np.isfinite(scale) or scale == 0:
        return np.nan
    return float(np.nanmean(np.abs(y_true - y_pred)) / scale)


def average_country_metrics(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    metric_cols = ["MAE", "RMSE", "sMAPE", "MASE"]
    existing = [c for c in metric_cols if c in df.columns]
    return df.groupby(group_cols, dropna=False)[existing].mean().reset_index()
