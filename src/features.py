from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_LAGS = (1, 3, 6, 12)


def add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out["year"] = out[date_col].dt.year
    out["month"] = out[date_col].dt.month
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    return out


def add_lag_features(
    df: pd.DataFrame,
    target_col: str,
    country_col: str = "country",
    date_col: str = "date",
) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out = out.sort_values([country_col, date_col])
    grouped = out.groupby(country_col, group_keys=False)[target_col]
    for lag in TARGET_LAGS:
        out[f"{target_col}_lag{lag}"] = grouped.shift(lag)
    out[f"{target_col}_diff_lag1"] = grouped.shift(1) - grouped.shift(2)
    out[f"{target_col}_diff_lag12"] = grouped.shift(1) - grouped.shift(13)
    return out


def add_lagged_annual_anchors(
    df: pd.DataFrame,
    anchor_cols: list[str],
    country_col: str = "country",
    year_col: str = "year",
) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values([country_col, year_col])
    for col in anchor_cols:
        out[f"{col}_annual_lag1"] = out.groupby(country_col)[col].shift(12)
    return out


def build_feature_frame(
    df: pd.DataFrame,
    target_col: str,
    country_col: str = "country",
    date_col: str = "date",
    annual_anchor_cols: list[str] | None = None,
) -> pd.DataFrame:
    annual_anchor_cols = annual_anchor_cols or []
    out = add_calendar_features(df, date_col=date_col)
    out = add_lag_features(out, target_col=target_col, country_col=country_col, date_col=date_col)
    if annual_anchor_cols:
        out = add_lagged_annual_anchors(out, annual_anchor_cols, country_col=country_col, year_col="year")
    return out
