from __future__ import annotations

import numpy as np
import pandas as pd


def seasonal_naive_forecast(train: pd.Series, horizon: int, seasonal_period: int = 12) -> np.ndarray:
    train = pd.Series(train).dropna().astype(float)
    if len(train) >= seasonal_period:
        pattern = train.iloc[-seasonal_period:].to_numpy()
        return np.resize(pattern, horizon)
    if len(train) > 0:
        return np.repeat(train.iloc[-1], horizon)
    return np.repeat(np.nan, horizon)


def last_observed_forecast(train: pd.Series, horizon: int) -> np.ndarray:
    train = pd.Series(train).dropna().astype(float)
    if len(train) == 0:
        return np.repeat(np.nan, horizon)
    return np.repeat(train.iloc[-1], horizon)


def ets_forecast(train: pd.Series, horizon: int, seasonal_period: int = 12) -> np.ndarray:
    train = pd.Series(train).dropna().astype(float)
    if len(train) < seasonal_period * 2:
        return last_observed_forecast(train, horizon)
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        model = ExponentialSmoothing(
            train,
            trend="add",
            damped_trend=True,
            seasonal="add",
            seasonal_periods=seasonal_period,
            initialization_method="estimated",
        )
        fit = model.fit(optimized=True)
        return fit.forecast(horizon).to_numpy()
    except Exception:
        return last_observed_forecast(train, horizon)
