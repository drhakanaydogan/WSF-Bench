from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover
    LGBMRegressor = None

from baselines import last_observed_forecast
from config import LIGHTGBM_PARAMS


@dataclass
class LightGBMRunResult:
    predictions: pd.DataFrame
    status: str
    failure_code: str | None
    message: str
    n_train: int
    n_test: int
    n_features: int


def _align_design_matrices(train_x: pd.DataFrame, test_x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_encoded = pd.get_dummies(train_x, drop_first=False)
    test_encoded = pd.get_dummies(test_x, drop_first=False)
    test_encoded = test_encoded.reindex(columns=train_encoded.columns, fill_value=0)
    return train_encoded, test_encoded


def _fallback_predictions(test_df: pd.DataFrame, train_df: pd.DataFrame, target_col: str, country_col: str) -> pd.Series:
    last_values = train_df.sort_values("date").groupby(country_col)[target_col].last()
    return test_df[country_col].map(last_values)


def fit_predict_pooled_lightgbm(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    train_mask: pd.Series,
    test_mask: pd.Series,
    country_col: str = "country",
) -> LightGBMRunResult:
    train_df = df.loc[train_mask].copy()
    test_df = df.loc[test_mask].copy()

    if train_df.empty or test_df.empty:
        preds = _fallback_predictions(test_df, train_df, target_col, country_col)
        return LightGBMRunResult(test_df.assign(prediction=preds), "fallback", "E1", "empty effective train or test design", len(train_df), len(test_df), 0)

    available_features = [c for c in feature_cols if c in df.columns and not train_df[c].isna().all()]
    train_df = train_df.dropna(subset=[target_col] + available_features)
    test_df = test_df.dropna(subset=available_features)

    if train_df.empty or test_df.empty:
        preds = _fallback_predictions(test_df, df.loc[train_mask], target_col, country_col)
        return LightGBMRunResult(test_df.assign(prediction=preds), "fallback", "E1", "empty effective design after feature filtering", len(train_df), len(test_df), 0)

    try:
        train_x, test_x = _align_design_matrices(train_df[available_features], test_df[available_features])
    except Exception as exc:
        preds = _fallback_predictions(test_df, df.loc[train_mask], target_col, country_col)
        return LightGBMRunResult(test_df.assign(prediction=preds), "fallback", "E2", f"matrix alignment failure: {exc}", len(train_df), len(test_df), 0)

    if train_x.empty or test_x.empty or train_x.shape[1] == 0:
        preds = _fallback_predictions(test_df, df.loc[train_mask], target_col, country_col)
        return LightGBMRunResult(test_df.assign(prediction=preds), "fallback", "E2", "empty aligned feature matrix", len(train_df), len(test_df), train_x.shape[1] if not train_x.empty else 0)

    if LGBMRegressor is None:
        preds = _fallback_predictions(test_df, df.loc[train_mask], target_col, country_col)
        return LightGBMRunResult(test_df.assign(prediction=preds), "fallback", "E3", "LightGBM is not installed", len(train_df), len(test_df), train_x.shape[1])

    try:
        model = LGBMRegressor(**LIGHTGBM_PARAMS)
        model.fit(train_x, train_df[target_col].astype(float))
        preds = model.predict(test_x)
        return LightGBMRunResult(test_df.assign(prediction=np.asarray(preds, dtype=float)), "valid", "V0", "execution-valid pooled LightGBM run", len(train_df), len(test_df), train_x.shape[1])
    except Exception as exc:
        preds = _fallback_predictions(test_df, df.loc[train_mask], target_col, country_col)
        return LightGBMRunResult(test_df.assign(prediction=preds), "fallback", "E3", f"model fitting or prediction exception: {exc}", len(train_df), len(test_df), train_x.shape[1])
