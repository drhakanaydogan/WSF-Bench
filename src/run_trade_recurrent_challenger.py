from __future__ import annotations

import math
import random

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from config import LIGHTGBM_PARAMS, PROTOCOL_FILE, TRADE_RECURRENT_RESULTS_FILE, TRADE_SPLITS, TRADE_TARGETS
from run_leaksafe_benchmark import (
    add_leaksafe_lgbm_features,
    compute_metrics,
    country_baseline_eval,
    fallback_predictions,
    make_train_dict,
    mase,
    pooled_lgbm_eval_leaksafe,
    summarise_outputs,
)


def build_lstm_sequences(df: pd.DataFrame, target: str, lookback: int, split_start: str, split_end: str):
    start = pd.Timestamp(split_start)
    end = pd.Timestamp(split_end)
    seq_rows = []
    train_cache = {}
    for country, group in df[["country", "date", target]].dropna().sort_values("date").groupby("country"):
        y = group[target].to_numpy(dtype=float)
        dates = group["date"].to_numpy()
        train_cache[country] = group.loc[group["date"] < start, target].to_numpy(dtype=float)
        for i in range(lookback, len(group)):
            seq_rows.append({"country": country, "date": pd.Timestamp(dates[i]), "y_true": y[i], "seq": y[i - lookback:i]})
    seq_df = pd.DataFrame(seq_rows)
    if seq_df.empty:
        return pd.DataFrame(), pd.DataFrame(), train_cache
    return seq_df[seq_df["date"] < start].copy(), seq_df[(seq_df["date"] >= start) & (seq_df["date"] <= end)].copy(), train_cache


def lstm_eval(df: pd.DataFrame, target: str, split_start: str, split_end: str, lookback: int = 12, seed: int = 42):
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.models import Sequential

    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    train_df, test_df, train_cache = build_lstm_sequences(df, target, lookback, split_start, split_end)
    if train_df.empty or test_df.empty:
        return pd.DataFrame(columns=["country", "date", "y_true", "y_pred", "model"]), train_cache

    x_train = np.stack(train_df["seq"].to_numpy()).astype(float)
    y_train = train_df["y_true"].to_numpy(dtype=float)
    x_test = np.stack(test_df["seq"].to_numpy()).astype(float)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train).reshape(x_train.shape[0], lookback, 1)
    x_test_scaled = scaler.transform(x_test).reshape(x_test.shape[0], lookback, 1)

    model = Sequential([
        LSTM(32, input_shape=(lookback, 1)),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        x_train_scaled,
        y_train,
        epochs=40,
        batch_size=32,
        validation_split=0.2,
        verbose=0,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
    )
    pred = model.predict(x_test_scaled, verbose=0).ravel()
    pred_df = test_df[["country", "date", "y_true"]].copy()
    pred_df["y_pred"] = pred
    pred_df["model"] = "LSTM"
    return pred_df, train_cache


def run_trade_recurrent_challenger() -> dict[str, pd.DataFrame]:
    trade = pd.read_excel(PROTOCOL_FILE, sheet_name="trade_panel_8c")
    trade["date"] = pd.to_datetime(trade["date"])
    trade["year"] = trade["date"].dt.year
    trade["month"] = trade["date"].dt.month
    fao_cols = [col for col in trade.columns if col.startswith("fao_")]
    metrics = []
    predictions = []
    logs = []
    for target_col, target_label in TRADE_TARGETS.items():
        for split_id, split_start, split_end in TRADE_SPLITS:
            for model_name in ["Seasonal Naive", "ETS"]:
                pred_df, train_source = country_baseline_eval(trade, target_col, split_start, split_end, model_name)
                metrics.append(compute_metrics(pred_df, train_source, split_id, "Trade_8c", target_label))
                predictions.append(pred_df.assign(panel="Trade_8c", split_id=split_id, target=target_label))
            pred_df, train_source = pooled_lgbm_eval_leaksafe(trade, target_col, fao_cols, split_start, split_end, "Trade_8c", split_id, target_label, logs)
            metrics.append(compute_metrics(pred_df, train_source, split_id, "Trade_8c", target_label))
            predictions.append(pred_df.assign(panel="Trade_8c", split_id=split_id, target=target_label))
            pred_df, train_source = lstm_eval(trade, target_col, split_start, split_end, lookback=12, seed=42)
            metrics.append(compute_metrics(pred_df, train_source, split_id, "Trade_8c", target_label))
            predictions.append(pred_df.assign(panel="Trade_8c", split_id=split_id, target=target_label))
    country_metrics = pd.concat(metrics, ignore_index=True)
    all_predictions = pd.concat(predictions, ignore_index=True)
    execution_log = pd.DataFrame(logs)
    return summarise_outputs(country_metrics, all_predictions, execution_log)


def main() -> None:
    outputs = run_trade_recurrent_challenger()
    with pd.ExcelWriter(TRADE_RECURRENT_RESULTS_FILE, engine="xlsxwriter") as writer:
        for sheet_name, table in outputs.items():
            table.to_excel(writer, sheet_name=sheet_name, index=False)


if __name__ == "__main__":
    main()
