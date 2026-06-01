from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def make_univariate_sequences(values: pd.Series, sequence_length: int = 12):
    arr = pd.Series(values).dropna().astype(float).to_numpy()
    x, y = [], []
    for i in range(sequence_length, len(arr)):
        x.append(arr[i - sequence_length:i])
        y.append(arr[i])
    if not x:
        return np.empty((0, sequence_length, 1)), np.empty((0,))
    return np.asarray(x, dtype=float)[..., None], np.asarray(y, dtype=float)


def fit_predict_lstm(train_values, test_values, sequence_length: int = 12, seed: int = 42):
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import LSTM, Dropout, Dense
    from tensorflow.keras.callbacks import EarlyStopping

    np.random.seed(seed)
    tf.random.set_seed(seed)

    combined = pd.concat([pd.Series(train_values), pd.Series(test_values)], ignore_index=True)
    train_x, train_y = make_univariate_sequences(pd.Series(train_values), sequence_length)
    if train_x.shape[0] == 0:
        return np.repeat(np.nan, len(test_values))

    scaler = StandardScaler()
    train_x_2d = train_x.reshape(train_x.shape[0], sequence_length)
    train_x_scaled = scaler.fit_transform(train_x_2d).reshape(train_x.shape)

    model = Sequential([
        LSTM(32, input_shape=(sequence_length, 1)),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        train_x_scaled,
        train_y,
        epochs=40,
        batch_size=32,
        validation_split=0.2,
        verbose=0,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
    )

    preds = []
    history = pd.Series(train_values).dropna().astype(float).tolist()
    for value in pd.Series(test_values).astype(float).tolist():
        window = np.asarray(history[-sequence_length:], dtype=float).reshape(1, sequence_length)
        window_scaled = scaler.transform(window).reshape(1, sequence_length, 1)
        pred = float(model.predict(window_scaled, verbose=0)[0, 0])
        preds.append(pred)
        history.append(value)
    return np.asarray(preds, dtype=float)
