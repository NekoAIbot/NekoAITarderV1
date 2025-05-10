# app/models/nn_models.py

import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Flatten, Dense, LSTM, Dropout
)
from tensorflow.keras.callbacks import EarlyStopping

from .feature_builder import build_features

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

class BaseNN:
    def __init__(self, name, build_fn):
        self.name     = name
        self.build_fn = build_fn
        self.lookback = 20
        self.model    = None

    def _windowed_data(self, df: pd.DataFrame, news_s: pd.Series):
        df2 = df.reset_index(drop=True).copy()
        news = pd.Series(news_s.values, index=df2.index)
        df2["target"] = (df2["close"].shift(-1) > df2["close"]).astype(int)
        df2.dropna(inplace=True)

        feat_df = build_features(df2, news)
        arr     = feat_df.values
        Xs, ys  = [], []
        for i in range(self.lookback, len(arr)-1):
            Xs.append(arr[i-self.lookback:i])
            ys.append(int(df2["target"].iloc[i]))
        return np.array(Xs), np.array(ys)

    def fit(self, df: pd.DataFrame, news_s: pd.Series):
        X, y = self._windowed_data(df, news_s)
        m    = self.build_fn(input_shape=X.shape[1:])
        es   = EarlyStopping(patience=5, restore_best_weights=True)
        m.fit(
            X, y,
            epochs=30, batch_size=32,
            validation_split=0.1,
            callbacks=[es],
            verbose=1
        )
        # save in Keras native format
        m.save(MODEL_DIR / f"{self.name}.keras")
        self.model = m

    def predict(self, df: pd.DataFrame, news: float):
        df2 = df.tail(self.lookback).copy()
        news_s = pd.Series(news, index=df2.index)
        feat_df = build_features(df2, news_s)
        if feat_df.shape[0] < self.lookback:
            return {"signal": "HOLD", "confidence": 0.0, "predicted_change": 0.0}
        X = feat_df.values.reshape(1, self.lookback, -1)
        p = float(self.model.predict(X)[0, 0])
        sig = "BUY" if p > 0.5 else "SELL"
        return {
            "signal":           sig,
            "confidence":       max(p, 1-p) * 100,
            "predicted_change": (p - (1-p)) * 100,
        }

def build_cnn(input_shape):
    m = Sequential([
        Input(shape=input_shape),
        Conv1D(32, 3, activation="relu"),
        MaxPooling1D(2),
        Flatten(),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    m.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
        run_eagerly=True    # optional, but safe if you hit TF graph issues
    )
    return m

def build_lstm(input_shape):
    m = Sequential([
        Input(shape=input_shape),
        LSTM(64),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ])
    m.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
        run_eagerly=True    # **this is the key fix**
    )
    return m

DenseModel = lambda: BaseNN("dense", lambda input_shape: build_cnn(input_shape))
CNNModel   = lambda: BaseNN("cnn",   build_cnn)
LSTMModel  = lambda: BaseNN("lstm",  build_lstm)
