# app/models/cnn_model.py

import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from .feature_builder import build_features

MODEL_DIR  = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_FILE = MODEL_DIR / "cnn_model.keras"

class CNNModel:
    def __init__(self, lookback: int = 20):
        self.lookback = lookback
        if MODEL_FILE.exists():
            # load without optimizer
            self.model = load_model(MODEL_FILE, compile=False)
        else:
            self.model = Sequential([
                Input(shape=(lookback, 13)),
                Conv1D(32, 3, activation="relu"),
                MaxPooling1D(2),
                Flatten(),
                Dense(32, activation="relu"),
                Dropout(0.2),
                Dense(1, activation="sigmoid"),
            ])

    def _prepare(self, df: pd.DataFrame, news: pd.Series):
        # reset index so df2.index is 0..n-1
        df2 = df.reset_index(drop=True).copy()
        n   = len(df2)

        # build a position-aligned news series
        vals = news.values
        if len(vals) < n:
            vals = np.concatenate([vals, np.zeros(n - len(vals))])
        else:
            vals = vals[:n]
        news_s = pd.Series(vals, index=df2.index)

        feat   = build_features(df2, news_s)
        closes = df2["close"].values

        Xs, ys = [], []
        arr    = feat.values
        for i in range(self.lookback, len(arr) - 1):
            Xs.append(arr[i-self.lookback:i])
            ys.append(int(closes[i+1] > closes[i]))

        return np.array(Xs), np.array(ys)

    def fit(self, df: pd.DataFrame, news: pd.Series):
        X, y = self._prepare(df, news)
        self.model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"],
            run_eagerly=True
        )
        es = EarlyStopping(patience=5, restore_best_weights=True)
        self.model.fit(
            X, y,
            validation_split=0.1,
            epochs=20,
            batch_size=32,
            callbacks=[es],
            verbose=1
        )
        self.model.save(MODEL_FILE)

    def predict(self, df: pd.DataFrame, news: float):
        # reset index
        df2 = df.reset_index(drop=True).copy()
        n   = len(df2)

        # position-aligned constant news
        vals = np.full(n, news, dtype=float)
        news_s = pd.Series(vals, index=df2.index)

        feat = build_features(df2, news_s)
        if len(feat) < self.lookback:
            return {"signal":"HOLD","confidence":0.0,"predicted_change":0.0}

        Xp = feat.values[-self.lookback:].reshape(1, self.lookback, -1)
        p  = float(self.model.predict(Xp)[0,0])
        sig = "BUY" if p > 0.5 else "SELL"
        return {
            "signal": sig,
            "confidence": p * 100,
            "predicted_change": (p - 0.5) * 200
        }

__all__ = ["CNNModel"]
