# File: app/models/cnn_model.py

import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    GlobalAveragePooling1D,
    Dropout,
    Dense,
)
from tensorflow.keras.callbacks import EarlyStopping
from .feature_builder import build_features

MODEL_DIR  = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_FILE = MODEL_DIR / "cnn_model.keras"

class CNNModel:
    def __init__(self, lookback: int = 20, filters: int = 32, kernel_size: int = 3):
        self.lookback = lookback
        self.filters  = filters
        self.kernel_size = kernel_size

        # If there's already a saved model *and* its lookback matches, load it.
        if MODEL_FILE.exists():
            try:
                # we'll check input shape to avoid mismatch
                m = load_model(MODEL_FILE, compile=False)
                in_shape = m.input_shape  # (None, lookback, features)
                if in_shape[1] == lookback:
                    self.model = m
                    return
            except Exception:
                pass

        # Otherwise (or on mismatch) build fresh:
        self._build_model()

    def _build_model(self):
        self.model = Sequential([
            Input(shape=(self.lookback, 13)),
            Conv1D(self.filters, self.kernel_size, activation="relu", padding="same"),
            Dropout(0.2),
            # this replaces Flatten() so we no longer depend on lookback * filters
            GlobalAveragePooling1D(),
            Dense(32, activation="relu"),
            Dropout(0.2),
            Dense(1, activation="sigmoid"),
        ])

    def _prepare(self, df: pd.DataFrame, news: pd.Series):
        # align news by timestamp
        news_s = news.reindex(df.index).fillna(0.0)
        df2 = df.reset_index(drop=True)
        n2  = news_s.reset_index(drop=True)

        feat = build_features(df2, n2)
        Xs, ys = [], []
        closes = df2["close"].values
        arr     = feat.values
        for i in range(self.lookback, len(arr)-1):
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
            batch_size=getattr(self, "batch_size", 32),
            callbacks=[es],
            verbose=1
        )
        self.model.save(MODEL_FILE)

    def predict(self, df: pd.DataFrame, news: float):
        idx = df.index[-self.lookback:]
        news_s = pd.Series(news, index=idx).reindex(df.index).fillna(0.0)
        df2 = df.reset_index(drop=True)
        n2  = news_s.reset_index(drop=True).iloc[-self.lookback:]
        feat = build_features(df2, n2.to_frame(name=0).iloc[:,0])
        if feat.shape[0] < self.lookback:
            return {"signal":"HOLD","confidence":0.0,"predicted_change":0.0}
        Xp = feat.values[-self.lookback:].reshape(1, self.lookback, -1)
        p  = float(self.model.predict(Xp)[0,0])
        sig = "BUY" if p>0.5 else "SELL"
        return {
            "signal": sig,
            "confidence": p*100,
            "predicted_change": (p-0.5)*200
        }

__all__ = ["CNNModel"]
