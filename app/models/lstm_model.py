# app/models/lstm_model.py

import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from .feature_builder import build_features

MODEL_DIR  = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_FILE = MODEL_DIR / "lstm_model.h5"

class LSTMModel:
    def __init__(self, lookback: int = 20):
        self.lookback = lookback
        if MODEL_FILE.exists():
            self.model = load_model(MODEL_FILE)
        else:
            self.model = Sequential([
                Input(shape=(lookback, 13)),
                LSTM(64, return_sequences=False),
                Dropout(0.2),
                Dense(32, activation="relu"),
                Dropout(0.2),
                Dense(1, activation="sigmoid"),
            ])
            self.model.compile("adam", "binary_crossentropy", metrics=["accuracy"])

    def _prepare(self, df: pd.DataFrame, news: pd.Series):
        df2 = df.reset_index(drop=True).copy()
        news_s = pd.Series(news.values, index=df2.index)
        feat = build_features(df2, news_s)
        Xs, ys = [], []
        closes = df2["close"].values
        arr     = feat.values
        for i in range(self.lookback, len(arr)-1):
            Xs.append(arr[i-self.lookback:i])
            ys.append(int(closes[i+1] > closes[i]))
        return np.array(Xs), np.array(ys)

    def fit(self, df: pd.DataFrame, news: pd.Series):
        X, y = self._prepare(df, news)
        es   = EarlyStopping(patience=5, restore_best_weights=True)
        self.model.fit(X, y,
                       validation_split=0.1,
                       epochs=20,
                       batch_size=32,
                       callbacks=[es],
                       verbose=1)
        self.model.save(MODEL_FILE)

    def predict(self, df: pd.DataFrame, news: float):
        # build a news series of constant value across last lookback
        df2 = df.reset_index(drop=True).copy()
        idx = df2.index[-self.lookback:]
        news_s = pd.Series([news]*self.lookback, index=idx)
        feat = build_features(df2, news_s)
        if len(feat) < self.lookback:
            return {"signal":"HOLD","confidence":0.0,"predicted_change":0.0}
        Xp = feat.values[-self.lookback:].reshape(1, self.lookback, -1)
        p  = float(self.model.predict(Xp)[0,0])
        sig = "BUY" if p>0.5 else "SELL"
        return {"signal":sig, "confidence":p*100, "predicted_change":(p-0.5)*200}

__all__ = ["LSTMModel"]
