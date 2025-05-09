import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

MODEL_DIR  = Path(__file__).parent / "models"
MODEL_FILE = MODEL_DIR / "lstm_model.joblib"

class LSTMModel:
    def __init__(self, lookback=20):
        self.lookback = lookback
        MODEL_DIR.mkdir(exist_ok=True)
        if MODEL_FILE.exists():
            self.model = joblib.load(MODEL_FILE)
        else:
            # build a simple stacked LSTM
            m = Sequential([
                LSTM(64, input_shape=(lookback, 7), return_sequences=True),
                Dropout(0.2),
                LSTM(32),
                Dropout(0.2),
                Dense(1, activation="sigmoid")
            ])
            m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
            self.model = m

    def _make_sequences(self, X: pd.DataFrame):
        arr = X.values
        seqs = []
        for i in range(self.lookback, len(arr)):
            seqs.append(arr[i-self.lookback : i])
        return np.stack(seqs)

    def fit(self, df: pd.DataFrame, news: pd.Series):
        # featurize exactly as XGBModel does (ret1,ret3,ma5,ma20,...,sentiment)
        from .xgb_model import MomentumModel as Feat
        feat = Feat.featurize(df, news)
        labels = (df["close"].shift(-1) > df["close"]).astype(int).iloc[-len(feat):].values
        X_seq = self._make_sequences(feat)
        es = EarlyStopping(patience=3, restore_best_weights=True)
        self.model.fit(X_seq, labels, epochs=20, batch_size=32, validation_split=0.1, callbacks=[es], verbose=0)
        joblib.dump(self.model, MODEL_FILE)

    def predict(self, df: pd.DataFrame, news: float):
        from .xgb_model import MomentumModel as Feat
        feat = Feat.featurize(df, news)
        if len(feat) < self.lookback:
            return {"signal": "HOLD", "confidence": 0.0}
        seq = feat.values[-self.lookback:].reshape(1, self.lookback, -1)
        p = float(self.model.predict(seq)[0])
        sig = "BUY" if p > 0.5 else "SELL"
        return {"signal": sig, "confidence": p * 100}
