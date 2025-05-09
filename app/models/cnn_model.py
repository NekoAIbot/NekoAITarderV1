import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dense, Dropout

MODEL_DIR  = Path(__file__).parent / "models"
MODEL_FILE = MODEL_DIR / "cnn_model.joblib"

class CNNModel:
    def __init__(self, lookback=20):
        self.lookback = lookback
        MODEL_DIR.mkdir(exist_ok=True)
        if MODEL_FILE.exists():
            self.model = joblib.load(MODEL_FILE)
        else:
            m = Sequential([
                Conv1D(64, kernel_size=3, activation="relu", input_shape=(lookback,7)),
                Dropout(0.2),
                Conv1D(32, kernel_size=3, activation="relu"),
                GlobalMaxPooling1D(),
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
        from .xgb_model import MomentumModel as Feat
        feat = Feat.featurize(df, news)
        y    = (df["close"].shift(-1) > df["close"]).astype(int).iloc[-len(feat):].values
        X_seq= self._make_sequences(feat)
        self.model.fit(X_seq, y, epochs=20, batch_size=32, validation_split=0.1, verbose=0)
        joblib.dump(self.model, MODEL_FILE)

    def predict(self, df: pd.DataFrame, news: float):
        from .xgb_model import MomentumModel as Feat
        feat = Feat.featurize(df, news)
        if len(feat) < self.lookback:
            return {"signal": "HOLD", "confidence": 0.0}
        seq = feat.values[-self.lookback:].reshape(1, self.lookback, -1)
        p   = float(self.model.predict(seq)[0])
        sig = "BUY" if p > 0.5 else "SELL"
        return {"signal": sig, "confidence": p * 100}
