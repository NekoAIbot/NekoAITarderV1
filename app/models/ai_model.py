import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

MODEL_DIR = Path(__file__).parent / "models"
MODEL_FILE = MODEL_DIR / "rf_model.joblib"

class AIModel:
    """
    Trains a RandomForest to predict next-bar direction.
    """
    def __init__(self):
        if MODEL_FILE.exists():
            # load existing model
            self.pipeline = joblib.load(MODEL_FILE)
        else:
            # fresh pipeline: scaler + RF
            self.pipeline = Pipeline([
                ("scaler",   StandardScaler()),
                ("clf",      RandomForestClassifier(
                    n_estimators=200,
                    max_depth=5,
                    random_state=42,
                    n_jobs=-1
                )),
            ])

    @staticmethod
    def featurize(df: pd.DataFrame, news: float) -> pd.DataFrame:
        """Compute technical indicators + news sentiment column."""
        X = df.copy()
        # 1) returns
        X["ret1"]  = X["close"].pct_change(1)
        X["ret3"]  = X["close"].pct_change(3)
        # 2) moving averages
        X["ma5"]   = X["close"].rolling(5).mean()
        X["ma20"]  = X["close"].rolling(20).mean()
        # 3) RSI (14)
        delta = X["close"].diff()
        up    = delta.clip(lower=0)
        down  = -delta.clip(upper=0)
        roll_up   = up.rolling(14).mean()
        roll_down = down.rolling(14).mean()
        rs = roll_up / roll_down
        X["rsi"] = 100 - (100 / (1 + rs))
        # 4) ATR (14)
        X["tr"] = np.maximum.reduce([
            X["high"] - X["low"],
            (X["high"] - X["close"].shift()).abs(),
            (X["low"]  - X["close"].shift()).abs(),
        ])
        X["atr"] = X["tr"].rolling(14).mean()
        # 5) news sentiment (constant per sample)
        X["sentiment"] = news
        # drop NaNs
        X = X.dropna()
        # select features
        feats = ["ret1","ret3","ma5","ma20","rsi","atr","sentiment"]
        return X[feats]

    def fit(self, df: pd.DataFrame, news_series: pd.Series):
        """
        Train on historical DF.
        `news_series` must align index to df.
        """
        # build label: next-bar up/down
        df2 = df.copy()
        df2["target"] = np.sign(df2["close"].shift(-1) - df2["close"])
        df2 = df2.dropna()
        # featurize
        # we’ll assume `news_series` has same index
        X = self.featurize(df2, news_series.loc[df2.index])
        y = df2.loc[X.index, "target"].apply(lambda x: 1 if x>0 else 0)
        # fit pipeline
        self.pipeline.fit(X, y)
        # persist
        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump(self.pipeline, MODEL_FILE)

    def predict(self, df: pd.DataFrame, news: float) -> dict:
        """
        Given the last 100 rows + a single news sentiment value,
        return signal, prob, and predicted_change.
        """
        # featurize last sample
        feat = self.featurize(df, news).iloc[[-1]]
        probs = self.pipeline.predict_proba(feat)[0]
        # class 1 = up, class 0 = down
        up_prob = float(probs[1])
        down_prob = float(probs[0])
        sig = "BUY" if up_prob > down_prob else "SELL"
        # predicted_change: use model probability differential
        pred_change = (up_prob - down_prob) * 100
        conf = max(up_prob, down_prob) * 100
        return {
            "signal":           sig,
            "confidence":       conf,
            "predicted_change": pred_change,
        }
