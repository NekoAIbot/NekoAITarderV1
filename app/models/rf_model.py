# app/models/rf_model.py

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from .feature_builder import build_features

MODEL_DIR  = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_FILE = MODEL_DIR / "rf_model.joblib"

class RFModel:
    def __init__(self):
        self.lookback = 20
        if MODEL_FILE.exists():
            self.pipeline = joblib.load(MODEL_FILE)
        else:
            self.pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("clf",    RandomForestClassifier(
                    n_estimators=200,
                    max_depth=5,
                    random_state=42,
                    n_jobs=-1
                )),
            ])

    def fit(self, df: pd.DataFrame, news: pd.Series):
        df2 = df.reset_index(drop=True).copy()
        news_s = pd.Series(news.values, index=df2.index)
        df2["target"] = (df2["close"].shift(-1) > df2["close"]).astype(int)
        df2.dropna(inplace=True)
        X = build_features(df2, news_s)
        y = df2.loc[X.index, "target"]
        self.pipeline.fit(X, y)
        joblib.dump(self.pipeline, MODEL_FILE)

    def predict(self, df: pd.DataFrame, news: float):
        df2 = df.reset_index(drop=True).copy()
        news_s = pd.Series(news, index=df2.index)
        feats = build_features(df2, news_s)
        if feats.empty:
            return {"signal":"HOLD","confidence":0.0,"predicted_change":0.0}
        p = self.pipeline.predict_proba(feats.iloc[[-1]])[0]
        up, dn = p[1], p[0]
        sig = "BUY" if up>dn else "SELL"
        return {"signal":sig, "confidence":max(up,dn)*100, "predicted_change":(up-dn)*100}

__all__ = ["RFModel"]
