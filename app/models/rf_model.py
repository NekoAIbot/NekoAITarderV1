import numpy as np, pandas as pd, joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from .xgb_model import MomentumModel as Feat

MODEL_DIR  = Path(__file__).parent/"models"
MODEL_FILE = MODEL_DIR/"rf_model.joblib"

class RFModel:
    def __init__(self):
        self.lookback = 20
        MODEL_DIR.mkdir(exist_ok=True)
        if MODEL_FILE.exists():
            self.pipeline = joblib.load(MODEL_FILE)
        else:
            self.pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(n_estimators=200, max_depth=5,
                                               random_state=42, n_jobs=-1))
            ])

    def fit(self, df, news):
        df2 = df.copy()
        df2["target"] = (df2["close"].shift(-1)>df2["close"]).astype(int)
        df2.dropna(inplace=True)
        X = Feat.featurize(df2, news)
        y = df2.loc[X.index,"target"]
        self.pipeline.fit(X,y)
        joblib.dump(self.pipeline, MODEL_FILE)

    def predict(self, df, news):
        X = Feat.featurize(df, news)
        if X.empty:
            return {"signal":"HOLD","confidence":0.0,"predicted_change":0.0}
        p = self.pipeline.predict_proba(X.iloc[[-1]])[0]
        up, dn = p[1],p[0]
        return {"signal":"BUY" if up>dn else "SELL",
                "confidence":max(up,dn)*100,
                "predicted_change":(up-dn)*100}
