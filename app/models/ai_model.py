# app/models/ai_model.py

import pandas as pd
import numpy as np
from pathlib import Path
import joblib

# xgboost may need installing: pip install xgboost
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

MODEL_DIR  = Path(__file__).parent / "models"
MODEL_FILE = MODEL_DIR / "xgb_model.joblib"

class MomentumModel:
    """
    Featurizes price + news into a rich technical dataset,
    then uses XGBoost to predict next-bar direction.
    """
    def __init__(self):
        # need at least 20 bars for all indicators
        self.lookback = 20

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        if MODEL_FILE.exists():
            self.pipeline = joblib.load(MODEL_FILE)
        else:
            self.pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("clf",    XGBClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric="logloss"
                )),
            ])

    @staticmethod
    def featurize(df: pd.DataFrame, news):
        X = df.copy()
        # 1) returns
        X["ret1"]  = X["close"].pct_change(1)
        X["ret3"]  = X["close"].pct_change(3)
        # 2) moving averages
        X["ma5"]    = X["close"].rolling(5).mean()
        X["ma20"]   = X["close"].rolling(20).mean()
        X["ma_diff"]= X["ma5"] - X["ma20"]
        # 3) RSI (14)
        delta      = X["close"].diff()
        up         = delta.clip(lower=0)
        down       = -delta.clip(upper=0)
        roll_up    = up.rolling(14).mean()
        roll_down  = down.rolling(14).mean()
        rs         = roll_up / roll_down
        X["rsi"]   = 100 - (100 / (1 + rs))
        # 4) ATR (14)
        tr = pd.concat([
            (X["high"] - X["low"]),
            (X["high"] - X["close"].shift()).abs(),
            (X["low"]  - X["close"].shift()).abs()
        ], axis=1).max(axis=1)
        X["atr"]   = tr.rolling(14).mean()
        # 5) ADX (14)
        up_move    = X["high"] - X["high"].shift()
        down_move  = X["low"].shift() - X["low"]
        plus_dm    = np.where((up_move>down_move)&(up_move>0), up_move, 0.0)
        minus_dm   = np.where((down_move>up_move)&(down_move>0), down_move, 0.0)
        atr14      = X["atr"]
        plus_di    = 100 * pd.Series(plus_dm, index=X.index).rolling(14).mean() / atr14
        minus_di   = 100 * pd.Series(minus_dm, index=X.index).rolling(14).mean() / atr14
        dx         = 100 * (abs(plus_di-minus_di)/(plus_di+minus_di)).replace([np.inf, -np.inf], 0)
        X["adx"]   = dx.rolling(14).mean()
        # 6) Bollinger Band width (20,2)
        mb         = X["close"].rolling(20).mean()
        sd         = X["close"].rolling(20).std()
        X["bb_width"] = ((mb + 2*sd) - (mb - 2*sd)) / mb
        # 7) MACD (12,26,9)
        ema12      = X["close"].ewm(span=12, adjust=False).mean()
        ema26      = X["close"].ewm(span=26, adjust=False).mean()
        macd       = ema12 - ema26
        sig_line   = macd.ewm(span=9, adjust=False).mean()
        X["macd"]      = macd
        X["macd_hist"] = macd - sig_line
        # 8) On-Balance Volume
        obv        = (np.sign(X["close"].diff()) * X["volume"]).cumsum()
        X["obv"]   = obv
        # 9) news sentiment
        if isinstance(news, pd.Series):
            X["sentiment"] = news.reindex(X.index).fillna(0.0)
        else:
            X["sentiment"] = float(news)
        # drop NaNs and select
        X = X.dropna()
        feats = [
            "ret1","ret3","ma5","ma20","ma_diff",
            "rsi","atr","adx","bb_width",
            "macd","macd_hist","obv","sentiment"
        ]
        return X[feats]

    def fit(self, df: pd.DataFrame, news: pd.Series):
        df2 = df.copy()
        df2["target"] = np.where(df2["close"].shift(-1) > df2["close"], 1, 0)
        df2 = df2.dropna()

        X = self.featurize(df2, news)
        y = df2.loc[X.index, "target"]

        self.pipeline.fit(X, y)
        joblib.dump(self.pipeline, MODEL_FILE)

    def predict(self, df: pd.DataFrame, news: float):
        # build features
        feat = self.featurize(df, news)
        # not enough data?
        if feat.empty:
            return {"signal": "HOLD", "confidence": 0.0, "predicted_change": 0.0}

        row    = feat.iloc[[-1]]
        proba  = self.pipeline.predict_proba(row)[0]
        up, dn = proba[1], proba[0]
        sig    = "BUY" if up > dn else "SELL"

        return {
            "signal":           sig,
            "confidence":       max(up, dn) * 100,
            "predicted_change": (up - dn) * 100,
        }

__all__ = ["MomentumModel"]
