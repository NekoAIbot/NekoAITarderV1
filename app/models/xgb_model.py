import numpy as np, pandas as pd, joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

MODEL_DIR  = Path(__file__).parent/"models"
MODEL_FILE = MODEL_DIR/"xgb_model.joblib"

class MomentumModel:
    def __init__(self):
        self.lookback = 20
        MODEL_DIR.mkdir(exist_ok=True)
        if MODEL_FILE.exists():
            self.pipeline = joblib.load(MODEL_FILE)
        else:
            self.pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", XGBClassifier(n_estimators=300,
                                      max_depth=6,
                                      learning_rate=0.05,
                                      subsample=0.8,
                                      colsample_bytree=0.8,
                                      random_state=42,
                                      use_label_encoder=False,
                                      eval_metric="logloss"))
            ])

    @staticmethod
    def featurize(df: pd.DataFrame, news):
        X = df.copy()
        X["ret1"] = X["close"].pct_change(1)
        X["ret3"] = X["close"].pct_change(3)
        X["ma5"]  = X["close"].rolling(5).mean()
        X["ma20"] = X["close"].rolling(20).mean()
        X["ma_diff"] = X["ma5"] - X["ma20"]
        delta = X["close"].diff()
        up, down = delta.clip(lower=0), -delta.clip(upper=0)
        X["rsi"] = 100 - (100/(1+(up.rolling(14).mean()/down.rolling(14).mean())))
        tr = pd.concat([X["high"]-X["low"],
                        (X["high"]-X["close"].shift()).abs(),
                        (X["low"]-X["close"].shift()).abs()],axis=1).max(axis=1)
        X["atr"] = tr.rolling(14).mean()
        upm = X["high"]-X["high"].shift();   downm = X["low"].shift()-X["low"]
        plus_dm = np.where((upm>downm)&(upm>0),upm,0.0)
        minus_dm= np.where((downm>upm)&(downm>0),downm,0.0)
        plus_di = 100*pd.Series(plus_dm,index=X.index).rolling(14).mean()/X["atr"]
        minus_di= 100*pd.Series(minus_dm,index=X.index).rolling(14).mean()/X["atr"]
        dx = 100*(abs(plus_di-minus_di)/(plus_di+minus_di)).replace([np.inf,-np.inf],0)
        X["adx"] = dx.rolling(14).mean()
        mb = X["close"].rolling(20).mean(); sd = X["close"].rolling(20).std()
        X["bb_width"] = ((mb+2*sd)-(mb-2*sd))/mb
        ema12 = X["close"].ewm(span=12,adjust=False).mean()
        ema26 = X["close"].ewm(span=26,adjust=False).mean()
        macd = ema12-ema26; sig = macd.ewm(span=9,adjust=False).mean()
        X["macd"] = macd; X["macd_hist"] = macd-sig
        X["obv"] = (np.sign(X["close"].diff())*X.get("volume",0)).cumsum()
        if isinstance(news,pd.Series):
            vals = news.values; X["sentiment"] = vals[-len(X):]
        else:
            X["sentiment"] = float(news)
        return X.dropna()[[
            "ret1","ret3","ma5","ma20","ma_diff",
            "rsi","atr","adx","bb_width",
            "macd","macd_hist","obv","sentiment"
        ]]

    def fit(self, df, news):
        df2 = df.copy()
        df2["target"] = (df2["close"].shift(-1)>df2["close"]).astype(int)
        df2.dropna(inplace=True)
        X = self.featurize(df2, news)
        y = df2.loc[X.index,"target"]
        self.pipeline.fit(X,y)
        joblib.dump(self.pipeline, MODEL_FILE)

    def predict(self, df, news):
        feat = self.featurize(df, news)
        if feat.empty:
            return {"signal":"HOLD","confidence":0.0,"predicted_change":0.0}
        p = self.pipeline.predict_proba(feat.iloc[[-1]])[0]
        up,dn = p[1],p[0]
        return {"signal":"BUY" if up>dn else "SELL",
                "confidence":max(up,dn)*100,
                "predicted_change":(up-dn)*100}
