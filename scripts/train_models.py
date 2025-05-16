#!/usr/bin/env python3
import os, warnings
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
import joblib

# suppress TF/XGB logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", ".*use_label_encoder.*")

# project imports
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config                import FOREX_MAJORS, CRYPTO_ASSETS
from app.market_data       import fetch_market_data
from app.news              import get_news_sentiment
from app.backtester        import backtest_symbol
from app.models.xgb_model  import MomentumModel as XGBModel
from app.models.lstm_model import LSTMModel
from app.models.cnn_model  import CNNModel

SYMBOLS = FOREX_MAJORS + CRYPTO_ASSETS

# ──────────────────────────────────────────────────────────────────────────────
# 1) FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────────────────
class TechnicalsTransformer(BaseEstimator, TransformerMixin):
    """Compute SMA, RSI, ATR and drop raw OHLC."""
    def fit(self, X, y=None): return self
    def transform(self, X):
        df = X.copy()
        df["sma5"]  = df["close"].rolling(5).mean()
        df["sma20"] = df["close"].rolling(20).mean()
        delta = df["close"].diff()
        up, dn = delta.clip(lower=0), -delta.clip(upper=0)
        rs = up.ewm(14).mean()/(dn.ewm(14).mean()+1e-8)
        df["rsi"] = 100 - 100/(1+rs)
        tr = pd.concat([
            df["high"]-df["low"],
            (df["high"]-df["close"].shift()).abs(),
            (df["low"]-df["close"].shift()).abs()
        ], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()
        return df[["sma5","sma20","rsi","atr"]].bfill().ffill()

class SentimentTransformer(BaseEstimator, TransformerMixin):
    """Fetch single headline-score per bar."""
    def fit(self, X, y=None): return self
    def transform(self, X):
        # X is DataFrame with index and a "symbol" column
        scores = []
        for ts, sym in zip(X.index, X["symbol"]):
            try:
                s = get_news_sentiment(sym)
            except:
                s = 0.0
            scores.append(s)
        return pd.DataFrame({"news": scores}, index=X.index)

def build_feature_pipeline():
    return FeatureUnion([
        ("tech", Pipeline([
            ("select", ColumnSelector(["open","high","low","close"])),
            ("tech", TechnicalsTransformer()),
            ("scale", StandardScaler())
        ])),
        ("sent", Pipeline([
            ("select", ColumnSelector(["symbol"])),
            ("news", SentimentTransformer()),
            ("scale", StandardScaler())
        ]))
    ])

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, cols): self.cols = cols
    def fit(self, X, y=None): return self
    def transform(self, X): return X[self.cols]

# ──────────────────────────────────────────────────────────────────────────────
# 2) DATA LOADING & SPLIT
# ──────────────────────────────────────────────────────────────────────────────
def load_all_data():
    frames, news = [], []
    for sym in SYMBOLS:
        df = fetch_market_data(sym).assign(symbol=sym)
        frames.append(df)
        # news placeholder, merged later via transformer
    return pd.concat(frames).sort_index()

def time_series_cv(df, n_splits=3, train_size=0.6, test_size=0.2):
    """Yield (train_idx, test_idx) for rolling CV."""
    N = len(df)
    step = int((N - train_size*N - test_size*N)/(n_splits-1))
    for i in range(n_splits):
        start = int(i*step)
        train_end = start + int(train_size*N)
        test_end  = train_end + int(test_size*N)
        yield (df.index[start:train_end], df.index[train_end:test_end])

# ──────────────────────────────────────────────────────────────────────────────
# 3) TRAIN / BACKTEST
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_model(name, model, df, feature_pipe, min_confidence, fee):
    records = []
    for fold, (tr_idx, te_idx) in enumerate(time_series_cv(df)):
        Xtr, Xte = df.loc[tr_idx], df.loc[te_idx]
        ytr = (Xtr["close"].shift(-1) > Xtr["close"]).astype(int)
        # train feature pipeline + model
        Xt_tr = feature_pipe.fit_transform(Xtr)
        model.lookback = 20
        if hasattr(model, "pipeline"):
            # XGBModel expects full df,news inputs
            model.fit(Xtr, None)
        else:
            model.batch_size = 32
            model.fit(Xtr, None)
        # backtest on te_idx
        rets = []
        for sym in Xte["symbol"].unique():
            pf = backtest_symbol(
                symbol         = sym,
                model          = model,
                fee_per_trade  = fee,
                min_confidence = min_confidence
            )
            rets.extend(pf.tolist())
        rets = np.array(rets)
        if len(rets)<2:
            metrics = {"fold":fold, "n_trades":len(rets), "pl":rets.sum(),
                       "sharpe":np.nan, "sortino":np.nan}
        else:
            sharpe  = rets.mean()/(rets.std(ddof=1)+1e-8)
            downs   = rets[rets<0]
            sortino = rets.mean()/(downs.std(ddof=1)+1e-8 if len(downs)>1 else 1e-8)
            metrics = {"fold":fold, "n_trades":len(rets),
                       "pl":rets.sum(), "sharpe":sharpe, "sortino":sortino}
        records.append(metrics)
    dfm = pd.DataFrame(records).assign(model=name)
    return dfm

if __name__=="__main__":
    df = load_all_data()
    feature_pipe = build_feature_pipeline()
    results = []

    # instantiate models
    configs = {
        "XGBoost": (XGBModel(),   {"min_confidence":0.5, "fee":0.0001}),
        "LSTM":    (LSTMModel(),  {"min_confidence":0.4, "fee":0.0001}),
        "CNN":     (CNNModel(),   {"min_confidence":0.4, "fee":0.0001}),
    }

    for name, (model, params) in configs.items():
        print(f"\n>>> Evaluating {name}")
        dfm = evaluate_model(
            name=name,
            model=model,
            df=df,
            feature_pipe=feature_pipe,
            min_confidence=params["min_confidence"],
            fee=params["fee"]
        )
        results.append(dfm)
        # save model
        joblib.dump(model, ROOT/f"models/{name}_final.pkl")

    # compile & report
    report = pd.concat(results, ignore_index=True)
    report.to_csv(ROOT/"backtest_report.csv", index=False)
    summary = (report.groupby("model")
                     .agg(trades=("n_trades","sum"),
                          avg_pl=("pl","mean"),
                          avg_sharpe=("sharpe","mean"),
                          avg_sortino=("sortino","mean")))
    print("\n=== AGGREGATED RESULTS ===")
    print(summary)
    summary.to_csv(ROOT/"backtest_summary.csv")
    print("\nFull report written to backtest_report.csv/backtest_summary.csv")
