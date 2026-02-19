#!/usr/bin/env python3
import os
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# PROJECT IMPORTS
# ------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import FOREX_MAJORS, CRYPTO_ASSETS
from app.market_data import fetch_market_data
from app.models.xgb_model import MomentumModel as XGBModel
from app.models.lstm_model import LSTMModel
from app.models.cnn_model import CNNModel
from app.news import fetch_news  # MUST return dataframe with columns: ["timestamp","symbol","sentiment"]

SYMBOLS = FOREX_MAJORS + CRYPTO_ASSETS

LOOKAHEAD = 1
ATR_PERIOD = 14
MIN_CONFIDENCE = 0.55
FEE = 0.0001
RISK_PER_TRADE = 0.01
SENTIMENT_LOOKBACK_HOURS = 24


# ------------------------------------------------------------------
# 1️⃣ TECHNICAL FEATURE ENGINEER
# ------------------------------------------------------------------

class TechnicalFeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        df = df.copy()

        df["sma5"] = df["close"].rolling(5).mean()
        df["sma20"] = df["close"].rolling(20).mean()

        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        rs = gain.rolling(14).mean() / (loss.rolling(14).mean() + 1e-9)
        df["rsi"] = 100 - (100 / (1 + rs))

        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs()
        ], axis=1).max(axis=1)

        df["atr"] = tr.rolling(ATR_PERIOD).mean()
        df["returns"] = df["close"].pct_change()

        df = df.dropna()

        return df


# ------------------------------------------------------------------
# 2️⃣ LEAK-FREE SENTIMENT ENGINE (BUILT-IN FILTER)
# ------------------------------------------------------------------

class SentimentEngine:

    def __init__(self, symbols):
        self.symbols = symbols
        self.news_cache = {}
        self._load_news()

    def _load_news(self):
        """
        Preload historical news per symbol.
        fetch_news() must return:
        DataFrame with columns:
        ["timestamp","symbol","sentiment"]
        """
        news_df = fetch_news()

        news_df["timestamp"] = pd.to_datetime(news_df["timestamp"])
        news_df = news_df.sort_values("timestamp")

        for sym in self.symbols:
            self.news_cache[sym] = news_df[news_df["symbol"] == sym].copy()

    def get_sentiment(self, symbol, current_time):
        """
        Only uses news BEFORE the candle timestamp.
        """
        if symbol not in self.news_cache:
            return 0.0

        news_df = self.news_cache[symbol]

        start_time = current_time - timedelta(hours=SENTIMENT_LOOKBACK_HOURS)

        relevant_news = news_df[
            (news_df["timestamp"] < current_time) &
            (news_df["timestamp"] >= start_time)
        ]

        if len(relevant_news) == 0:
            return 0.0

        return relevant_news["sentiment"].mean()


def attach_sentiment(df, sentiment_engine):

    sentiments = []

    for idx, row in df.iterrows():
        score = sentiment_engine.get_sentiment(
            symbol=row["symbol"],
            current_time=idx
        )
        sentiments.append(score)

    df["sentiment"] = sentiments
    return df


# ------------------------------------------------------------------
# 3️⃣ TARGET CREATION
# ------------------------------------------------------------------

def create_target(df):
    df["target"] = (df["close"].shift(-LOOKAHEAD) > df["close"]).astype(int)
    df = df.dropna()
    return df


# ------------------------------------------------------------------
# 4️⃣ DATA LOADER
# ------------------------------------------------------------------

def load_all_data():
    frames = []
    for sym in SYMBOLS:
        df = fetch_market_data(sym)
        df["symbol"] = sym
        frames.append(df)

    df = pd.concat(frames).sort_index()
    return df


# ------------------------------------------------------------------
# 5️⃣ WALK-FORWARD VALIDATION
# ------------------------------------------------------------------

def walk_forward_validation(df, model):

    tscv = TimeSeriesSplit(n_splits=5)
    results = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):

        train = df.iloc[train_idx]
        test = df.iloc[test_idx]

        X_train = train.drop(columns=["target"])
        y_train = train["target"]

        X_test = test.drop(columns=["target"])
        y_test = test["target"]

        model.fit(X_train, y_train)

        probs = model.predict_proba(X_test)[:, 1]
        signals = probs > MIN_CONFIDENCE

        returns = test["returns"].values[signals]

        if len(returns) > 1:
            sharpe = np.sqrt(252) * returns.mean() / (returns.std() + 1e-9)
        else:
            sharpe = 0

        results.append({
            "fold": fold,
            "trades": len(returns),
            "pl": returns.sum(),
            "sharpe": sharpe
        })

    return pd.DataFrame(results)


# ------------------------------------------------------------------
# 6️⃣ ENSEMBLE MODEL
# ------------------------------------------------------------------

def build_ensemble():

    xgb = XGBModel().get_model()
    lstm = LSTMModel().get_model()
    cnn = CNNModel().get_model()

    ensemble = StackingClassifier(
        estimators=[
            ("xgb", xgb),
            ("lstm", lstm),
            ("cnn", cnn)
        ],
        final_estimator=LogisticRegression(),
        passthrough=True
    )

    return ensemble


# ------------------------------------------------------------------
# 7️⃣ MAIN EXECUTION
# ------------------------------------------------------------------

if __name__ == "__main__":

    print("Loading market data...")
    df = load_all_data()

    print("Engineering technical features...")
    df = TechnicalFeatureEngineer().fit_transform(df)

    print("Loading and aligning sentiment...")
    sentiment_engine = SentimentEngine(SYMBOLS)
    df = attach_sentiment(df, sentiment_engine)

    print("Creating target...")
    df = create_target(df)

    feature_cols = [
        "sma5","sma20","rsi","atr",
        "returns","sentiment"
    ]

    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    ensemble = build_ensemble()

    print("Running walk-forward validation...")
    results = walk_forward_validation(df[feature_cols + ["target","returns"]], ensemble)

    print("\n=== VALIDATION RESULTS ===")
    print(results)

    print("\nTraining final model on full dataset...")
    X_full = df[feature_cols]
    y_full = df["target"]

    ensemble.fit(X_full, y_full)

    joblib.dump(ensemble, ROOT / "models/hedge_fund_ensemble.pkl")
    joblib.dump(scaler, ROOT / "models/hedge_fund_scaler.pkl")

    print("\nModel + scaler saved successfully.")
