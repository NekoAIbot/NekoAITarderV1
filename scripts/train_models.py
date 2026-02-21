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
from app.news import fetch_news

SYMBOLS = FOREX_MAJORS + CRYPTO_ASSETS

LOOKAHEAD = 1
ATR_PERIOD = 14
MIN_CONFIDENCE = 0.55
FEE = 0.0001
SENTIMENT_LOOKBACK_HOURS = 24
INITIAL_CAPITAL = 1.0
RISK_PER_TRADE = 0.01  # 1% capital risked per trade

# ------------------------------------------------------------------
# 1️⃣ TECHNICAL FEATURES
# ------------------------------------------------------------------
class TechnicalFeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        df = df.copy()
        # SMA/EMA
        df["sma5"] = df["close"].rolling(5).mean()
        df["sma20"] = df["close"].rolling(20).mean()
        df["ema5"] = df["close"].ewm(span=5).mean()
        df["ema20"] = df["close"].ewm(span=20).mean()
        # RSI
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        rs = gain.rolling(14).mean() / (loss.rolling(14).mean() + 1e-9)
        df["rsi"] = 100 - (100 / (1 + rs))
        # ATR
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs()
        ], axis=1).max(axis=1)
        df["atr"] = tr.rolling(ATR_PERIOD).mean()
        # Bollinger Bands
        df["bb_upper"] = df["sma20"] + 2 * df["close"].rolling(20).std()
        df["bb_lower"] = df["sma20"] - 2 * df["close"].rolling(20).std()
        df["bb_width"] = df["bb_upper"] - df["bb_lower"]
        # Momentum
        df["momentum5"] = df["close"] - df["close"].shift(5)
        df["momentum10"] = df["close"] - df["close"].shift(10)
        # Volatility
        df["volatility10"] = df["returns"].rolling(10).std()
        df["volatility20"] = df["returns"].rolling(20).std()
        # Returns
        df["returns"] = df["close"].pct_change()
        # ROC
        df["roc5"] = df["close"].pct_change(5)
        df["roc10"] = df["close"].pct_change(10)
        # Drop NA
        return df.dropna()

# ------------------------------------------------------------------
# 2️⃣ SENTIMENT ENGINE
# ------------------------------------------------------------------
class SentimentEngine:

    def __init__(self, symbols):
        self.symbols = symbols
        self.news_cache = {}
        self._load_news()

    def _load_news(self):
        news_df = fetch_news()
        if news_df is None or len(news_df) == 0:
            # fallback to zero sentiment
            news_df = pd.DataFrame(columns=["timestamp","symbol","sentiment"])
        news_df["timestamp"] = pd.to_datetime(news_df["timestamp"], utc=True)
        news_df = news_df.sort_values("timestamp")
        for sym in self.symbols:
            self.news_cache[sym] = news_df[news_df["symbol"] == sym].copy()

    def get_sentiment(self, symbol, current_time, windows=[1,6,24]):
        """Return multiple sentiment features (avg over 1h, 6h, 24h)"""
        if symbol not in self.news_cache:
            return {f"sentiment_{w}h": 0.0 for w in windows}
        news_df = self.news_cache[symbol]
        current_time = pd.to_datetime(current_time, utc=True)
        out = {}
        for w in windows:
            start_time = current_time - timedelta(hours=w)
            relevant = news_df[(news_df["timestamp"] < current_time) &
                               (news_df["timestamp"] >= start_time)]
            out[f"sentiment_{w}h"] = float(relevant["sentiment"].mean()) if len(relevant) > 0 else 0.0
        return out

def attach_sentiment(df, sentiment_engine):
    sentiments_list = []
    for idx, row in df.iterrows():
        sentiments = sentiment_engine.get_sentiment(row["symbol"], idx)
        sentiments_list.append(sentiments)
    sentiment_df = pd.DataFrame(sentiments_list, index=df.index)
    df = pd.concat([df, sentiment_df], axis=1)
    return df

# ------------------------------------------------------------------
# 3️⃣ TARGET
# ------------------------------------------------------------------
def create_target(df):
    df = df.copy()
    df["target"] = (df["close"].shift(-LOOKAHEAD) > df["close"]).astype(int)
    return df.dropna()

# ------------------------------------------------------------------
# 4️⃣ LOAD DATA
# ------------------------------------------------------------------
def load_all_data():
    frames = []
    for sym in SYMBOLS:
        df = fetch_market_data(sym)
        df.index = pd.to_datetime(df.index, utc=True)
        df["symbol"] = sym
        frames.append(df)
    return pd.concat(frames).sort_index()

# ------------------------------------------------------------------
# 5️⃣ METRICS
# ------------------------------------------------------------------
def compute_sharpe(returns):
    if len(returns) < 2:
        return 0.0
    std = np.std(returns)
    if std < 1e-8:
        return 0.0
    return np.sqrt(252) * np.mean(returns) / std

# ------------------------------------------------------------------
# 6️⃣ CAPITAL SIMULATION
# ------------------------------------------------------------------
def simulate_trading(test_df, signals):
    capital = INITIAL_CAPITAL
    equity_curve = []
    returns = test_df["returns"].values
    for i in range(len(signals)):
        if signals[i]:
            trade_return = returns[i] - FEE
            position_return = trade_return * RISK_PER_TRADE
            capital *= (1 + position_return)
        equity_curve.append(capital)
    equity_curve = np.array(equity_curve)
    total_return = capital - INITIAL_CAPITAL
    strategy_returns = np.diff(equity_curve) / equity_curve[:-1] if len(equity_curve) > 1 else np.zeros_like(equity_curve)
    return total_return, strategy_returns

# ------------------------------------------------------------------
# 7️⃣ WALK-FORWARD VALIDATION
# ------------------------------------------------------------------
def walk_forward_validation(df, feature_cols, model):
    tscv = TimeSeriesSplit(n_splits=5)
    results = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
        train = df.iloc[train_idx].copy()
        test = df.iloc[test_idx].copy()
        X_train = train[feature_cols]
        y_train = train["target"]
        X_test = test[feature_cols]
        y_test = test["target"]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        signals = probs > MIN_CONFIDENCE
        total_return, strategy_returns = simulate_trading(test, signals)
        sharpe = compute_sharpe(strategy_returns)
        results.append({
            "fold": fold,
            "trades": int(signals.sum()),
            "total_return_pct": float(total_return * 100),
            "sharpe": float(sharpe)
        })
        print(f"Fold {fold} | Trades: {signals.sum()} | Return: {total_return*100:.2f}% | Sharpe: {sharpe:.3f}")
    return pd.DataFrame(results)

# ------------------------------------------------------------------
# 8️⃣ ENSEMBLE
# ------------------------------------------------------------------
def build_ensemble():
    return StackingClassifier(
        estimators=[
            ("xgb", XGBModel().get_model()),
            ("lstm", LSTMModel().get_model()),
            ("cnn", CNNModel().get_model())
        ],
        final_estimator=LogisticRegression(),
        passthrough=True
    )

# ------------------------------------------------------------------
# 9️⃣ MAIN
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
    df = df.dropna()

    feature_cols = [c for c in df.columns if c not in ["symbol","target"]]

    ensemble = build_ensemble()

    print("Running walk-forward validation...")
    results = walk_forward_validation(df, feature_cols, ensemble)
    print("\n=== VALIDATION RESULTS ===")
    print(results)

    print("\nTraining final model on full dataset...")
    scaler = StandardScaler()
    X_full = scaler.fit_transform(df[feature_cols])
    y_full = df["target"]
    ensemble.fit(X_full, y_full)
    joblib.dump(ensemble, ROOT / "models/hedge_fund_ensemble.pkl")
    joblib.dump(scaler, ROOT / "models/hedge_fund_scaler.pkl")
    print("\nModel + scaler saved successfully.")