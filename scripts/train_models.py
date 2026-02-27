#!/usr/bin/env python3
# scripts/train_models.py

import sys
import warnings
from pathlib import Path
from multiprocessing import Process
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.market_data import fetch_market_data
from app.models.xgb_model import MomentumModel

# ==========================================
# CONFIG
# ==========================================

FX_SYMBOLS = ["EURUSD", "USDJPY", "GBPUSD", "USDCHF", "AUDUSD"]
CRYPTO_SYMBOLS = ["BTCUSDT", "ETHUSDT"]

TIMEFRAME = "1h"
HISTORY_LIMIT = 15000
ATR_PERIOD = 14

MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ==========================================
# SAFE SHARPE
# ==========================================

def compute_sharpe(returns):
    returns = np.array(returns)
    if len(returns) < 30:
        return 0.0
    std = np.std(returns)
    if std < 1e-6:
        return 0.0
    return np.mean(returns) / std

# ==========================================
# FEATURES
# ==========================================

def engineer_features(df):
    df = df.copy()

    df["returns"] = df["close"].pct_change()
    df["ema50"] = df["close"].ewm(span=50).mean()
    df["ema200"] = df["close"].ewm(span=200).mean()
    df["ema_slope"] = df["ema50"].pct_change()

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
    df["volatility20"] = df["returns"].rolling(20).std()
    df["vol_percentile"] = df["volatility20"].rolling(200).rank(pct=True)

    return df.dropna()

# ==========================================
# FX TARGET (Regression)
# ==========================================

def create_fx_target(df, lookahead):
    df = df.copy()
    df["forward_return"] = df["close"].shift(-lookahead) / df["close"] - 1
    return df.dropna()

# ==========================================
# CRYPTO TARGET (TP/SL Classification)
# ==========================================

def create_crypto_target(df, lookahead, rr_ratio):
    df = df.copy()
    df["long_target"] = 0

    for i in range(len(df) - lookahead):
        entry = df["close"].iloc[i]
        atr = df["atr"].iloc[i]

        sl = entry - atr
        tp = entry + atr * rr_ratio

        future = df.iloc[i+1:i+lookahead+1]

        for _, row in future.iterrows():
            if row["low"] <= sl:
                break
            if row["high"] >= tp:
                df.iloc[i, df.columns.get_loc("long_target")] = 1
                break

    return df

# ==========================================
# FX TRAINING (Regression Engine)
# ==========================================

def train_fx():

    print("\n========== TRAINING FX (REGRESSION) ==========\n")

    LOOKAHEAD = 18
    COST = 0.0002
    VOL_FILTER = 0.6

    portfolio_returns = []

    for symbol in FX_SYMBOLS:

        print(f"Processing {symbol}")

        df = fetch_market_data(symbol, TIMEFRAME, HISTORY_LIMIT)
        if df is None or len(df) < 500:
            continue

        df.index = pd.to_datetime(df.index, utc=True)

        df = engineer_features(df)
        df = create_fx_target(df, LOOKAHEAD)

        df = df.sort_index()

        split = int(len(df) * 0.7)
        train = df.iloc[:split]
        test = df.iloc[split:]

        feature_cols = [c for c in df.columns if c not in ["forward_return"]]

        model = MomentumModel()
        model.fit(train[feature_cols], train["forward_return"])

        preds = model.predict(test[feature_cols])

        returns = []

        for i, pred in enumerate(preds):

            if test["vol_percentile"].iloc[i] < VOL_FILTER:
                continue

            position = np.clip(pred * 5, -1, 1)  # scale prediction

            realized = test["forward_return"].iloc[i]

            r = position * realized - abs(position) * COST
            returns.append(r)

        sharpe = compute_sharpe(returns)
        print(f"{symbol} | Sharpe: {round(sharpe,4)} | Trades: {len(returns)}")

        portfolio_returns.extend(returns)

    portfolio_sharpe = compute_sharpe(portfolio_returns)

    print(f"\nFX Portfolio Sharpe: {round(portfolio_sharpe,4)}")
    print(f"Total Trades: {len(portfolio_returns)}")

    # Train final model on full FX dataset
    full_data = []

    for symbol in FX_SYMBOLS:
        df = fetch_market_data(symbol, TIMEFRAME, HISTORY_LIMIT)
        if df is None or len(df) < 500:
            continue
        df.index = pd.to_datetime(df.index, utc=True)
        df = engineer_features(df)
        df = create_fx_target(df, LOOKAHEAD)
        full_data.append(df)

    if full_data:
        full_df = pd.concat(full_data)
        feature_cols = [c for c in full_df.columns if c not in ["forward_return"]]
        final_model = MomentumModel()
        final_model.fit(full_df[feature_cols], full_df["forward_return"])
        joblib.dump(final_model, MODEL_DIR / "fx_model.pkl")
        print("FX model saved.")

# ==========================================
# CRYPTO TRAINING (TP Engine)
# ==========================================

def train_crypto():

    print("\n========== TRAINING CRYPTO (TP/SL) ==========\n")

    LOOKAHEAD = 8
    RR_RATIO = 2.0
    COST = 0.0005
    VOL_FILTER = 0.55

    portfolio_returns = []

    for symbol in CRYPTO_SYMBOLS:

        print(f"Processing {symbol}")

        df = fetch_market_data(symbol, TIMEFRAME, HISTORY_LIMIT)
        if df is None or len(df) < 500:
            continue

        df.index = pd.to_datetime(df.index, utc=True)

        df = engineer_features(df)
        df = create_crypto_target(df, LOOKAHEAD, RR_RATIO)
        df = df.dropna()

        df = df.sort_index()

        split = int(len(df) * 0.7)
        train = df.iloc[:split]
        test = df.iloc[split:]

        feature_cols = [c for c in df.columns if c not in ["long_target"]]

        model = MomentumModel()
        model.fit(train[feature_cols], train["long_target"])

        preds = model.predict(test[feature_cols])

        returns = []

        for i, pred in enumerate(preds):

            if pred < 0.55:
                continue

            if test["vol_percentile"].iloc[i] < VOL_FILTER:
                continue

            if test["long_target"].iloc[i] == 1:
                r = RR_RATIO - COST
            else:
                r = -1 - COST

            returns.append(r)

        sharpe = compute_sharpe(returns)
        print(f"{symbol} | Sharpe: {round(sharpe,4)} | Trades: {len(returns)}")

        portfolio_returns.extend(returns)

    portfolio_sharpe = compute_sharpe(portfolio_returns)

    print(f"\nCRYPTO Portfolio Sharpe: {round(portfolio_sharpe,4)}")
    print(f"Total Trades: {len(portfolio_returns)}")

    # Train final crypto model
    full_data = []

    for symbol in CRYPTO_SYMBOLS:
        df = fetch_market_data(symbol, TIMEFRAME, HISTORY_LIMIT)
        if df is None or len(df) < 500:
            continue
        df.index = pd.to_datetime(df.index, utc=True)
        df = engineer_features(df)
        df = create_crypto_target(df, LOOKAHEAD, RR_RATIO)
        df = df.dropna()
        full_data.append(df)

    if full_data:
        full_df = pd.concat(full_data)
        feature_cols = [c for c in full_df.columns if c not in ["long_target"]]
        final_model = MomentumModel()
        final_model.fit(full_df[feature_cols], full_df["long_target"])
        joblib.dump(final_model, MODEL_DIR / "crypto_model.pkl")
        print("CRYPTO model saved.")

# ==========================================
# PARALLEL EXECUTION
# ==========================================

if __name__ == "__main__":

    fx_process = Process(target=train_fx)
    crypto_process = Process(target=train_crypto)

    fx_process.start()
    crypto_process.start()

    fx_process.join()
    crypto_process.join()

    print("\nAll models trained in parallel successfully.")