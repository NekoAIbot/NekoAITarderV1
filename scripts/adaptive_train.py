#!/usr/bin/env python3
# scripts/adaptive_train.py

import sys
import warnings
from pathlib import Path
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

ALL_SYMBOLS = FX_SYMBOLS + CRYPTO_SYMBOLS

TIMEFRAME = "1h"
HISTORY_LIMIT = 15000
LOOKAHEAD = 18

COST_FX = 0.0002
COST_CRYPTO = 0.0005

ATR_PERIOD = 14

MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ==========================================
# SAFE SHARPE (ANNUALIZED)
# ==========================================

def compute_sharpe(returns):
    returns = np.array(returns)
    if len(returns) < 50:
        return 0.0

    std = np.std(returns)
    if std < 1e-8:
        return 0.0

    return np.mean(returns) / std

# ==========================================
# FEATURE ENGINEERING
# ==========================================

def engineer_features(df):
    df = df.copy()

    df["returns"] = df["close"].pct_change()

    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
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
    df["vol_percentile"] = (
        df["volatility20"]
        .rolling(200)
        .rank(pct=True)
    )

    return df.dropna()

# ==========================================
# TARGET
# ==========================================

def create_directional_target(df, lookahead):
    df = df.copy()
    df["forward_return"] = (
        df["close"].shift(-lookahead) / df["close"] - 1
    )
    return df.dropna()

# ==========================================
# REGIME DETECTION
# ==========================================

def detect_regime(row):
    if row["vol_percentile"] > 0.8 and abs(row["ema_slope"]) > 0.001:
        return "high_vol_trend"
    elif row["vol_percentile"] < 0.3:
        return "low_vol_range"
    else:
        return "normal"

REGIME_MULTIPLIER = {
    "high_vol_trend": 1.2,
    "normal": 1.0,
    "low_vol_range": 0.6,
}

# ==========================================
# ADAPTIVE THRESHOLD
# ==========================================

def compute_threshold(row):
    base = 0.0005
    vol_adj = row["volatility20"] * 0.5

    regime_adj = {
        "high_vol_trend": 0.8,
        "normal": 1.0,
        "low_vol_range": 1.3
    }[row["regime"]]

    return base * regime_adj + vol_adj

# ==========================================
# MAIN TRAINING
# ==========================================

def train_adaptive():

    print("\n========== STAGE 1 ADAPTIVE TRAINING ==========\n")

    portfolio_returns = []
    full_data = []

    for symbol in ALL_SYMBOLS:

        print(f"\nProcessing {symbol}")

        df = fetch_market_data(symbol, TIMEFRAME, HISTORY_LIMIT)
        if df is None or len(df) < 800:
            print("Insufficient data.")
            continue

        df.index = pd.to_datetime(df.index, utc=True)
        df = df.sort_index()

        df = engineer_features(df)
        df = create_directional_target(df, LOOKAHEAD)

        # REGIME ADDED AFTER TARGET (no leakage)
        df["regime"] = df.apply(detect_regime, axis=1)

        split = int(len(df) * 0.7)

        train = df.iloc[:split]
        test = df.iloc[split:]

        feature_cols = [
            c for c in df.columns
            if c not in ["forward_return", "regime"]
        ]

        model = MomentumModel()
        model.fit(train[feature_cols], train["forward_return"])

        preds = model.predict(test[feature_cols])

        returns = []

        for i in range(len(preds)):

            pred = preds[i]
            row = test.iloc[i]

            threshold = compute_threshold(row)

            if abs(pred) < threshold:
                continue

            direction = np.sign(pred)
            signal_strength = abs(pred)

            position = (
                0.5
                * signal_strength
                * REGIME_MULTIPLIER[row["regime"]]
            )

            position = np.clip(position, 0, 1)
            position *= direction

            realized = row["forward_return"]
            cost = COST_FX if symbol in FX_SYMBOLS else COST_CRYPTO

            pnl = position * realized - abs(position) * cost
            returns.append(pnl)

        sharpe = compute_sharpe(returns)
        print(f"{symbol} | Sharpe: {round(sharpe,3)} | Trades: {len(returns)}")

        portfolio_returns.extend(returns)
        full_data.append(df)

    portfolio_sharpe = compute_sharpe(portfolio_returns)

    print("\n========== PORTFOLIO RESULTS ==========")
    print(f"Portfolio Sharpe: {round(portfolio_sharpe,3)}")
    print(f"Total Trades: {len(portfolio_returns)}")

    # FINAL MODEL TRAINING
    if full_data:

        full_df = pd.concat(full_data).sort_index()

        feature_cols = [
            c for c in full_df.columns
            if c not in ["forward_return", "regime"]
        ]

        final_model = MomentumModel()
        final_model.fit(full_df[feature_cols], full_df["forward_return"])

        joblib.dump(final_model, MODEL_DIR / "adaptive_model.pkl")

        # Save feature list for live consistency
        joblib.dump(feature_cols, MODEL_DIR / "adaptive_features.pkl")

        print("Adaptive unified model saved.")
        print("Feature schema saved.")

# ==========================================
# ENTRY
# ==========================================

if __name__ == "__main__":
    train_adaptive()