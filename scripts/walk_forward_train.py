#!/usr/bin/env python3

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

# ================================
# CONFIG
# ================================

FX_SYMBOLS = ["EURUSD", "USDJPY", "GBPUSD", "USDCHF", "AUDUSD"]
CRYPTO_SYMBOLS = ["BTCUSDT", "ETHUSDT"]
ALL_SYMBOLS = FX_SYMBOLS + CRYPTO_SYMBOLS

TIMEFRAME = "1h"
HISTORY_LIMIT = 20000
LOOKAHEAD = 18

TRAIN_MONTHS = 3
TEST_MONTHS = 1

COST_FX = 0.0002
COST_CRYPTO = 0.0005

# ================================
# SHARPE
# ================================

def compute_sharpe(returns):
    returns = np.array(returns)
    if len(returns) < 50:
        return 0.0
    std = np.std(returns)
    if std < 1e-8:
        return 0.0
    return np.mean(returns) / std

# ================================
# FEATURES
# ================================

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

    df["volatility20"] = df["returns"].rolling(20).std()

    return df.dropna()

def create_target(df):
    df = df.copy()
    df["forward_return"] = df["close"].shift(-LOOKAHEAD) / df["close"] - 1
    return df.dropna()

# ================================
# WALK FORWARD
# ================================

def walk_forward_symbol(symbol):

    print(f"\nProcessing {symbol}")

    df = fetch_market_data(symbol, TIMEFRAME, HISTORY_LIMIT)
    if df is None or len(df) < 2500:
        print("Not enough data.")
        return []

    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()

    df = engineer_features(df)
    df = create_target(df)

    returns_all = []

    start_date = df.index.min()
    end_date = df.index.max()

    current_train_start = start_date

    while True:

        train_end = current_train_start + pd.DateOffset(months=TRAIN_MONTHS)
        test_end = train_end + pd.DateOffset(months=TEST_MONTHS)

        train_df = df[(df.index >= current_train_start) & (df.index < train_end)]
        test_df = df[(df.index >= train_end) & (df.index < test_end)]

        if len(test_df) < 200:
            break

        feature_cols = [c for c in df.columns if c != "forward_return"]

        model = MomentumModel()
        model.fit(train_df[feature_cols], train_df["forward_return"])

        preds = model.predict(test_df[feature_cols])

        for i in range(len(preds)):
            pred = preds[i]
            realized = test_df["forward_return"].iloc[i]

            direction = np.sign(pred)
            position = np.clip(abs(pred), 0, 1) * direction

            cost = COST_FX if symbol in FX_SYMBOLS else COST_CRYPTO
            pnl = position * realized - abs(position) * cost

            returns_all.append(pnl)

        current_train_start += pd.DateOffset(months=1)

    sharpe = compute_sharpe(returns_all)
    print(f"{symbol} Walk-Forward Sharpe: {round(sharpe,3)} | Trades: {len(returns_all)}")

    return returns_all

# ================================
# MAIN
# ================================

def main():

    print("\n========== WALK-FORWARD VALIDATION ==========\n")

    portfolio_returns = []

    for symbol in ALL_SYMBOLS:
        r = walk_forward_symbol(symbol)
        portfolio_returns.extend(r)

    portfolio_sharpe = compute_sharpe(portfolio_returns)

    print("\n========== PORTFOLIO WALK-FORWARD ==========")
    print(f"Portfolio Sharpe: {round(portfolio_sharpe,3)}")
    print(f"Total Trades: {len(portfolio_returns)}")

if __name__ == "__main__":
    main()