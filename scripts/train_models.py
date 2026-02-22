#!/usr/bin/env python3
# scripts/train_models.py

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

# =========================
# CONFIGURATION
# =========================

FX_SYMBOLS = ["EURUSD", "USDJPY", "GBPUSD", "USDCHF", "AUDUSD"]
CRYPTO_SYMBOLS = ["BTCUSDT", "ETHUSDT"]

# ✅ Restrict to 1H only
TIMEFRAMES = ["1h"]

# ✅ Increase depth for stronger learning
HISTORY_LIMITS = {
    "1h": 15000
}

ATR_PERIOD = 14
FX_COST = 0.0002
CRYPTO_COST = 0.0005

MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# =========================
# FEATURE ENGINEERING
# =========================

def engineer_features(df):
    df = df.copy()

    # Returns
    df["returns"] = df["close"].pct_change()

    # Trend EMAs
    df["ema20"] = df["close"].ewm(span=20).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()
    df["ema100"] = df["close"].ewm(span=100).mean()
    df["ema200"] = df["close"].ewm(span=200).mean()

    df["ema_trend"] = (df["ema50"] > df["ema200"]).astype(int)

    # RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / (loss.rolling(14).mean() + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    # True Range / ATR
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs()
    ], axis=1).max(axis=1)

    df["atr"] = tr.rolling(14).mean()

    # Volatility
    df["volatility20"] = df["returns"].rolling(20).std()

    # Breakout Structure (very important)
    df["high_50"] = df["high"].rolling(50).max()
    df["low_50"] = df["low"].rolling(50).min()

    df["breakout_up"] = (df["close"] > df["high_50"].shift()).astype(int)
    df["breakout_down"] = (df["close"] < df["low_50"].shift()).astype(int)

    # Trend regime
    df["trend_regime"] = (df["close"] > df["ema200"]).astype(int)

    return df.dropna()

# =========================
# 3-CLASS TARGET
# =========================

LOOKAHEAD_MAP = {
    "1h": 6
}

def create_target(df):
    df = df.copy()
    lookahead = LOOKAHEAD_MAP[df["timeframe"].iloc[0]]

    df["fwd_return"] = df["close"].shift(-lookahead) / df["close"] - 1

    # ✅ Stronger move threshold
    threshold = 1.75 * df["atr"] / df["close"]

    df["direction"] = 0
    df.loc[df["fwd_return"] > threshold, "direction"] = 1
    df.loc[df["fwd_return"] < -threshold, "direction"] = -1

    # Regime = expansion vs contraction
    df["abs_fwd"] = df["fwd_return"].abs()
    df["regime_target"] = (
        df["abs_fwd"] > df["abs_fwd"].rolling(100).median()
    ).astype(int)

    return df.dropna()

# =========================
# LOAD DATA
# =========================

def load_symbols(symbols, asset_type):
    frames = []

    for symbol in symbols:
        for tf in TIMEFRAMES:

            print(f"Fetching {symbol} {tf}")

            df = fetch_market_data(
                symbol,
                timeframe=tf,
                limit=HISTORY_LIMITS[tf]
            )

            if df is None or len(df) < 300:
                continue

            df.index = pd.to_datetime(df.index, utc=True)
            df["symbol"] = symbol
            df["timeframe"] = tf
            df["asset_type"] = asset_type

            df = engineer_features(df)
            df = create_target(df)

            frames.append(df)

    return pd.concat(frames).sort_index()

# =========================
# WALK FORWARD
# =========================

# =========================
# WALK FORWARD
# =========================

def walk_forward(df, cost):

    df = df.sort_index()
    unique_dates = df.index.unique()
    n_splits = 3
    split_size = len(unique_dates) // (n_splits + 1)

    results = []

    # Create EMA trend column for direction
    df["ema_trend"] = (df["ema20"] > df["ema200"]).astype(int)  # 1 = uptrend, 0 = downtrend

    for i in range(n_splits):

        train_end = split_size * (i + 1)
        test_end = split_size * (i + 2)

        train = df.loc[unique_dates[:train_end]]
        test = df.loc[unique_dates[train_end:test_end]]

        feature_cols = [
            c for c in df.columns
            if c not in ["symbol", "timeframe", "asset_type",
                         "direction", "regime_target",
                         "fwd_return", "abs_fwd", "ema_trend"]
        ]

        X_train = train[feature_cols]
        X_test = test[feature_cols]

        y_regime_train = train["regime_target"]

        fwd = test["fwd_return"].values

        # -------------------------
        # TRAIN ONLY EXPANSION MODEL
        # -------------------------
        regime_model = MomentumModel().get_model()
        regime_model.fit(X_train, y_regime_train)

        # Predict expansion probability
        expansion_prob = regime_model.predict_proba(X_test)[:, 1]

        # Trade only strong expansion
        trade_mask = expansion_prob > 0.7

        # Direction from EMA trend
        ema_trend_test = test["ema_trend"].values
        direction = np.where(ema_trend_test == 1, 1, -1)

        # Position sizing = expansion confidence
        position = direction * expansion_prob * trade_mask

        returns = position * fwd - (np.abs(position) > 0) * cost

        sharpe = 0 if len(returns) < 2 else np.mean(returns)/(np.std(returns)+1e-9)

        results.append({
            "fold": i,
            "sharpe": sharpe,
            "trades": int(np.sum(np.abs(position) > 0))
        })

        print(f"\nFold {i} Sharpe: {round(sharpe,4)} | Trades: {results[-1]['trades']}")

    return pd.DataFrame(results)

# =========================
# FINAL MODEL TRAINING
# =========================

if __name__ == "__main__":

    print("Loading FX data...")
    fx_df = load_symbols(FX_SYMBOLS, "fx")

    print("\nLoading Crypto data...")
    crypto_df = load_symbols(CRYPTO_SYMBOLS, "crypto")

    print("\nRunning FX Walk-Forward")
    fx_results = walk_forward(fx_df, FX_COST)
    print(fx_results)

    print("\nRunning Crypto Walk-Forward")
    crypto_results = walk_forward(crypto_df, CRYPTO_COST)
    print(crypto_results)

    print("\nTraining Final FX Expansion Model...")
    fx_features = [c for c in fx_df.columns if c not in
        ["symbol","timeframe","asset_type","direction",
         "regime_target","fwd_return","abs_fwd","ema_trend"]]

    fx_regime = MomentumModel().get_model()
    fx_regime.fit(fx_df[fx_features],
                  fx_df["regime_target"])

    joblib.dump(fx_regime, MODEL_DIR/"fx_expansion.pkl")

    print("Training Final Crypto Expansion Model...")
    crypto_features = [c for c in crypto_df.columns if c not in
        ["symbol","timeframe","asset_type","direction",
         "regime_target","fwd_return","abs_fwd","ema_trend"]]

    crypto_regime = MomentumModel().get_model()
    crypto_regime.fit(crypto_df[crypto_features],
                      crypto_df["regime_target"])

    joblib.dump(crypto_regime, MODEL_DIR/"crypto_expansion.pkl")

    print("\nAll models trained and saved successfully.")