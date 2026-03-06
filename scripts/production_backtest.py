#!/usr/bin/env python3

"""
NekoAITrader Production Backtest
Compatible with production_train.py regime models
"""

import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


# ================================
# FEATURE ENGINEERING
# ================================

def rsi(close, period=14):
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


def atr(df, period=14):

    prev_close = df.close.shift(1)

    tr = pd.concat([
        df.high - df.low,
        (df.high - prev_close).abs(),
        (df.low - prev_close).abs()
    ], axis=1).max(axis=1)

    return tr.rolling(period).mean()


def build_features(df):

    x = df.copy()

    # returns
    x["ret_1"] = x.close.pct_change(1)
    x["ret_3"] = x.close.pct_change(3)
    x["ret_12"] = x.close.pct_change(12)

    # EMAs
    x["ema_8"] = x.close.ewm(span=8, adjust=False).mean()
    x["ema_21"] = x.close.ewm(span=21, adjust=False).mean()

    x["ema_spread"] = (x.ema_8 - x.ema_21) / x.close

    # ATR
    x["atr"] = atr(x)
    x["atr_pct"] = x.atr / x.close

    # RSI
    x["rsi"] = rsi(x.close)

    # Volatility regime
    vol = x["ret_1"].rolling(200).std()
    pct = vol.rolling(500).rank(pct=True)

    x["vol_regime"] = pd.cut(
        pct,
        bins=[0.0, 0.33, 0.66, 1.0],
        labels=[0,1,2]
    ).astype(float)

    # Time features
    hours = x.index.hour

    x["hour_sin"] = np.sin(2*np.pi*hours/24)
    x["hour_cos"] = np.cos(2*np.pi*hours/24)

    # Day-of-week features (FIX)
    dow = x.index.dayofweek

    x["dow_sin"] = np.sin(2*np.pi*dow/7)
    x["dow_cos"] = np.cos(2*np.pi*dow/7)

    # Cross-sectional momentum (FIX)
    if "symbol" in x.columns:
        x["cs_momentum"] = x.groupby("symbol")["ret_12"].transform(lambda s: s - s.mean())
    else:
        x["cs_momentum"] = x["ret_12"]

    return x.replace([np.inf, -np.inf], np.nan)

# ================================
# DATA LOADER
# ================================

def load_fx_data():

    fx_dir = ROOT / "data/FX_DATA/PARQUET"

    frames = []

    for f in fx_dir.glob("*.parquet"):

        sym = f.stem

        df = pd.read_parquet(f)

        df.index = pd.to_datetime(df.index, utc=True)

        df["symbol"] = sym

        frames.append(df)

    if not frames:
        raise RuntimeError("No FX parquet files found")

    df = pd.concat(frames).sort_index()

    return df


# ================================
# BACKTEST ENGINE
# ================================

def run_backtest(models, features, horizon, top_pct):

    panel = load_fx_data()

    panel = build_features(panel)

    panel["target_return"] = panel.close.shift(-horizon)/panel.close - 1

    panel = panel.dropna()

    preds = []

    # ---------------------------
    # Generate predictions
    # ---------------------------

    for i in range(len(panel)):

        row = panel.iloc[i]

        regime = int(row["vol_regime"])

        if regime not in models:

            preds.append(0)

            continue

        model = models[regime]

        X = row[features].values.reshape(1, -1)

        try:
            pred = model.predict(X)[0]
        except Exception:
            pred = 0

        preds.append(pred)

    preds = np.array(preds)

    # ---------------------------
    # Signal filtering
    # ---------------------------

    if np.all(preds == 0):

        return {
            "error": "model produced zero predictions",
            "trades": 0
        }, np.array([])

    threshold = np.percentile(np.abs(preds), 100*(1-top_pct))

    signals = np.where(np.abs(preds) >= threshold, np.sign(preds), 0)

    # ---------------------------
    # Execute trades
    # ---------------------------

    trades = []

    equity = [1.0]

    for i, sig in enumerate(signals):

        if sig == 0:
            continue

        ret = panel.iloc[i]["target_return"] * sig

        trades.append(ret)

        equity.append(equity[-1]*(1+ret))

    equity = np.array(equity)

    trades = np.array(trades)

    if len(trades) == 0:

        result = {
            "error": "no trades executed",
            "trades": 0,
            "win_rate": 0,
            "avg_return": 0,
            "total_return": 0,
            "max_drawdown": 0,
            "sharpe": 0
        }

        return result, np.array([])

    wins = np.sum(trades > 0)

    sharpe = np.mean(trades)/(np.std(trades)+1e-9)*math.sqrt(252)

    result = {

        "trades": int(len(trades)),

        "win_rate": float(wins/len(trades)),

        "avg_return": float(np.mean(trades)),

        "total_return": float(equity[-1]-1),

        "max_drawdown": float(np.min(equity/np.maximum.accumulate(equity)-1)),

        "sharpe": float(sharpe)
    }

    return result, trades


# ================================
# MONTE CARLO
# ================================

def monte_carlo(trades, runs=500):

    if trades is None or len(trades) == 0:

        return {
            "mc_median_return": 0,
            "mc_worst_return": 0,
            "mc_best_return": 0
        }

    curves = []

    for _ in range(runs):

        shuffled = np.random.permutation(trades)

        equity = np.cumprod(1 + shuffled)

        if len(equity) == 0:
            continue

        curves.append(equity[-1])

    curves = np.array(curves)

    return {

        "mc_median_return": float(np.median(curves)-1),

        "mc_worst_return": float(np.min(curves)-1),

        "mc_best_return": float(np.max(curves)-1)
    }


# ================================
# MAIN
# ================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="models/production/fx_regime_models.joblib")

    parser.add_argument("--horizon", type=int, default=12)

    parser.add_argument("--top-pct", type=float, default=0.1)

    parser.add_argument("--output", default="models/production/backtest_report.json")

    args = parser.parse_args()

    print("Loading models...")

    bundle = joblib.load(args.model)

    models = bundle["models"]

    features = bundle["features"]

    print("Running backtest...")

    result, trades = run_backtest(

        models,
        features,
        args.horizon,
        args.top_pct
    )

    mc = monte_carlo(trades)

    result.update(mc)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w") as f:

        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))

    print(f"\nSaved report → {args.output}")


if __name__ == "__main__":

    main()