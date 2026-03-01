#!/usr/bin/env python3
"""
Institutional FX Regime-Aware Training Pipeline (Demo → Live → Prop Ready)

Features:
- 3 volatility regimes (percentile-based)
- Walk-forward leak-safe validation
- Auto-disable weak/negative alpha regimes (IC threshold)
- Top-N% signal filtering
- Saves only statistically valid regime models
- Deployment metadata included for position sizing
"""

from __future__ import annotations
import argparse, sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from xgboost import XGBRegressor

# ======================================
# PATH SETUP
# ======================================

ROOT = Path("/workspaces/NekoAITarderV1")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import FOREX_MAJORS

MODEL_DIR = ROOT / "models" / "production"
DATA_DIR = ROOT / "data/FX_DATA/PARQUET"

# ======================================
# CONFIG
# ======================================

@dataclass
class TrainConfig:
    limit: int
    horizon: int
    n_splits: int
    embargo: int
    train_fraction: float
    random_state: int
    top_pct_threshold: float
    min_ic_threshold: float

# ======================================
# UTILS
# ======================================

def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def load_fx(symbol):
    path = DATA_DIR / f"{symbol}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index, utc=True)
        return df.sort_index()
    return None

# ======================================
# FEATURE ENGINEERING
# ======================================

def build_features(df):
    x = df.copy()

    # Returns
    x["ret_1"] = x["close"].pct_change()
    x["ret_3"] = x["close"].pct_change(3)
    x["ret_12"] = x["close"].pct_change(12)

    # Trend
    x["ema_8"] = x["close"].ewm(span=8, adjust=False).mean()
    x["ema_21"] = x["close"].ewm(span=21, adjust=False).mean()
    x["ema_spread"] = (x["ema_8"] - x["ema_21"]) / x["close"]

    # Volatility
    tr = pd.concat([
        (x["high"] - x["low"]).abs(),
        (x["high"] - x["close"].shift()).abs(),
        (x["low"] - x["close"].shift()).abs()
    ], axis=1).max(axis=1)

    x["atr"] = tr.rolling(14).mean()
    x["atr_pct"] = x["atr"] / x["close"]

    # Vol regime (percentile bucket)
    vol = x["ret_1"].rolling(200).std()
    pct = vol.rolling(500).rank(pct=True)

    x["vol_regime"] = pd.cut(
        pct,
        bins=[0.0, 0.33, 0.66, 1.0],
        labels=[0, 1, 2]
    ).astype(float)

    # Session encoding
    hour = x.index.hour
    x["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    x["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    x["dow_sin"] = np.sin(2 * np.pi * x.index.dayofweek / 7)
    x["dow_cos"] = np.cos(2 * np.pi * x.index.dayofweek / 7)

    return x.replace([np.inf, -np.inf], np.nan)

# ======================================
# PANEL ASSEMBLY
# ======================================

def assemble_panel(cfg):
    rows = []

    for sym in FOREX_MAJORS:
        df = load_fx(sym)
        if df is None:
            print(f"[WARN] Missing {sym}")
            continue

        df = df.tail(cfg.limit)
        if len(df) < 1000:
            continue

        feat = build_features(df)
        feat["target_return"] = feat["close"].shift(-cfg.horizon) / feat["close"] - 1
        feat["symbol"] = sym
        feat = feat.dropna()

        print(f"[INFO] {sym}: usable rows = {len(feat)}")
        rows.append(feat)

    panel = pd.concat(rows).sort_index()
    panel["cs_momentum"] = panel.groupby(panel.index)["ret_3"].rank(pct=True)
    panel = panel.dropna()

    print(f"[INFO] Final panel size: {panel.shape}")
    return panel

# ======================================
# WALK FORWARD
# ======================================

def walk_forward(index, n_splits, train_fraction, embargo):
    ts = np.array(sorted(index.unique()))
    train_size = int(len(ts) * train_fraction)
    block = (len(ts) - train_size) // n_splits

    for i in range(n_splits):
        train_end = train_size + i * block
        test_start = train_end + embargo
        test_end = test_start + block
        yield set(ts[:train_end]), set(ts[test_start:test_end])

# ======================================
# MODEL
# ======================================

def build_model(seed):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", XGBRegressor(
            n_estimators=1500,
            max_depth=7,
            learning_rate=0.015,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            n_jobs=-1
        ))
    ])

# ======================================
# TRAIN
# ======================================

def train(cfg):
    panel = assemble_panel(cfg)
    features = [c for c in panel.columns if c not in {"target_return", "symbol"}]

    regime_models = {}
    regime_stats = {}

    for regime in sorted(panel["vol_regime"].unique()):
        subset = panel[panel["vol_regime"] == regime]
        print(f"\n[INFO] Training regime {regime} with {len(subset)} rows")

        model = build_model(cfg.random_state)
        ic_scores = []
        acc_scores = []

        for tr_ts, te_ts in walk_forward(subset.index, cfg.n_splits, cfg.train_fraction, cfg.embargo):
            tr = subset[subset.index.isin(tr_ts)]
            te = subset[subset.index.isin(te_ts)]

            if len(tr) < 500 or len(te) < 100:
                continue

            m = clone(model)
            m.fit(tr[features], tr["target_return"])

            preds = m.predict(te[features])
            true = te["target_return"].values

            ic = spearmanr(preds, true).correlation
            ic_scores.append(ic)

            threshold = np.percentile(np.abs(preds), 100*(1-cfg.top_pct_threshold))
            mask = np.abs(preds) >= threshold

            if mask.sum() > 0:
                acc_scores.append(np.mean(np.sign(preds[mask]) == np.sign(true[mask])))

        mean_ic = np.nanmean(ic_scores)
        mean_acc = np.nanmean(acc_scores)

        print(f"[REGIME {regime}] Mean IC: {mean_ic:.4f}, Top {int(cfg.top_pct_threshold*100)}% accuracy: {mean_acc:.4f}")

        regime_stats[int(regime)] = {
            "mean_ic": float(mean_ic),
            "mean_accuracy": float(mean_acc)
        }

        if mean_ic > cfg.min_ic_threshold:
            final_model = build_model(cfg.random_state)
            final_model.fit(subset[features], subset["target_return"])
            regime_models[int(regime)] = final_model
            print(f"[INFO] Regime {regime} ENABLED")
        else:
            print(f"[INFO] Regime {regime} DISABLED (IC below threshold)")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out = MODEL_DIR / "fx_regime_models.joblib"

    joblib.dump({
        "models": regime_models,
        "features": features,
        "created_at": utc_now(),
        "threshold": cfg.top_pct_threshold,
        "min_ic_threshold": cfg.min_ic_threshold,
        "regime_stats": regime_stats,
        "position_sizing_hint": "size = predicted_return / atr"
    }, out)

    print(f"\n[OK] Saved regime models -> {out}")

# ======================================
# CLI
# ======================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20000)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--embargo", type=int, default=2)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-pct-threshold", type=float, default=0.1)
    parser.add_argument("--min-ic-threshold", type=float, default=0.04)
    args = parser.parse_args()

    cfg = TrainConfig(
        limit=args.limit,
        horizon=args.horizon,
        n_splits=args.splits,
        embargo=args.embargo,
        train_fraction=args.train_fraction,
        random_state=args.seed,
        top_pct_threshold=args.top_pct_threshold,
        min_ic_threshold=args.min_ic_threshold
    )

    train(cfg)

if __name__ == "__main__":
    main()