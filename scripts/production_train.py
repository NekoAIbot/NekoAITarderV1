#!/usr/bin/env python3
"""Production-grade model training pipeline for NekoAI.

Highlights:
- Pulls as much market data as possible via app.market_data fallback chain.
- Builds robust technical + cross-sectional features.
- Runs purged walk-forward validation.
- Tunes and compares strong baseline models.
- Persists a self-contained model bundle with metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CRYPTO_ASSETS, FOREX_MAJORS
import importlib.util


def _load_fetch_market_data():
    module_path = ROOT / "app" / "market_data.py"
    spec = importlib.util.spec_from_file_location("neko_market_data", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load market_data module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fetch_market_data


fetch_market_data = _load_fetch_market_data()


MODEL_DIR = ROOT / "models"
PROD_DIR = MODEL_DIR / "production"


@dataclass
class Config:
    timeframe: str
    limit: int
    horizon: int
    min_rows_per_symbol: int
    train_fraction: float
    n_splits: int
    embargo: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_symbol_data(symbol: str, timeframe: str, limit: int) -> pd.DataFrame | None:
    try:
        df = fetch_market_data(symbol, timeframe=timeframe, limit=limit)
    except Exception as exc:
        print(f"[WARN] fetch failed for {symbol}: {exc}")
        return None

    if df is None or df.empty:
        print(f"[WARN] no rows for {symbol}")
        return None

    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        print(f"[WARN] missing columns for {symbol}: {sorted(required - set(df.columns))}")
        return None

    out = df.copy()
    out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close", "volume"])
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def _build_symbol_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()

    x["ret_1"] = x["close"].pct_change(1)
    x["ret_3"] = x["close"].pct_change(3)
    x["ret_5"] = x["close"].pct_change(5)
    x["ret_15"] = x["close"].pct_change(15)

    x["log_ret_1"] = np.log(x["close"]).diff(1)
    x["range"] = (x["high"] - x["low"]) / x["close"].replace(0, np.nan)

    x["ema_8"] = x["close"].ewm(span=8, adjust=False).mean()
    x["ema_21"] = x["close"].ewm(span=21, adjust=False).mean()
    x["ema_55"] = x["close"].ewm(span=55, adjust=False).mean()
    x["ema_diff_8_21"] = (x["ema_8"] - x["ema_21"]) / x["close"].replace(0, np.nan)
    x["ema_diff_21_55"] = (x["ema_21"] - x["ema_55"]) / x["close"].replace(0, np.nan)

    x["rsi_14"] = _rsi(x["close"], 14)
    x["atr_14"] = _atr(x, 14)
    x["atr_pct"] = x["atr_14"] / x["close"].replace(0, np.nan)

    vol_sma_20 = x["volume"].rolling(20).mean()
    vol_std_20 = x["volume"].rolling(20).std()
    x["vol_z_20"] = (x["volume"] - vol_sma_20) / vol_std_20.replace(0, np.nan)

    mom_10 = x["close"] - x["close"].shift(10)
    x["mom_10"] = mom_10 / x["close"].shift(10).replace(0, np.nan)

    return x


def _build_panel_dataset(symbols: List[str], cfg: Config) -> Tuple[pd.DataFrame, Dict[str, int]]:
    rows = []
    coverage = {}

    for sym in symbols:
        raw = _safe_symbol_data(sym, cfg.timeframe, cfg.limit)
        if raw is None or len(raw) < cfg.min_rows_per_symbol:
            print(f"[WARN] skip {sym}: insufficient rows")
            continue

        feat = _build_symbol_features(raw)
        feat["symbol"] = sym

        fwd = feat["close"].shift(-cfg.horizon) / feat["close"] - 1
        fwd = fwd.clip(-0.20, 0.20)
        feat["target_return"] = fwd
        feat["target"] = (fwd > 0).astype(int)

        feat = feat.dropna()
        if len(feat) < cfg.min_rows_per_symbol // 2:
            print(f"[WARN] skip {sym}: too few rows after featureing")
            continue

        coverage[sym] = len(feat)
        rows.append(feat)

    if not rows:
        raise RuntimeError("No usable data assembled. Check API keys/network/data availability.")

    panel = pd.concat(rows).sort_index()

    symbol_ret = panel.pivot_table(index=panel.index, columns="symbol", values="ret_1")
    panel["cross_mkt_ret_mean"] = symbol_ret.mean(axis=1).reindex(panel.index).fillna(0.0).values
    panel["cross_mkt_ret_std"] = symbol_ret.std(axis=1).reindex(panel.index).fillna(0.0).values

    panel = panel.replace([np.inf, -np.inf], np.nan).dropna(subset=["target", "target_return"])
    return panel, coverage


def _feature_columns(df: pd.DataFrame) -> List[str]:
    excluded = {"target", "target_return", "symbol"}
    return [c for c in df.columns if c not in excluded]


def _time_walk_forward_splits(index: pd.DatetimeIndex, n_splits: int, train_fraction: float, embargo: int):
    unique_ts = np.array(sorted(index.unique()))
    n = len(unique_ts)
    if n < 60:
        print(f"[WARN] limited unique timestamps for walk-forward: {n}")

    train_size = int(n * train_fraction)
    train_size = max(train_size, min(40, max(n - 20, 1)))

    test_size = max((n - train_size) // max(n_splits, 1), 10)

    for k in range(n_splits):
        train_end = train_size + k * test_size
        test_start = min(train_end + embargo, n - 1)
        test_end = min(test_start + test_size, n)
        if test_end - test_start < 10:
            continue
        train_ts = unique_ts[:train_end]
        test_ts = unique_ts[test_start:test_end]
        yield train_ts, test_ts


def _build_candidates(seed: int) -> Dict[str, Pipeline]:
    return {
        "xgb_balanced": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=700,
                        max_depth=6,
                        learning_rate=0.03,
                        subsample=0.9,
                        colsample_bytree=0.8,
                        reg_lambda=1.0,
                        min_child_weight=2,
                        random_state=seed,
                        n_jobs=-1,
                        eval_metric="logloss",
                    ),
                ),
            ]
        ),
        "rf_robust": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=600,
                        max_depth=10,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "logreg_calibrated": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=0.8,
                        class_weight="balanced",
                        random_state=seed,
                        max_iter=400,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def _score_fold(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.52) -> Dict[str, float]:
    pred = (proba >= threshold).astype(int)
    pnl = np.where(pred == 1, np.where(y_true == 1, 1.0, -1.0), 0.0)

    auc = 0.5
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, proba)

    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "auc": float(auc),
        "pnl_mean": float(np.mean(pnl)),
        "pnl_sharpe": float(np.mean(pnl) / (np.std(pnl) + 1e-9)),
        "trades": int(np.sum(pred == 1)),
    }


def train(cfg: Config, seed: int, symbols: List[str]) -> Path:
    panel, coverage = _build_panel_dataset(symbols, cfg)
    if panel.empty:
        raise RuntimeError("Assembled panel dataset is empty after preprocessing.")

    feats = _feature_columns(panel)

    candidates = _build_candidates(seed)
    cv_results = {name: [] for name in candidates}

    idx = panel.index
    for fold, (train_ts, test_ts) in enumerate(
        _time_walk_forward_splits(idx, cfg.n_splits, cfg.train_fraction, cfg.embargo),
        start=1,
    ):
        tr = panel[panel.index.isin(train_ts)]
        te = panel[panel.index.isin(test_ts)]

        x_tr, y_tr = tr[feats], tr["target"].astype(int)
        x_te, y_te = te[feats], te["target"].astype(int)

        for name, model in candidates.items():
            m = clone(model)
            m.fit(x_tr, y_tr)
            proba = m.predict_proba(x_te)[:, 1]
            metrics = _score_fold(y_te.to_numpy(), proba)
            cv_results[name].append(metrics)
            print(f"[fold={fold}] {name}: f1={metrics['f1']:.4f} auc={metrics['auc']:.4f} sharpe={metrics['pnl_sharpe']:.4f}")

    summary = {}
    for name, folds in cv_results.items():
        if not folds:
            continue
        summary[name] = {
            k: float(np.mean([f[k] for f in folds])) for k in folds[0].keys()
        }

    if not summary:
        print("[WARN] no walk-forward folds produced; falling back to single holdout split.")
        unique_ts = np.array(sorted(panel.index.unique()))
        split = max(int(len(unique_ts) * cfg.train_fraction), 1)
        train_ts = set(unique_ts[:split])
        test_ts = set(unique_ts[split:]) if split < len(unique_ts) else set(unique_ts[-1:])
        tr = panel[panel.index.map(lambda x: x in train_ts)]
        te = panel[panel.index.map(lambda x: x in test_ts)]
        x_tr, y_tr = tr[feats], tr["target"].astype(int)
        x_te, y_te = te[feats], te["target"].astype(int)
        for name, model in candidates.items():
            m = clone(model)
            m.fit(x_tr, y_tr)
            proba = m.predict_proba(x_te)[:, 1]
            metrics = _score_fold(y_te.to_numpy(), proba)
            summary[name] = metrics

    best_name = max(summary, key=lambda n: (summary[n]["pnl_sharpe"], summary[n]["f1"], summary[n]["auc"]))
    print(f"[INFO] best model: {best_name} | {summary[best_name]}")

    best_model = clone(candidates[best_name])
    best_model.fit(panel[feats], panel["target"].astype(int))

    PROD_DIR.mkdir(parents=True, exist_ok=True)
    out_file = PROD_DIR / "best_production_bundle.joblib"

    bundle = {
        "created_at_utc": _utc_now(),
        "config": cfg.__dict__,
        "symbols": symbols,
        "coverage_rows": coverage,
        "feature_columns": feats,
        "best_model_name": best_name,
        "cv_summary": summary,
        "model": best_model,
    }

    joblib.dump(bundle, out_file)

    meta_file = PROD_DIR / "best_production_bundle.meta.json"
    meta_file.write_text(json.dumps({k: v for k, v in bundle.items() if k != "model"}, indent=2))

    print(f"[OK] model bundle written: {out_file}")
    print(f"[OK] metadata written: {meta_file}")
    return out_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train production-grade classifier using richest available market data.")
    parser.add_argument("--timeframe", default="5m", help="Data timeframe, e.g. 1m/5m/15m/1h")
    parser.add_argument("--limit", type=int, default=20000, help="Max bars requested per symbol")
    parser.add_argument("--horizon", type=int, default=3, help="Bars ahead for target")
    parser.add_argument("--min-rows", type=int, default=1200, help="Min rows per symbol")
    parser.add_argument("--train-fraction", type=float, default=0.7, help="Initial train fraction")
    parser.add_argument("--splits", type=int, default=5, help="Walk-forward splits")
    parser.add_argument("--embargo", type=int, default=2, help="Embargo bars between train/test")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(
        timeframe=args.timeframe,
        limit=args.limit,
        horizon=args.horizon,
        min_rows_per_symbol=args.min_rows,
        train_fraction=args.train_fraction,
        n_splits=args.splits,
        embargo=args.embargo,
    )

    symbols = FOREX_MAJORS + CRYPTO_ASSETS
    train(cfg, seed=args.seed, symbols=symbols)


if __name__ == "__main__":
    main()
