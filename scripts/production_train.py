#!/usr/bin/env python3
"""Institutional-style training pipeline with strict data-quality gates.

Key design goals:
- NEVER train on synthetic fallback candles.
- Fetch richest available data from multiple providers with strict provenance.
- Validate data quality before modeling.
- Tune models with walk-forward CV and trading-oriented objective.
- Persist reproducible bundle with metadata and thresholds.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CRYPTO_ASSETS, FOREX_MAJORS


# -------------------------
# Dynamic import of market_data without importing app package side-effects
# -------------------------
def _load_market_module():
    module_path = ROOT / "app" / "market_data.py"
    spec = importlib.util.spec_from_file_location("neko_market_data", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load market_data module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MARKET = _load_market_module()


MODEL_DIR = ROOT / "models" / "production"


@dataclass
class TrainConfig:
    timeframe: str
    limit: int
    horizon: int
    min_rows_per_symbol: int
    min_symbols: int
    n_splits: int
    embargo: int
    train_fraction: float
    random_state: int


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    required = ["open", "high", "low", "close", "volume"]
    if not set(required).issubset(df.columns):
        raise ValueError(f"Missing OHLCV columns, got: {list(df.columns)}")
    out = df[required].copy()
    out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
    out = out.dropna()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def _candle_quality_checks(df: pd.DataFrame, symbol: str, min_rows: int) -> None:
    if len(df) < min_rows:
        raise ValueError(f"{symbol}: insufficient rows ({len(df)} < {min_rows})")

    if (df["close"] <= 0).any() or (df[["open", "high", "low"]] <= 0).any().any():
        raise ValueError(f"{symbol}: non-positive prices detected")

    if (df["high"] < df["low"]).any():
        raise ValueError(f"{symbol}: high<low anomaly")

    flat_ratio = float((df["close"].diff().abs() < 1e-12).mean())
    if flat_ratio > 0.98:
        raise ValueError(f"{symbol}: suspiciously flat series ({flat_ratio:.3f})")

    nan_ratio = float(df.isna().mean().mean())
    if nan_ratio > 0.01:
        raise ValueError(f"{symbol}: too many NaN values ({nan_ratio:.4f})")


def _provider_attempts(symbol: str, timeframe: str, limit: int) -> Iterable[Tuple[str, Any]]:
    # Intentionally avoid MARKET.fetch_market_data because it can synthesize random fallback candles.
    yield "yfinance", lambda: MARKET.fetch_yfinance(symbol, timeframe, limit)
    yield "twelvedata", lambda: MARKET.fetch_twelvedata(symbol, timeframe, limit)
    yield "alphavantage", lambda: MARKET.fetch_alphavantage(symbol, timeframe, limit)


def fetch_real_data(symbol: str, timeframe: str, limit: int, min_rows: int) -> Tuple[pd.DataFrame, str]:
    errors: List[str] = []
    for provider, fn in _provider_attempts(symbol, timeframe, limit):
        try:
            df = _ensure_ohlcv(fn())
            _candle_quality_checks(df, symbol, min_rows)
            return df, provider
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{provider}: {exc}")

    raise RuntimeError(f"{symbol}: all providers failed quality checks -> {' | '.join(errors)}")


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ru = up.ewm(alpha=1 / period, adjust=False).mean()
    rd = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = ru / rd.replace(0, np.nan)
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


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["ret_1"] = x["close"].pct_change(1)
    x["ret_3"] = x["close"].pct_change(3)
    x["ret_6"] = x["close"].pct_change(6)
    x["ret_12"] = x["close"].pct_change(12)

    x["log_ret_1"] = np.log(x["close"]).diff(1)
    x["log_ret_3"] = np.log(x["close"]).diff(3)

    x["ema_8"] = x["close"].ewm(span=8, adjust=False).mean()
    x["ema_21"] = x["close"].ewm(span=21, adjust=False).mean()
    x["ema_55"] = x["close"].ewm(span=55, adjust=False).mean()
    x["ema_spread_8_21"] = (x["ema_8"] - x["ema_21"]) / x["close"].replace(0, np.nan)
    x["ema_spread_21_55"] = (x["ema_21"] - x["ema_55"]) / x["close"].replace(0, np.nan)

    x["rsi_14"] = _rsi(x["close"], 14)
    x["atr_14"] = _atr(x, 14)
    x["atr_pct"] = x["atr_14"] / x["close"].replace(0, np.nan)
    x["hl_range"] = (x["high"] - x["low"]) / x["close"].replace(0, np.nan)

    vma20 = x["volume"].rolling(20).mean()
    vstd20 = x["volume"].rolling(20).std()
    x["vol_z20"] = (x["volume"] - vma20) / vstd20.replace(0, np.nan)

    mom10 = x["close"] - x["close"].shift(10)
    x["mom_10"] = mom10 / x["close"].shift(10).replace(0, np.nan)

    return x


def assemble_panel(cfg: TrainConfig, symbols: List[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows: List[pd.DataFrame] = []
    provenance: Dict[str, str] = {}
    coverage: Dict[str, int] = {}

    for sym in symbols:
        try:
            raw, provider = fetch_real_data(sym, cfg.timeframe, cfg.limit, cfg.min_rows_per_symbol)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] {sym}: {exc}")
            continue

        feat = build_features(raw)
        fwd = feat["close"].shift(-cfg.horizon) / feat["close"] - 1
        feat["target_return"] = fwd.clip(-0.05, 0.05)
        feat["target"] = (feat["target_return"] > 0).astype(int)
        feat["symbol"] = sym
        feat = feat.replace([np.inf, -np.inf], np.nan).dropna()

        if len(feat) < cfg.min_rows_per_symbol // 2:
            print(f"[WARN] {sym}: too few rows after feature build ({len(feat)})")
            continue

        provenance[sym] = provider
        coverage[sym] = int(len(feat))
        rows.append(feat)

    if len(rows) < cfg.min_symbols:
        raise RuntimeError(
            f"Insufficient high-quality symbols ({len(rows)} < {cfg.min_symbols}). "
            "Refusing to train to protect model quality."
        )

    panel = pd.concat(rows).sort_index()

    pivot_ret1 = panel.pivot_table(index=panel.index, columns="symbol", values="ret_1")
    panel["cross_ret_mean"] = pivot_ret1.mean(axis=1).reindex(panel.index).fillna(0.0).values
    panel["cross_ret_std"] = pivot_ret1.std(axis=1).reindex(panel.index).fillna(0.0).values

    panel = panel.replace([np.inf, -np.inf], np.nan).dropna(subset=["target", "target_return"])

    meta = {
        "provenance": provenance,
        "coverage_rows": coverage,
        "rows_total": int(len(panel)),
        "symbols_used": sorted(list(coverage.keys())),
    }
    return panel, meta


def feature_columns(df: pd.DataFrame) -> List[str]:
    excluded = {"target", "target_return", "symbol"}
    return [c for c in df.columns if c not in excluded]


def walk_forward_splits(index: pd.DatetimeIndex, n_splits: int, train_fraction: float, embargo: int):
    ts = np.array(sorted(index.unique()))
    n = len(ts)
    if n < 120:
        raise RuntimeError("Too few unique timestamps for robust walk-forward CV")

    train_size = max(int(n * train_fraction), 80)
    test_size = max((n - train_size) // max(n_splits, 1), 20)

    for i in range(n_splits):
        train_end = train_size + i * test_size
        test_start = min(train_end + embargo, n - 1)
        test_end = min(test_start + test_size, n)
        if test_end - test_start < 10:
            continue
        yield set(ts[:train_end]), set(ts[test_start:test_end])


def build_candidates(seed: int) -> Dict[str, Pipeline]:
    return {
        "xgb": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=900,
                        max_depth=6,
                        learning_rate=0.025,
                        subsample=0.9,
                        colsample_bytree=0.85,
                        reg_lambda=1.2,
                        min_child_weight=2,
                        random_state=seed,
                        n_jobs=-1,
                        eval_metric="logloss",
                    ),
                ),
            ]
        ),
        "rf": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=900,
                        max_depth=10,
                        min_samples_leaf=3,
                        class_weight="balanced_subsample",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "logreg": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=0.7,
                        class_weight="balanced",
                        random_state=seed,
                        max_iter=500,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def trading_objective(y_true: np.ndarray, proba: np.ndarray, threshold: float, cost_bps: float) -> Dict[str, float]:
    pred = (proba >= threshold).astype(int)
    costs = cost_bps / 10000.0

    pnl = np.where(pred == 1, np.where(y_true == 1, 1.0 - costs, -1.0 - costs), 0.0)
    trade_count = int(np.sum(pred == 1))
    win_rate = float(np.mean(pnl[pred == 1] > 0)) if trade_count else 0.0

    auc = 0.5
    if len(np.unique(y_true)) > 1:
        auc = float(roc_auc_score(y_true, proba))

    mean_pnl = float(np.mean(pnl))
    sharpe = float(mean_pnl / (np.std(pnl) + 1e-9))

    # conservative combined objective
    score = (0.55 * sharpe) + (0.35 * auc) + (0.10 * (win_rate - 0.5))

    return {
        "score": score,
        "auc": auc,
        "sharpe": sharpe,
        "mean_pnl": mean_pnl,
        "trade_count": float(trade_count),
        "win_rate": win_rate,
    }


def threshold_search(y_true: np.ndarray, proba: np.ndarray, cost_bps: float) -> Tuple[float, Dict[str, float]]:
    candidates = np.arange(0.52, 0.71, 0.01)
    best_t, best_m = 0.55, None
    for t in candidates:
        m = trading_objective(y_true, proba, float(t), cost_bps)
        if best_m is None or m["score"] > best_m["score"]:
            best_t, best_m = float(t), m
    assert best_m is not None
    return best_t, best_m


def train(cfg: TrainConfig, symbols: List[str], cost_bps: float) -> Path:
    panel, data_meta = assemble_panel(cfg, symbols)
    feats = feature_columns(panel)

    candidates = build_candidates(cfg.random_state)
    folds_by_model: Dict[str, List[Dict[str, float]]] = {k: [] for k in candidates}

    for fold, (train_ts, test_ts) in enumerate(
        walk_forward_splits(panel.index, cfg.n_splits, cfg.train_fraction, cfg.embargo),
        start=1,
    ):
        tr = panel[panel.index.map(lambda x: x in train_ts)]
        te = panel[panel.index.map(lambda x: x in test_ts)]
        x_tr, y_tr = tr[feats], tr["target"].astype(int).to_numpy()
        x_te, y_te = te[feats], te["target"].astype(int).to_numpy()

        for name, model in candidates.items():
            m = clone(model)
            m.fit(x_tr, y_tr)
            p = m.predict_proba(x_te)[:, 1]
            thr, metrics = threshold_search(y_te, p, cost_bps)
            metrics["threshold"] = thr
            folds_by_model[name].append(metrics)
            print(
                f"[fold={fold}] {name}: score={metrics['score']:.4f} "
                f"auc={metrics['auc']:.4f} sharpe={metrics['sharpe']:.4f} "
                f"thr={thr:.2f}"
            )

    summary: Dict[str, Dict[str, float]] = {}
    for name, fold_metrics in folds_by_model.items():
        if not fold_metrics:
            continue
        keys = fold_metrics[0].keys()
        summary[name] = {k: float(np.mean([m[k] for m in fold_metrics])) for k in keys}

    if not summary:
        raise RuntimeError("No valid folds completed. Increase data quality/size.")

    best_name = max(summary, key=lambda n: summary[n]["score"])
    best_threshold = summary[best_name]["threshold"]
    print(f"[INFO] selected model={best_name} threshold={best_threshold:.2f}")

    best_model = clone(candidates[best_name])
    best_model.fit(panel[feats], panel["target"].astype(int).to_numpy())

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bundle_path = MODEL_DIR / "best_production_bundle.joblib"

    bundle = {
        "created_at_utc": utc_now(),
        "config": cfg.__dict__,
        "symbols": data_meta["symbols_used"],
        "feature_columns": feats,
        "provenance": data_meta["provenance"],
        "coverage_rows": data_meta["coverage_rows"],
        "rows_total": data_meta["rows_total"],
        "selection_cost_bps": cost_bps,
        "best_model_name": best_name,
        "best_threshold": best_threshold,
        "cv_summary": summary,
        "model": best_model,
    }

    joblib.dump(bundle, bundle_path)

    meta_path = MODEL_DIR / "best_production_bundle.meta.json"
    meta_path.write_text(json.dumps({k: v for k, v in bundle.items() if k != "model"}, indent=2))

    print(f"[OK] bundle: {bundle_path}")
    print(f"[OK] metadata: {meta_path}")
    return bundle_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Strict production training pipeline (real data only).")
    p.add_argument("--timeframe", default="5m")
    p.add_argument("--limit", type=int, default=10000)
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument("--min-rows", type=int, default=1200)
    p.add_argument("--min-symbols", type=int, default=8)
    p.add_argument("--splits", type=int, default=5)
    p.add_argument("--embargo", type=int, default=2)
    p.add_argument("--train-fraction", type=float, default=0.7)
    p.add_argument("--cost-bps", type=float, default=3.5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        timeframe=args.timeframe,
        limit=args.limit,
        horizon=args.horizon,
        min_rows_per_symbol=args.min_rows,
        min_symbols=args.min_symbols,
        n_splits=args.splits,
        embargo=args.embargo,
        train_fraction=args.train_fraction,
        random_state=args.seed,
    )

    symbols = FOREX_MAJORS + CRYPTO_ASSETS
    train(cfg, symbols, cost_bps=args.cost_bps)


if __name__ == "__main__":
    main()
