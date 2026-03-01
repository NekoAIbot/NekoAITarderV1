#!/usr/bin/env python3
"""Institutional-style training pipeline with strict data-quality gates."""

from __future__ import annotations

import argparse
import importlib.util
import json
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
    data_mode: str
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


def load_local_parquet(symbol: str, timeframe: str) -> pd.DataFrame | None:
    fx_dir = ROOT / "data" / "raw" / "fx"
    crypto_dir = ROOT / "data" / "raw" / "crypto"

    if symbol.endswith("USDT"):
        f = crypto_dir / f"{symbol}_{timeframe}.parquet"
        return pd.read_parquet(f) if f.exists() else None

    files = sorted(fx_dir.glob(f"{symbol}_M1_*.parquet"))
    if not files:
        return None
    return pd.concat([pd.read_parquet(fp) for fp in files]).sort_index()


def _provider_attempts(symbol: str, timeframe: str, limit: int) -> Iterable[Tuple[str, Any]]:
    yield "yfinance", lambda: MARKET.fetch_yfinance(symbol, timeframe, limit)
    yield "twelvedata", lambda: MARKET.fetch_twelvedata(symbol, timeframe, limit)
    yield "alphavantage", lambda: MARKET.fetch_alphavantage(symbol, timeframe, limit)


def fetch_real_data(symbol: str, timeframe: str, limit: int, min_rows: int, data_mode: str) -> Tuple[pd.DataFrame, str]:
    errors: List[str] = []

    local_df = load_local_parquet(symbol, timeframe)
    if local_df is not None:
        try:
            df = _ensure_ohlcv(local_df)
            _candle_quality_checks(df, symbol, min_rows)
            return df.tail(limit), "local_parquet"
        except Exception as exc:  # noqa: BLE001
            errors.append(f"local_parquet: {exc}")

    if data_mode == "parquet-only":
        msg = " | ".join(errors) if errors else "no local parquet file"
        raise RuntimeError(f"{symbol}: {msg}")

    for provider, fn in _provider_attempts(symbol, timeframe, limit):
        try:
            df = _ensure_ohlcv(fn())
            _candle_quality_checks(df, symbol, min_rows)
            return df, provider
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{provider}: {exc}")

    raise RuntimeError(f"{symbol}: all providers failed -> {' | '.join(errors)}")


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
        [(df["high"] - df["low"]).abs(), (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
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
    x["ema_8"] = x["close"].ewm(span=8, adjust=False).mean()
    x["ema_21"] = x["close"].ewm(span=21, adjust=False).mean()
    x["ema_55"] = x["close"].ewm(span=55, adjust=False).mean()
    x["ema_spread_8_21"] = (x["ema_8"] - x["ema_21"]) / x["close"].replace(0, np.nan)
    x["ema_spread_21_55"] = (x["ema_21"] - x["ema_55"]) / x["close"].replace(0, np.nan)
    x["rsi_14"] = _rsi(x["close"], 14)
    x["atr_14"] = _atr(x, 14)
    x["atr_pct"] = x["atr_14"] / x["close"].replace(0, np.nan)
    x["hl_range"] = (x["high"] - x["low"]) / x["close"].replace(0, np.nan)
    x["vol_z20"] = (x["volume"] - x["volume"].rolling(20).mean()) / x["volume"].rolling(20).std().replace(0, np.nan)
    x["mom_10"] = (x["close"] - x["close"].shift(10)) / x["close"].shift(10).replace(0, np.nan)
    return x


def assemble_panel(cfg: TrainConfig, symbols: List[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows: List[pd.DataFrame] = []
    provenance: Dict[str, str] = {}
    coverage: Dict[str, int] = {}

    for sym in symbols:
        try:
            raw, provider = fetch_real_data(sym, cfg.timeframe, cfg.limit, cfg.min_rows_per_symbol, cfg.data_mode)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] {sym}: {exc}")
            continue

        feat = build_features(raw)
        feat["target_return"] = (feat["close"].shift(-cfg.horizon) / feat["close"] - 1).clip(-0.05, 0.05)
        feat["target"] = (feat["target_return"] > 0).astype(int)
        feat["symbol"] = sym
        feat = feat.replace([np.inf, -np.inf], np.nan).dropna()
        if len(feat) < cfg.min_rows_per_symbol // 2:
            continue

        rows.append(feat)
        provenance[sym] = provider
        coverage[sym] = int(len(feat))

    if len(rows) < cfg.min_symbols:
        raise RuntimeError(f"Insufficient high-quality symbols ({len(rows)} < {cfg.min_symbols})")

    panel = pd.concat(rows).sort_index()
    p = panel.pivot_table(index=panel.index, columns="symbol", values="ret_1")
    panel["cross_ret_mean"] = p.mean(axis=1).reindex(panel.index).fillna(0.0).values
    panel["cross_ret_std"] = p.std(axis=1).reindex(panel.index).fillna(0.0).values
    panel = panel.replace([np.inf, -np.inf], np.nan).dropna(subset=["target", "target_return"])

    return panel, {"provenance": provenance, "coverage_rows": coverage, "rows_total": int(len(panel)), "symbols_used": sorted(coverage)}


def feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in {"target", "target_return", "symbol"}]


def walk_forward_splits(index: pd.DatetimeIndex, n_splits: int, train_fraction: float, embargo: int):
    ts = np.array(sorted(index.unique()))
    n = len(ts)
    if n < 120:
        raise RuntimeError("Too few unique timestamps")
    train_size = max(int(n * train_fraction), 80)
    test_size = max((n - train_size) // max(n_splits, 1), 20)
    for i in range(n_splits):
        tr_end = train_size + i * test_size
        te_start = min(tr_end + embargo, n - 1)
        te_end = min(te_start + test_size, n)
        if te_end - te_start < 10:
            continue
        yield set(ts[:tr_end]), set(ts[te_start:te_end])


def build_candidates(seed: int) -> Dict[str, Pipeline]:
    return {
        "xgb": Pipeline([("imp", SimpleImputer(strategy="median")), ("m", XGBClassifier(n_estimators=700, max_depth=6, learning_rate=0.03, subsample=0.9, colsample_bytree=0.85, random_state=seed, n_jobs=-1, eval_metric="logloss"))]),
        "rf": Pipeline([("imp", SimpleImputer(strategy="median")), ("m", RandomForestClassifier(n_estimators=700, max_depth=10, min_samples_leaf=3, class_weight="balanced_subsample", random_state=seed, n_jobs=-1))]),
        "logreg": Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("m", LogisticRegression(C=0.7, class_weight="balanced", random_state=seed, max_iter=500, n_jobs=-1))]),
    }


def trading_objective(y_true: np.ndarray, proba: np.ndarray, threshold: float, cost_bps: float) -> Dict[str, float]:
    pred = (proba >= threshold).astype(int)
    costs = cost_bps / 10000.0
    pnl = np.where(pred == 1, np.where(y_true == 1, 1.0 - costs, -1.0 - costs), 0.0)
    auc = float(roc_auc_score(y_true, proba)) if len(np.unique(y_true)) > 1 else 0.5
    sharpe = float(np.mean(pnl) / (np.std(pnl) + 1e-9))
    trade_count = int(np.sum(pred == 1))
    win_rate = float(np.mean(pnl[pred == 1] > 0)) if trade_count else 0.0
    score = 0.55 * sharpe + 0.35 * auc + 0.10 * (win_rate - 0.5)
    return {"score": score, "auc": auc, "sharpe": sharpe, "trade_count": float(trade_count), "win_rate": win_rate}


def threshold_search(y_true: np.ndarray, proba: np.ndarray, cost_bps: float) -> Tuple[float, Dict[str, float]]:
    best_t, best_m = 0.55, None
    for t in np.arange(0.52, 0.71, 0.01):
        m = trading_objective(y_true, proba, float(t), cost_bps)
        if best_m is None or m["score"] > best_m["score"]:
            best_t, best_m = float(t), m
    return best_t, best_m


def train(cfg: TrainConfig, symbols: List[str], cost_bps: float) -> Path:
    panel, meta = assemble_panel(cfg, symbols)
    feats = feature_columns(panel)
    candidates = build_candidates(cfg.random_state)
    folds = {k: [] for k in candidates}

    for fold, (tr_ts, te_ts) in enumerate(walk_forward_splits(panel.index, cfg.n_splits, cfg.train_fraction, cfg.embargo), start=1):
        tr = panel[panel.index.map(lambda x: x in tr_ts)]
        te = panel[panel.index.map(lambda x: x in te_ts)]
        x_tr, y_tr = tr[feats], tr["target"].astype(int).to_numpy()
        x_te, y_te = te[feats], te["target"].astype(int).to_numpy()

        for name, model in candidates.items():
            m = clone(model)
            m.fit(x_tr, y_tr)
            p = m.predict_proba(x_te)[:, 1]
            thr, metrics = threshold_search(y_te, p, cost_bps)
            metrics["threshold"] = thr
            folds[name].append(metrics)
            print(f"[fold={fold}] {name}: score={metrics['score']:.4f} auc={metrics['auc']:.4f} thr={thr:.2f}")

    summary = {name: {k: float(np.mean([m[k] for m in vals])) for k in vals[0]} for name, vals in folds.items() if vals}
    if not summary:
        raise RuntimeError("No valid folds completed")

    best_name = max(summary, key=lambda n: summary[n]["score"])
    best_threshold = summary[best_name]["threshold"]
    best_model = clone(candidates[best_name])
    best_model.fit(panel[feats], panel["target"].astype(int).to_numpy())

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out = MODEL_DIR / "best_production_bundle.joblib"
    bundle = {
        "created_at_utc": utc_now(),
        "config": cfg.__dict__,
        "symbols": meta["symbols_used"],
        "feature_columns": feats,
        "provenance": meta["provenance"],
        "coverage_rows": meta["coverage_rows"],
        "rows_total": meta["rows_total"],
        "selection_cost_bps": cost_bps,
        "best_model_name": best_name,
        "best_threshold": best_threshold,
        "cv_summary": summary,
        "model": best_model,
    }
    joblib.dump(bundle, out)
    (MODEL_DIR / "best_production_bundle.meta.json").write_text(json.dumps({k: v for k, v in bundle.items() if k != "model"}, indent=2))
    print(f"[OK] bundle: {out}")
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Strict production training pipeline")
    p.add_argument("--timeframe", default="1m")
    p.add_argument("--data-mode", choices=["parquet-only", "parquet-or-api"], default="parquet-only")
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


def main():
    args = parse_args()
    cfg = TrainConfig(
        timeframe=args.timeframe,
        data_mode=args.data_mode,
        limit=args.limit,
        horizon=args.horizon,
        min_rows_per_symbol=args.min_rows,
        min_symbols=args.min_symbols,
        n_splits=args.splits,
        embargo=args.embargo,
        train_fraction=args.train_fraction,
        random_state=args.seed,
    )
    train(cfg, FOREX_MAJORS + CRYPTO_ASSETS, cost_bps=args.cost_bps)


if __name__ == "__main__":
    main()
