#!/usr/bin/env python3
"""Strict OOS backtest using local parquet (preferred) or API data."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_market_module():
    module_path = ROOT / "app" / "market_data.py"
    spec = importlib.util.spec_from_file_location("neko_market_data", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load market_data module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MARKET = _load_market_module()


@dataclass
class BTConfig:
    timeframe: str
    data_mode: str
    limit: int
    horizon: int
    threshold: float | None
    spread_bps: float
    slippage_bps: float
    fee_bps: float
    min_rows_per_symbol: int


def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    req = ["open", "high", "low", "close", "volume"]
    if not set(req).issubset(df.columns):
        raise ValueError("OHLCV columns missing")
    out = df[req].copy()
    out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
    out = out.dropna()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


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
    errs: List[str] = []

    local_df = load_local_parquet(symbol, timeframe)
    if local_df is not None:
        try:
            df = _ensure_ohlcv(local_df)
            if len(df) >= min_rows and (df["close"] > 0).all():
                return df.tail(limit), "local_parquet"
            raise ValueError(f"quality fail rows={len(df)}")
        except Exception as exc:  # noqa: BLE001
            errs.append(f"local_parquet:{exc}")

    if data_mode == "parquet-only":
        raise RuntimeError(f"{symbol}: {' | '.join(errs) if errs else 'missing parquet'}")

    for provider, fn in _provider_attempts(symbol, timeframe, limit):
        try:
            df = _ensure_ohlcv(fn())
            if len(df) < min_rows:
                raise ValueError(f"too few rows ({len(df)}<{min_rows})")
            if (df["close"] <= 0).any():
                raise ValueError("non-positive close")
            return df, provider
        except Exception as exc:  # noqa: BLE001
            errs.append(f"{provider}:{exc}")

    raise RuntimeError(f"{symbol}: data fetch failed ({' | '.join(errs)})")


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
    tr = pd.concat([(df["high"] - df["low"]).abs(), (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()], axis=1).max(axis=1)
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


def assemble_panel(symbols: List[str], cfg: BTConfig) -> Tuple[pd.DataFrame, Dict[str, str]]:
    rows: List[pd.DataFrame] = []
    provenance: Dict[str, str] = {}
    for sym in symbols:
        try:
            raw, provider = fetch_real_data(sym, cfg.timeframe, cfg.limit, cfg.min_rows_per_symbol, cfg.data_mode)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] {sym}: {exc}")
            continue

        f = build_features(raw)
        f["target_return"] = (f["close"].shift(-cfg.horizon) / f["close"] - 1).clip(-0.05, 0.05)
        f["target"] = (f["target_return"] > 0).astype(int)
        f["symbol"] = sym
        f = f.replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(f)
        provenance[sym] = provider

    if not rows:
        raise RuntimeError("No real high-quality data available for backtest")

    panel = pd.concat(rows).sort_index()
    p = panel.pivot_table(index=panel.index, columns="symbol", values="ret_1")
    panel["cross_ret_mean"] = p.mean(axis=1).reindex(panel.index).fillna(0.0).values
    panel["cross_ret_std"] = p.std(axis=1).reindex(panel.index).fillna(0.0).values
    panel = panel.replace([np.inf, -np.inf], np.nan).dropna(subset=["target", "target_return"])
    return panel, provenance


def max_drawdown(equity: np.ndarray) -> float:
    peaks = np.maximum.accumulate(equity) if len(equity) else np.array([1.0])
    dd = (equity - peaks) / np.maximum(peaks, 1e-9) if len(equity) else np.array([0.0])
    return float(dd.min())


def ann_factor(tf: str) -> float:
    return float({"1m": 365 * 24 * 60, "5m": 365 * 24 * 12, "15m": 365 * 24 * 4, "30m": 365 * 24 * 2, "1h": 365 * 24, "4h": 365 * 6, "1d": 365}.get(tf, 365 * 24 * 12))


def run_backtest(bundle_path: Path, cfg: BTConfig) -> Dict[str, Any]:
    bundle = joblib.load(bundle_path)
    model = bundle["model"]
    symbols = bundle["symbols"]
    feats = bundle["feature_columns"]
    threshold = cfg.threshold if cfg.threshold is not None else float(bundle.get("best_threshold", 0.55))

    panel, provenance = assemble_panel(symbols, cfg)
    ts = np.array(sorted(panel.index.unique()))
    oos = panel[panel.index.map(lambda x: x in set(ts[int(len(ts) * 0.7):]))].copy()
    if oos.empty:
        raise RuntimeError("No OOS rows available")

    X = oos[feats]
    ret = np.clip(oos["target_return"].to_numpy(), -0.20, 0.20)
    proba = model.predict_proba(X)[:, 1]
    signal = (proba >= threshold).astype(int)

    costs = (cfg.spread_bps + cfg.slippage_bps + cfg.fee_bps) / 10000.0
    strat_ret = np.where(signal == 1, ret - costs, 0.0)
    strat_ret = np.clip(strat_ret, -0.20, 0.20)

    equity = np.cumprod(1 + strat_ret)
    market = np.cumprod(1 + ret)
    trades = int(np.sum(signal == 1))
    wins = int(np.sum(strat_ret > 0))
    sharpe = float((np.mean(strat_ret) / (np.std(strat_ret) + 1e-12)) * math.sqrt(ann_factor(cfg.timeframe)))

    result: Dict[str, Any] = {
        "model_name": bundle.get("best_model_name"),
        "threshold": threshold,
        "rows_oos": int(len(oos)),
        "trades": trades,
        "win_rate": float(wins / trades) if trades else 0.0,
        "avg_trade_return": float(np.mean(strat_ret[signal == 1])) if trades else 0.0,
        "total_return": float(equity[-1] - 1) if len(equity) else 0.0,
        "market_return": float(market[-1] - 1) if len(market) else 0.0,
        "max_drawdown": max_drawdown(equity),
        "sharpe": sharpe,
        "costs_total_bps": cfg.spread_bps + cfg.slippage_bps + cfg.fee_bps,
        "symbols": symbols,
        "provenance": provenance,
    }

    by_symbol: Dict[str, Dict[str, float]] = {}
    for sym, sdf in oos.groupby("symbol"):
        xs = sdf[feats]
        rs = np.clip(sdf["target_return"].to_numpy(), -0.20, 0.20)
        ps = model.predict_proba(xs)[:, 1]
        sg = (ps >= threshold).astype(int)
        sr = np.clip(np.where(sg == 1, rs - costs, 0.0), -0.20, 0.20)
        eq = np.cumprod(1 + sr)
        by_symbol[sym] = {
            "rows": float(len(sdf)),
            "trades": float(np.sum(sg == 1)),
            "return": float(eq[-1] - 1) if len(eq) else 0.0,
            "win_rate": float(np.mean(sr[sg == 1] > 0)) if np.sum(sg == 1) else 0.0,
        }

    result["by_symbol"] = by_symbol
    return result


def parse_args():
    p = argparse.ArgumentParser(description="Strict production OOS backtest")
    p.add_argument("--bundle", default="models/production/best_production_bundle.joblib")
    p.add_argument("--timeframe", default="1m")
    p.add_argument("--data-mode", choices=["parquet-only", "parquet-or-api"], default="parquet-only")
    p.add_argument("--limit", type=int, default=10000)
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--spread-bps", type=float, default=2.0)
    p.add_argument("--slippage-bps", type=float, default=1.0)
    p.add_argument("--fee-bps", type=float, default=0.5)
    p.add_argument("--min-rows", type=int, default=900)
    p.add_argument("--output", default="models/production/latest_backtest.json")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = BTConfig(
        timeframe=args.timeframe,
        data_mode=args.data_mode,
        limit=args.limit,
        horizon=args.horizon,
        threshold=args.threshold,
        spread_bps=args.spread_bps,
        slippage_bps=args.slippage_bps,
        fee_bps=args.fee_bps,
        min_rows_per_symbol=args.min_rows,
    )

    result = run_backtest(Path(args.bundle), cfg)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"[OK] saved report: {out}")


if __name__ == "__main__":
    main()
