#!/usr/bin/env python3
"""Strict OOS backtest for production bundle with realistic risk controls.

Highlights:
- Uses only real provider data (never synthetic fallback path).
- Uses model-selected decision threshold from training bundle.
- Cost-aware performance and per-symbol sanity reporting.
"""

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


def _provider_attempts(symbol: str, timeframe: str, limit: int) -> Iterable[Tuple[str, Any]]:
    yield "yfinance", lambda: MARKET.fetch_yfinance(symbol, timeframe, limit)
    yield "twelvedata", lambda: MARKET.fetch_twelvedata(symbol, timeframe, limit)
    yield "alphavantage", lambda: MARKET.fetch_alphavantage(symbol, timeframe, limit)


def fetch_real_data(symbol: str, timeframe: str, limit: int, min_rows: int) -> Tuple[pd.DataFrame, str]:
    errs: List[str] = []
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
    x["mom_10"] = (x["close"] - x["close"].shift(10)) / x["close"].shift(10).replace(0, np.nan)
    return x


def assemble_panel(symbols: List[str], cfg: BTConfig) -> Tuple[pd.DataFrame, Dict[str, str]]:
    rows: List[pd.DataFrame] = []
    provenance: Dict[str, str] = {}

    for sym in symbols:
        try:
            raw, provider = fetch_real_data(sym, cfg.timeframe, cfg.limit, cfg.min_rows_per_symbol)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] {sym}: {exc}")
            continue

        f = build_features(raw)
        fwd = f["close"].shift(-cfg.horizon) / f["close"] - 1
        f["target_return"] = fwd.clip(-0.05, 0.05)
        f["target"] = (f["target_return"] > 0).astype(int)
        f["symbol"] = sym
        f = f.replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(f)
        provenance[sym] = provider

    if not rows:
        raise RuntimeError("No real high-quality data available for backtest.")

    panel = pd.concat(rows).sort_index()
    pivot_ret1 = panel.pivot_table(index=panel.index, columns="symbol", values="ret_1")
    panel["cross_ret_mean"] = pivot_ret1.mean(axis=1).reindex(panel.index).fillna(0.0).values
    panel["cross_ret_std"] = pivot_ret1.std(axis=1).reindex(panel.index).fillna(0.0).values
    panel = panel.replace([np.inf, -np.inf], np.nan).dropna(subset=["target", "target_return"])
    return panel, provenance


def max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peaks = np.maximum.accumulate(equity)
    dd = (equity - peaks) / np.maximum(peaks, 1e-9)
    return float(dd.min())


def ann_factor(tf: str) -> float:
    m = {
        "1m": 365 * 24 * 60,
        "5m": 365 * 24 * 12,
        "15m": 365 * 24 * 4,
        "30m": 365 * 24 * 2,
        "1h": 365 * 24,
        "4h": 365 * 6,
        "1d": 365,
    }
    return float(m.get(tf, 365 * 24 * 12))


def run_backtest(bundle_path: Path, cfg: BTConfig) -> Dict[str, Any]:
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_path}")

    bundle = joblib.load(bundle_path)
    model = bundle["model"]
    symbols = bundle["symbols"]
    feats = bundle["feature_columns"]

    threshold = cfg.threshold if cfg.threshold is not None else float(bundle.get("best_threshold", 0.55))

    panel, provenance = assemble_panel(symbols, cfg)

    ts = np.array(sorted(panel.index.unique()))
    split = int(len(ts) * 0.7)
    oos_ts = set(ts[split:])
    oos = panel[panel.index.map(lambda x: x in oos_ts)].copy()
    if oos.empty:
        raise RuntimeError("No OOS rows available")

    X = oos[feats]
    y = oos["target"].astype(int).to_numpy()
    ret = oos["target_return"].to_numpy()

    proba = model.predict_proba(X)[:, 1]
    signal = (proba >= threshold).astype(int)

    costs = (cfg.spread_bps + cfg.slippage_bps + cfg.fee_bps) / 10000.0
    strat_ret = np.where(signal == 1, ret - costs, 0.0)

    # conservative clip to avoid reporting absurd compounding outliers from bad data slices
    strat_ret = np.clip(strat_ret, -0.20, 0.20)
    ret = np.clip(ret, -0.20, 0.20)

    equity = np.cumprod(1 + strat_ret)
    market = np.cumprod(1 + ret)

    trades = int(np.sum(signal == 1))
    wins = int(np.sum(strat_ret > 0))

    mean_r = float(np.mean(strat_ret))
    std_r = float(np.std(strat_ret))
    sharpe = float((mean_r / (std_r + 1e-12)) * math.sqrt(ann_factor(cfg.timeframe)))

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
        sr = np.where(sg == 1, rs - costs, 0.0)
        sr = np.clip(sr, -0.20, 0.20)
        eq = np.cumprod(1 + sr)

        by_symbol[sym] = {
            "rows": float(len(sdf)),
            "trades": float(np.sum(sg == 1)),
            "return": float(eq[-1] - 1) if len(eq) else 0.0,
            "win_rate": float(np.mean(sr[sg == 1] > 0)) if np.sum(sg == 1) else 0.0,
        }

    result["by_symbol"] = by_symbol
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Strict production OOS backtest")
    p.add_argument("--bundle", default="models/production/best_production_bundle.joblib")
    p.add_argument("--timeframe", default="5m")
    p.add_argument("--limit", type=int, default=10000)
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--spread-bps", type=float, default=2.0)
    p.add_argument("--slippage-bps", type=float, default=1.0)
    p.add_argument("--fee-bps", type=float, default=0.5)
    p.add_argument("--min-rows", type=int, default=900)
    p.add_argument("--output", default="models/production/latest_backtest.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = BTConfig(
        timeframe=args.timeframe,
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
