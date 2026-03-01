#!/usr/bin/env python3
"""Production-grade backtest for the production model bundle.

- Loads saved model bundle.
- Rebuilds exact feature pipeline across symbols.
- Simulates probability-threshold trading with spread/slippage/fees.
- Outputs detailed metrics + per-symbol breakdown.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


@dataclass
class BTConfig:
    timeframe: str
    limit: int
    horizon: int
    threshold: float
    spread_bps: float
    slippage_bps: float
    fee_bps: float


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
    x["mom_10"] = (x["close"] - x["close"].shift(10)) / x["close"].shift(10).replace(0, np.nan)
    return x


def _load_bundle(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"bundle not found: {path}")
    return joblib.load(path)


def _assemble(symbols: List[str], cfg: BTConfig) -> pd.DataFrame:
    rows = []
    for sym in symbols:
        try:
            df = fetch_market_data(sym, timeframe=cfg.timeframe, limit=cfg.limit)
        except Exception as exc:
            print(f"[WARN] data fetch failed for {sym}: {exc}")
            continue

        if df is None or len(df) < 500:
            continue

        d = df.copy()
        d.index = pd.to_datetime(d.index, utc=True, errors="coerce")
        d = d.dropna()
        d = d[~d.index.duplicated(keep="last")].sort_index()

        f = _build_symbol_features(d)
        f["symbol"] = sym
        fwd = f["close"].shift(-cfg.horizon) / f["close"] - 1
        f["target_return"] = fwd.clip(-0.20, 0.20)
        f["target"] = (f["target_return"] > 0).astype(int)
        f = f.dropna()
        rows.append(f)

    if not rows:
        raise RuntimeError("No data for backtest.")

    panel = pd.concat(rows).sort_index()
    symbol_ret = panel.pivot_table(index=panel.index, columns="symbol", values="ret_1")
    panel["cross_mkt_ret_mean"] = symbol_ret.mean(axis=1).reindex(panel.index).fillna(0.0).values
    panel["cross_mkt_ret_std"] = symbol_ret.std(axis=1).reindex(panel.index).fillna(0.0).values
    panel = panel.replace([np.inf, -np.inf], np.nan).dropna(subset=["target", "target_return"])
    return panel


def _drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peaks = np.maximum.accumulate(equity)
    dd = (equity - peaks) / np.maximum(peaks, 1e-9)
    return float(dd.min())


def _annualization_factor(timeframe: str) -> float:
    mapping = {
        "1m": 365 * 24 * 60,
        "5m": 365 * 24 * 12,
        "15m": 365 * 24 * 4,
        "30m": 365 * 24 * 2,
        "1h": 365 * 24,
        "4h": 365 * 6,
        "1d": 365,
    }
    return float(mapping.get(timeframe, 365 * 24 * 12))


def run_backtest(bundle_path: Path, cfg: BTConfig) -> Dict:
    bundle = _load_bundle(bundle_path)
    model = bundle["model"]
    symbols = bundle["symbols"]
    feats = bundle["feature_columns"]

    panel = _assemble(symbols, cfg)
    if panel.empty:
        raise RuntimeError("Backtest panel is empty after preprocessing.")

    # Use last 30% as OOS holdout to mimic production behavior
    unique_ts = np.array(sorted(panel.index.unique()))
    split = int(len(unique_ts) * 0.7)
    oos_ts = set(unique_ts[split:])
    oos = panel[panel.index.map(lambda x: x in oos_ts)].copy()

    X = oos[feats]
    y = oos["target"].astype(int).to_numpy()
    ret = oos["target_return"].to_numpy()

    proba = model.predict_proba(X)[:, 1]
    go_long = (proba >= cfg.threshold).astype(int)

    costs = (cfg.spread_bps + cfg.slippage_bps + cfg.fee_bps) / 10000.0
    strategy_ret = np.where(go_long == 1, ret - costs, 0.0)

    equity = np.cumprod(1 + strategy_ret)
    market_equity = np.cumprod(1 + ret)

    ann = _annualization_factor(cfg.timeframe)
    mean_r = np.mean(strategy_ret)
    std_r = np.std(strategy_ret)
    sharpe = (mean_r / (std_r + 1e-12)) * math.sqrt(ann)

    wins = int(np.sum(strategy_ret > 0))
    trades = int(np.sum(go_long == 1))

    result = {
        "model_name": bundle.get("best_model_name"),
        "symbols": symbols,
        "rows_oos": int(len(oos)),
        "trades": trades,
        "win_rate": float(wins / trades) if trades else 0.0,
        "avg_trade_return": float(np.mean(strategy_ret[go_long == 1])) if trades else 0.0,
        "total_return": float(equity[-1] - 1) if len(equity) else 0.0,
        "market_return": float(market_equity[-1] - 1) if len(market_equity) else 0.0,
        "max_drawdown": float(_drawdown(equity)),
        "sharpe": float(sharpe),
        "threshold": cfg.threshold,
        "costs_total_bps": cfg.spread_bps + cfg.slippage_bps + cfg.fee_bps,
    }

    by_symbol = {}
    for sym, sdf in oos.groupby("symbol"):
        x_s = sdf[feats]
        r_s = sdf["target_return"].to_numpy()
        p_s = model.predict_proba(x_s)[:, 1]
        g_s = (p_s >= cfg.threshold).astype(int)
        s_ret = np.where(g_s == 1, r_s - costs, 0.0)
        eq_s = np.cumprod(1 + s_ret)
        by_symbol[sym] = {
            "rows": int(len(sdf)),
            "trades": int(np.sum(g_s == 1)),
            "return": float(eq_s[-1] - 1) if len(eq_s) else 0.0,
        }

    result["by_symbol"] = by_symbol
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run production-grade out-of-sample backtest")
    p.add_argument("--bundle", default="models/production/best_production_bundle.joblib")
    p.add_argument("--timeframe", default="5m")
    p.add_argument("--limit", type=int, default=20000)
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument("--threshold", type=float, default=0.55)
    p.add_argument("--spread-bps", type=float, default=2.0)
    p.add_argument("--slippage-bps", type=float, default=1.0)
    p.add_argument("--fee-bps", type=float, default=0.5)
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
    )

    result = run_backtest(Path(args.bundle), cfg)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print(json.dumps(result, indent=2))
    print(f"[OK] saved backtest report to {out}")


if __name__ == "__main__":
    main()
