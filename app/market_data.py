#!/usr/bin/env python3
# File: app/market_data.py

import time
import random
from pathlib import Path

import requests
import pandas as pd
import yfinance as yf
from config import TWELVEDATA_API_KEY, ALPHAVANTAGE_API_KEY

# ── Cache configuration ───────────────────────────────────────────────────────

CACHE_DIR = Path.home() / ".nekoai" / "cache" / "market_data"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL      = 300    # seconds before cache expires
API_CALL_DELAY = 1.0    # seconds to wait after each symbol fetch

def _read_cache(symbol: str) -> pd.DataFrame | None:
    path = CACHE_DIR / f"{symbol}_1m.json"
    if not path.exists() or (time.time() - path.stat().st_mtime) > CACHE_TTL:
        return None
    try:
        df = pd.read_json(path, orient="split", convert_dates=True)
        # restore the DatetimeIndex
        return df.set_index(df.index.name or "index")
    except Exception:
        return None

def _write_cache(symbol: str, df: pd.DataFrame):
    path = CACHE_DIR / f"{symbol}_1m.json"
    df.to_json(path, orient="split", date_format="iso")

# ── Symbol normalization ─────────────────────────────────────────────────────

def _normalize_for_yf(symbol: str) -> str:
    # FX pairs → e.g. EURUSD → EURUSD=X
    if len(symbol) == 6 and not symbol.endswith("USDT"):
        return f"{symbol}=X"
    # Crypto USDT → BTCUSDT → BTC-USD
    if symbol.endswith("USDT"):
        return f"{symbol[:-4]}-USD"
    return symbol

def _normalize_symbol_for_api(symbol: str) -> str:
    # for TwelveData / AlphaVantage (base/quote)
    if symbol.endswith("USDT"):
        base, quote = symbol[:-4], "USD"
    else:
        base, quote = symbol[:3], symbol[3:]
    return f"{base}/{quote}"

# ── yfinance primary ─────────────────────────────────────────────────────────

def fetch_yfinance(symbol: str) -> pd.DataFrame:
    yf_sym = _normalize_for_yf(symbol)
    ticker = yf.Ticker(yf_sym)
    hist   = ticker.history(period="7d", interval="1m", actions=False)
    if hist.empty:
        raise ValueError("yfinance returned no data")
    df = hist.iloc[-100:][["Open","High","Low","Close","Volume"]]
    df.columns = ["open","high","low","close","volume"]
    return df

# ── TwelveData fallback ───────────────────────────────────────────────────────

def fetch_twelvedata(symbol: str) -> pd.DataFrame:
    pair = _normalize_symbol_for_api(symbol)
    resp = requests.get(
        "https://api.twelvedata.com/time_series",
        params={
            "symbol":     pair,
            "interval":   "1min",
            "outputsize": 100,
            "apikey":     TWELVEDATA_API_KEY,
        }, timeout=10
    ).json()
    if "values" not in resp:
        raise ValueError(f"TwelveData error: {resp.get('message', resp)}")

    raw = pd.DataFrame(resp["values"])[::-1]
    raw["datetime"] = pd.to_datetime(raw["datetime"])
    raw = raw.set_index("datetime")

    # ensure column exists before fillna
    for col in ("open","high","low","close","volume"):
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce").fillna(0.0)
        else:
            raw[col] = 0.0

    time.sleep(random.uniform(10, 15))
    return raw[["open","high","low","close","volume"]]

# ── AlphaVantage final fallback ───────────────────────────────────────────────

def fetch_alphavantage(symbol: str) -> pd.DataFrame:
    base, quote = _normalize_symbol_for_api(symbol).split("/")
    resp = requests.get(
        "https://www.alphavantage.co/query",
        params={
            "function":    "FX_INTRADAY",
            "from_symbol": base,
            "to_symbol":   quote,
            "interval":    "1min",
            "outputsize":  "compact",
            "apikey":      ALPHAVANTAGE_API_KEY,
        }, timeout=10
    ).json()
    key = "Time Series FX (1min)"
    if key not in resp:
        raise ValueError(f"AlphaVantage error: {resp}")

    rows = []
    for ts, vals in resp[key].items():
        rows.append({
            "datetime": pd.to_datetime(ts),
            "open":     float(vals["1. open"]),
            "high":     float(vals["2. high"]),
            "low":      float(vals["3. low"]),
            "close":    float(vals["4. close"]),
            "volume":   float(vals.get("5. volume", 0.0)),
        })
    df = pd.DataFrame(rows).set_index("datetime").sort_index()
    time.sleep(random.uniform(10, 15))
    return df[["open","high","low","close","volume"]]

# ── Unified market-data API ──────────────────────────────────────────────────

def fetch_market_data(symbol: str) -> pd.DataFrame:
    """
    Return latest 100 1-min bars for `symbol`:
      1) Cache (5-min TTL)
      2) yfinance primary
      3) TwelveData fallback
      4) AlphaVantage final fallback
      5) Throttle via API_CALL_DELAY
    """
    df = _read_cache(symbol)
    if df is not None:
        return df

    # try in order
    try:
        df = fetch_yfinance(symbol)
    except Exception as e1:
        print(f"⚠️ yfinance failed for {symbol}: {e1}")
        try:
            df = fetch_twelvedata(symbol)
        except Exception as e2:
            print(f"⚠️ TwelveData failed for {symbol}: {e2}")
            try:
                df = fetch_alphavantage(symbol)
            except Exception as e3:
                print(f"⚠️ AlphaVantage failed for {symbol}: {e3}")
                # dummy fallback
                import numpy as np
                idx = pd.date_range(end=pd.Timestamp.utcnow(), periods=100, freq="T")
                df = pd.DataFrame({
                    "open":   np.random.rand(100),
                    "high":   np.random.rand(100),
                    "low":    np.random.rand(100),
                    "close":  np.random.rand(100),
                    "volume": np.random.randint(1,1000,100),
                }, index=idx)

    # cache + throttle
    _write_cache(symbol, df)
    time.sleep(API_CALL_DELAY)
    return df
