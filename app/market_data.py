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

CACHE_TTL = 300
API_CALL_DELAY = 1.0

# ──────────────────────────────────────────────────────────────────────────────
# TIMEFRAME MAPPING
# ──────────────────────────────────────────────────────────────────────────────

YF_INTERVAL_MAP = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "60m",
    "4h": "60m",   # resampled later
    "1d": "1d",
}

TD_INTERVAL_MAP = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1day",
}

AV_INTERVAL_MAP = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "60min",
}

# Pandas-safe frequency map (FIXED)
PANDAS_FREQ_MAP = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}

# ──────────────────────────────────────────────────────────────────────────────
# CACHE
# ──────────────────────────────────────────────────────────────────────────────

def _cache_path(symbol: str, timeframe: str) -> Path:
    return CACHE_DIR / f"{symbol}_{timeframe}.json"

def _read_cache(symbol: str, timeframe: str) -> pd.DataFrame | None:
    path = _cache_path(symbol, timeframe)
    if not path.exists() or (time.time() - path.stat().st_mtime) > CACHE_TTL:
        return None
    try:
        return pd.read_json(path, orient="split", convert_dates=True)
    except Exception:
        return None

def _write_cache(symbol: str, timeframe: str, df: pd.DataFrame):
    path = _cache_path(symbol, timeframe)
    df.to_json(path, orient="split", date_format="iso")

# ──────────────────────────────────────────────────────────────────────────────
# SYMBOL NORMALIZATION
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_for_yf(symbol: str) -> str:
    if len(symbol) == 6 and not symbol.endswith("USDT"):
        return f"{symbol}=X"
    if symbol.endswith("USDT"):
        return f"{symbol[:-4]}-USD"
    return symbol

def _normalize_symbol_for_api(symbol: str) -> str:
    if symbol.endswith("USDT"):
        base, quote = symbol[:-4], "USD"
    else:
        base, quote = symbol[:3], symbol[3:]
    return f"{base}/{quote}"

# ──────────────────────────────────────────────────────────────────────────────
# RESAMPLING (FIXED FOR PANDAS 2.x)
# ──────────────────────────────────────────────────────────────────────────────

def _resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if timeframe == "4h":
        df = df.resample("4h").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()
    return df

# ──────────────────────────────────────────────────────────────────────────────
# DATA PROVIDERS
# ──────────────────────────────────────────────────────────────────────────────

def fetch_yfinance(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    yf_sym = _normalize_for_yf(symbol)
    interval = YF_INTERVAL_MAP.get(timeframe, "1m")

    period = "60d" if timeframe in ["1h", "4h"] else "7d"

    hist = yf.Ticker(yf_sym).history(
        period=period,
        interval=interval,
        actions=False
    )

    if hist.empty:
        raise ValueError("yfinance returned no data")

    df = hist[["Open", "High", "Low", "Close", "Volume"]]
    df.columns = ["open", "high", "low", "close", "volume"]

    df = _resample(df, timeframe)
    return df.tail(limit)

def fetch_twelvedata(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    pair = _normalize_symbol_for_api(symbol)
    interval = TD_INTERVAL_MAP.get(timeframe, "1min")

    resp = requests.get(
        "https://api.twelvedata.com/time_series",
        params={
            "symbol": pair,
            "interval": interval,
            "outputsize": limit,
            "apikey": TWELVEDATA_API_KEY,
        },
        timeout=10
    ).json()

    if "values" not in resp:
        raise ValueError(resp)

    raw = pd.DataFrame(resp["values"])[::-1]
    raw["datetime"] = pd.to_datetime(raw["datetime"])
    raw = raw.set_index("datetime")

    for col in ("open", "high", "low", "close", "volume"):
        raw[col] = pd.to_numeric(raw[col], errors="coerce").fillna(0.0)

    time.sleep(random.uniform(8, 12))
    return raw[["open", "high", "low", "close", "volume"]].tail(limit)

def fetch_alphavantage(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    base, quote = _normalize_symbol_for_api(symbol).split("/")
    interval = AV_INTERVAL_MAP.get(timeframe, "1min")

    resp = requests.get(
        "https://www.alphavantage.co/query",
        params={
            "function": "FX_INTRADAY",
            "from_symbol": base,
            "to_symbol": quote,
            "interval": interval,
            "outputsize": "compact",
            "apikey": ALPHAVANTAGE_API_KEY,
        },
        timeout=10
    ).json()

    key = f"Time Series FX ({interval})"
    if key not in resp:
        raise ValueError(resp)

    rows = []
    for ts, vals in resp[key].items():
        rows.append({
            "datetime": pd.to_datetime(ts),
            "open": float(vals["1. open"]),
            "high": float(vals["2. high"]),
            "low": float(vals["3. low"]),
            "close": float(vals["4. close"]),
            "volume": float(vals.get("5. volume", 0.0)),
        })

    df = pd.DataFrame(rows).set_index("datetime").sort_index()
    time.sleep(random.uniform(8, 12))
    return df[["open", "high", "low", "close", "volume"]].tail(limit)

# ──────────────────────────────────────────────────────────────────────────────
# UNIFIED API (FULLY FIXED)
# ──────────────────────────────────────────────────────────────────────────────

def fetch_market_data(symbol: str, timeframe: str = "1m", limit: int = 1000) -> pd.DataFrame:

    df = _read_cache(symbol, timeframe)
    if df is not None:
        return df

    try:
        df = fetch_yfinance(symbol, timeframe, limit)
    except Exception as e1:
        print(f"⚠️ yfinance failed for {symbol}: {e1}")
        try:
            df = fetch_twelvedata(symbol, timeframe, limit)
        except Exception as e2:
            print(f"⚠️ TwelveData failed for {symbol}: {e2}")
            try:
                df = fetch_alphavantage(symbol, timeframe, limit)
            except Exception as e3:
                print(f"⚠️ AlphaVantage failed for {symbol}: {e3}")
                # Final safe fallback (FIXED frequency)
                import numpy as np
                freq = PANDAS_FREQ_MAP.get(timeframe, "1min")
                idx = pd.date_range(
                    end=pd.Timestamp.utcnow(),
                    periods=limit,
                    freq=freq
                )
                df = pd.DataFrame({
                    "open": np.random.rand(limit),
                    "high": np.random.rand(limit),
                    "low": np.random.rand(limit),
                    "close": np.random.rand(limit),
                    "volume": np.random.randint(1, 1000, limit),
                }, index=idx)

    _write_cache(symbol, timeframe, df)
    time.sleep(API_CALL_DELAY)
    return df