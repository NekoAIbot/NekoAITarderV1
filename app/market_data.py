#!/usr/bin/env python3
# File: app/market_data.py

import json
import time
import random
from pathlib import Path

import requests
import pandas as pd
from config import TWELVEDATA_API_KEY, ALPHAVANTAGE_API_KEY

# Cache configuration
CACHE_DIR = Path.home() / ".nekoai" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL = 300  # seconds

def _read_cache(symbol: str) -> pd.DataFrame | None:
    path = CACHE_DIR / f"{symbol}_1m.json"
    if not path.exists() or (time.time() - path.stat().st_mtime) > CACHE_TTL:
        return None
    return pd.read_json(path)

def _write_cache(symbol: str, df: pd.DataFrame):
    path = CACHE_DIR / f"{symbol}_1m.json"
    df.to_json(path, orient="records")

def fetch_twelvedata(symbol: str) -> pd.DataFrame:
    """Fetch 1-min bars from TwelveData, drop datetime, pause 10–15s."""
    pair = f"{symbol[:3]}/{symbol[3:]}"
    resp = requests.get(
        "https://api.twelvedata.com/time_series",
        params={
            "symbol":     pair,
            "interval":   "1min",
            "outputsize": 100,
            "apikey":     TWELVEDATA_API_KEY,
        },
        timeout=10
    ).json()

    if "values" not in resp:
        raise ValueError(f"TwelveData error: {resp.get('message', resp)}")

    df = pd.DataFrame(resp["values"])[::-1].reset_index(drop=True)

    # Force-drop any timestamp
    if "datetime" in df.columns:
        df = df.drop(columns=["datetime"])

    # Ensure exactly OHLCV
    for col in ("open", "high", "low", "close", "volume"):
        if col not in df.columns:
            df[col] = 0.0

    df = df[["open", "high", "low", "close", "volume"]].astype("float")

    time.sleep(random.uniform(10, 15))
    return df

def fetch_alphavantage(symbol: str) -> pd.DataFrame:
    """Fetch 1-min FX bars from AlphaVantage, pause 10–15s."""
    resp = requests.get(
        "https://www.alphavantage.co/query",
        params={
            "function":    "FX_INTRADAY",
            "from_symbol": symbol[:3],
            "to_symbol":   symbol[3:],
            "interval":    "1min",
            "outputsize":  "compact",
            "apikey":      ALPHAVANTAGE_API_KEY,
        },
        timeout=10
    ).json()

    key = "Time Series FX (1min)"
    if key not in resp:
        raise ValueError(f"AlphaVantage error: {resp}")

    records = []
    for _, vals in resp[key].items():
        records.append({
            "open":   float(vals["1. open"]),
            "high":   float(vals["2. high"]),
            "low":    float(vals["3. low"]),
            "close":  float(vals["4. close"]),
            "volume": float(vals.get("5. volume", 0.0)),
        })

    df = pd.DataFrame(records)

    # Ensure exactly OHLCV
    for col in ("open", "high", "low", "close", "volume"):
        if col not in df.columns:
            df[col] = 0.0

    df = df[["open", "high", "low", "close", "volume"]].astype("float")

    time.sleep(random.uniform(10, 15))
    return df

def fetch_market_data(symbol: str) -> pd.DataFrame:
    """
    Return latest 100 1-min bars for `symbol`, with:
      1) Cache (5 min)
      2) fetch_twelvedata() → pause
      3) fallback fetch_alphavantage() → pause
      4) dummy fallback
    """
    # 1) Cache
    df = _read_cache(symbol)
    if df is not None:
        return df

    # 2) TwelveData
    try:
        df = fetch_twelvedata(symbol)
    except Exception as e:
        print(f"⚠️ TwelveData failed for {symbol}: {e}")
        # 3) AlphaVantage fallback
        try:
            df = fetch_alphavantage(symbol)
        except Exception as e2:
            print(f"⚠️ AlphaVantage failed for {symbol}: {e2}")
            import numpy as np
            df = pd.DataFrame({
                "open":   np.random.rand(100),
                "high":   np.random.rand(100),
                "low":    np.random.rand(100),
                "close":  np.random.rand(100),
                "volume": np.random.randint(1, 1000, size=100),
            })

    # 4) Cache & return
    _write_cache(symbol, df)
    return df
