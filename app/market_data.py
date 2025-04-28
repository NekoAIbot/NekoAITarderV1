# app/market_data.py
import os
import requests
import pandas as pd
from config import TWELVEDATA_API_KEY

def fetch_twelvedata(symbol):
    # Convert "EURUSD" → "EUR/USD"
    pair = f"{symbol[:3]}/{symbol[3:]}"
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol":    pair,
        "interval":  "1min",
        "outputsize":100,
        "apikey":    TWELVEDATA_API_KEY,
        "format":    "JSON"
    }
    resp = requests.get(url, params=params, timeout=10).json()
    if "values" not in resp:
        raise ValueError(f"TwelveData error: {resp.get('message', resp)}")
    
    df = pd.DataFrame(resp["values"])
    df = df.iloc[::-1].reset_index(drop=True)  # oldest → newest

    # Ensure all necessary columns exist
    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            raise ValueError(f"Missing expected column: {col}")

    if "volume" not in df.columns:
        df["volume"] = 1000  # or use np.random.randint(500,1500, size=len(df)) for random volume

    df = df.astype({
        "open": "float",
        "high": "float",
        "low": "float",
        "close": "float",
        "volume": "float"
    })
    return df
