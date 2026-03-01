#!/usr/bin/env python3
"""Download crypto OHLCV and save parquet files for training/backtesting."""

import argparse
from pathlib import Path
import pandas as pd
import yfinance as yf

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "DOTUSDT", "XRPUSDT"]
YF_MAP = {
    "BTCUSDT": "BTC-USD",
    "ETHUSDT": "ETH-USD",
    "BNBUSDT": "BNB-USD",
    "SOLUSDT": "SOL-USD",
    "ADAUSDT": "ADA-USD",
    "DOTUSDT": "DOT-USD",
    "XRPUSDT": "XRP-USD",
}


def to_interval(tf: str) -> str:
    m = {"1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "60m", "1d": "1d"}
    return m.get(tf, "1m")


def download_symbol(symbol: str, timeframe: str, period: str, out_dir: Path):
    yf_symbol = YF_MAP.get(symbol, symbol)
    interval = to_interval(timeframe)
    hist = yf.Ticker(yf_symbol).history(period=period, interval=interval, actions=False)
    if hist.empty:
        print(f"[WARN] no data for {symbol}")
        return

    df = hist[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index().dropna()

    out_path = out_dir / f"{symbol}_{timeframe}.parquet"
    df.to_parquet(out_path)
    print(f"[OK] wrote {out_path} rows={len(df)}")


def parse_args():
    p = argparse.ArgumentParser(description="Download crypto data to parquet")
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    p.add_argument("--timeframe", default="1m")
    p.add_argument("--period", default="60d")
    p.add_argument("--output-dir", default="data/raw/crypto")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for s in args.symbols:
        try:
            download_symbol(s, args.timeframe, args.period, out_dir)
        except Exception as exc:
            print(f"[WARN] {s}: {exc}")


if __name__ == "__main__":
    main()
