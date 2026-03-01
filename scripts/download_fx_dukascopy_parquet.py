#!/usr/bin/env python3
"""Download Dukascopy FX tick data and store monthly M1 parquet files."""

import argparse
import os
import lzma
import struct
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

REQUEST_TIMEOUT = 10


def get_url(pair, year, month_zero, day, hour):
    return (
        f"https://datafeed.dukascopy.com/datafeed/"
        f"{pair}/{year}/{month_zero:02d}/{day:02d}/{hour:02d}h_ticks.bi5"
    )


def download_hour(pair, year, month_zero, day, hour):
    url = get_url(pair, year, month_zero, day, hour)
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return None

        decompressed = lzma.decompress(r.content)

        rows = []
        for i in range(0, len(decompressed), 20):
            chunk = decompressed[i : i + 20]
            if len(chunk) < 20:
                continue
            ms, bid, ask, bid_vol, ask_vol = struct.unpack(">I I I f f", chunk)
            rows.append([ms, bid / 100000, ask / 100000, bid_vol, ask_vol])

        if not rows:
            return None

        df = pd.DataFrame(rows, columns=["ms", "bid", "ask", "bid_vol", "ask_vol"])
        base_time = datetime(year, month_zero + 1, day, hour)
        df["datetime"] = base_time + pd.to_timedelta(df["ms"], unit="ms")
        df = df.set_index("datetime")
        return df
    except Exception:
        return None


def process_month(pair, year, month_zero, out_dir: Path, max_workers: int):
    out_path = out_dir / f"{pair}_M1_{year}_{month_zero+1:02d}.parquet"
    if out_path.exists():
        print(f"Skipping {pair} {year}-{month_zero+1:02d} (exists)")
        return

    print(f"Processing {pair} {year}-{month_zero+1:02d}")
    start = datetime(year, month_zero + 1, 1)
    end = (start + pd.offsets.MonthEnd(1)).to_pydatetime() + timedelta(days=1)

    all_frames = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        current = start
        while current < end:
            for hour in range(24):
                futures.append(
                    executor.submit(
                        download_hour,
                        pair,
                        current.year,
                        current.month - 1,
                        current.day,
                        hour,
                    )
                )
            current += timedelta(days=1)

        for f in as_completed(futures):
            df = f.result()
            if df is not None:
                all_frames.append(df)

    if not all_frames:
        print(f"  No data for {pair} {year}-{month_zero+1:02d}")
        return

    ticks = pd.concat(all_frames).sort_index()
    m1 = ticks["bid"].resample("1min").ohlc()
    m1["volume"] = ticks["bid_vol"].resample("1min").sum()
    m1 = m1.dropna()
    m1.to_parquet(out_path)
    print(f"  Saved {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Download Dukascopy FX data to parquet")
    p.add_argument("--pairs", nargs="+", default=["EURUSD", "USDJPY", "GBPUSD", "USDCHF", "AUDUSD"])
    p.add_argument("--start-year", type=int, default=2016)
    p.add_argument("--end-year", type=int, default=2025)
    p.add_argument("--output-dir", default="data/raw/fx")
    p.add_argument("--max-workers", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for pair in args.pairs:
        print(f"\n===== STARTING {pair} =====")
        for year in range(args.start_year, args.end_year + 1):
            for month_zero in range(12):
                process_month(pair, year, month_zero, out_dir, args.max_workers)


if __name__ == "__main__":
    main()
