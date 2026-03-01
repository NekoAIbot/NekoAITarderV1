import os
import lzma
import struct
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

# ================= CONFIG =================
PAIRS = ["EURUSD", "USDJPY", "GBPUSD", "USDCHF", "AUDUSD"]
START_YEAR = 2016
END_YEAR = 2025
OUTPUT_DIR = "data/raw/fx"
MAX_WORKERS = 10          # per month parallel hours
REQUEST_TIMEOUT = 10
# ==========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_url(pair, year, month, day, hour):
    # Dukascopy months are 0-based
    return (
        f"https://datafeed.dukascopy.com/datafeed/"
        f"{pair}/{year}/{month:02d}/{day:02d}/{hour:02d}h_ticks.bi5"
    )


def download_hour(pair, year, month, day, hour):
    url = get_url(pair, year, month, day, hour)

    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return None

        decompressed = lzma.decompress(r.content)

        rows = []
        for i in range(0, len(decompressed), 20):
            chunk = decompressed[i:i + 20]
            if len(chunk) < 20:
                continue

            ms, bid, ask, bid_vol, ask_vol = struct.unpack(">I I I f f", chunk)

            rows.append([
                ms,
                bid / 100000,
                ask / 100000,
                bid_vol,
                ask_vol
            ])

        if not rows:
            return None

        df = pd.DataFrame(rows, columns=["ms", "bid", "ask", "bid_vol", "ask_vol"])

        base_time = datetime(year, month + 1, day, hour)
        df["datetime"] = base_time + pd.to_timedelta(df["ms"], unit="ms")
        df.set_index("datetime", inplace=True)

        return df

    except Exception:
        return None


def process_month(pair, year, month):
    out_path = os.path.join(
        OUTPUT_DIR,
        f"{pair}_M1_{year}_{month+1:02d}.parquet"
    )

    # Skip if already downloaded
    if os.path.exists(out_path):
        print(f"Skipping {pair} {year}-{month+1:02d} (exists)")
        return

    print(f"Processing {pair} {year}-{month+1:02d}")

    start = datetime(year, month + 1, 1)
    end = (start + pd.offsets.MonthEnd(1)).to_pydatetime() + timedelta(days=1)

    all_frames = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
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
                        hour
                    )
                )
            current += timedelta(days=1)

        for future in as_completed(futures):
            df = future.result()
            if df is not None:
                all_frames.append(df)

    if not all_frames:
        print(f"  No data for {pair} {year}-{month+1:02d}")
        return

    ticks = pd.concat(all_frames)
    ticks.sort_index(inplace=True)

    # Convert ticks → M1 OHLC
    m1 = ticks["bid"].resample("1min").ohlc()
    m1["volume"] = ticks["bid_vol"].resample("1min").sum()

    m1.dropna(inplace=True)

    m1.to_parquet(out_path)
    print(f"  Saved {out_path}")


def main():
    for pair in PAIRS:
        print(f"\n===== STARTING {pair} =====")
        for year in range(START_YEAR, END_YEAR + 1):
            for month in range(12):
                process_month(pair, year, month)


if __name__ == "__main__":
    main()