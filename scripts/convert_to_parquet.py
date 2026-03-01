import os
import zipfile
import glob
import pandas as pd

# Paths for BTC and ETH
PAIRS = {
    "BTCUSDT": "data/binance/btcusdt/1h",
    "ETHUSDT": "data/binance/ethusdt/1h"
}

OUTPUT_BASE = "data/processed"
os.makedirs(OUTPUT_BASE, exist_ok=True)

def convert_zip(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as z:
        csv_files = [f for f in z.namelist() if f.lower().endswith(".csv")]
        if not csv_files:
            return None

        csv_name = csv_files[0]
        with z.open(csv_name) as f:
            df = pd.read_csv(f)

    # Standardize columns (Binance style)
    df.columns = [c.strip().lower() for c in df.columns]

    rename = {
        "open_time": "datetime",
        "close_time": "datetime",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume"
    }
    df.rename(columns={k:v for k,v in rename.items() if k in df.columns}, inplace=True)

    # Convert timestamp if present
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], unit="ms", errors="ignore")
        df.set_index("datetime", inplace=True)

    return df

def process_pair(pair, raw_dir):
    output_dir = os.path.join(OUTPUT_BASE, pair.lower(), "1h")
    os.makedirs(output_dir, exist_ok=True)

    all_dfs = []

    zip_files = glob.glob(os.path.join(raw_dir, "*.zip"))
    if not zip_files:
        print(f"No ZIP files found for {pair} in {raw_dir}")
        return

    for zip_path in zip_files:
        print(f"Processing {zip_path}")
        df = convert_zip(zip_path)
        if df is not None:
            all_dfs.append(df)

    if not all_dfs:
        print(f"No valid CSVs found in ZIPs for {pair}")
        return

    full = pd.concat(all_dfs)
    full.sort_index(inplace=True)

    out_path = os.path.join(output_dir, f"{pair}_1H.parquet")
    full.to_parquet(out_path)
    print(f"Saved: {out_path}")

def main():
    for pair, raw_dir in PAIRS.items():
        process_pair(pair, raw_dir)

if __name__ == "__main__":
    main()