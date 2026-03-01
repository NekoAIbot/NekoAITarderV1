#!/usr/bin/env python3
import zipfile
import pandas as pd
from pathlib import Path
from collections import defaultdict

ROOT = Path("data/FX_DATA/HISTDATA")
OUT_DIR = Path("data/FX_DATA/PARQUET")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def detect_pair(zip_path: Path):
    """
    Extract pair from filenames like:
    HISTDATA_COM_ASCII_USDJPY_M12016.zip
    """
    parts = zip_path.stem.split("_")

    if "ASCII" in parts:
        idx = parts.index("ASCII")
        if idx + 1 < len(parts):
            return parts[idx + 1].upper()

    return None


def parse_csv_from_zip(zip_path):
    frames = []

    with zipfile.ZipFile(zip_path, "r") as z:
        for name in z.namelist():
            if not name.lower().endswith(".csv"):
                continue

            with z.open(name) as f:
                df = pd.read_csv(
                    f,
                    sep=";",
                    header=None,
                    names=["datetime", "open", "high", "low", "close", "volume"],
                )

            # Proper datetime format for HISTDATA
            df["datetime"] = pd.to_datetime(
                df["datetime"],
                format="%Y%m%d %H%M%S",
                errors="coerce"
            )

            df.dropna(subset=["datetime"], inplace=True)
            df.set_index("datetime", inplace=True)

            frames.append(df)

    if frames:
        return pd.concat(frames)

    return None


def main():
    # Phase 1: Scan & bucket by pair
    buckets = defaultdict(list)

    print("Scanning ZIP files...")
    for zip_path in ROOT.rglob("*.zip"):
        pair = detect_pair(zip_path)
        if not pair:
            print(f"Skipping {zip_path.name} (pair not detected)")
            continue

        print(f"Reading {zip_path.name} -> {pair}")
        df = parse_csv_from_zip(zip_path)
        if df is not None:
            buckets[pair].append(df)

    # Phase 2: Merge and write per pair
    print("\nWriting parquet files...")
    for pair, frames in buckets.items():
        print(f"Merging {pair} ({len(frames)} files)")

        df = pd.concat(frames)
        df = df[~df.index.duplicated(keep="first")]
        df.sort_index(inplace=True)

        out_file = OUT_DIR / f"{pair}.parquet"
        df.to_parquet(out_file, engine="pyarrow", compression="snappy")

        print(f"Saved: {out_file} | rows: {len(df)}")


if __name__ == "__main__":
    main()