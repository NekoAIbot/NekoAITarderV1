#!/usr/bin/env python3
# scripts/train_model.py

import sys
from pathlib import Path
import joblib
import pandas as pd

# allow imports from project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import FOREX_MAJORS, CRYPTO_ASSETS
from app.market_data   import fetch_market_data
from app.news          import get_news_series
from app.models.xgb_model import MomentumModel

def main():
    symbols = FOREX_MAJORS + CRYPTO_ASSETS
    dfs, news_segs = [], []

    print("🛠 Building unified training set…")
    for sym in symbols:
        print(f"  – {sym}")
        df = fetch_market_data(sym)                    # must return a pd.DataFrame with a DatetimeIndex
        ns = get_news_series(sym, df.index)            # get one sentiment value per bar
        df["symbol"] = sym
        dfs.append(df)
        news_segs.append(ns)

    # concatenate
    big_df       = pd.concat(dfs)
    big_news     = pd.concat(news_segs).sort_index()
    print(f"→ training on {len(big_df)} total rows")

    # fit our XGBoost model
    model        = MomentumModel()
    model.fit(big_df, big_news)

    # persist
    out_path     = ROOT / "app" / "models" / "models" / "xgb_model.joblib"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model.pipeline, out_path)

    print("✅ Done, model saved to", out_path)

if __name__ == "__main__":
    main()
