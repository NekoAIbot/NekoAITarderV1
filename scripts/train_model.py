#!/usr/bin/env python3
import sys
from pathlib import Path
import joblib
import pandas as pd

# allow imports from project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import FOREX_MAJORS, CRYPTO_ASSETS
from app.models.ai_model import MomentumModel
from app.market_data import fetch_market_data
from app.news import get_news_sentiment

def main():
    symbols = FOREX_MAJORS + CRYPTO_ASSETS
    model   = MomentumModel()
    diagnostics = []

    for sym in symbols:
        print(f"🛠  Training on {sym} …")

        # 1) load your price data
        df = fetch_market_data(sym)

        # 2) fetch one sentiment per symbol, then align it with df.index
        news_score  = get_news_sentiment(sym)
        news_series = pd.Series(news_score, index=df.index)

        # 3) fit the model
        model.fit(df, news_series)

        # quick in-sample check
        preds = model.predict(df, news_score)
        diagnostics.append((sym, preds['signal'], preds['predicted_change']))

    # 4) persist the trained pipeline
    MODEL_PATH = ROOT / "models" / "rf_model.joblib"
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model.pipeline, MODEL_PATH)

    print("✅ Training complete. Model saved to", MODEL_PATH)
    print("Diagnostics (last-bar predictions):")
    for sym, sig, change in diagnostics:
        print(f"  • {sym}: {sig} ({change:.2f}%)")

if __name__ == "__main__":
    main()
