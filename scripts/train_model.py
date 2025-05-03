#!/usr/bin/env python3
import pandas as pd
from app.market_data import fetch_market_data
from app.news import get_news_sentiment
from app.models.ai_model import AIModel
from config import FOREX_MAJORS, CRYPTO_ASSETS

def main(symbols):
    model = AIModel()
    for sym in symbols:
        print(f"🛠  Training on {sym} …")
        df = fetch_market_data(sym)
        # Build a (constant) news_series for each bar
        news_scores = pd.Series(
            [get_news_sentiment(sym)] * len(df),
            index=df.index
        )
        model.fit(df, news_scores)
    print("✅ Training complete.")

if __name__ == "__main__":
    # override any args: always train on all defined assets
    from config import FOREX_MAJORS, CRYPTO_ASSETS
    all_syms = FOREX_MAJORS + CRYPTO_ASSETS
    main(all_syms)
