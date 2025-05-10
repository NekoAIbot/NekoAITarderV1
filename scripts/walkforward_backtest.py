#!/usr/bin/env python3
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# project root → allow “app” imports
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import FOREX_MAJORS, CRYPTO_ASSETS
from app.market_data import fetch_twelvedata
from app.news           import get_news_series
from app.models.xgb_model import XGBModel
from app.backtester      import backtest_symbol

def walkforward(symbol, model, window_size=500, step=100):
    df = fetch_twelvedata(symbol)
    dates = df.index
    results = []
    for start in range(0, len(df) - window_size, step):
        train_df = df.iloc[start : start + window_size]
        test_df  = df.iloc[start + window_size : start + window_size + step]

        news_train = get_news_series(symbol, train_df.index)
        model.fit(train_df, news_train)

        # backtest on test_df slice
        pl = []
        for t in range(model.lookback, len(test_df) - 2):
            window = test_df.iloc[: t+1]
            out = model.predict(window, news=0.0)
            if out["signal"] == "HOLD":
                continue
            entry   = test_df["open"].iloc[t+1]
            exit_   = test_df["open"].iloc[t+2]
            direction = 1 if out["signal"] == "BUY" else -1
            pl.append(direction * (exit_ - entry))
        results.append(np.mean(pl) if pl else 0.0)

    idx = [dates[i] for i in range(0, len(df) - window_size, step)]
    return pd.Series(results, index=idx)

if __name__ == "__main__":
    symbols = FOREX_MAJORS + CRYPTO_ASSETS
    for sym in symbols:
        print(f"\n▶ Walk-forward {sym}")
        model = XGBModel()
        wf = walkforward(sym, model)
        print(wf.describe())
