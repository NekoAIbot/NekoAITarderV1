#!/usr/bin/env python3
"""Simple walk-forward CV utility for time series model validation."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.market_data import fetch_market_data
from app.models.ai_model import MomentumModel


def walk_forward(symbol: str = "EURUSD", train_size: int = 600, test_size: int = 200, step: int = 200):
    try:
        df = fetch_market_data(symbol, timeframe="1m", limit=2000)
    except Exception:
        idx = pd.date_range(end=pd.Timestamp.utcnow(), periods=2000, freq="min")
        arr = np.cumsum(np.random.randn(2000)) + 100
        df = pd.DataFrame({"open": arr, "high": arr + 0.1, "low": arr - 0.1, "close": arr, "volume": np.random.randint(1, 100, 2000)}, index=idx)
    model = MomentumModel()

    scores = []
    i = 0
    while i + train_size + test_size < len(df):
        train = df.iloc[i : i + train_size]
        test = df.iloc[i + train_size : i + train_size + test_size]
        try:
            model.fit(train, pd.Series(0.0, index=train.index))
        except Exception:
            i += step
            continue

        wins = 0
        trades = 0
        for t in range(model.lookback, len(test) - 1):
            out = model.predict(test.iloc[: t + 1], 0.0)
            sig = out["signal"]
            if sig == "HOLD":
                continue
            entry = test["close"].iloc[t]
            nxt = test["close"].iloc[t + 1]
            pnl = (nxt - entry) if sig == "BUY" else (entry - nxt)
            wins += int(pnl > 0)
            trades += 1
        if trades:
            scores.append(wins / trades)
        i += step

    mean_score = float(np.mean(scores)) if scores else 0.0
    print(f"Walk-forward win-rate ({symbol}): {mean_score:.4f} over {len(scores)} windows")
    return mean_score


if __name__ == "__main__":
    walk_forward(sys.argv[1] if len(sys.argv) > 1 else "EURUSD")
