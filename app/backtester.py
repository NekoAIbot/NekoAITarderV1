# app/backtester.py

import pandas as pd
import numpy as np
from app.market_data import fetch_twelvedata
from app.trade_logger import log_trade
from app.models.ai_model import MomentumModel

# app/backtester.py (excerpt)

def backtest_symbol(symbol: str,
                    model,
                    fee_per_trade: float = 0.0002,   # e.g. 0.02% per side
                    slippage: float     = 0.0001):   # e.g. 1 pip for FX
    df = fetch_twelvedata(symbol)
    opens = df["open"].values
    trades = []
    for t in range(model.lookback, len(df)-2):
        window = df.iloc[:t+1]
        out = model.predict(window, news=0.0)
        sig = out["signal"]
        if sig=="HOLD": continue

        entry_price = opens[t+1] + (slippage if sig=="BUY" else -slippage)
        exit_price  = opens[t+2] - (slippage if sig=="BUY" else -slippage)
        raw_pl      = (exit_price - entry_price) * (1 if sig=="BUY" else -1)
        profit      = raw_pl - fee_per_trade*entry_price*2    # round‐trip
        trades.append(profit)
        log_trade(...)

    return np.array(trades)

def summarize(trades: np.ndarray):
    """Return summary stats for a numpy array of P/Ls."""
    if trades.size == 0:
        return {"n":0}
    wins = trades > 0
    return {
        "n":          int(trades.size),
        "wins":       int(wins.sum()),
        "win_rate":   float(wins.mean() * 100),
        "avg_pl":     float(trades.mean()),
        "total_pl":   float(trades.sum()),
        "std_pl":     float(trades.std()),
    }
