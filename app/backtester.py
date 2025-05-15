#!/usr/bin/env python3
import numpy as np
from app.market_data import fetch_market_data
from app.trade_logger import log_trade

def backtest_symbol(symbol: str,
                    model,
                    fee_per_trade: float = 0.0,
                    min_confidence: float = 0.0):
    """
    Runs a next-bar backtest on 1-min data for `symbol`.
    For each bar t, predict on bars[:t], enter at open(t+1), exit at open(t+2).
    Only trades if model.confidence >= min_confidence.
    """
    df      = fetch_market_data(symbol)
    opens   = df["open"].values
    profits = []

    # t runs from lookback .. len(df)-3 so that open[t+2] exists
    for t in range(model.lookback, len(df) - 2):
        window = df.iloc[: t + 1]
        out    = model.predict(window, news=0.0)
        sig    = out["signal"]
        conf   = out["confidence"] / 100.0

        if sig == "HOLD" or conf < min_confidence:
            continue

        entry = opens[t + 1]
        exit_ = opens[t + 2]
        direction = 1 if sig == "BUY" else -1
        raw_pl    = direction * (exit_ - entry)
        profit    = raw_pl - fee_per_trade
        is_win    = profit > 0

        # log each trade
        log_trade(
            symbol      = symbol,
            signal      = sig,
            volume      = 1.0,
            entry_price = entry,
            exit_price  = exit_,
            profit      = profit,
            win         = is_win,
        )

        profits.append(profit)

    return np.array(profits)
