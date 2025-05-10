import pandas as pd
import numpy as np
from app.market_data import fetch_twelvedata
from app.trade_logger import log_trade
from app.models.xgb_model import MomentumModel

def backtest_symbol(symbol: str,
                    model: MomentumModel,
                    fee_per_trade: float = 0.0):
    """
    Runs a next-bar backtest on 1-min data for `symbol`.
    For each bar t, predict on bars[:t], enter at open(t+1), exit at open(t+2).
    """
    df = fetch_twelvedata(symbol)
    opens = df["open"].values

    trades = []
    for t in range(model.lookback, len(df) - 2):
        window = df.iloc[:t+1]
        out = model.predict(window, news=0.0)
        sig = out["signal"]
        if sig == "HOLD":
            continue

        entry_price = opens[t+1]
        exit_price  = opens[t+2]
        direction   = 1 if sig == "BUY" else -1
        raw_pl      = direction * (exit_price - entry_price)
        profit      = raw_pl - fee_per_trade
        win         = profit > 0

        # ★ Correctly pass all required args to log_trade
        log_trade(
            symbol=symbol,
            signal=sig,
            volume=1.0,
            entry_price=entry_price,
            exit_price=exit_price,
            profit=profit,
            win=win
        )
        trades.append(profit)

    return np.array(trades)
