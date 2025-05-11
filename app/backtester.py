import numpy as np
from app.market_data    import fetch_twelvedata
from app.trade_logger   import log_trade

def backtest_symbol(symbol: str,
                    model,
                    fee_per_trade: float = 0.0,
                    min_confidence: float = 0.6):
    """
    Next-bar backtest. Only trade when model.confidence >= min_confidence.
    """
    df    = fetch_twelvedata(symbol)
    opens = df["open"].values
    profits = []

    for t in range(model.lookback, len(df)-2):
        window = df.iloc[:t+1]
        out    = model.predict(window, news=0.0)
        if out["signal"]=="HOLD" or out["confidence"]/100 < min_confidence:
            continue

        entry = opens[t+1]
        exit_ = opens[t+2]
        dirn  = 1 if out["signal"]=="BUY" else -1
        raw_pl= dirn*(exit_ - entry)
        profit = raw_pl - fee_per_trade
        win    = profit>0

        log_trade(
            symbol=symbol,
            signal=out["signal"],
            volume=1.0,
            entry_price=entry,
            exit_price=exit_,
            profit=profit,
            win=win
        )
        profits.append(profit)

    return np.array(profits)
