import pandas as pd
import numpy as np
from datetime import datetime
from app.market_data import fetch_twelvedata
from app.trade_logger import log_trade
from app.models.ai_model import MomentumModel  # Removed BaseModel

def backtest_symbol(symbol: str,
                    model: MomentumModel,  # Use specific model type
                    fee_per_trade: float = 0.0):
    """
    Runs a simple next-bar backtest on 1-min data for `symbol`.
    For each bar t, predict on bars[:t], then enter at open(t+1), exit at open(t+2).
    """
    # 1) Fetch historical bars
    df = fetch_twelvedata(symbol)
    opens = df["open"].values
    closes = df["close"].values

    trades = []
    for t in range(model.lookback + 1, len(df) - 2):
        window = pd.DataFrame({
            "open": opens[:t+1],
            "high": df["high"][:t+1],
            "low": df["low"][:t+1],
            "close": closes[:t+1],
            "volume": df["volume"][:t+1],
        })
        out = model.predict(window, news=0.0)
        sig = out["signal"]
        if sig == "HOLD":
            continue

        entry_price = opens[t+1]
        exit_price = opens[t+2]
        direction = 1 if sig == "BUY" else -1
        raw_pl = direction * (exit_price - entry_price)
        profit = raw_pl - fee_per_trade
        win = profit > 0

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

def summarize(trades: np.ndarray):
    if trades.size == 0:
        return {"n": 0}
    wins = trades > 0
    return {
        "n": len(trades),
        "wins": int(wins.sum()),
        "win_rate": float(wins.mean() * 100),
        "avg_pl": float(trades.mean()),
        "total_pl": float(trades.sum()),
        "std_pl": float(trades.std()),
    }
