# app/models/momentum_model.py

import pandas as pd
from app.news import get_news_sentiment

class MomentumModel:
    """
    Simple heuristic: compare last close vs N bars ago.
    Outputs signal, predicted_change (%), confidence, and news_sentiment.
    """
    def __init__(self, lookback: int = 5):
        self.lookback = lookback

    def predict(self, data: pd.DataFrame, symbol: str = None) -> dict:
        """
        data: must have a 'close' column
        symbol: the ticker (e.g. "EURUSD") so we can fetch live sentiment
        returns: {"signal","predicted_change","confidence","news_sentiment"}
        """
        result = {"signal": "HOLD", "predicted_change": 0.0, "confidence": 0.0, "news_sentiment": 0.0}

        # 1) momentum signal
        if len(data) > self.lookback:
            close = data["close"]
            past  = close.iloc[-(self.lookback+1)]
            now   = close.iloc[-1]
            pct   = (now - past) / past * 100  # % change over lookback

            if pct > 0:
                sig = "BUY"
            elif pct < 0:
                sig = "SELL"
            else:
                sig = "HOLD"
            conf = min(abs(pct) * 2, 100.0)

            result.update(signal=sig, predicted_change=pct, confidence=conf)

        # 2) live news sentiment
        if symbol:
            try:
                ns = get_news_sentiment(symbol)
            except Exception:
                ns = 0.0
            result["news_sentiment"] = ns

        return result
