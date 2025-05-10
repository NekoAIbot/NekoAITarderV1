# app/models/basic_model.py
import numpy as np

class BasicModel:
    def __init__(self, model_path=None):
        # TODO: load your real model(s)
        pass

    def predict(self, df):
        # Stub → replace with your trained AI
        import random
        sig = random.choice(["BUY","SELL","HOLD"])
        conf = float(random.random()*100)
        if len(df)>=2:
            p,l = df["close"].iloc[-2], df["close"].iloc[-1]
            pct = (l-p)/p*100
        else:
            pct = 0.0
        sentiment = random.choice(["Positive","Neutral","Negative"])
        return {
            "signal": sig,
            "confidence": conf,
            "predicted_change": pct,
            "news_sentiment": sentiment
        }

# app/models/momentum_model.py

import numpy as np

class MomentumModel:
    """
    Simple heuristic: predicted_change = last close vs 5-period ago.
    Replace with your real AI model later.
    """
    def __init__(self, lookback: int = 5):
        self.lookback = lookback

    def predict(self, data):
        # data is a DataFrame with columns ['open','high','low','close','volume']
        if len(data) <= self.lookback:
            return {"signal": "HOLD", "predicted_change": 0.0, "confidence": 0.0}
        close = data["close"]
        past = close.iloc[-(self.lookback+1)]
        now  = close.iloc[-1]
        pct  = (now - past) / past * 100
        # simple signal: buy if positive momentum, sell if negative
        sig = "BUY" if pct > 0 else "SELL" if pct < 0 else "HOLD"
        # confidence scaled to abs(pct) capped at 100%
        conf = min(abs(pct)*2, 100.0)
        return {"signal": sig, "predicted_change": pct, "confidence": conf}
