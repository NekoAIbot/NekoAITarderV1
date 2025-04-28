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
