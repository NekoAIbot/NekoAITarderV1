# app/models/basic_model.py
import numpy as np

class BasicModel:
    """
    Stub for your custom-trained AI:
      - signal:          'BUY'/'SELL'/'HOLD'
      - confidence:      float 0–100 (%)
      - predicted_change: float (percent)
      - news_sentiment:  'Positive'/'Neutral'/'Negative'
    """
    def __init__(self, model_path=None):
        # TODO: load your trained model(s) here
        pass

    def predict(self, df):
        # ▼ Replace this stub with your model inference
        import random
        signal = random.choice(["BUY","SELL","HOLD"])
        confidence = float(random.random()*100)
        if len(df)>=2:
            prev, last = df["close"].iloc[-2], df["close"].iloc[-1]
            predicted_change = (last-prev)/prev*100
        else:
            predicted_change = 0.0
        news_sentiment = random.choice(["Positive","Neutral","Negative"])
        # ▲
        return {
            "signal": signal,
            "confidence": confidence,
            "predicted_change": predicted_change,
            "news_sentiment": news_sentiment
        }
