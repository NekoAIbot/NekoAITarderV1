# app/models/basic_model.py
import numpy as np

class BasicModel:
    """
    Stub for your custom-trained AI:
      - signal:     'BUY' / 'SELL' / 'HOLD'
      - confidence: float 0–100 (%)
      - predicted_change: float (percent)
      - news_sentiment:   'Positive' / 'Neutral' / 'Negative'
    """

    def __init__(self, model_path=None):
        # Load your trained model here (e.g., joblib.load)
        pass

    def predict(self, df):
        # ▼ Replace this stub with your model inference
        signal = np.random.choice(["BUY","SELL","HOLD"])
        confidence = float(np.random.rand() * 100)
        if len(df) >= 2:
            prev, last = df["close"].iloc[-2], df["close"].iloc[-1]
            predicted_change = (last - prev) / prev * 100
        else:
            predicted_change = 0.0
        news_sentiment = np.random.choice(["Positive","Neutral","Negative"])
        # ▲
        return {
            "signal": signal,
            "confidence": confidence,
            "predicted_change": predicted_change,
            "news_sentiment": news_sentiment
        }
