import numpy as np
from .rf_model   import RFModel
from .xgb_model  import MomentumModel as XGBModel
from .lstm_model import LSTMModel
from .cnn_model  import CNNModel

class EnsembleModel:
    def __init__(self, lookback=20):
        # instantiate each with same lookback
        self.models = [
            RFModel(), XGBModel(), LSTMModel(), CNNModel()
        ]
        self.lookback = lookback

    def predict(self, df, news):
        # collect each model's probability for “up”
        ups = []
        dns = []
        for m in self.models:
            out = m.predict(df, news)
            # we assume .predict returns {"signal", "confidence", "predicted_change"}
            # but we need proba. If pipeline: we can fetch proba:
            if hasattr(m, "pipeline"):
                feat = m.featurize(df, news).iloc[[-1]]
                p0,p1 = m.pipeline.predict_proba(feat)[0]
            else:
                # for Keras: we treat confidence as p(up)
                p1 = out["confidence"]/100
                p0 = 1 - p1
            ups.append(p1)
            dns.append(p0)

        avg_up = np.mean(ups)
        avg_dn = np.mean(dns)
        signal = "BUY" if avg_up>avg_dn else "SELL"
        return {
            "signal": signal,
            "confidence": max(avg_up,avg_dn)*100,
            "predicted_change": (avg_up-avg_dn)*100
        }
