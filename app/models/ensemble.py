# app/models/ensemble.py

from .xgb_model        import XGBModel
from .lstm_model       import LSTMModel
from .cnn_model        import CNNModel
from .dense_nn_model   import DenseNNModel

class Ensemble:
    def __init__(self):
        self.models = {
            "xgb":   XGBModel(),
            "lstm":  LSTMModel(),
            "cnn":   CNNModel(),
            "dense": DenseNNModel(),
        }

    def backtest_all(self, symbol, fee_per_trade=0.0):
        from app.backtester import backtest_symbol
        results = {}
        for name, m in self.models.items():
            pl = backtest_symbol(symbol, m, fee_per_trade=fee_per_trade)
            win_rate = (pl>0).mean() * 100 if len(pl) else 0.0
            total_pl = pl.sum() if len(pl) else 0.0
            results[name] = {"win_rate": win_rate, "total_pl": total_pl}
        return results

    def best_model_for(self, symbol):
        stats = self.backtest_all(symbol)
        best = max(stats.items(), key=lambda kv: kv[1]["win_rate"])[0]
        return best, self.models[best]
