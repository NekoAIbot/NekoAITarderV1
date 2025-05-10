#!/usr/bin/env python3
import sys
from pathlib import Path

# allow imports from project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import FOREX_MAJORS, CRYPTO_ASSETS
from app.backtester      import backtest_symbol
from app.models.rf_model import RFModel
from app.models.xgb_model import MomentumModel as XGBModel
from app.models.lstm_model import LSTMModel
from app.models.cnn_model import CNNModel

SYMBOLS = FOREX_MAJORS + CRYPTO_ASSETS

def compute_stats(pl):
    n    = len(pl)
    wins = sum(1 for p in pl if p>0)
    return {
        "n":        n,
        "win_rate": wins/n*100 if n else 0.0,
        "total":    sum(pl),
        "avg":      sum(pl)/n if n else 0.0,
    }

def run_model(name, model):
    print(f"\n🔍 Backtesting {name}")
    all_pl = []
    for s in SYMBOLS:
        all_pl += list(backtest_symbol(s, model, fee_per_trade=0.0))
    st = compute_stats(all_pl)
    print(f"→ {name}: Trades={st['n']}, WinRate={st['win_rate']:.1f}%, "
          f"AvgPL={st['avg']:.5f}, TotalPL={st['total']:.5f}")

if __name__=="__main__":
    rf   = RFModel()
    xgb  = XGBModel()
    lstm = LSTMModel()
    cnn  = CNNModel()

    run_model("RF", rf)
    run_model("XGB", xgb)
    run_model("LSTM", lstm)
    run_model("CNN", cnn)
