#!/usr/bin/env python3
# scripts/tune_xgb.py

import sys
from pathlib import Path
import optuna

# project root → allow “app” imports
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import FOREX_MAJORS, CRYPTO_ASSETS
from app.backtester import backtest_symbol
from app.models.xgb_model import MomentumModel
import numpy as np

def objective(trial):
    # sample hyper-parameters
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth":    trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.2),
        "subsample":    trial.suggest_uniform("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
    }

    # build a fresh model for each trial
    model = MomentumModel()
    model.pipeline.named_steps["clf"].set_params(**params)

    # backtest over a small validation set
    syms = FOREX_MAJORS[:2]  # just two for speed; expand as needed
    pl_total = 0.0
    ntrades  = 0
    for s in syms:
        pl = backtest_symbol(s, model, fee_per_trade=0.0)
        pl_total += np.nansum(pl)
        ntrades  += len(pl)

    # objective: maximize profit per trade
    return pl_total / max(1, ntrades)

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    print("Best params:", study.best_params)
    print("Best value:", study.best_value)
