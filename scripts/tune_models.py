#!/usr/bin/env python3
import sys
from pathlib import Path
import optuna
import numpy as np
import pandas as pd

# allow imports from project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import FOREX_MAJORS, CRYPTO_ASSETS
from app.market_data    import fetch_market_data
from app.news           import get_news_sentiment
from app.backtester     import backtest_symbol
from app.models.xgb_model  import MomentumModel as XGBModel
from app.models.lstm_model import LSTMModel
from app.models.cnn_model  import CNNModel

SYMBOLS = FOREX_MAJORS + CRYPTO_ASSETS

def build_dataset():
    dfs, news = [], []
    for sym in SYMBOLS:
        df = fetch_market_data(sym)
        try:
            s = get_news_sentiment(sym)
        except:
            s = 0.0
        dfs.append(df.assign(symbol=sym))
        news.append(pd.Series(s,index=df.index))
    big_df   = pd.concat(dfs)
    big_news = pd.concat(news).sort_index()
    return big_df, big_news

def objective(trial):
    df, news = build_dataset()
    split = int(0.8 * len(df))
    tdf, vdf = df.iloc[:split], df.iloc[split:]
    tnews, vnews = news.iloc[:split], news.iloc[split:]

    model_type = trial.suggest_categorical("model", ["xgb","lstm","cnn"])
    if model_type=="xgb":
        params = {
            "n_estimators":    trial.suggest_int("n_estimators",100,500,step=50),
            "max_depth":       trial.suggest_int("max_depth",3,10),
            "learning_rate":   trial.suggest_loguniform("learning_rate",1e-3,0.2),
            "subsample":       trial.suggest_uniform("subsample",0.5,1.0),
            "colsample_bytree":trial.suggest_uniform("colsample_bytree",0.5,1.0)
        }
        m = XGBModel()
        m.pipeline.set_params(**{
            "clf__n_estimators":    params["n_estimators"],
            "clf__max_depth":       params["max_depth"],
            "clf__learning_rate":   params["learning_rate"],
            "clf__subsample":       params["subsample"],
            "clf__colsample_bytree":params["colsample_bytree"]
        })
        m.fit(tdf, tnews)
    elif model_type=="lstm":
        m = LSTMModel(); m.fit(tdf, tnews)
    else:
        m = CNNModel(); m.fit(tdf, tnews)

    # backtest on validation slice
    total_pl = 0.0
    for sym in pd.unique(vdf["symbol"]):
        total_pl += sum(backtest_symbol(sym, m, fee_per_trade=0.0))

    return - total_pl  # maximize P/L

if __name__=="__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=40)
    print("Best params:", study.best_params)
