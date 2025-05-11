#!/usr/bin/env python3
import sys
from pathlib import Path
import optuna
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config            import FOREX_MAJORS, CRYPTO_ASSETS
from app.market_data   import fetch_market_data
from app.news          import get_news_sentiment
from app.backtester    import backtest_symbol
from app.models.rf_model   import RFModel
from app.models.xgb_model  import MomentumModel as XGBModel
from app.models.lstm_model import LSTMModel
from app.models.cnn_model  import CNNModel

SYMBOLS = FOREX_MAJORS + CRYPTO_ASSETS

def build_dataset():
    dfs, news = [], []
    for sym in SYMBOLS:
        df = fetch_market_data(sym)
        try: s = get_news_sentiment(sym)
        except: s = 0.0
        dfs.append(df.assign(symbol=sym))
        news.append(pd.Series(s, index=df.index))
    big_df   = pd.concat(dfs)
    big_news = pd.concat(news).sort_index()
    return big_df, big_news

def objective(trial):
    df, news = build_dataset()
    split   = int(0.8*len(df))
    tdf, vdf = df.iloc[:split], df.iloc[split:]
    tnews, vnews = news.iloc[:split], news.iloc[split:]

    model_name = trial.suggest_categorical("model", ["rf","xgb","lstm","cnn"])
    if model_name=="rf":
        n_est = trial.suggest_int("rf_n_estimators",100,300,step=50)
        md    = trial.suggest_int("rf_max_depth", 3, 10)
        m = RFModel()
        m.pipeline.set_params(clf__n_estimators=n_est, clf__max_depth=md)
        m.fit(tdf, tnews)

    elif model_name=="xgb":
        params = {
            "n_estimators":    trial.suggest_int("xgb_n_estimators",100,500,50),
            "max_depth":       trial.suggest_int("xgb_max_depth",3,10),
            "learning_rate":   trial.suggest_loguniform("xgb_lr",1e-3,0.2),
            "subsample":       trial.suggest_uniform("xgb_sub",0.5,1.0),
            "colsample_bytree":trial.suggest_uniform("xgb_col",0.5,1.0)
        }
        m = XGBModel()
        m.pipeline.set_params(**{
            "clf__n_estimators":    params["n_estimators"],
            "clf__max_depth":       params["max_depth"],
            "clf__learning_rate":   params["learning_rate"],
            "clf__subsample":       params["subsample"],
            "clf__colsample_bytree":params["colsample_bytree"],
        })
        m.fit(tdf, tnews)

    elif model_name=="lstm":
        look = trial.suggest_int("lstm_lookback",10,50,10)
        drop = trial.suggest_uniform("lstm_dropout",0.1,0.5)
        units= trial.suggest_int("lstm_units",32,128,32)
        m = LSTMModel(lookback=look)
        m.model.layers[1].units = units
        m.model.layers[1].dropout = drop
        m.model.compile("adam","binary_crossentropy",metrics=["accuracy"],run_eagerly=True)
        m.fit(tdf, tnews)

    else:  # cnn
        look = trial.suggest_int("cnn_lookback",10,50,10)
        m = CNNModel(lookback=look)
        m.fit(tdf, tnews)

    # validation backtest
    total_pl = 0.0
    for s in pd.unique(vdf["symbol"]):
        subdf = vdf[vdf["symbol"]==s]
        total_pl += sum(backtest_symbol(s, m, fee_per_trade=0.0, min_confidence=0.6))
    return - total_pl

if __name__=="__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)
    print("Best parameters:", study.best_params)
