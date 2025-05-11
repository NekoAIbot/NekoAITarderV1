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

from config                import FOREX_MAJORS, CRYPTO_ASSETS
from app.market_data       import fetch_market_data
from app.news              import get_news_sentiment
from app.backtester        import backtest_symbol
from app.models.xgb_model  import MomentumModel as XGBModel
from app.models.lstm_model import LSTMModel
from app.models.cnn_model  import CNNModel

SYMBOLS = FOREX_MAJORS + CRYPTO_ASSETS

def build_dataset():
    """Fetch all symbols, return one big DataFrame (DatetimeIndex) + aligned news Series."""
    dfs, news = [], []
    for sym in SYMBOLS:
        df = fetch_market_data(sym)  # must return a DataFrame indexed by Timestamp
        try:
            s = get_news_sentiment(sym)
        except:
            s = 0.0
        dfs.append(df.assign(symbol=sym))
        news.append(pd.Series(s, index=df.index))
    big_df   = pd.concat(dfs).sort_index()
    big_news = pd.concat(news).sort_index()
    return big_df, big_news

def objective(trial):
    df, news = build_dataset()

    # time-based split at 80%
    split = int(0.8 * len(df))
    # slice and *preserve* the original DatetimeIndex on each
    tdf = df.iloc[:split].copy()
    vdf = df.iloc[split:].copy()
    tdf.index = df.index[:split]
    vdf.index = df.index[split:]

    # now slice news by those exact timestamps
    tnews = news.loc[tdf.index]
    vnews = news.loc[vdf.index]

    choice = trial.suggest_categorical("model", ["xgb", "lstm", "cnn"])
    if choice == "xgb":
        params = {
            "n_estimators":     trial.suggest_int("xgb_n_estimators", 100, 500, step=50),
            "max_depth":        trial.suggest_int("xgb_max_depth", 3, 10),
            "learning_rate":    trial.suggest_float("xgb_lr", 1e-3, 0.2, log=True),
            "subsample":        trial.suggest_float("xgb_sub", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("xgb_col", 0.5, 1.0),
        }
        m = XGBModel()
        m.pipeline.set_params(
            clf__n_estimators=params["n_estimators"],
            clf__max_depth=params["max_depth"],
            clf__learning_rate=params["learning_rate"],
            clf__subsample=params["subsample"],
            clf__colsample_bytree=params["colsample_bytree"],
        )
        m.fit(tdf, tnews)

    elif choice == "lstm":
        m = LSTMModel()
        m.fit(tdf, tnews)

    else:  # cnn
        m = CNNModel()
        m.fit(tdf, tnews)

    # backtest on the holdout
    total_pl = 0.0
    for sym in pd.unique(vdf["symbol"]):
        subdf = vdf[vdf["symbol"] == sym].drop(columns="symbol")
        total_pl += subdf.pipe(lambda d: backtest_symbol(sym, m, fee_per_trade=0.0)).sum()

    # Optuna minimizes, so return negative P/L to maximize profit
    return - total_pl

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=40)

    print("Best params:", study.best_params)
    print("Retraining on full dataset…")

    # final retrain
    df, news = build_dataset()
    best = study.best_params

    if best["model"] == "xgb":
        final = XGBModel()
        final.pipeline.set_params(
            clf__n_estimators=best["xgb_n_estimators"],
            clf__max_depth=best["xgb_max_depth"],
            clf__learning_rate=best["xgb_lr"],
            clf__subsample=best["xgb_sub"],
            clf__colsample_bytree=best["xgb_col"],
        )
    elif best["model"] == "lstm":
        final = LSTMModel()
    else:
        final = CNNModel()

    final.fit(df, news)

    # save in the proper format
    out = ROOT / "app" / "models" / (
        f"{best['model']}_tuned_model.joblib"
        if best["model"] == "xgb"
        else f"{best['model']}_tuned_model.keras"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(final, "pipeline"):
        import joblib
        joblib.dump(final.pipeline, out)
    else:
        final.model.save(out)

    print("Final tuned model saved to", out)
