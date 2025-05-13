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
    dfs, news = [], []
    for sym in SYMBOLS:
        df = fetch_market_data(sym)
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

    # 80/20 time split
    split = int(0.8 * len(df))
    tdf, vdf = df.iloc[:split].copy(), df.iloc[split:].copy()
    tdf.index, vdf.index = df.index[:split], df.index[split:]
    tnews, vnews         = news.iloc[:split], news.iloc[split:]

    # common HPs
    lookback   = trial.suggest_int("lookback", 10, 50)
    min_conf   = trial.suggest_float("min_confidence", 0.5, 0.9)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    model_type = trial.suggest_categorical("model", ["xgb", "lstm", "cnn"])

    if model_type == "xgb":
        # XGBoost hyperparams
        params = {
            "n_estimators":     trial.suggest_int("xgb_n_estimators", 50, 500, step=50),
            "max_depth":        trial.suggest_int("xgb_max_depth", 3, 12),
            "learning_rate":    trial.suggest_float("xgb_lr", 1e-3, 0.2, log=True),
            "subsample":        trial.suggest_float("xgb_sub", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("xgb_col", 0.5, 1.0),
        }
        m = XGBModel()
        m.pipeline.set_params(
            clf__n_estimators=     params["n_estimators"],
            clf__max_depth=        params["max_depth"],
            clf__learning_rate=    params["learning_rate"],
            clf__subsample=        params["subsample"],
            clf__colsample_bytree= params["colsample_bytree"],
        )
        m.lookback = lookback
        m.fit(tdf, tnews)

    elif model_type == "lstm":
        m = LSTMModel(lookback=lookback)
        m.batch_size = batch_size
        m.fit(tdf, tnews)

    else:  # cnn
        m = CNNModel(lookback=lookback)
        m.batch_size = batch_size
        m.fit(tdf, tnews)

    # backtest on validation slice
    all_profits = []
    for sym in pd.unique(vdf["symbol"]):
        pf = backtest_symbol(
            symbol         = sym,
            model          = m,
            fee_per_trade  = 0.0,
            min_confidence = min_conf
        )
        all_profits.extend(pf.tolist())

    if len(all_profits) < 2:
        return 0.0

    rets   = np.array(all_profits)
    sharpe = np.mean(rets) / (np.std(rets, ddof=1) + 1e-8)
    # maximize Sharpe → minimize negative Sharpe
    return -sharpe

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)

    print("Best params:", study.best_params)
    print("Retraining on full dataset…")

    df, news = build_dataset()
    best = study.best_params

    # build final model
    lookback   = best.get("lookback", 20)
    batch_size = best.get("batch_size", 32)
    min_conf   = best.get("min_confidence", 0.6)

    if best["model"] == "xgb":
        final = XGBModel(); final.lookback = lookback
        final.pipeline.set_params(
            clf__n_estimators=     best["xgb_n_estimators"],
            clf__max_depth=        best["xgb_max_depth"],
            clf__learning_rate=    best["xgb_lr"],
            clf__subsample=        best["xgb_sub"],
            clf__colsample_bytree= best["xgb_col"],
        )
    elif best["model"] == "lstm":
        final = LSTMModel(lookback=lookback); final.batch_size = batch_size
    else:
        final = CNNModel(lookback=lookback); final.batch_size = batch_size

    final.fit(df, news)

    ext = "joblib" if best["model"] == "xgb" else "keras"
    out = ROOT / "app" / "models" / f"{best['model']}_tuned.{ext}"
    out.parent.mkdir(exist_ok=True)
    if hasattr(final, "pipeline"):
        import joblib; joblib.dump(final.pipeline, out)
    else:
        final.model.save(out)

    print("Final tuned model saved to", out)
