#!/usr/bin/env python3
# ── PLACE THIS AT THE VERY TOP ───────────────────────────────────────────────
import os, sys, warnings, contextlib, io
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

warnings.filterwarnings("ignore", ".*Unable to register.*")
warnings.filterwarnings("ignore", ".*computation placer already registered.*")
warnings.filterwarnings("ignore", ".*use_label_encoder.*")

# Monkey‑patch Keras fit to always run with verbose=0
_K = tf.keras.Model
_orig_fit = _K.fit
def _silent_fit(self, *args, **kwargs):
    kwargs.setdefault("verbose", 0)
    return _orig_fit(self, *args, **kwargs)
_K.fit = _silent_fit

# ── UTILITY FOR SILENT FITTING (sklearn/XGB) ────────────────────────────────
def silent_fit(model, *args, **kwargs):
    if "verbose" not in kwargs:
        kwargs["verbose"] = 0
    buf_out, buf_err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
        try:
            return model.fit(*args, **kwargs)
        except TypeError as e:
            if "verbose" in str(e):
                kwargs.pop("verbose", None)
                return model.fit(*args, **kwargs)
            raise

# ── REGULAR IMPORTS ─────────────────────────────────────────────────────────
import optuna, numpy as np, pandas as pd
from pathlib import Path
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
    return pd.concat(dfs).sort_index(), pd.concat(news).sort_index()

def enrich_features(df):
    df = df.copy()
    df["sma5"]  = df["close"].rolling(5).mean()
    df["sma20"] = df["close"].rolling(20).mean()
    delta = df["close"].diff()
    up, dn = delta.clip(lower=0), -delta.clip(upper=0)
    rs = up.ewm(14).mean() / (dn.ewm(14).mean() + 1e-8)
    df["rsi"] = 100 - 100/(1+rs)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    return df.bfill().ffill()

def objective(trial):
    df, news = build_dataset()
    split = int(0.8*len(df))
    tdf, vdf = df.iloc[:split], df.iloc[split:]
    tnews, vnews = news.iloc[:split], news.iloc[split:]

    # hyperparams
    lookback   = trial.suggest_int("lookback", 10, 50)
    min_conf   = trial.suggest_float("min_confidence", 0.0, 0.9)
    batch_size = trial.suggest_categorical("batch_size", [16,32,64])
    model_type = trial.suggest_categorical("model", ["xgb","lstm","cnn"])

    # choose & train
    if model_type == "xgb":
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
        silent_fit(m, tdf, tnews)

    elif model_type == "lstm":
        m = LSTMModel(lookback=lookback)
        m.batch_size = batch_size
        silent_fit(m, tdf, tnews)

    else:  # cnn
        m = CNNModel(lookback=lookback)
        m.batch_size = batch_size
        silent_fit(m, tdf, tnews)

    # backtest & score
    profits = []
    for sym in pd.unique(vdf["symbol"]):
        pf = backtest_symbol(sym, m, fee_per_trade=0.0, min_confidence=min_conf)
        profits.extend(pf.tolist())
    profits = np.array(profits)

    if len(profits) < 5:
        pl = float(profits.sum()) if profits.size else 0.0
        print(f"Trial#{trial.number}: few trades ({len(profits)}), P/L={pl:.4f}")
        return -pl

    sharpe = profits.mean() / (profits.std(ddof=1) + 1e-8)
    print(f"Trial#{trial.number}: Sharpe={sharpe:.4f}")
    return -sharpe

if __name__=="__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100, show_progress_bar=False)

    print("\nBest params:", study.best_params)
    print("Retraining on full dataset…")

    df, news = build_dataset()
    best = study.best_params

    if best["model"] == "xgb":
        final = XGBModel(); final.lookback = best["lookback"]
        final.pipeline.set_params(
            clf__n_estimators=     best["xgb_n_estimators"],
            clf__max_depth=        best["xgb_max_depth"],
            clf__learning_rate=    best["xgb_lr"],
            clf__subsample=        best["xgb_sub"],
            clf__colsample_bytree= best["xgb_col"],
        )
    elif best["model"] == "lstm":
        final = LSTMModel(lookback=best["lookback"])
        final.batch_size = best["batch_size"]
    else:
        final = CNNModel(lookback=best["lookback"])
        final.batch_size = best["batch_size"]

    silent_fit(final, df, news)

    ext = "joblib" if best["model"] == "xgb" else "keras"
    out = ROOT / "app" / "models" / f"{best['model']}_tuned.{ext}"
    out.parent.mkdir(exist_ok=True)
    if hasattr(final, "pipeline"):
        import joblib; joblib.dump(final.pipeline, out)
    else:
        final.model.save(out)

    print("Final tuned model saved to", out)
