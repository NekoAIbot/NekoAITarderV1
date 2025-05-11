# app/trainer.py

import time, shutil, joblib
import pandas as pd
from pathlib import Path

from config            import FOREX_MAJORS, CRYPTO_ASSETS
from app.market_data   import fetch_market_data
from app.news          import get_news_sentiment
from app.backtester    import backtest_symbol
from app.models.rf_model   import RFModel
from app.models.xgb_model  import MomentumModel as XGBModel
from app.models.lstm_model import LSTMModel
from app.models.cnn_model  import CNNModel

SYMBOLS = FOREX_MAJORS + CRYPTO_ASSETS
MODEL_DIR = Path(__file__).parent / "models"

def build_dataset():
    dfs, news = [], []
    for sym in SYMBOLS:
        df = fetch_market_data(sym)
        try: s = get_news_sentiment(sym)
        except: s = 0.0
        dfs.append(df.assign(symbol=sym))
        news.append(pd.Series(s, index=df.index))
        time.sleep(1)
    big_df   = pd.concat(dfs)
    big_news = pd.concat(news).sort_index()
    return big_df, big_news

def train_all_and_select_best() -> tuple[str, object]:
    """
    Trains RF, XGB, LSTM, CNN; backtests each; returns (best_name, best_model_instance).
    Also persists each and a copy as best_model.*
    """
    df, news = build_dataset()

    # train
    models = {
        "RF":   RFModel(),
        "XGB":  XGBModel(),
        "LSTM": LSTMModel(),
        "CNN":  CNNModel(),
    }
    for name, m in models.items():
        m.fit(df, news)
        # save each under app/models/<name.lower()>_best.*
        ext = "joblib" if hasattr(m, "pipeline") else "keras"
        path = MODEL_DIR / f"{name.lower()}_best.{ext}"
        MODEL_DIR.mkdir(exist_ok=True)
        if hasattr(m, "pipeline"):
            joblib.dump(m.pipeline, path)
        else:
            m.model.save(path)

    # evaluate
    results = {}
    for name, m in models.items():
        all_pl = []
        for s in SYMBOLS:
            all_pl += list(backtest_symbol(s, m, fee_per_trade=0.0, min_confidence=0.6))
        results[name] = sum(all_pl)

    best_name = max(results, key=results.get)
    best_model = models[best_name]

    # copy to best_model.*
    src = MODEL_DIR / f"{best_name.lower()}_best.{ 'joblib' if best_name in ('RF','XGB') else 'keras'}"
    dst = MODEL_DIR / f"best_model.{src.suffix.lstrip('.')}"
    shutil.copy(src, dst)

    return best_name, best_model
