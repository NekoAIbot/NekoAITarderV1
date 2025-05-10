#!/usr/bin/env python3
# enable TF eager mode for all functions (so .fit / .save work correctly)
import tensorflow as tf
tf.config.run_functions_eagerly(True)

import sys
import time
import shutil
from pathlib import Path
import joblib
import pandas as pd

# allow “app” imports
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
    print("🛠 Building unified training set…")
    for i, sym in enumerate(SYMBOLS):
        print(f"  – {sym}")
        df = fetch_market_data(sym)
        try:
            s = get_news_sentiment(sym)
        except Exception:
            s = 0.0
        ns = pd.Series(s, index=df.index)
        df["symbol"] = sym
        dfs.append(df)
        news.append(ns)
        time.sleep(1)
    big_df   = pd.concat(dfs)
    big_news = pd.concat(news).sort_index()
    print(f"→ training on {len(big_df)} rows")
    return big_df, big_news

def train_and_save(name, ModelClass, df, news):
    print(f"\n🔧 Training {name}…")
    m = ModelClass()
    m.fit(df, news)

    # determine extension & path
    if hasattr(m, "pipeline"):
        ext = "joblib"
    else:
        ext = "keras"

    out = ROOT / "app" / "models" / f"{name.lower()}_best.{ext}"
    out.parent.mkdir(exist_ok=True)

    if hasattr(m, "pipeline"):
        joblib.dump(m.pipeline, out)
    else:
        # Keras .save will pick extension
        m.model.save(out)

    print(f"✅ Saved {name} to {out}")
    return m

def evaluate(name, model):
    print(f"\n🧪 Backtesting {name}…")
    all_pl = []
    for s in SYMBOLS:
        all_pl += list(backtest_symbol(s, model, fee_per_trade=0.0))
    wins  = sum(1 for p in all_pl if p > 0)
    wrate = wins / len(all_pl) * 100 if all_pl else 0.0
    tot   = sum(all_pl)
    print(f"→ {name}: Trades={len(all_pl)}, WinRate={wrate:.1f}%, TotalPL={tot:.3f}")
    return tot

def main():
    df, news = build_dataset()

    rf   = train_and_save("RF",   RFModel,   df, news)
    xgb  = train_and_save("XGB",  XGBModel,  df, news)
    lstm = train_and_save("LSTM", LSTMModel, df, news)
    cnn  = train_and_save("CNN",  CNNModel,  df, news)

    results = {
        "RF":   evaluate("RF",   rf),
        "XGB":  evaluate("XGB",  xgb),
        "LSTM": evaluate("LSTM", lstm),
        "CNN":  evaluate("CNN",  cnn),
    }

    best = max(results, key=results.get)
    print(f"\n🏆 Best model: {best}")

    ext = "joblib" if best in ("RF", "XGB") else "keras"
    src = ROOT / "app" / "models" / f"{best.lower()}_best.{ext}"
    dst = ROOT / "app" / "models" / f"best_model.{ext}"

    # use shutil.copy for all files
    shutil.copy(src, dst)
    print(f"✅ Best model saved to {dst}")

if __name__ == "__main__":
    main()
