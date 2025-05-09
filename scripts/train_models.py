#!/usr/bin/env python3
import sys, time, shutil
from pathlib import Path

import joblib
import pandas as pd

# allow imports from project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import FOREX_MAJORS, CRYPTO_ASSETS
from app.market_data  import fetch_market_data
from app.news         import get_news_sentiment
from app.backtester   import backtest_symbol

from app.models.rf_model   import RFModel
from app.models.xgb_model  import MomentumModel as XGBModel
from app.models.lstm_model import LSTMModel
from app.models.cnn_model  import CNNModel
# from app.models.rl_agent   import RLAgent  # if/when you implement

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
        dfs.append(df.assign(symbol=sym))
        news.append(pd.Series(s, index=df.index))
        time.sleep(1)  # throttle NewsAPI
    big_df   = pd.concat(dfs)
    big_news = pd.concat(news).sort_index()
    print(f"→ training on {len(big_df)} rows")
    return big_df, big_news

def train_and_save(name, ModelClass, df, news):
    print(f"\n🔧 Training {name}…")
    model = ModelClass()
    model.fit(df, news)
    out = ROOT / "app" / "models" / f"{name.lower()}_best.joblib"
    out.parent.mkdir(exist_ok=True)
    # RF & XGB expose `.pipeline`, neural nets are saved internally
    artifact = getattr(model, "pipeline", model)
    joblib.dump(artifact, out)
    print(f"✅ Saved {name} to {out}")
    return model

def evaluate(name, model):
    print(f"🧪 Backtesting {name}…")
    all_pl = []
    for sym in SYMBOLS:
        pl = backtest_symbol(sym, model, fee_per_trade=0.0)
        all_pl.extend(pl)
    wins   = sum(1 for p in all_pl if p>0)
    wrate  = wins/len(all_pl)*100 if all_pl else 0.0
    tot_pl = sum(all_pl)
    print(f"→ {name}: Trades={len(all_pl)}, WinRate={wrate:.1f}%, TotalPL={tot_pl:.3f}")
    return tot_pl

def main():
    df, news = build_dataset()

    # train each model
    rf_model   = train_and_save("RF", RFModel,   df, news)
    xgb_model  = train_and_save("XGB", XGBModel, df, news)
    lstm_model = train_and_save("LSTM", LSTMModel, df, news)
    cnn_model  = train_and_save("CNN", CNNModel, df, news)
    # rl_model   = train_and_save("RL", RLAgent,  df, news)

    # evaluate & pick best
    results = {
        "RF":    evaluate("RF", rf_model),
        "XGB":   evaluate("XGB", xgb_model),
        "LSTM":  evaluate("LSTM", lstm_model),
        "CNN":   evaluate("CNN", cnn_model),
    }
    best = max(results, key=results.get)
    print(f"\n🏆 Best model: {best} (TotalPL={results[best]:.3f})")

    # copy best artifact to best_model.joblib
    src = ROOT/"app"/"models"/f"{best.lower()}_best.joblib"
    dst = ROOT/"app"/"models"/"best_model.joblib"
    shutil.copy(src, dst)
    print(f"✅ Best model saved to {dst}")

if __name__=="__main__":
    main()
