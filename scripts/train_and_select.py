#!/usr/bin/env python3
import sys, time
from pathlib import Path
import joblib, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from config                import FOREX_MAJORS, CRYPTO_ASSETS
from app.market_data       import fetch_market_data
from app.news              import get_news_sentiment
from app.models.xgb_model  import XGBModel
from app.models.rf_model   import RFModel
from app.models.nn_models  import DenseModel, CNNModel, LSTMModel
from app.backtester        import backtest_symbol

def compute_stats(pl):
    n    = len(pl)
    tot  = sum(pl)
    return tot

def main():
    symbols = FOREX_MAJORS + CRYPTO_ASSETS
    # load data once
    dfs, news = {}, {}
    for s in symbols:
        dfs[s]   = fetch_market_data(s)
        try:
            news[s] = get_news_sentiment(s)
        except:
            news[s] = 0.0
        time.sleep(1)

    # define your roster
    roster = {
        "xgb":   XGBModel(),
        "rf":    RFModel(),
        "dense": DenseModel(),
        "cnn":   CNNModel(),
        "lstm":  LSTMModel(),
    }

    results = {}
    for name, mdl in roster.items():
        print(f"🏋️ Training {name} …")
        # unify training across symbols
        big_df   = pd.concat([dfs[s].assign(symbol=s) for s in symbols])
        big_news = pd.Series(
            [news[s] for s in symbols for _ in dfs[s].index],
            index=big_df.index
        )
        mdl.fit(big_df, big_news)

        print(f"🔍 Backtesting {name} …")
        all_pl = []
        for s in symbols:
            pl = backtest_symbol(s, mdl, fee_per_trade=0.0)
            all_pl.extend(pl.tolist())
        results[name] = compute_stats(all_pl)
        print(f"→ {name}: TotalPL={results[name]:.4f}")

    # pick best
    champion = max(results, key=results.get)
    print(f"\n🏆 Champion: {champion} (TotalPL={results[champion]:.4f})")

    # persist champion pipeline (or model directory)
    out = ROOT/"app"/"models"/"champion"
    if champion in ("xgb","rf"):
        joblib.dump(roster[champion].pipeline, out.with_suffix(".joblib"))
    else:
        # for keras, save the saved_model dir under champion name
        roster[champion].model.save(out)
    print("✅ Champion saved to", out)

if __name__=="__main__":
    main()
