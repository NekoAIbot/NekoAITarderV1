#!/usr/bin/env python3
# scripts/run_backtest.py

import sys
from pathlib import Path

# project root → allow “app” imports
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import FOREX_MAJORS, CRYPTO_ASSETS
from app.backtester       import backtest_symbol
from app.models.xgb_model import MomentumModel

def compute_stats(pl_list):
    n       = len(pl_list)
    total   = sum(pl_list)
    avg     = total / n if n else 0.0
    wins    = sum(1 for x in pl_list if x > 0)
    win_pct = (wins / n * 100) if n else 0.0
    return {"n": n, "total_pl": total, "avg_pl": avg, "win_rate": win_pct}

def main(symbols):
    model     = MomentumModel()
    overall   = []

    for sym in symbols:
        print(f"\n🔍 Backtesting {sym} …")
        pl_list = backtest_symbol(sym, model, fee_per_trade=0.0)
        stats   = compute_stats(pl_list)
        overall.extend(pl_list)

        print(f"→ {sym}: Trades={stats['n']}, WinRate={stats['win_rate']:.1f}%, "
              f"AvgPL={stats['avg_pl']:.5f}, TotalPL={stats['total_pl']:.5f}")

    tot = compute_stats(overall)
    print(f"\n📈 Overall: Trades={tot['n']}, WinRate={tot['win_rate']:.1f}%," 
          f" TotalPL={tot['total_pl']:.5f}")

if __name__ == "__main__":
    if len(sys.argv)>1:
        syms = sys.argv[1:]
    else:
        syms = FOREX_MAJORS + CRYPTO_ASSETS
    main(syms)
