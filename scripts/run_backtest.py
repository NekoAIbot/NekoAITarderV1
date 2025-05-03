#!/usr/bin/env python3
# File: scripts/run_backtest.py

import sys
from pathlib import Path

# allow imports from project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_today_symbols
from app.models.ai_model import MomentumModel
from app.backtester import backtest_symbol

def compute_stats(pl_list):
    """Given a list of per-trade P/L, return stats dict."""
    n       = len(pl_list)
    total   = sum(pl_list)
    avg     = total / n if n else 0.0
    wins    = sum(1 for x in pl_list if x > 0)
    win_pct = (wins / n * 100) if n else 0.0
    return {
        "n":         n,
        "total_pl":  total,
        "avg_pl":    avg,
        "win_rate":  win_pct,
    }

def main(symbols):
    model      = MomentumModel(lookback=5)
    overall_pl = []

    for sym in symbols:
        print(f"\n🔍 Backtesting {sym} …")
        pl_list = backtest_symbol(sym, model, fee_per_trade=0.0)
        stats   = compute_stats(pl_list)
        overall_pl.extend(pl_list)

        print(
            f"→ {sym}: Trades={stats['n']}, "
            f"WinRate={stats['win_rate']:.1f}%, "
            f"AvgPL={stats['avg_pl']:.5f}, "
            f"TotalPL={stats['total_pl']:.5f}"
        )

    total_stats = compute_stats(overall_pl)
    print(
        f"\n📈 Overall: TotalTrades={total_stats['n']}, "
        f"WinRate={total_stats['win_rate']:.1f}%, "
        f"TotalPL={total_stats['total_pl']:.5f}"
    )

if __name__ == "__main__":
    # If the user passed symbols on the command line, use those.
    # Otherwise, grab the full list from config.get_today_symbols()
    if len(sys.argv) > 1:
        syms = sys.argv[1:]
    else:
        syms = get_today_symbols()
    main(syms)
