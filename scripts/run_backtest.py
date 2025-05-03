#!/usr/bin/env python3
import sys
from pathlib import Path

# allow imports from project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import FOREX_MAJORS, CRYPTO_ASSETS
from app.models.ai_model import MomentumModel as AIModel
from app.backtester import backtest_symbol

def compute_stats(pl_list):
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
    model      = AIModel()
    overall_pl = []

    for sym in symbols:
        print(f"\n🔍 Backtesting {sym} …")
        # backtest_symbol returns a list of individual trade P/Ls
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
    # If user passed symbols on the command line, use those.
    # Otherwise backtest the full universe.
    if len(sys.argv) > 1:
        syms = sys.argv[1:]
    else:
        syms = FOREX_MAJORS + CRYPTO_ASSETS
    main(syms)
