# app/state.py
import time
from collections import defaultdict
from app.db import get_state, set_state

start_time = time.time()
trades_today = int(get_state("trades_today", "0"))
wins_today = int(get_state("wins_today", "0"))
losses_today = int(get_state("losses_today", "0"))
daily_pnl = float(get_state("daily_pnl", "0.0"))
symbol_trade_counter = defaultdict(int)


def _persist() -> None:
    set_state("trades_today", str(trades_today))
    set_state("wins_today", str(wins_today))
    set_state("losses_today", str(losses_today))
    set_state("daily_pnl", str(daily_pnl))


def increment_trade_count(symbol=None, win=None, profit: float = 0.0):
    global trades_today, wins_today, losses_today, daily_pnl
    trades_today += 1
    daily_pnl += float(profit)
    if symbol:
        symbol_trade_counter[symbol] += 1
    if win is True:
        wins_today += 1
    elif win is False:
        losses_today += 1
    _persist()


def get_bot_status():
    uptime = time.time() - start_time
    h, rem = divmod(int(uptime), 3600)
    m, s = divmod(rem, 60)
    up_str = f"{h}h {m}m {s}s"
    top = sorted(symbol_trade_counter.items(), key=lambda x: -x[1])[:3]
    top_symbols = [f"{sym}({cnt})" for sym, cnt in top]
    return up_str, trades_today, top_symbols, wins_today, losses_today, daily_pnl


def daily_summary():
    up, tr, tops, w, l, pnl = get_bot_status()
    wr = (w / tr * 100) if tr else 0.0
    msg = (
        "📋 Daily Trading Summary\n\n"
        f"🕒 Uptime: {up}\n"
        f"📊 Trades: {tr}  ✅ {w}  ❌ {l}\n"
        f"🎯 Win Rate: {wr:.2f}%\n"
        f"💰 Daily P/L: {pnl:.5f}\n"
    )
    if tops:
        msg += f"🏆 Top: {', '.join(tops)}\n"
    msg += "\n📅 Tomorrow awaits!"
    return msg


def reset_daily_trades():
    global trades_today, wins_today, losses_today, symbol_trade_counter, daily_pnl
    trades_today = wins_today = losses_today = 0
    daily_pnl = 0.0
    symbol_trade_counter = defaultdict(int)
    _persist()
    print("✅ Daily stats reset")
