# app/state.py
import time
from collections import defaultdict

start_time = time.time()
trades_today = wins_today = losses_today = 0
symbol_trade_counter = defaultdict(int)

def increment_trade_count(symbol=None, win=None):
    global trades_today, wins_today, losses_today
    trades_today += 1
    if symbol:
        symbol_trade_counter[symbol] += 1
    if win is True:
        wins_today += 1
    elif win is False:
        losses_today += 1

def get_bot_status():
    uptime = time.time() - start_time
    h, rem = divmod(int(uptime),3600)
    m, s   = divmod(rem,60)
    up_str = f"{h}h {m}m {s}s"
    top = sorted(symbol_trade_counter.items(), key=lambda x:-x[1])[:3]
    top_symbols = [f"{sym}({cnt})" for sym,cnt in top]
    return up_str, trades_today, top_symbols, wins_today, losses_today

def daily_summary():
    up, tr, tops, w, l = get_bot_status()
    wr = (w/tr*100) if tr else 0.0
    msg = (
        "📋 Daily Trading Summary\n\n"
        f"🕒 Uptime: {up}\n"
        f"📊 Trades: {tr}  ✅ {w}  ❌ {l}\n"
        f"🎯 Win Rate: {wr:.2f}%\n"
    )
    if tops:
        msg += f"🏆 Top: {', '.join(tops)}\n"
    msg += "\n📅 Tomorrow awaits!"
    return msg

def reset_daily_trades():
    global trades_today, wins_today, losses_today, symbol_trade_counter
    trades_today = wins_today = losses_today = 0
    symbol_trade_counter = defaultdict(int)
    print("✅ Daily stats reset")
