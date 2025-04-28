# app/state.py

import time
import random
from collections import defaultdict

start_time = time.time()
trades_today = 0
wins_today = 0
losses_today = 0
symbol_trade_counter = defaultdict(int)

def increment_trade_count(symbol=None, win=None):
    """Increment trades and optionally track wins/losses."""
    global trades_today, wins_today, losses_today
    trades_today += 1
    if symbol:
        symbol_trade_counter[symbol] += 1
    if win is not None:
        if win:
            wins_today += 1
        else:
            losses_today += 1

def get_bot_status():
    uptime_seconds = time.time() - start_time
    hours, remainder = divmod(int(uptime_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    uptime = f"{hours}h {minutes}m {seconds}s"

    top_symbols = sorted(symbol_trade_counter.items(), key=lambda x: x[1], reverse=True)
    top_symbols = [f"{sym} ({count})" for sym, count in top_symbols[:3]]

    return uptime, trades_today, top_symbols, wins_today, losses_today

def daily_summary():
    """Return daily trading summary as a message string."""
    uptime, trades_today, top_symbols, wins_today, losses_today = get_bot_status()

    win_rate = (wins_today / trades_today) * 100 if trades_today else 0

    message = (
        "📋 Daily Trading Summary\n\n"
        f"🕒 Uptime: {uptime}\n"
        f"📊 Total Trades Today: {trades_today}\n"
        f"✅ Wins: {wins_today}\n"
        f"❌ Losses: {losses_today}\n"
        f"🎯 Win Rate: {win_rate:.2f}%\n"
    )

    if top_symbols:
        message += f"🏆 Top Traded Symbols: {', '.join(top_symbols)}\n"

    message += "\n📅 See you tomorrow for more profits! 🚀"

    return message

def reset_daily_trades():
    """Reset trades counter and stats at midnight UTC."""
    global trades_today, wins_today, losses_today, symbol_trade_counter
    trades_today = 0
    wins_today = 0
    losses_today = 0
    symbol_trade_counter = defaultdict(int)
    print("✅ Trades and stats reset for new day!")
