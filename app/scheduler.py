# app/scheduler.py

import schedule
import time
import random

from config import TESTING_MODE, TRADING_INTERVAL_MINUTES
from app.trading import trading_job
from app.telegram_bot import send_message, send_message_channel
from app.state import get_bot_status, reset_daily_trades, daily_summary

def heartbeat_job():
    """Send heartbeat message to Telegram."""
    uptime, trades_today, top_symbols, wins, losses = get_bot_status()
    messages = [
        "💓 Still kicking! Monitoring markets carefully.",
        "🧠 NekoAI is analyzing data... All systems nominal.",
        "🛰️ Heartbeat check: we are alive and hunting 📈📉.",
        "🐾 Still stalking the Forex jungle... patience!",
        "🎯 Online, focused, waiting for the perfect opportunity."
    ]
    random_message = random.choice(messages)
    report = (
        f"{random_message}\n\n"
        f"🕒 Uptime: {uptime}\n"
        f"📊 Trades today: {trades_today}\n"
        f"🏆 Top Symbols: {', '.join(top_symbols) if top_symbols else '–'}\n"
        f"✅ Wins: {wins}    ❌ Losses: {losses}\n"
    )
    send_message(report)

def daily_summary_job():
    """Post daily trading summary to Telegram Channel."""
    summary = daily_summary()
    send_message_channel(summary)

def run_scheduler():
    """Run all scheduled tasks."""
    # 1) Trading job: every 30s if testing, else every N minutes
    if TESTING_MODE:
        schedule.every(30).seconds.do(trading_job)
        print("⚡ TEST MODE: trading every 30 seconds.")
    else:
        schedule.every(TRADING_INTERVAL_MINUTES).minutes.do(trading_job)
        print(f"Scheduler started: trading every {TRADING_INTERVAL_MINUTES} minutes.")

    # 2) Heartbeat every 1 hour
    schedule.every(1).hours.do(heartbeat_job)

    # 3) Reset counters at midnight UTC
    schedule.every().day.at("00:00").do(reset_daily_trades)

    # 4) Daily summary at 23:59 UTC
    schedule.every().day.at("23:59").do(daily_summary_job)

    # Loop
    while True:
        schedule.run_pending()
        time.sleep(1)
