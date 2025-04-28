# app/scheduler.py
import schedule
import time
import random

from config import TESTING_MODE, TRADING_INTERVAL_MINUTES
from app.trading import trading_job
from app.telegram_bot import send_message, send_message_channel
from app.state import get_bot_status, reset_daily_trades, daily_summary

def heartbeat_job():
    up, tr, tops, w, l = get_bot_status()
    lines = [
        "💓 All systems go.",
        "🛰️ Ping: still tracking markets.",
        "🎯 Eyes on the prize!",
        "🐾 Stalking the pips."
    ]
    msg = random.choice(lines) + "\n\n"
    msg += f"🕒 Uptime: {up}\n📊 Trades: {tr}  ✅{w}  ❌{l}\n"
    msg += f"🏆 Top: {', '.join(tops) if tops else '–'}"
    send_message(msg)

def daily_summary_job():
    send_message_channel(daily_summary())

def run_scheduler():
    if TESTING_MODE:
        schedule.every(30).seconds.do(trading_job)
        print("⚡ TEST MODE: trading every 30s")
    else:
        schedule.every(TRADING_INTERVAL_MINUTES).minutes.do(trading_job)
        print(f"Scheduler: trading every {TRADING_INTERVAL_MINUTES}m")
    schedule.every().hour.do(heartbeat_job)
    schedule.every().day.at("00:00").do(reset_daily_trades)
    schedule.every().day.at("23:59").do(daily_summary_job)
    while True:
        schedule.run_pending()
        time.sleep(1)
