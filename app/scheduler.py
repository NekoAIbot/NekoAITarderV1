#!/usr/bin/env python3
import schedule
import time
import random

from config                import TESTING_MODE, TRADING_INTERVAL_MINUTES
from app.trading           import trading_job
from app.telegram_bot      import send_message, send_message_channel
from app.state             import get_bot_status, reset_daily_trades, daily_summary
from app.mt5_handler       import initialize_mt5, monitor_positions, shutdown_mt5

def heartbeat_job():
    up, tr, tops, w, l = get_bot_status()
    lines = ["💓 All systems go.", "🛰️ Tracking markets.", "🎯 Aiming high."]
    msg = (
        random.choice(lines)
        + f"\nUptime: {up}\nTrades: {tr} ✅{w} ❌{l}"
        + f"\nTop: {', '.join(tops) or '–'}"
    )
    send_message(msg)

def daily_summary_job():
    send_message_channel(daily_summary())

def run_scheduler():
    # 1) Initialize MT5 once
    mt5 = initialize_mt5()

    # 2) Schedule trading cycles
    if TESTING_MODE:
        schedule.every(30).seconds.do(trading_job)
        print("⚡ TEST MODE: trading every 30s")
    else:
        schedule.every(TRADING_INTERVAL_MINUTES).minutes.do(trading_job)
        print(f"Scheduler: trading every {TRADING_INTERVAL_MINUTES} minutes")

    # 3) Heartbeat once an hour
    schedule.every().hour.do(heartbeat_job)

    # 4) Daily housekeeping
    schedule.every().day.at("00:00").do(reset_daily_trades)
    schedule.every().day.at("23:59").do(daily_summary_job)

    # 5) Monitor positions every minute
    schedule.every(5).seconds.do(lambda: monitor_positions(mt5))

    # 6) Main loop (and clean shutdown)
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    finally:
        shutdown_mt5(mt5)
