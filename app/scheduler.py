# app/scheduler.py
import schedule 
import time 
import random
import subprocess
from config import TESTING_MODE, TRADING_INTERVAL_MINUTES
from app.trading            import trading_job
from app.telegram_bot       import send_message, send_message_channel
from app.state              import get_bot_status, reset_daily_trades, daily_summary

def heartbeat_job():
    up, total_trades, top_syms, wins, losses = get_bot_status()
    lines = [
        "💓 All systems go.",
        "🛰️ Tracking markets.",
        "🎯 Aiming high."
    ]
    msg = random.choice(lines) + "\n"
    msg += f"⏱ Uptime: {up}\n"
    msg += f"📊 Trades: {total_trades}  ✅{wins}  ❌{losses}\n"
    msg += f"🏆 Top Symbols: {', '.join(top_syms) or '–'}"
    send_message(msg)

def daily_summary_job():
    send_message_channel(daily_summary())

def retrain_backtest_job():
    """
    Every 6 hours: retrain all models, run backtest, and send summary to your Telegram channel.
    """
    send_message_channel("🔄 Starting scheduled retrain & backtest…")
    # retrain
    try:
        out = subprocess.check_output(
            ["python", "scripts/train_models.py"],
            stderr=subprocess.STDOUT,
            text=True
        )
        send_message_channel(f"✅ Retrain complete:\n```\n{out}\n```")
    except subprocess.CalledProcessError as e:
        send_message_channel(f"❌ Retrain failed:\n```\n{e.output}\n```")
        return

    # backtest with the new best model
    try:
        out = subprocess.check_output(
            ["python", "scripts/run_backtest.py"],
            stderr=subprocess.STDOUT,
            text=True
        )
        send_message_channel(f"📈 Backtest results:\n```\n{out}\n```")
    except subprocess.CalledProcessError as e:
        send_message_channel(f"❌ Backtest failed:\n```\n{e.output}\n```")

def run_scheduler():
    # trading job
    if TESTING_MODE:
        schedule.every(30).seconds.do(trading_job)
        print("⚡ TEST MODE: trading every 30s")
    else:
        schedule.every(TRADING_INTERVAL_MINUTES).minutes.do(trading_job)
        print(f"Scheduler: trading every {TRADING_INTERVAL_MINUTES} minutes")

    # heartbeat & daily summary
    schedule.every().hour.do(heartbeat_job)
    schedule.every().day.at("00:00").do(reset_daily_trades)
    schedule.every().day.at("23:59").do(daily_summary_job)

    # automatic retrain & backtest every 6 hours
    schedule.every(6).hours.do(retrain_backtest_job)
    print("Scheduler: retrain & backtest every 6 hours")

    # main loop
    while True:
        schedule.run_pending()
        time.sleep(1)
