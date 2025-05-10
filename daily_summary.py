import pandas as pd
from datetime import datetime
from app.trade_logger import LOG_FILE
from app.telegram_bot import send_message_channel

def send_daily_summary():
    df = pd.read_csv(LOG_FILE, parse_dates=["timestamp"])
    today = datetime.utcnow().date()
    today_df = df[df.timestamp.dt.date == today]
    if today_df.empty:
        msg = "📊 Daily Summary:\nNo trades executed today."
    else:
        wins = today_df.profit > 0
        wr = wins.mean() * 100
        avg_ret = (today_df.profit / (today_df.entry * today_df.volume)).mean() * 100
        msg = (
            f"📊 Daily Summary ({today} UTC)\n"
            f"Trades: {len(today_df)}\n"
            f"Win Rate: {wr:.1f}%\n"
            f"Avg Return per Trade: {avg_ret:.2f}%\n"
            f"Total P/L: {today_df.profit.sum():.5f}"
        )
    send_message_channel(msg)
