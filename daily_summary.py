from app.trade_logger import build_daily_summary
from app.telegram_bot import send_message_channel


def send_daily_summary():
    summary = build_daily_summary()
    send_message_channel(summary["message"])
