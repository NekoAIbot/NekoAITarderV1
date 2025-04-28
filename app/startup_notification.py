# app/startup_notification.py
from datetime import datetime
from app.telegram_bot import send_message, send_message_channel

def send_startup_message():
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    msg = f"🤖 NekoAI Bot ONLINE\n🕒 {now}"
    send_message(msg)
    send_message_channel(msg)
