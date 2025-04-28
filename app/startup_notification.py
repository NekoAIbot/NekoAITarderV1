# app/startup_notification.py

from datetime import datetime
from app.telegram_bot import send_message, send_message_channel

def send_startup_message():
    """Send bot startup notification to personal chat and channel."""
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    message = (
        "🤖 NekoAI Trader Bot is now ONLINE!\n\n"
        f"🕒 {now}\n\n"
        "Ready to hunt 📈📉."
    )

    # Send to your personal chat
    send_message(message)
    # ALSO send to your channel
    send_message_channel(message)
