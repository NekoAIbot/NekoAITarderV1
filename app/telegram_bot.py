# app/telegram_bot.py

import os

try:
    import requests
except ImportError:
    requests = None

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID")
API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

def _post(text: str, chat_id: str, parse_mode: str = "HTML"):
    """Internal helper for sending Telegram messages."""
    if requests is None:
        # Fallback if requests not installed
        print("⚠️ Cannot send Telegram message: `requests` module not installed.")
        print(f"[Telegram->{chat_id}]: {text}")
        return

    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": parse_mode,
    }
    resp = None
    try:
        resp = requests.post(API_URL, data=payload, timeout=10)
        resp.raise_for_status()
    except requests.Timeout:
        print("⚠️ Telegram send timeout—skipping this message.")
    except requests.RequestException as e:
        err_text = getattr(resp, "text", "<no response>")
        print("❌ Telegram send failed:", e, err_text)


def send_message(text: str, parse_mode: str = "HTML"):
    """Send a direct message to the personal chat."""
    _post(text, TELEGRAM_CHAT_ID, parse_mode)


def send_message_channel(text: str, parse_mode: str = "HTML"):
    """Send a message to the Telegram channel."""
    _post(text, TELEGRAM_CHANNEL_ID, parse_mode)