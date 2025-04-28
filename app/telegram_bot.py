# app/telegram_bot.py

import os
import requests

TELEGRAM_BOT_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID     = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_CHANNEL_ID  = os.getenv("TELEGRAM_CHANNEL_ID")

API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

def _post(text: str, chat_id: str):
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown"  # or HTML if you prefer
    }
    resp = requests.post(API_URL, data=payload, timeout=10)
    try:
        resp.raise_for_status()
    except requests.RequestException as e:
        print("❌ Telegram send failed:", e, resp.text)

def send_message(text: str):
    """Send a DM to your personal chat."""
    _post(text, TELEGRAM_CHAT_ID)

def send_message_channel(text: str):
    """Send a post to your Telegram channel."""
    _post(text, TELEGRAM_CHANNEL_ID)
