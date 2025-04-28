# app/error_handler.py

import traceback
from app.telegram_bot import send_message

def report_exception(e):
    """Format and send error to Telegram."""
    error_text = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
    message = f"🚨 Bot Error Detected 🚨\n\n{error_text[:4000]}"  # Telegram max = 4096 chars
    send_message(message)
