# app/error_handler.py

import traceback
from app.telegram_bot import send_message, send_message_channel

def report_exception(e):
    error_text = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
    formatted = f"🚨 Bot Error 🚨\n\n<pre>{error_text[:4000]}</pre>"
    send_message(formatted, parse_mode=None)
    send_message_channel(formatted, parse_mode=None)
