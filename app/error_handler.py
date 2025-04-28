# app/error_handler.py
import traceback
from app.telegram_bot import send_message, send_message_channel

def report_exception(e):
    err = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
    msg = f"🚨 Bot Error 🚨\n\n<pre>{err[:4000]}</pre>"
    send_message(msg)
    send_message_channel(msg)
