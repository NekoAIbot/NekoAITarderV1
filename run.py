# run.py

# =========================
# LOAD ENV FIRST (CRITICAL)
# =========================
from dotenv import load_dotenv
load_dotenv()  # MUST be first – before any app imports

import os

# Optional one-time debug (remove after verification)
print("BOT =", os.getenv("TELEGRAM_BOT_TOKEN"))
print("CHAT =", os.getenv("TELEGRAM_CHAT_ID"))

# =========================
# NOW import app modules
# =========================
from app.startup_notification import send_startup_message
from app.error_handler import report_exception
from app.scheduler import run_scheduler


if __name__ == "__main__":
    try:
        send_startup_message()
        run_scheduler()
    except Exception as e:
        report_exception(e)
        raise
