# run.py

# =========================
# LOAD ENV FIRST (CRITICAL)
# =========================
from dotenv import load_dotenv
load_dotenv()  # MUST be first – before any app imports

import os

# =========================
# NOW import app modules
# =========================
from app.logging_utils import setup_logging
from app.db import init_db
from app.startup_notification import send_startup_message
from app.error_handler import report_exception
from app.scheduler import run_scheduler


if __name__ == "__main__":
    setup_logging()
    init_db()
    try:
        send_startup_message()
        run_scheduler()
    except Exception as e:
        report_exception(e)
        raise
