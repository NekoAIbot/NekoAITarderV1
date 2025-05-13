# app/main.py
import sys, os
# ensure project root on PYTHONPATH so `app.*` imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.startup_notification import send_startup_message
from app.error_handler       import report_exception
from app.scheduler           import run_scheduler

def main():
    send_startup_message()
    run_scheduler()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        report_exception(e)
        raise
