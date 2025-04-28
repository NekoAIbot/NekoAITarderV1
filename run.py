# run.py

from app.main import main
from app.error_handler import report_exception
from app.startup_notification import send_startup_message

if __name__ == "__main__":
    try:
        send_startup_message()  # ⬅️ Send online notification
        main()
    except Exception as e:
        report_exception(e)
        raise
