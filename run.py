# run.py
from app.main import main
from app.startup_notification import send_startup_message
from app.error_handler import report_exception

if __name__ == "__main__":
    try:
        send_startup_message()
        main()
    except Exception as e:
        report_exception(e)
        raise
