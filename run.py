# run.py
from app.startup_notification import send_startup_message
from app.error_handler       import report_exception
from app.scheduler           import run_scheduler

if __name__=="__main__":
    try:
        send_startup_message()
        run_scheduler()
    except Exception as e:
        report_exception(e)
        raise
