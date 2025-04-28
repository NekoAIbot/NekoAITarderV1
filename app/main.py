# app/main.py

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.scheduler import run_scheduler

def main():
    run_scheduler()

if __name__ == "__main__":
    main()
