# app/trade_logger.py

import csv
from pathlib import Path
from datetime import datetime

# will live next to this file
LOG_FILE = Path(__file__).parent / "trades.csv"
HEADERS = [
    "timestamp",
    "symbol",
    "signal",
    "volume",
    "entry_price",
    "exit_price",
    "profit",
    "win",
]

def _ensure_file():
    if not LOG_FILE.exists():
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(HEADERS)

def log_trade(
    symbol: str,
    signal: str,
    volume: float,
    entry_price: float,
    exit_price: float,
    profit: float,
    win: bool
):
    """Append one trade’s details to trades.csv."""
    _ensure_file()
    timestamp = datetime.utcnow().isoformat()
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp,
            symbol,
            signal,
            volume,
            entry_price,
            exit_price,
            profit,
            int(win),
        ])
