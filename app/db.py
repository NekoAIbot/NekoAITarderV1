import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

DB_PATH = Path(__file__).resolve().parent / "nekoai.db"


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runtime_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT,
                signal TEXT,
                volume REAL,
                entry_price REAL,
                exit_price REAL,
                profit REAL,
                win INTEGER
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT,
                signal TEXT,
                volume REAL,
                status TEXT,
                broker_ticket TEXT,
                details TEXT
            )
            """
        )


def set_state(key: str, value: str) -> None:
    init_db()
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO runtime_state(key, value, updated_at)
            VALUES(?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=CURRENT_TIMESTAMP
            """,
            (key, str(value)),
        )


def get_state(key: str, default: str) -> str:
    init_db()
    try:
        with get_conn() as conn:
            row = conn.execute("SELECT value FROM runtime_state WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default
    except sqlite3.OperationalError:
        return default


def insert_trade(timestamp: str, symbol: str, signal: str, volume: float, entry_price: float, exit_price: float, profit: float, win: bool) -> None:
    init_db()
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO trades(timestamp, symbol, signal, volume, entry_price, exit_price, profit, win)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (timestamp, symbol, signal, volume, entry_price, exit_price, profit, int(bool(win))),
        )


def insert_order(timestamp: str, symbol: str, signal: str, volume: float, status: str, broker_ticket: str = "", details: str = "") -> None:
    init_db()
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO orders(timestamp, symbol, signal, volume, status, broker_ticket, details)
            VALUES(?, ?, ?, ?, ?, ?, ?)
            """,
            (timestamp, symbol, signal, volume, status, broker_ticket, details),
        )
