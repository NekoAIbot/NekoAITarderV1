import csv
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = PROJECT_ROOT / "trade_activity.csv"

HEADERS = [
    "timestamp_utc",
    "symbol",
    "signal",
    "status",
    "volume",
    "entry_price",
    "exit_price",
    "profit",
    "win",
    "model_name",
    "indicator_used",
    "indicator_reason",
    "strategy_explanation",
    "sl_price",
    "tp1_price",
    "tp2_price",
    "tp3_price",
    "sl_calculation",
    "tp_calculation",
    "news_sentiment",
    "predicted_change_pct",
    "confidence_pct",
    "failure_reason",
    "notes",
]


def _ensure_file():
    if not LOG_FILE.exists():
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(HEADERS)


def log_trade_event(
    *,
    symbol: str,
    signal: str,
    status: str,
    volume: Optional[float] = None,
    entry_price: Optional[float] = None,
    exit_price: Optional[float] = None,
    profit: Optional[float] = None,
    win: Optional[bool] = None,
    model_name: str = "",
    indicator_used: str = "",
    indicator_reason: str = "",
    strategy_explanation: str = "",
    sl_price: Optional[float] = None,
    tp1_price: Optional[float] = None,
    tp2_price: Optional[float] = None,
    tp3_price: Optional[float] = None,
    sl_calculation: str = "",
    tp_calculation: str = "",
    news_sentiment: Optional[float] = None,
    predicted_change_pct: Optional[float] = None,
    confidence_pct: Optional[float] = None,
    failure_reason: str = "",
    notes: str = "",
):
    """Append a rich trading lifecycle event to project-level CSV."""
    _ensure_file()
    row = [
        datetime.now(timezone.utc).isoformat(),
        symbol,
        signal,
        status,
        "" if volume is None else volume,
        "" if entry_price is None else entry_price,
        "" if exit_price is None else exit_price,
        "" if profit is None else profit,
        "" if win is None else int(bool(win)),
        model_name,
        indicator_used,
        indicator_reason,
        strategy_explanation,
        "" if sl_price is None else sl_price,
        "" if tp1_price is None else tp1_price,
        "" if tp2_price is None else tp2_price,
        "" if tp3_price is None else tp3_price,
        sl_calculation,
        tp_calculation,
        "" if news_sentiment is None else news_sentiment,
        "" if predicted_change_pct is None else predicted_change_pct,
        "" if confidence_pct is None else confidence_pct,
        failure_reason,
        notes,
    ]

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def log_trade(symbol, signal, volume, entry_price, exit_price, profit, win):
    """Backward-compatible minimal logger wrapper used by backtester."""
    log_trade_event(
        symbol=symbol,
        signal=signal,
        status="closed",
        volume=volume,
        entry_price=entry_price,
        exit_price=exit_price,
        profit=profit,
        win=win,
        notes="legacy log_trade call",
    )


def _safe_float(v):
    try:
        if v is None or v == "":
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def build_daily_summary(target_date=None):
    """Return metrics dict and Telegram-ready summary for given UTC date."""
    _ensure_file()
    if target_date is None:
        target_date = datetime.now(timezone.utc).date()

    rows = []
    with open(LOG_FILE, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            ts = r.get("timestamp_utc", "")
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                continue
            if dt.date() == target_date:
                rows.append(r)

    if not rows:
        msg = f"📊 Daily Trading Summary ({target_date} UTC)\nNo signals/trades recorded today."
        return {"trades": 0, "signals": 0, "message": msg}

    signals = len(rows)
    executed = [r for r in rows if r.get("status") in {"executed", "closed"}]
    closed = [r for r in rows if r.get("status") == "closed"]
    failed = [r for r in rows if r.get("status") == "failed"]

    pnl_values = [_safe_float(r.get("profit")) for r in closed]
    pnl_values = [p for p in pnl_values if p is not None]
    total_pnl = sum(pnl_values) if pnl_values else 0.0

    win_vals = [r.get("win") for r in closed if r.get("win") in {"0", "1", 0, 1}]
    wins = sum(int(w) for w in win_vals)
    losses = len(win_vals) - wins
    win_rate = (wins / len(win_vals) * 100) if win_vals else 0.0

    assets = [r.get("symbol") for r in rows if r.get("symbol")]
    counter = Counter(assets)
    top_asset = counter.most_common(1)[0][0] if counter else "N/A"

    avg_conf = [_safe_float(r.get("confidence_pct")) for r in rows]
    avg_conf = [x for x in avg_conf if x is not None]
    avg_conf_v = sum(avg_conf) / len(avg_conf) if avg_conf else 0.0

    avg_move = [_safe_float(r.get("predicted_change_pct")) for r in rows]
    avg_move = [x for x in avg_move if x is not None]
    avg_move_v = sum(avg_move) / len(avg_move) if avg_move else 0.0

    msg = (
        f"📊 Daily Trading Summary ({target_date} UTC)\n"
        f"Signals Logged: {signals}\n"
        f"Trades Executed: {len(executed)}\n"
        f"Trades Closed: {len(closed)}\n"
        f"Failed Trade Attempts: {len(failed)}\n"
        f"Total PnL: {total_pnl:.5f}\n"
        f"Wins/Losses: {wins}/{losses} (WR {win_rate:.1f}%)\n"
        f"Most Traded Asset: {top_asset}\n"
        f"Avg Confidence: {avg_conf_v:.2f}%\n"
        f"Avg Predicted Change: {avg_move_v:.2f}%"
    )

    return {
        "signals": signals,
        "trades": len(executed),
        "closed": len(closed),
        "failed": len(failed),
        "total_pnl": total_pnl,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "most_traded_asset": top_asset,
        "avg_confidence_pct": avg_conf_v,
        "avg_predicted_change_pct": avg_move_v,
        "message": msg,
    }
