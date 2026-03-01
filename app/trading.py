import pandas as pd
import numpy as np
import random
import time
import os
import logging
from datetime import datetime

from config import get_today_symbols, SL_AMOUNT, TP_AMOUNT, USE_MOCK_MT5, MIN_CONFIDENCE_PCT
from app.market_data import fetch_market_data
from app.mt5_handler import initialize_mt5, shutdown_mt5, open_trade, close_trade
from app.models.ai_model import MomentumModel as BasicModel
from app.news import get_news_sentiment
from app.telegram_bot import send_message, send_message_channel
from app.state import increment_trade_count, get_bot_status
from app.risk_manager import RiskManager
from app.id_manager import IDManager
from app.db import insert_order
from app.risk_engine import RiskEngine

MOCK_TRADE_HOLD_SECONDS = int(os.getenv("MOCK_TRADE_HOLD_SECONDS", 120))
PRE_SIGNAL_WAIT        = 30  # seconds

logger = logging.getLogger(__name__)

def trading_job():
    """Main trading execution: pre-signal, AI signal, execute trades."""
    symbols = get_today_symbols()
    logger.info(f"Trading cycle started | symbols={symbols}")

    mt5   = initialize_mt5()
    model = BasicModel()
    rm    = RiskManager()
    idm   = IDManager()
    risk  = RiskEngine()

    for sym in symbols:
        # 1) Pre-signal alert
        pre = (
            "⚠️ Risk Alert:\n"
            "Market conditions indicate heightened risk.\n"
            "Ensure proper risk management.\n"
            "⏳ Preparing trade signal..."
        )
        send_message(pre)
        send_message_channel(pre)
        time.sleep(PRE_SIGNAL_WAIT)

        # 2) Fetch data & live news sentiment
        df  = fetch_market_data(sym)
        ns  = get_news_sentiment(sym)

        # 3) AI predict (pass in news)
        out  = model.predict(df, ns)
        sig  = out.get("signal", "HOLD").upper()
        conf = out.get("confidence", 0.0)
        pc   = out.get("predicted_change", 0.0)

        # risk guardrails
        if conf < MIN_CONFIDENCE_PCT:
            logger.info(f"Skipping {sym}: confidence {conf:.1f}% < {MIN_CONFIDENCE_PCT:.1f}%")
            continue

        _, _, _, _, _, pnl = get_bot_status()
        allowed, reason = risk.can_open(sym, pnl)
        if not allowed:
            msg = f"🛑 Risk engine blocked {sym}: {reason} (pnl={pnl:.5f})"
            logger.warning(msg)
            send_message(msg)
            send_message_channel(msg)
            if reason == "daily_loss_limit":
                break
            continue

        sid = idm.next()

        # 4) Determine entry price
        entry = None
        if mt5 and not USE_MOCK_MT5:
            import MetaTrader5 as mt5mod
            mt5mod.symbol_select(sym, True)
            tick = mt5mod.symbol_info_tick(sym)
            if tick and ((sig == "BUY" and tick.ask is not None)
                         or (sig == "SELL" and tick.bid is not None)):
                entry = tick.ask if sig == "BUY" else tick.bid
            else:
                print(f"❌ No tick data for {sym}, falling back to last close.")
        if entry is None:
            entry = df["close"].iloc[-1]

        # 5) Display-only SL/TP
        pip = 0.01 if sym.endswith("JPY") else 0.0001
        sl_price  = entry - SL_AMOUNT * pip if sig == "BUY" else entry + SL_AMOUNT * pip
        tp1_price = entry + TP_AMOUNT * pip if sig == "BUY" else entry - TP_AMOUNT * pip
        tp2_price = entry + 2 * TP_AMOUNT * pip if sig == "BUY" else entry - 2 * TP_AMOUNT * pip
        tp3_price = entry + 3 * TP_AMOUNT * pip if sig == "BUY" else entry - 3 * TP_AMOUNT * pip

        # 6) Send boxed main signal
        box = (
            "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
            "┃ 🚀 NekoAIBot Trade Signal 🚀 ┃\n"
            "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n\n"
            f"Signal ID: {sid}\n"
            f"Pair/Asset:       {sym}\n"
            f"Predicted Change: {pc:.2f}%\n"
            f"News Sentiment:   {ns:.1f}%\n"
            f"AI Signal:        {sig}\n"
            f"Confidence:       {conf:.1f}%\n\n"
            f"Entry:     {entry:.5f}\n"
            f"Stop Loss: {sl_price:.5f}\n"
            "——————————————\n"
            "Take Profits:\n"
            f"  • TP1: {tp1_price:.5f}\n"
            f"  • TP2: {tp2_price:.5f}\n"
            f"  • TP3: {tp3_price:.5f}\n\n"
            "⚠️ Risk Warning: Trading involves significant risk.\n\n"
            "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
            "┃   NekoAIBot - Next-Gen Trading   ┃\n"
            "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛"
        )
        send_message(f"<pre>{box}</pre>")
        send_message_channel(f"<pre>{box}</pre>")

        # 7) Execute trade
        volume = rm.get_lot()
        insert_order(datetime.utcnow().isoformat(), sym, sig, volume, "submitted")
        risk.on_open(sym)
        res    = open_trade(mt5, sym, sig, volume)
        if not res or (hasattr(res, "retcode") and res.retcode != 0):
            insert_order(datetime.utcnow().isoformat(), sym, sig, volume, "failed")
            risk.on_close(sym)
            continue
        ticket = getattr(res, "order", None)
        insert_order(datetime.utcnow().isoformat(), sym, sig, volume, "filled", str(ticket or ""))

        # 8) Hold then close
        time.sleep(MOCK_TRADE_HOLD_SECONDS)
        close_trade(mt5, ticket, sym)
        risk.on_close(sym)

        # 9) Profit & stats
        profit = 0.0
        if mt5 and hasattr(mt5, "history_deals_get"):
            try:
                h = mt5.history_deals_get(datetime.utcnow(), datetime.utcnow())
                profit = h[-1].profit
            except:
                pass
        else:
            profit = random.choice([TP_AMOUNT * pip, -SL_AMOUNT * pip])

        win = profit > 0
        increment_trade_count(sym, win=win, profit=profit)
        rm.adjust(win)
        logger.info(f"{sym} {'WIN' if win else 'LOSS'} ({profit:.5f})")

    shutdown_mt5(mt5)
