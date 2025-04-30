# app/trading.py

import pandas as pd
import numpy as np
import random
import time
import os
from datetime import datetime

from config import get_today_symbols, SL_AMOUNT, TP_AMOUNT, USE_MOCK_MT5
from app.market_data import fetch_twelvedata
from app.mt5_handler import initialize_mt5, shutdown_mt5, open_trade, close_trade
from app.models.basic_model import BasicModel
from app.telegram_bot import send_message, send_message_channel
from app.state import increment_trade_count
from app.risk_manager import RiskManager
from app.id_manager import IDManager

MOCK_TRADE_HOLD_SECONDS = int(os.getenv("MOCK_TRADE_HOLD_SECONDS", 120))
PRE_SIGNAL_WAIT        = 30  # seconds


def fetch_market_data(mt5, symbol):
    if mt5 and not USE_MOCK_MT5:
        try:
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 100)
            if rates is not None and len(rates):
                return pd.DataFrame(rates)
        except Exception as e:
            print(f"[Error] MT5 data fetch for {symbol}: {e}")
    try:
        return fetch_twelvedata(symbol)
    except Exception as e:
        print(f"[Error] TwelveData fetch for {symbol}: {e}")
    return pd.DataFrame({
        "open":   np.random.rand(100),
        "high":   np.random.rand(100),
        "low":    np.random.rand(100),
        "close":  np.random.rand(100),
        "volume": np.random.randint(1,1000,size=100),
    })


def trading_job():
    """Main trading execution: pre-signal, AI signal, execute trades."""
    symbols = get_today_symbols()
    print(f"[{datetime.utcnow()}] Trading cycle: {symbols}")

    mt5   = initialize_mt5()
    model = BasicModel()
    rm    = RiskManager()
    idm   = IDManager()

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

        # 2) Fetch data & AI predict
        df  = fetch_market_data(mt5, sym)
        out = model.predict(df)
        sig = out.get("signal", "HOLD").upper()
        conf= out.get("confidence", 0.0)
        pc  = out.get("predicted_change", 0.0)
        ns  = out.get("news_sentiment", "N/A")
        sid = idm.next()

        # 3) Determine entry price for display
        if mt5 and not USE_MOCK_MT5:
            import MetaTrader5 as mt5mod
            tick  = mt5mod.symbol_info_tick(sym)
            entry = tick.ask if sig == "BUY" else tick.bid
        else:
            entry = df["close"].iloc[-1]

        # 4) Display-only SL/TP
        pip = 0.01 if sym.endswith("JPY") else 0.0001
        sl_price  = entry - SL_AMOUNT * pip if sig == "BUY" else entry + SL_AMOUNT * pip
        tp1_price = entry + TP_AMOUNT * pip if sig == "BUY" else entry - TP_AMOUNT * pip
        tp2_price = entry + 2 * TP_AMOUNT * pip if sig == "BUY" else entry - 2 * TP_AMOUNT * pip
        tp3_price = entry + 3 * TP_AMOUNT * pip if sig == "BUY" else entry - 3 * TP_AMOUNT * pip

        # 5) Send boxed main signal
        box = (
            "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
            "┃ 🚀 NekoAIBot Trade Signal 🚀 ┃\n"
            "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n\n"
            f"Signal ID: {sid}\n"
            f"Pair/Asset:       {sym}\n"
            f"Predicted Change: {pc:.2f}%\n"
            f"News Sentiment:   {ns}\n"
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

        # 6) Execute trade with SL/TP auto detection
        volume = rm.get_lot()
        res    = open_trade(mt5, sym, sig, volume)
        if not res or (hasattr(res, "retcode") and res.retcode != 0):
            continue
        ticket = getattr(res, "order", None)

        # 7) Hold then close
        time.sleep(MOCK_TRADE_HOLD_SECONDS)
        close_trade(mt5, ticket, sym)

        # 8) Compute profit and update stats
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
        increment_trade_count(sym, win=win)
        rm.adjust(win)
        print(f"🏁 {sym} {'WIN' if win else 'LOSS'} ({profit:.5f})")

    shutdown_mt5(mt5)
