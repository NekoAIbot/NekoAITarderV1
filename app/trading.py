# app/trading.py
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime

from config import get_today_symbols, SL_AMOUNT, TP_AMOUNT, USE_MOCK_MT5
from app.market_data import fetch_twelvedata
from app.mt5_handler import initialize_mt5, shutdown_mt5, open_trade
from app.models.basic_model import BasicModel
from app.telegram_bot import send_message, send_message_channel
from app.state import increment_trade_count
from app.risk_manager import RiskManager
from app.id_manager import IDManager

MOCK_TRADE_HOLD_SECONDS = int(os.getenv("MOCK_TRADE_HOLD_SECONDS", 120))
PRE_SIGNAL_WAIT        = 30

def fetch_market_data(mt5, symbol):
    if mt5:
        data = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 100)
        return pd.DataFrame(data)
    else:
        return fetch_twelvedata(symbol)

def trading_job():
    symbols = get_today_symbols()
    print(f"[{datetime.utcnow()}] Cycle: {symbols}")

    mt5   = initialize_mt5()
    model = BasicModel()
    rm    = RiskManager()
    idm   = IDManager()

    for sym in symbols:
        # 1) pre-signal
        pre = (
            "⚠️ Risk Alert:\n"
            "Market conditions indicate heightened risk.\n"
            "Ensure proper risk management.\n"
            "⏳ Preparing signal..."
        )
        send_message(pre); send_message_channel(pre)
        time.sleep(PRE_SIGNAL_WAIT)

        # 2) fetch data & predict
        df  = fetch_market_data(mt5, sym)
        out = model.predict(df)
        sig, conf, pc, ns = out["signal"], out["confidence"], out["predicted_change"], out["news_sentiment"]
        sid = idm.next()

        # 3) entry & levels
        if mt5 and not USE_MOCK_MT5:
            tick  = mt5.symbol_info_tick(sym)
            entry = tick.ask if sig=="BUY" else tick.bid
        else:
            entry = df["close"].iloc[-1]
        sl  = entry - SL_AMOUNT
        tp1 = entry + TP_AMOUNT
        tp2 = entry + 2*TP_AMOUNT
        tp3 = entry + 3*TP_AMOUNT

        # 4) boxed signal
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
            f"Stop Loss: {sl:.5f}\n"
            "——————————————\n"
            "Take Profits:\n"
            f"  • TP1: {tp1:.5f}\n"
            f"  • TP2: {tp2:.5f}\n"
            f"  • TP3: {tp3:.5f}\n\n"
            "⚠️ Risk Warning: Trading involves significant risk.\n\n"
            "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
            "┃   NekoAIBot - Next-Gen Trading   ┃\n"
            "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛"
        )
        boxed = f"<pre>{box}</pre>"
        send_message(boxed); send_message_channel(boxed)

        # 5) execute
        vol = rm.get_lot()
        if mt5 and not USE_MOCK_MT5:
            res = open_trade(mt5, sym, sig, vol, sl, tp1)
        else:
            res = open_trade(mt5, sym, sig, vol, 0, 0)

        if res.retcode != 0:
            continue

        ticket = res.order
        time.sleep(MOCK_TRADE_HOLD_SECONDS)

        # close
        from app.trading import close_trade
        res_close, _ = close_trade(mt5, ticket, sym)
        profit = None
        try:
            history = mt5.history_deals_get(datetime.now(), datetime.now())
            profit  = history[-1].profit
        except:
            profit = 0.0

        win = profit > 0
        increment_trade_count(sym, win)
        rm.adjust(win)
        print(f"🏁 {sym} {'WIN' if win else 'LOSS'} ({profit:.2f})")

    shutdown_mt5(mt5)

