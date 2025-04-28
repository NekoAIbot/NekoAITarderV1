# app/trading.py

import pandas as pd
import numpy as np
import random
import time
import os
from datetime import datetime
from config import get_today_symbols, SL_AMOUNT, TP_AMOUNT
from app.mt5_handler import initialize_mt5, shutdown_mt5
from app.models.basic_model import BasicModel
from app.telegram_bot import send_message, send_message_channel
from app.state import increment_trade_count
from app.risk_manager import RiskManager

MOCK_TRADE_HOLD_SECONDS = int(os.getenv("MOCK_TRADE_HOLD_SECONDS", 120))
PRE_SIGNAL_WAIT = 30  # seconds

def fetch_market_data(mt5, symbol):
    """Fetch real market data or fallback to dummy data."""
    if mt5:
        try:
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 100)
            if rates is not None and len(rates) > 0:
                df = pd.DataFrame(rates)
                return df
        except Exception as e:
            print(f"[Error] Fetching live market data failed: {e}")

    # fallback if MT5 unavailable
    return pd.DataFrame({
        "open":   np.random.rand(100),
        "high":   np.random.rand(100),
        "low":    np.random.rand(100),
        "close":  np.random.rand(100),
        "volume": np.random.randint(1, 1000, size=100),
    })

def open_mock_trade(mt5, symbol, signal, volume):
    """Open a mock trade."""
    if signal.lower() == "buy":
        order_type = mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(symbol).ask
    else:
        order_type = mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(symbol).bid

    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "deviation": 10,
        "magic": 123456,
        "comment": "Test Trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    res = mt5.order_send(req)
    return res, price

def close_mock_trade(mt5, ticket, symbol):
    """Close the opened mock trade and return exit price."""
    pos = mt5.positions_get(ticket=ticket)
    if not pos:
        return None, None
    pos = pos[0]
    if pos.type == mt5.ORDER_TYPE_BUY:
        exit_price = mt5.symbol_info_tick(symbol).bid
        close_type = mt5.ORDER_TYPE_SELL
    else:
        exit_price = mt5.symbol_info_tick(symbol).ask
        close_type = mt5.ORDER_TYPE_BUY

    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "position": ticket,
        "symbol": symbol,
        "volume": pos.volume,
        "type": close_type,
        "price": exit_price,
        "deviation": 10,
        "magic": 123456,
        "comment": "Close Test Trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    res = mt5.order_send(req)
    return res, exit_price

def check_trade_profit(mt5, ticket):
    """Get final profit after closing trade."""
    history = mt5.history_deals_get(datetime.now(), datetime.now())
    for deal in history:
        if deal.order == ticket:
            return deal.profit
    return None

def trading_job():
    """Main trading execution job."""
    symbols = get_today_symbols()
    print(f"[{datetime.utcnow()}] Running trading job for: {symbols}")

    mt5 = initialize_mt5()
    model = BasicModel()
    rm = RiskManager()

    for sym in symbols:
        # Pre-signal
        pre = (
            "⚠️ Risk Alert:\n"
            "Market conditions indicate heightened risk.\n"
            "Ensure proper risk management before proceeding.\n"
            "⏳ Preparing to drop a trade signal..."
        )
        send_message(pre)
        send_message_channel(pre)

        time.sleep(PRE_SIGNAL_WAIT)

        # Fetch Data + Predict
        df = fetch_market_data(mt5, sym)
        prediction = model.predict(df)
        ai_raw = prediction.get("signal", "HOLD").upper()
        confidence = prediction.get("confidence", 0.0)

        # Real unique ID with milliseconds
        ts = int(datetime.utcnow().timestamp() * 1000)
        signal_id = f"NekoAITrader_{ts}"

        # Entry & targets
        if mt5:
            entry = (
                mt5.symbol_info_tick(sym).ask if ai_raw == "BUY"
                else mt5.symbol_info_tick(sym).bid
            )
        else:
            entry = df["close"].iloc[-1]

        sl = entry - SL_AMOUNT
        tp1 = entry + TP_AMOUNT
        tp2 = entry + 2 * TP_AMOUNT
        tp3 = entry + 3 * TP_AMOUNT

        # Main signal formatting
        main = (
            "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
            "┃ 🚀 NekoAIBot Trade Signal 🚀 ┃\n"
            "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n\n"
            f"Signal ID: {signal_id}\n"
            f"Pair/Asset:   {sym}\n"
            "Predicted Change: N/A\n"
            "News Sentiment:    N/A\n"
            f"AI Signal:     {ai_raw}\n"
            f"Confidence:     {confidence:.2f}%\n\n"
            f"Entry:      {entry:.5f}\n"
            f"Stop Loss:  {sl:.5f}\n"
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
        boxed = f"<pre>{main}</pre>"
        send_message(boxed)
        send_message_channel(boxed)

        # Execute Trade
        volume = rm.get_lot()
        if mt5:
            res_open, _ = open_mock_trade(mt5, sym, ai_raw, volume)
            if res_open.retcode != mt5.TRADE_RETCODE_DONE:
                continue

            ticket = res_open.order
            print(f"⏳ Waiting {MOCK_TRADE_HOLD_SECONDS}s before closing…")
            time.sleep(MOCK_TRADE_HOLD_SECONDS)

            _, exit_price = close_mock_trade(mt5, ticket, sym)
            profit = check_trade_profit(mt5, ticket)
            win = (profit or 0) > 0
            increment_trade_count(sym, win=win)
            rm.adjust(win)
            print(f"🏁 {sym} trade {'WIN' if win else 'LOSS'} ({profit:.2f})")
        else:
            win = random.choice([True, False])
            increment_trade_count(sym, win=win)
            rm.adjust(win)

    shutdown_mt5(mt5)

# extra utility
def close_trade(mt5, ticket, symbol):
    """
    Close a given MT5 position at market price.
    Returns (result, exit_price).
    """
    positions = mt5.positions_get(ticket=ticket)
    if not positions:
        return None, None

    pos = positions[0]
    import MetaTrader5 as mt5mod

    if pos.type == mt5mod.ORDER_TYPE_BUY:
        exit_price = mt5mod.symbol_info_tick(symbol).bid
        close_type = mt5mod.ORDER_TYPE_SELL
    else:
        exit_price = mt5mod.symbol_info_tick(symbol).ask
        close_type = mt5mod.ORDER_TYPE_BUY

    request = {
        "action":      mt5mod.TRADE_ACTION_DEAL,
        "position":    ticket,
        "symbol":      symbol,
        "volume":      pos.volume,
        "type":        close_type,
        "price":       exit_price,
        "deviation":   10,
        "magic":       123456,
        "comment":     "Close Trade",
        "type_time":   mt5mod.ORDER_TIME_GTC,
        "type_filling":mt5mod.ORDER_FILLING_IOC,
    }
    result = mt5mod.order_send(request)
    return result, exit_price
