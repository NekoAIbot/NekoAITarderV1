# app/trading.py

import pandas as pd
import numpy as np
import random
import time
import os
from datetime import datetime

from config import get_today_symbols
from app.mt5_handler import initialize_mt5, shutdown_mt5
from app.models.basic_model import BasicModel
from app.telegram_bot import send_message
from app.state import increment_trade_count

# Read mock trade configs
MOCK_TRADE_VOLUME = float(os.getenv("MOCK_TRADE_VOLUME", 0.01))
MOCK_TRADE_HOLD_SECONDS = int(os.getenv("MOCK_TRADE_HOLD_SECONDS", 120))

def fetch_market_data(mt5, symbol):
    """Fetch real or dummy data for a single symbol."""
    if mt5:
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 100)
        return pd.DataFrame(rates)
    # dummy fallback
    df = pd.DataFrame({
        "open":   np.random.rand(100),
        "high":   np.random.rand(100),
        "low":    np.random.rand(100),
        "close":  np.random.rand(100),
        "volume": np.random.randint(1, 1000, size=100),
    })
    return df

def open_mock_trade(mt5, symbol, signal):
    """Open a small test trade."""
    if signal.lower() == "buy":
        order_type = mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(symbol).ask
    else:
        order_type = mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(symbol).bid

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": MOCK_TRADE_VOLUME,
        "type": order_type,
        "price": price,
        "deviation": 10,
        "magic": 123456,
        "comment": "Test Trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"✅ Mock trade opened: {symbol} Ticket {result.order}")
    else:
        print(f"❌ Trade open failed for {symbol} - Code {result.retcode}")
    return result

def close_mock_trade(mt5, ticket, symbol):
    """Close the opened mock trade."""
    position = mt5.positions_get(ticket=ticket)
    if not position:
        print(f"⚠️ No open position found for ticket {ticket}")
        return None

    position = position[0]
    price = mt5.symbol_info_tick(symbol).bid if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).ask

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "position": ticket,
        "symbol": symbol,
        "volume": position.volume,
        "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
        "price": price,
        "deviation": 10,
        "magic": 123456,
        "comment": "Close Test Trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"✅ Trade closed: {symbol} Ticket {ticket}")
        return result
    else:
        print(f"❌ Trade close failed for {symbol} Ticket {ticket}")
        return None

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

    for sym in symbols:
        df = fetch_market_data(mt5, sym)
        signal = model.predict(df)
        msg = f"{sym}: SIGNAL → {signal.upper()}"
        send_message(msg)

        if mt5:
            result = open_mock_trade(mt5, sym, signal)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                continue

            ticket = result.order

            print(f"⏳ Waiting {MOCK_TRADE_HOLD_SECONDS} seconds before closing...")
            time.sleep(MOCK_TRADE_HOLD_SECONDS)

            close_mock_trade(mt5, ticket, sym)
            final_profit = check_trade_profit(mt5, ticket)

            if final_profit is not None:
                win = final_profit > 0
                increment_trade_count(symbol=sym, win=win)
                print(f"🏁 {sym} trade result: {'WIN' if win else 'LOSS'} ({final_profit:.2f} profit)")
            else:
                print(f"⚠️ Could not determine trade outcome for {sym}, assuming loss.")
                increment_trade_count(symbol=sym, win=False)
        else:
            # fallback random win/loss
            simulated_result = random.choice([True, False])
            increment_trade_count(symbol=sym, win=simulated_result)

    shutdown_mt5(mt5)
