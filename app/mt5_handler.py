#!/usr/bin/env python3
# File: app/mt5_handler.py

import os
import time
from config import USE_MOCK_MT5, MT5_LOGIN, MT5_PASSWORD, MT5_SERVER

def initialize_mt5():
    """Initialize and log in to MT5, or return None if using mock."""
    if USE_MOCK_MT5:
        print("⚠️ MOCK MT5 enabled—skipping real MT5 init.")
        return None

    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("❌ MetaTrader5 not installed.")
        return None

    if not mt5.initialize():
        print(f"❌ MT5 initialize failed: {mt5.last_error()}")
        return None

    if not mt5.login(login=int(MT5_LOGIN), password=MT5_PASSWORD, server=MT5_SERVER):
        print(f"❌ MT5 login failed: {mt5.last_error()}")
        mt5.shutdown()
        return None

    print("✅ MT5 initialized & logged in.")
    return mt5

def compute_trade_levels(entry_price: float,
                         opposite_price: float,
                         side: str,
                         symbol: str,
                         mt5=None) -> dict:
    """
    Compute SL/TP levels with a default 50% buffer,
    but a 100% buffer for USDCAD & NZDUSD to overcome their stricter freeze.
    """
    # Static fallback params
    fallback = {
        'EURUSD': {'digits':5, 'point':1e-5, 'trade_stops_level':20},
        'USDJPY': {'digits':3, 'point':1e-3, 'trade_stops_level':20},
        'USDCAD': {'digits':5, 'point':1e-5, 'trade_stops_level':20},
        'NZDUSD': {'digits':5, 'point':1e-5, 'trade_stops_level':20},
    }
    p = fallback.get(symbol, {})
    digits   = p.get('digits', 5)
    point    = p.get('point',   1e-5)
    min_pts  = p.get('trade_stops_level', 20)
    min_dist = min_pts * point

    # Override from live MT5 if available
    if mt5:
        try:
            import MetaTrader5 as mt5mod
            if mt5mod.symbol_select(symbol, True):
                info = mt5mod.symbol_info(symbol)
                if info:
                    digits   = info.digits
                    point    = info.point
                    min_pts  = getattr(info, 'trade_stops_level', min_pts)
                    min_dist = min_pts * point
        except Exception as e:
            print(f"[Warning] live params fetch failed for {symbol}: {e}")

    # SL/TP in pips from env
    sl_pips = float(os.getenv('SL_AMOUNT', 2))
    tp_pips = float(os.getenv('TP_AMOUNT', 3))
    pip_sz  = 0.01 if 'JPY' in symbol.upper() else 0.0001

    # raw offsets
    raw_sl_off = sl_pips * pip_sz
    raw_tp_off = tp_pips * pip_sz

    # base points
    base_sl_pts = max(int(raw_sl_off/point), min_pts)
    base_tp_pts = max(int(raw_tp_off/point), min_pts)

    # apply buffer: 2× for USDCAD & NZDUSD, else 1.5×
    buf = 2.0 if symbol.upper() in ('USDCAD', 'NZDUSD', 'USDJPY') else 1.5
    sl_off = base_sl_pts * point * buf
    tp_off = base_tp_pts * point * buf

    if side.upper() == 'BUY':
        sl_raw = entry_price - sl_off
        tp_raw = entry_price + tp_off
    else:
        sl_raw = entry_price + sl_off
        tp_raw = entry_price - tp_off

    # Debug logging
    print(f"[Debug] {symbol} | side={side} | entry={entry_price} | opposite={opposite_price}")
    print(f"[Debug] digits={digits}, point={point}, min_pts={min_pts}, min_dist={min_dist}")
    print(f"[Debug] Offsets → SL={sl_off}, TP={tp_off}")
    print(f"[Debug] Pre-round → SL={sl_raw}, TP={tp_raw}")

    # round to broker precision
    sl_level = round(sl_raw, digits)
    tp_level = round(tp_raw, digits)

    return {'sl_level': sl_level, 'tp_level': tp_level}

def open_trade(mt5, symbol: str, signal: str, volume: float):
    """Send a market order, iterating fill modes until success."""
    if signal.upper() == "HOLD":
        print(f"ℹ️ HOLD signal for {symbol}; skipping.")
        return None

    if mt5 is None:
        print(f"🔸 MOCK {signal.upper()} {symbol} vol={volume}")
        return {'mock': True}

    import MetaTrader5 as mt5mod

    if not mt5mod.symbol_select(symbol, True):
        print(f"❌ symbol_select failed for {symbol}")
        return None

    tick = mt5mod.symbol_info_tick(symbol)
    if not tick:
        print(f"❌ No tick data for {symbol}")
        return None

    entry    = tick.ask if signal.upper() == 'BUY' else tick.bid
    opposite = tick.bid if signal.upper() == 'BUY' else tick.ask

    levels = compute_trade_levels(entry, opposite, signal, symbol, mt5)
    sl, tp = levels['sl_level'], levels['tp_level']
    print(f"[Debug] Order params → entry={entry}, SL={sl}, TP={tp}, vol={volume}")

    deviation = int(os.getenv('MT5_DEVIATION', 20))
    modes = [
        mt5mod.ORDER_FILLING_RETURN,
        mt5mod.ORDER_FILLING_IOC,
        mt5mod.ORDER_FILLING_FOK,
    ]

    for idx, mode in enumerate(modes):
        req = {
            'action':       mt5mod.TRADE_ACTION_DEAL,
            'symbol':       symbol,
            'volume':       volume,
            'type':         mt5mod.ORDER_TYPE_BUY if signal.upper()=='BUY' else mt5mod.ORDER_TYPE_SELL,
            'price':        entry,
            'sl':           sl,
            'tp':           tp,
            'deviation':    deviation,
            'magic':        123456,
            'comment':      'Live Trade',
            'type_time':    mt5mod.ORDER_TIME_GTC,
            'type_filling': mode,
        }
        res = mt5mod.order_send(req)
        if res.retcode == mt5mod.TRADE_RETCODE_DONE:
            print(f"✅ Opened {signal.upper()} {symbol}: ticket={res.order}, SL={sl}, TP={tp} (mode={idx})")
            return res
        else:
            print(f"⚠️ Mode {idx} failed (retcode={res.retcode}); retrying…")
            time.sleep(1)

    print(f"❌ All fill modes failed for {symbol}")
    return None

def close_trade(mt5, ticket, symbol):
    """Close an existing position (or simulate it)."""
    if mt5 is None:
        print(f"🔸 MOCK close ticket={ticket} for {symbol}")
        return {'mock': True}, None

    import MetaTrader5 as mt5mod
    positions = mt5.positions_get(ticket=ticket)
    if not positions:
        print(f"❌ No position for ticket {ticket}")
        return None, None

    pos = positions[0]
    order_type = (mt5mod.ORDER_TYPE_SELL
                  if pos.type == mt5mod.ORDER_TYPE_BUY
                  else mt5mod.ORDER_TYPE_BUY)
    tick = mt5mod.symbol_info_tick(symbol)
    if not tick:
        print(f"❌ No tick data when closing {symbol}")
        return None, None

    price = tick.bid if order_type==mt5mod.ORDER_TYPE_BUY else tick.ask
    req = {
        'action':       mt5mod.TRADE_ACTION_DEAL,
        'position':     ticket,
        'symbol':       symbol,
        'volume':       pos.volume,
        'type':         order_type,
        'price':        price,
        'deviation':    10,
        'magic':        123456,
        'comment':      'Close Trade',
        'type_time':    mt5mod.ORDER_TIME_GTC,
        'type_filling': mt5mod.ORDER_FILLING_IOC,
    }
    res = mt5mod.order_send(req)
    if res.retcode == mt5mod.TRADE_RETCODE_DONE:
        print(f"✅ Trade closed: ticket {ticket}")
    else:
        print(f"❌ Close failed (retcode={res.retcode})")
    return res, price

def shutdown_mt5(mt5_module):
    """Cleanly shut down MT5 connection."""
    if mt5_module:
        mt5_module.shutdown()
        print("ℹ️ MT5 shut down.")
