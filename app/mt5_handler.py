#!/usr/bin/env python3
# File: app/mt5_handler.py

import os
import time
import math
from config import USE_MOCK_MT5, MT5_LOGIN, MT5_PASSWORD, MT5_SERVER
from app.market_data import fetch_market_data

def initialize_mt5():
    """Initialize and log in to MT5 (or return None if using mock)."""
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


def get_symbol_properties(mt5mod, symbol: str):
    """Auto-detect the broker’s exact symbol name and return (name, info)."""
    variations = [
        symbol.replace('USDT', 'USD').upper(),
        symbol.replace('/', '').upper(),
        symbol.upper()
    ]
    for sym in variations:
        if mt5mod.symbol_select(sym, True):
            info = mt5mod.symbol_info(sym)
            if info:
                return sym, info
    return None, None


def compute_trade_levels(entry_price: float,
                         side: str,
                         info,
                         symbol: str) -> dict:
    """
    Calculate SL/TP:
    - For FX (EURUSD, USDJPY, USDCAD, NZDUSD) we respect the broker's trade_stops_level.
    - For everything else we clamp that to our own pip‐based minimum.
    """
    digits = info.digits
    point  = info.point
    raw_min_dist = info.trade_stops_level * point

    # our pip‐based minimum from SL_AMOUNT
    sl_pips = float(os.getenv('SL_AMOUNT', 2))
    pip_sz  = 0.01 if 'JPY' in symbol.upper() else 0.0001
    min_dist_by_pip = sl_pips * pip_sz

    fx_pairs = {'EURUSD','USDJPY','USDCAD','NZDUSD'}
    if symbol.upper() not in fx_pairs:
        min_dist = min(raw_min_dist, min_dist_by_pip)
    else:
        min_dist = raw_min_dist

    # risk‐based distance = 1% of entry
    risk_dist = entry_price * 0.01
    sl_dist = max(risk_dist, min_dist)
    tp_dist = sl_dist * 1.5

    if side.upper() == 'BUY':
        sl = entry_price - sl_dist
        tp = entry_price + tp_dist
    else:
        sl = entry_price + sl_dist
        tp = entry_price - tp_dist

    return {
        'sl': round(sl, digits),
        'tp': round(tp, digits),
        'digits': digits
    }


def normalize_volume(desired: float, info) -> float:
    """Raise volume up to the broker’s minimum/step—never skip."""
    min_vol  = info.volume_min
    step     = info.volume_step or min_vol
    if desired <= min_vol:
        return min_vol
    steps = math.floor((desired - min_vol) / step)
    return round(min_vol + steps * step, 5)


def open_trade(mt5, symbol: str, signal: str, strat_vol: float):
    """Universal trade: detect symbol, normalize vol, compute levels, try all fillings."""
    import MetaTrader5 as mt5mod
    sig = signal.upper()
    if sig == "HOLD":
        print(f"ℹ️ Skipping HOLD for {symbol}")
        return None

    broker_sym, info = get_symbol_properties(mt5mod, symbol)
    if not broker_sym:
        print(f"❌ Symbol not found: {symbol}")
        return None

    # fetch live tick or fallback to last-close
    tick = None
    for _ in range(3):
        tick = mt5mod.symbol_info_tick(broker_sym)
        if tick: break
        time.sleep(0.5)
    if not tick:
        print(f"❌ No tick data for {broker_sym}, using last-close")
        df = fetch_market_data(symbol)
        entry = df["close"].iat[-1]
    else:
        entry = tick.ask if sig=='BUY' else tick.bid

    vol = normalize_volume(strat_vol, info)
    if vol != strat_vol:
        print(f"🔄 Adjusted volume {strat_vol:.5f}→{vol:.5f}")

    lvl = compute_trade_levels(entry, sig, info, broker_sym)
    print(f"[Debug] {broker_sym} | {sig} | entry={entry:.{lvl['digits']}f} SL={lvl['sl']} TP={lvl['tp']} vol={vol}")

    # always try all three filling modes
    modes = [mt5mod.ORDER_FILLING_RETURN, mt5mod.ORDER_FILLING_IOC, mt5mod.ORDER_FILLING_FOK]
    req = {
        'action':    mt5mod.TRADE_ACTION_DEAL,
        'symbol':    broker_sym,
        'volume':    vol,
        'type':      mt5mod.ORDER_TYPE_BUY if sig=='BUY' else mt5mod.ORDER_TYPE_SELL,
        'price':     entry,
        'sl':        lvl['sl'],
        'tp':        lvl['tp'],
        'deviation': int(os.getenv('MT5_DEVIATION', 500)),
        'magic':     123456,
        'comment':   'AutoTrade',
        'type_time': mt5mod.ORDER_TIME_GTC,
    }
    for mode in modes:
        req['type_filling'] = mode
        res = mt5mod.order_send(req)
        if res.retcode == mt5mod.TRADE_RETCODE_DONE:
            print(f"✅ {sig} {broker_sym} ticket={res.order}")
            return res
        print(f"⚠️ Mode {mode} failed: {res.comment}")
        time.sleep(0.5)

    print(f"❌ All modes failed for {broker_sym}")
    return None


def close_trade(mt5, ticket, symbol):
    """Close position—retry price, use IOC."""
    if mt5 is None:
        print(f"🔸 MOCK close {ticket} for {symbol}")
        return {'mock':True}, None

    import MetaTrader5 as mt5mod
    positions = mt5mod.positions_get(ticket=ticket)
    if not positions:
        print(f"❌ No position {ticket}")
        return None, None
    pos = positions[0]
    broker_sym = pos.symbol
    side = 'SELL' if pos.type==mt5mod.ORDER_TYPE_BUY else 'BUY'

    price = None
    for _ in range(3):
        tick = mt5mod.symbol_info_tick(broker_sym)
        if tick:
            price = tick.bid if side=='BUY' else tick.ask
            break
        time.sleep(0.5)
    if price is None:
        price = pos.price_open

    req = {
        'action':       mt5mod.TRADE_ACTION_DEAL,
        'position':     ticket,
        'symbol':       broker_sym,
        'volume':       pos.volume,
        'type':         mt5mod.ORDER_TYPE_BUY if side=='BUY' else mt5mod.ORDER_TYPE_SELL,
        'price':        price,
        'deviation':    50,
        'magic':        123456,
        'comment':      'Close Trade',
        'type_time':    mt5mod.ORDER_TIME_GTC,
        'type_filling': mt5mod.ORDER_FILLING_IOC,
    }
    res = mt5mod.order_send(req)
    if res.retcode in (mt5mod.TRADE_RETCODE_DONE, mt5mod.TRADE_RETCODE_DONE_PARTIAL):
        print(f"✅ Closed {ticket}@{price:.5f}")
        return res, price
    print(f"❌ Close failed: {res.comment}")
    return None, None


def shutdown_mt5(mt5_module):
    """Cleanly shut down MT5 connection."""
    if mt5_module:
        mt5_module.shutdown()
        print("ℹ️ MT5 shut down.")
