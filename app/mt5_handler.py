#!/usr/bin/env python3
# File: app/mt5_handler.py

import os
import time
import math
import requests
from config import USE_MOCK_MT5, MT5_LOGIN, MT5_PASSWORD, MT5_SERVER
from app.market_data import fetch_market_data
from app.telegram_bot import send_message_channel as _raw_send

def send_telegram(msg: str):
    """
    Send a Telegram message, backing off automatically on 429 Too Many Requests.
    """
    try:
        _raw_send(msg)
    except requests.HTTPError as e:
        if e.response.status_code == 429:
            retry = int(
                e.response
                 .json()
                 .get("parameters", {})
                 .get("retry_after", 1)
            )
            print(f"⚠️ Telegram 429, retry after {retry}s")
            time.sleep(retry)
            _raw_send(msg)
        else:
            print(f"❌ Telegram send failed: {e}")

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
    """Auto-detect broker’s exact symbol name and return (name, info)."""
    for candidate in (
        symbol.replace("USDT", "USD").upper(),
        symbol.replace("/", "").upper(),
        symbol.upper(),
    ):
        if mt5mod.symbol_select(candidate, True):
            info = mt5mod.symbol_info(candidate)
            if info:
                return candidate, info
    return None, None

def _get_env_float(name: str, default: float) -> float:
    raw = os.getenv(name, str(default))
    val = raw.split("#", 1)[0].strip()
    try:
        return float(val)
    except ValueError:
        return default

def compute_trade_levels(entry_price, side, info, symbol):
    digits       = info.digits
    point        = info.point
    raw_min_dist = info.trade_stops_level * point

    sl_pips    = _get_env_float("SL_AMOUNT", 2.0)
    pip_sz     = 0.01 if "JPY" in symbol else 0.0001
    min_by_pip = sl_pips * pip_sz

    fx_pairs = {"EURUSD", "USDJPY", "USDCAD", "NZDUSD"}
    min_dist = raw_min_dist if symbol.upper() in fx_pairs else min(raw_min_dist, min_by_pip)

    risk_dist = entry_price * 0.01
    sl_dist   = max(risk_dist, min_dist)
    tp_dist   = sl_dist * 1.5

    if side == "BUY":
        sl = entry_price - sl_dist
        tp = entry_price + tp_dist
    else:
        sl = entry_price + sl_dist
        tp = entry_price - tp_dist

    return {"sl": round(sl, digits), "tp": round(tp, digits), "digits": digits}

def normalize_volume(desired, info):
    min_vol = info.volume_min
    step    = info.volume_step or min_vol
    if desired <= min_vol:
        return min_vol
    steps = math.floor((desired - min_vol) / step)
    return round(min_vol + steps * step, 5)

def open_trade(mt5, symbol, signal, strat_vol):
    """
    Try to open a trade.
    If all filling modes fail due to position-limit, auto-close profitable positions.
    """
    import MetaTrader5 as mt5mod

    sig = signal.upper()
    if sig == "HOLD":
        print(f"ℹ️ Skipping HOLD for {symbol}")
        return None

    broker_sym, info = get_symbol_properties(mt5mod, symbol)
    if not broker_sym:
        print(f"❌ Symbol not found: {symbol}")
        return None

    # fetch entry price
    tick = None
    for _ in range(3):
        tick = mt5mod.symbol_info_tick(broker_sym)
        if tick:
            break
        time.sleep(0.5)
    entry = (tick.ask if sig == "BUY" else tick.bid) if tick else fetch_market_data(symbol)["close"].iat[-1]

    vol = normalize_volume(strat_vol, info)
    if vol != strat_vol:
        print(f"🔄 Adjusted volume {strat_vol:.5f}→{vol:.5f}")

    lvl = compute_trade_levels(entry, sig, info, broker_sym)
    print(f"[Debug] {broker_sym}|{sig}|entry={entry:.{lvl['digits']}f} SL={lvl['sl']} TP={lvl['tp']} vol={vol}")

    modes = [mt5mod.ORDER_FILLING_RETURN, mt5mod.ORDER_FILLING_IOC, mt5mod.ORDER_FILLING_FOK]
    req = {
        "action":      mt5mod.TRADE_ACTION_DEAL,
        "symbol":      broker_sym,
        "volume":      vol,
        "type":        mt5mod.ORDER_TYPE_BUY if sig == "BUY" else mt5mod.ORDER_TYPE_SELL,
        "price":       entry,
        "sl":          lvl["sl"],
        "tp":          lvl["tp"],
        "deviation":   int(os.getenv("MT5_DEVIATION", 500)),
        "magic":       123456,
        "comment":     "AutoTrade",
        "type_time":   mt5mod.ORDER_TIME_GTC,
    }

    for mode in modes:
        req["type_filling"] = mode
        res = mt5mod.order_send(req)
        if res.retcode == mt5mod.TRADE_RETCODE_DONE:
            msg = f"✅ {sig} {broker_sym} ticket={res.order} vol={vol:.2f}"
            print(msg)
            send_telegram(msg)
            return res
        print(f"⚠️ Mode {mode} failed: {res.comment}")
        time.sleep(0.5)

    # all modes failed
    err = getattr(res, "comment", "<no error>")
    print(f"❌ All modes failed for {broker_sym}: {err}")
    send_telegram(f"❌ All fills failed for {broker_sym}: {err}")

    if "Position limit reached" in err:
        pos_list = mt5mod.positions_get(symbol=broker_sym) or []
        lines = [f"⚠️ Position limit on {broker_sym}. Current positions:"]
        for p in pos_list:
            side_p = "BUY" if p.type == mt5mod.ORDER_TYPE_BUY else "SELL"
            lines.append(
                f" • ticket={p.ticket} side={side_p} vol={p.volume:.2f} open={p.price_open:.{info.digits}f}"
            )
        send_telegram("\n".join(lines))

        # auto-close winning ones
        for p in pos_list:
            side_p = "BUY" if p.type == mt5mod.ORDER_TYPE_BUY else "SELL"
            tick = mt5mod.symbol_info_tick(broker_sym)
            if not tick:
                continue
            price_now = tick.bid if p.type == mt5mod.ORDER_TYPE_BUY else tick.ask
            raw_pl = (price_now - p.price_open) * (1 if side_p == "BUY" else -1)
            if raw_pl > 0:
                _, _ = close_trade(mt5, p.ticket, broker_sym)

    return None

def close_trade(mt5, ticket, symbol):
    """Close a given position."""
    import MetaTrader5 as mt5mod

    if mt5 is None:
        print(f"🔸 MOCK close {ticket} for {symbol}")
        return {"mock": True}, None

    positions = mt5mod.positions_get(ticket=ticket)
    if not positions:
        print(f"❌ No position {ticket}")
        return None, None
    pos = positions[0]
    side = "SELL" if pos.type == mt5mod.ORDER_TYPE_BUY else "BUY"

    for _ in range(3):
        t = mt5mod.symbol_info_tick(symbol)
        if t:
            price = t.bid if side == "BUY" else t.ask
            break
        time.sleep(0.5)
    else:
        price = pos.price_open

    req = {
        "action":       mt5mod.TRADE_ACTION_DEAL,
        "position":     ticket,
        "symbol":       symbol,
        "volume":       pos.volume,
        "type":         mt5mod.ORDER_TYPE_BUY if side == "BUY" else mt5mod.ORDER_TYPE_SELL,
        "price":        price,
        "deviation":    50,
        "magic":        123456,
        "comment":      "Close Trade",
        "type_time":    mt5mod.ORDER_TIME_GTC,
        "type_filling": mt5mod.ORDER_FILLING_IOC,
    }
    res = mt5mod.order_send(req)
    if res.retcode in (mt5mod.TRADE_RETCODE_DONE, mt5mod.TRADE_RETCODE_DONE_PARTIAL):
        msg = f"✅ Closed {ticket}@{price:.5f}"
        print(msg)
        send_telegram(msg)
        return res, price

    print(f"❌ Close failed: {res.comment}")
    send_telegram(f"❌ Close failed for {ticket}: {res.comment}")
    return None, None

def monitor_positions(mt5):
    """
    Periodically call to report P/L on all open positions.
    - First run dumps an “initial snapshot” of every position.
    - Subsequent runs only report if P/L moves >0.01% (0.0001).
    - Aggregates all changes into one Telegram message per run.
    """
    import MetaTrader5 as mt5mod
    if mt5 is None:
        return

    # initial snapshot
    if not hasattr(monitor_positions, "_initial"):
        monitor_positions._initial = True
        monitor_positions._cache   = {}
        for p in mt5mod.positions_get() or []:
            ticket = p.ticket
            symbol = p.symbol
            side   = "BUY" if p.type==mt5mod.ORDER_TYPE_BUY else "SELL"
            open_p = p.price_open
            tick   = mt5mod.symbol_info_tick(symbol)
            if not tick: 
                continue
            now_p = tick.bid if side=="BUY" else tick.ask
            pl    = (now_p - open_p)*(1 if side=="BUY" else -1)*p.volume
            msg   = f"🔍 Initial pos {ticket} {symbol} {side} vol={p.volume:.2f} P/L={pl:.5f}"
            print(msg)
            send_telegram(msg)
            monitor_positions._cache[ticket] = pl

    # gather updates this run
    updates = []
    for p in mt5mod.positions_get() or []:
        ticket = p.ticket
        symbol = p.symbol
        side   = "BUY" if p.type==mt5mod.ORDER_TYPE_BUY else "SELL"
        open_p = p.price_open
        tick   = mt5mod.symbol_info_tick(symbol)
        if not tick:
            continue
        now_p = tick.bid if side=="BUY" else tick.ask
        pl    = (now_p - open_p)*(1 if side=="BUY" else -1)*p.volume
        last  = monitor_positions._cache.get(ticket, 0.0)
        # report if moved >0.01%
        if abs(pl-last)/(abs(last)+1e-8) > 0.0001:
            updates.append((ticket, symbol, side, p.volume, pl))
            monitor_positions._cache[ticket] = pl

    if updates:
        lines = ["📊 Position P/L updates:"]
        for t, s, sd, v, pl in updates:
            lines.append(f" • #{t} {s} {sd} vol={v:.2f} P/L={pl:.5f}")
        msg = "\n".join(lines)
        print(msg)
        send_telegram(msg)

def shutdown_mt5(mt5_module):
    """Cleanly shut down MT5 connection."""
    if mt5_module:
        mt5_module.shutdown()
        print("ℹ️ MT5 shut down.")

__all__ = [
    "initialize_mt5", "get_symbol_properties", "compute_trade_levels",
    "normalize_volume", "open_trade", "close_trade",
    "monitor_positions", "shutdown_mt5"
]
