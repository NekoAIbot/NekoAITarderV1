# app/mt5_handler.py
from config import USE_MOCK_MT5, MT5_LOGIN, MT5_PASSWORD, MT5_SERVER

def initialize_mt5():
    if USE_MOCK_MT5:
        print("MOCK MT5 enabled—skipping real MT5 init.")
        return None
    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("MetaTrader5 not installed—skipping.")
        return None
    if not mt5.initialize(login=int(MT5_LOGIN),
                          password=MT5_PASSWORD,
                          server=MT5_SERVER):
        print("MT5 init failed:", mt5.last_error())
        return None
    print("MT5 initialized.")
    return mt5

def shutdown_mt5(mt5_module):
    if mt5_module:
        mt5_module.shutdown()
        print("MT5 shut down.")

def open_trade(mt5, symbol, signal, volume, sl, tp):
    """Place a real MT5 order with SL/TP."""
    import MetaTrader5 as mt5mod
    order_type = mt5mod.ORDER_TYPE_BUY if signal.lower()=="buy" else mt5mod.ORDER_TYPE_SELL
    tick = mt5mod.symbol_info_tick(symbol)
    price = tick.ask if order_type==mt5mod.ORDER_TYPE_BUY else tick.bid
    request = {
        "action":     mt5mod.TRADE_ACTION_DEAL,
        "symbol":     symbol,
        "volume":     volume,
        "type":       order_type,
        "price":      price,
        "sl":         sl,
        "tp":         tp,
        "deviation":  10,
        "magic":      123456,
        "comment":    "Live Trade",
        "type_time":  mt5mod.ORDER_TIME_GTC,
        "type_filling":mt5mod.ORDER_FILLING_IOC,
    }
    res = mt5mod.order_send(request)
    if res.retcode == mt5mod.TRADE_RETCODE_DONE:
        print(f"✅ Opened trade {symbol} Ticket {res.order}")
    else:
        print(f"❌ Open failed {symbol} Code {res.retcode}")
    return res
