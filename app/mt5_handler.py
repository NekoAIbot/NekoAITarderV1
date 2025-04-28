# app/mt5_handler.py
from config import USE_MOCK_MT5, MT5_LOGIN, MT5_PASSWORD, MT5_SERVER

def initialize_mt5():
    if USE_MOCK_MT5:
        print("MOCK MT5 enabled—skipping real MT5 init.")
        return None

    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("MetaTrader5 not installed – skipping.")
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
