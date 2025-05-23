# config.py
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name, str(default))
    # drop inline comments and strip whitespace
    cleaned = raw.split("#", 1)[0].strip()
    try:
        return float(cleaned)
    except ValueError:
        return default

# Telegram
TELEGRAM_BOT_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID     = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_CHANNEL_ID  = os.getenv("TELEGRAM_CHANNEL_ID")

# MT5
MT5_LOGIN    = os.getenv("MT5_LOGIN")
MT5_PASSWORD = os.getenv("MT5_PASSWORD")
MT5_SERVER   = os.getenv("MT5_SERVER")
USE_MOCK_MT5 = os.getenv("USE_MOCK_MT5", "false").lower() == "true"

# API Keys
FOREX_API_KEY        = os.getenv("FOREX_API_KEY")
TWELVEDATA_API_KEY   = os.getenv("TWELVEDATA_API_KEY")
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
NEWSAPI_KEY          = os.getenv("NEWSAPI_KEY")
ECONOMIC_API_KEY     = os.getenv("ECONOMIC_API_KEY")
OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY")

# Feature flags
EXHAUSTIVE_SEARCH = os.getenv("EXHAUSTIVE_SEARCH", "false").lower() == "true"
TESTING_MODE      = os.getenv("TESTING_MODE",      "false").lower() == "true"

# Trading interval
TRADING_INTERVAL_MINUTES = int(os.getenv("TRADING_INTERVAL_MINUTES", "5"))

# Position-sizing
LOT_MIN            = _get_float("LOT_MIN",            0.01)
LOT_MAX            = _get_float("LOT_MAX",            0.20)
LOT_BASE           = _get_float("LOT_BASE",           LOT_MIN)
LOT_ADJUST_PERCENT = _get_float("LOT_ADJUST_PERCENT", 10.0)

# SL/TP defaults
SL_AMOUNT = _get_float("SL_AMOUNT", 2.0)
TP_AMOUNT = _get_float("TP_AMOUNT", 3.0)

# Assets
FOREX_MAJORS  = ["EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD","USDCHF","NZDUSD"]
CRYPTO_ASSETS = ["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","ADAUSDT","DOTUSDT","XRPUSDT"]

def get_today_symbols():
    """
    Weekdays: trade forex; weekends: trade crypto.
    """
    weekday = datetime.utcnow().weekday()
    return FOREX_MAJORS if weekday < 5 else CRYPTO_ASSETS
