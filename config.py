import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()  # load from .env

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
TESTING_MODE      = os.getenv("TESTING_MODE", "false").lower() == "true"

# Trading interval (minutes) when NOT in testing
TRADING_INTERVAL_MINUTES = int(os.getenv("TRADING_INTERVAL_MINUTES", 5))

# Asset lists
FOREX_MAJORS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD",
    "USDCAD", "USDCHF", "NZDUSD"
]
CRYPTO_ASSETS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT",
    "ADAUSDT", "DOTUSDT", "MATICUSDT"
]

def get_today_symbols():
    """Weekdays → Forex majors; Weekends → Crypto assets."""
    weekday = datetime.utcnow().weekday()  # 0=Mon … 6=Sun
    return FOREX_MAJORS if weekday < 5 else CRYPTO_ASSETS
