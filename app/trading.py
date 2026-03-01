import pandas as pd
import numpy as np
import random
import time
import os
from datetime import datetime
import joblib

from config import get_today_symbols, SL_AMOUNT, TP_AMOUNT, USE_MOCK_MT5
from app.market_data import fetch_market_data
from app.mt5_handler import initialize_mt5, shutdown_mt5, open_trade, close_trade
from app.news import get_news_sentiment
from app.telegram_bot import send_message, send_message_channel
from app.state import increment_trade_count
from app.risk_manager import RiskManager
from app.id_manager import IDManager

MOCK_TRADE_HOLD_SECONDS = int(os.getenv("MOCK_TRADE_HOLD_SECONDS", 120))
PRE_SIGNAL_WAIT        = 30  # seconds
MODEL_PATH = "/workspaces/NekoAITarderV1/models/production/fx_regime_models.joblib"

# Load production models bundle and normalize structure to
# production_models[regime] = {"model": model, "features": [...]}
if os.path.exists(MODEL_PATH):
    bundle = joblib.load(MODEL_PATH)
    raw_models = bundle.get("models", {})
    global_features = bundle.get("features", [])
    production_models = {}
    for k, m in raw_models.items():
        try:
            rk = int(k)
        except Exception:
            rk = k
        production_models[rk] = {"model": m, "features": global_features}
else:
    production_models = {}


def detect_regime(df):
    """Detect numeric volatility regime (0,1,2) from price series."""
    s = df.copy()
    ret = s["close"].pct_change()
    vol = ret.rolling(200).std()
    pct = vol.rolling(500).rank(pct=True)
    regime = pd.cut(pct, bins=[0.0, 0.33, 0.66, 1.0], labels=[0, 1, 2]).astype(float)
    return int(regime.iloc[-1])

from app.models.ai_model import MomentumModel as BasicModel  # fallback


def trading_job():
    symbols = get_today_symbols()
    print(f"[{datetime.utcnow()}] Trading cycle: {symbols}")

    mt5   = initialize_mt5()
    rm    = RiskManager()
    idm   = IDManager()

    for sym in symbols:
        pre = (
            "⚠️ Risk Alert:\n"
            "Market conditions indicate heightened risk.\n"
            "Ensure proper risk management.\n"
            "⏳ Preparing trade signal..."
        )
        send_message(pre)
        send_message_channel(pre)
        time.sleep(PRE_SIGNAL_WAIT)

        df  = fetch_market_data(sym)
        ns  = get_news_sentiment(sym)

        # Use production model if available
        regime = detect_regime(df)
        if regime in production_models:
            pm = production_models[regime]
            features = pm["features"]
            latest = df.iloc[-1:]
            pc = pm["model"].predict(latest[features])[0]
            sig = "BUY" if pc > 0 else "SELL"
            conf = abs(pc)
        else:
            model = BasicModel()
            out = model.predict(df, ns)
            sig = out.get("signal", "HOLD").upper()
            conf = out.get("confidence", 0.0)
            pc = out.get("predicted_change", 0.0)

        sid = idm.next()

        entry = None
        if mt5 and not USE_MOCK_MT5:
            try:
                import MetaTrader5 as mt5mod
                mt5mod.symbol_select(sym, True)
                tick = mt5mod.symbol_info_tick(sym)
                if tick and ((sig == "BUY" and tick.ask is not None) or (sig == "SELL" and tick.bid is not None)):
                    entry = tick.ask if sig == "BUY" else tick.bid
            except ImportError:
                entry = df['close'].iloc[-1]

        if entry is None:
            entry = df['close'].iloc[-1]

        pip = 0.01 if sym.endswith("JPY") else 0.0001
        sl_price  = entry - SL_AMOUNT * pip if sig == "BUY" else entry + SL_AMOUNT * pip
        tp1_price = entry + TP_AMOUNT * pip if sig == "BUY" else entry - TP_AMOUNT * pip
        tp2_price = entry + 2 * TP_AMOUNT * pip if sig == "BUY" else entry - 2 * TP_AMOUNT * pip
        tp3_price = entry + 3 * TP_AMOUNT * pip if sig == "BUY" else entry - 3 * TP_AMOUNT * pip

        box = (
            f"┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
            f"┃ 🚀 NekoAIBot Trade Signal 🚀 ┃\n"
            f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n\n"
            f"Signal ID: {sid}\n"
            f"Pair/Asset: {sym}\n"
            f"Predicted Change: {pc*100:.2f}%\n"
            f"News Sentiment: {ns:.1f}%\n"
            f"AI Signal: {sig}\n"
            f"Confidence: {conf*100:.1f}%\n\n"
            f"Entry: {entry:.5f}\n"
            f"Stop Loss: {sl_price:.5f}\n"
            f"——————————————\nTake Profits:\n"
            f"  • TP1: {tp1_price:.5f}\n"
            f"  • TP2: {tp2_price:.5f}\n"
            f"  • TP3: {tp3_price:.5f}\n\n"
            f"⚠️ Risk Warning: Trading involves significant risk.\n\n"
            f"┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
            f"┃   NekoAIBot - Next-Gen Trading   ┃\n"
            f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛"
        )
        send_message(f"<pre>{box}</pre>")
        send_message_channel(f"<pre>{box}</pre>")

        volume = rm.get_lot()
        res    = open_trade(mt5, sym, sig, volume)
        if not res or (hasattr(res, "retcode") and res.retcode != 0):
            continue
        ticket = getattr(res, "order", None)

        time.sleep(MOCK_TRADE_HOLD_SECONDS)
        close_trade(mt5, ticket, sym)

        profit = random.choice([TP_AMOUNT * pip, -SL_AMOUNT * pip])
        win = profit > 0
        increment_trade_count(sym, win=win)
        rm.adjust(win)
        print(f"🏁 {sym} {'WIN' if win else 'LOSS'} ({profit:.5f})")

    shutdown_mt5(mt5)
