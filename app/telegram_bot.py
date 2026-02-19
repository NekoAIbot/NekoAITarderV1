# app/telegram_bot.py

import os
import sys

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    requests = None


# =========================
# Environment configuration
# =========================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID")

TELEGRAM_ENABLED = all([TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID])

if TELEGRAM_ENABLED:
    API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
else:
    API_URL = None


# =========================
# Create resilient session
# =========================
session = None

if requests is not None:
    session = requests.Session()

    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["POST"],
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)


# =========================
# Startup validation
# =========================
def telegram_health_check():
    if not TELEGRAM_ENABLED:
        print("⚠️ Telegram disabled: missing BOT TOKEN or CHAT ID")
        return False

    if requests is None:
        print("⚠️ Telegram disabled: `requests` module not installed")
        return False

    try:
        r = session.post(
            API_URL,
            data={"chat_id": TELEGRAM_CHAT_ID, "text": "✅ Telegram connected"},
            timeout=5,
        )
        r.raise_for_status()
        print("✅ Telegram health check passed")
        return True
    except Exception as e:
        print("❌ Telegram health check failed:", e)
        return False


# Run once on import
telegram_health_check()


# =========================
# Internal sender
# =========================
def _post(text: str, chat_id: str, parse_mode: str = "HTML"):
    if not TELEGRAM_ENABLED:
        return

    if not chat_id:
        print("⚠️ Telegram chat_id missing. Message skipped.")
        return

    if session is None:
        print("⚠️ Requests session unavailable. Telegram disabled.")
        return

    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }

    resp = None  # Prevent UnboundLocalError

    try:
        resp = session.post(API_URL, data=payload, timeout=10)
        resp.raise_for_status()

    except requests.Timeout:
        print("⚠️ Telegram send timeout — message dropped.")

    except requests.ConnectionError as e:
        # DNS failure / network down
        print("❌ Telegram connection error:", e)

    except requests.RequestException as e:
        err_text = resp.text if resp is not None else "No response from Telegram"
        print("❌ Telegram send failed:", e)
        print("🔎 Telegram response:", err_text)

    except Exception as e:
        # Catch-all safety net (prevents bot crash)
        print("🔥 Unexpected Telegram error:", e)


# =========================
# Public API
# =========================
def send_message(text: str, parse_mode: str = "HTML"):
    """Send a direct Telegram message."""
    _post(text, TELEGRAM_CHAT_ID, parse_mode)


def send_message_channel(text: str, parse_mode: str = "HTML"):
    """Send a message to a Telegram channel."""
    _post(text, TELEGRAM_CHANNEL_ID, parse_mode)
