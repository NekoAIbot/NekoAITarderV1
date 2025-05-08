# app/news.py

import os
from datetime import datetime, timedelta
import pandas as pd

from newsapi import NewsApiClient
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# initialize clients
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
if not NEWSAPI_KEY:
    raise RuntimeError("Please set NEWSAPI_KEY in your environment")
newsapi = NewsApiClient(api_key=NEWSAPI_KEY)
vader   = SentimentIntensityAnalyzer()

# map symbols to search queries (tweak as you like)
QUERY_MAP = {
    **{f"{c}USD": c for c in ["EUR","GBP","AUD","CAD","CHF","JPY","NZD"]},
    **{f"{c}USDT": c for c in ["BTC","ETH","BNB","SOL","ADA","DOT","XRP"]},
}

def _fetch_headlines(symbol: str,
                     from_dt: datetime,
                     to_dt:   datetime) -> list[str]:
    """Call NewsAPI everything endpoint for given time window."""
    q = QUERY_MAP.get(symbol, symbol)
    res = newsapi.get_everything(
        q=q,
        from_param=from_dt.isoformat(),
        to=to_dt.isoformat(),
        language="en",
        page_size=100,
    )
    return [a["title"] for a in res.get("articles", [])]

def _score_headlines(headlines: list[str]) -> float:
    """Return average VADER compound score, or 0 if none."""
    if not headlines:
        return 0.0
    scores = [vader.polarity_scores(t)["compound"] for t in headlines]
    return sum(scores) / len(scores)

def get_news_sentiment(symbol: str) -> float:
    """
    Real‐time sentiment: fetch headlines from last hour, return avg compound*100.
    """
    to_dt   = datetime.utcnow()
    from_dt = to_dt - timedelta(hours=1)
    hl     = _fetch_headlines(symbol, from_dt, to_dt)
    return _score_headlines(hl) * 100.0

def get_news_series(symbol: str, index: pd.DatetimeIndex) -> pd.Series:
    """
    For historical backtests/training: for each timestamp in `index`,
    fetch headlines in the prior bar window (assumes 1-min bars).
    Returns a Series of same length.
    """
    sentiments = []
    for ts in index:
        # you may adjust the window to match your bar interval
        from_dt = ts - timedelta(minutes=1)
        to_dt   = ts
        hl      = _fetch_headlines(symbol, from_dt, to_dt)
        sentiments.append(_score_headlines(hl) * 100.0)
    return pd.Series(sentiments, index=index)
