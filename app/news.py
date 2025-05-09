# app/news.py

import os
import time
from datetime import datetime, timedelta
import pandas as pd

import feedparser
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from newsapi import NewsApiClient, newsapi_exception

# initialize clients
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
newsapi = NewsApiClient(api_key=NEWSAPI_KEY) if NEWSAPI_KEY else None
vader   = SentimentIntensityAnalyzer()

# map symbols to search queries
QUERY_MAP = {
    **{f"{c}USD": c for c in ["EUR","GBP","AUD","CAD","CHF","JPY","NZD"]},
    **{f"{c}USDT": c for c in ["BTC","ETH","BNB","SOL","ADA","DOT","XRP"]},
}

def _fetch_rss_headlines(symbol: str) -> list[str]:
    """Fetch titles from Google News RSS for a given symbol query."""
    q = QUERY_MAP.get(symbol, symbol)
    url = f"https://news.google.com/rss/search?q={q}"
    feed = feedparser.parse(url)
    if feed.bozo or not feed.entries:
        return []
    return [entry.title for entry in feed.entries]

def _fetch_newsapi_headlines(symbol: str,
                            from_dt: datetime,
                            to_dt:   datetime) -> list[str]:
    """Fallback: call NewsAPI everything endpoint for given time window."""
    if not newsapi:
        return []
    q = QUERY_MAP.get(symbol, symbol)
    try:
        res = newsapi.get_everything(
            q=q,
            from_param=from_dt.isoformat(),
            to=to_dt.isoformat(),
            language="en",
            page_size=50,
        )
    except newsapi_exception.NewsAPIException:
        return []
    return [a["title"] for a in res.get("articles", [])]

def _score_headlines(headlines: list[str]) -> float:
    """Return average VADER compound score, or 0 if none."""
    if not headlines:
        return 0.0
    scores = [vader.polarity_scores(t)["compound"] for t in headlines]
    return sum(scores) / len(scores)

def get_news_sentiment(symbol: str) -> float:
    """
    Real‐time sentiment: try RSS from last hour, fallback to NewsAPI.
    Returns avg compound*100.
    """
    # 1) RSS (no timestamp)
    rss = _fetch_rss_headlines(symbol)
    if rss:
        return _score_headlines(rss) * 100.0

    # 2) fallback to NewsAPI
    now = datetime.utcnow()
    hl = _fetch_newsapi_headlines(symbol, now - timedelta(hours=1), now)
    return _score_headlines(hl) * 100.0

def get_news_series(symbol: str, index: pd.DatetimeIndex) -> pd.Series:
    """
    For historical backtests/training: for each timestamp in `index`,
    try RSS + fallback to NewsAPI in that bar window.
    Returns a Series aligned to `index`.
    """
    sentiments = []
    for ts in index:
        # 1–minute window
        from_dt = ts - timedelta(minutes=1)
        to_dt   = ts

        # try RSS once per symbol per training run
        if not sentiments:
            rss = _fetch_rss_headlines(symbol)
            if rss:
                score = _score_headlines(rss) * 100.0
                sentiments = [score] * len(index)
                break

        # otherwise fallback per-bar (but throttle)
        hl = _fetch_newsapi_headlines(symbol, from_dt, to_dt)
        sentiments.append(_score_headlines(hl) * 100.0)
        time.sleep(1)  # throttle NewsAPI calls

    # if we filled once via RSS, `sentiments` is full; else build from appended
    if len(sentiments) != len(index):
        sentiments = sentiments[:len(index)]

    return pd.Series(sentiments, index=index)
