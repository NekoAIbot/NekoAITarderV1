#!/usr/bin/env python3
# File: app/news.py

import os
import time
from datetime import datetime, timedelta
import pandas as pd
import feedparser

from dotenv import load_dotenv
from newsapi import NewsApiClient, newsapi_exception

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -------------------------------------------------------------------
# LOAD ENV VARIABLES
# -------------------------------------------------------------------

load_dotenv()
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

newsapi = NewsApiClient(api_key=NEWSAPI_KEY) if NEWSAPI_KEY else None

# -------------------------------------------------------------------
# LOAD FINBERT MODEL (FINANCIAL SENTIMENT)
# -------------------------------------------------------------------

FINBERT_MODEL = "ProsusAI/finbert"

tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)

model.eval()

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------

LOOKBACK_DAYS = 30
REQUEST_THROTTLE = 1

QUERY_MAP = {
    **{f"{c}USD": c for c in ["EUR", "GBP", "AUD", "CAD", "CHF", "JPY", "NZD"]},
    **{f"{c}USDT": c for c in ["BTC", "ETH", "BNB", "SOL", "ADA", "DOT", "XRP"]},
}

# -------------------------------------------------------------------
# FINBERT SENTIMENT SCORING
# -------------------------------------------------------------------

def finbert_score(text: str) -> float:
    """
    Returns sentiment score:
    Positive → +1
    Neutral  →  0
    Negative → -1
    Weighted by probability
    """

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]

    # FinBERT label order:
    # 0 = negative
    # 1 = neutral
    # 2 = positive

    negative = probs[0].item()
    neutral = probs[1].item()
    positive = probs[2].item()

    return positive - negative  # weighted directional sentiment


# -------------------------------------------------------------------
# BULK NEWS FETCH FOR TRAINING
# -------------------------------------------------------------------

def _fetch_newsapi_bulk(symbol: str,
                        from_dt: datetime,
                        to_dt: datetime) -> list[dict]:

    if not newsapi:
        return []

    q = QUERY_MAP.get(symbol, symbol)

    try:
        res = newsapi.get_everything(
            q=q,
            from_param=from_dt.strftime("%Y-%m-%dT%H:%M:%S"),
            to=to_dt.strftime("%Y-%m-%dT%H:%M:%S"),
            language="en",
            page_size=100,
            sort_by="publishedAt"
        )
    except newsapi_exception.NewsAPIException:
        return []

    records = []

    for article in res.get("articles", []):
        title = article.get("title")
        published = article.get("publishedAt")

        if not title or not published:
            continue

        timestamp = pd.to_datetime(published)

        sentiment = finbert_score(title)

        records.append({
            "timestamp": timestamp,
            "symbol": symbol,
            "sentiment": sentiment
        })

    return records


# -------------------------------------------------------------------
# REQUIRED FUNCTION FOR TRAINING SCRIPT
# -------------------------------------------------------------------

def fetch_news() -> pd.DataFrame:
    """
    Fetch bulk historical financial news.
    Returns DataFrame:
    ["timestamp","symbol","sentiment"]
    """

    if not newsapi:
        return pd.DataFrame(columns=["timestamp", "symbol", "sentiment"])

    all_records = []

    now = datetime.utcnow()
    start = now - timedelta(days=LOOKBACK_DAYS)

    symbols = list(QUERY_MAP.keys())

    for symbol in symbols:
        records = _fetch_newsapi_bulk(symbol, start, now)
        all_records.extend(records)
        time.sleep(REQUEST_THROTTLE)

    if not all_records:
        return pd.DataFrame(columns=["timestamp", "symbol", "sentiment"])

    df = pd.DataFrame(all_records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    return df.reset_index(drop=True)


# -------------------------------------------------------------------
# LEAK-FREE HISTORICAL ALIGNMENT
# -------------------------------------------------------------------

def get_news_series(symbol: str,
                    index: pd.DatetimeIndex,
                    lookback_hours: int = 24) -> pd.Series:

    news_df = fetch_news()

    if news_df.empty:
        return pd.Series([0.0] * len(index), index=index)

    news_df = news_df[news_df["symbol"] == symbol]

    sentiments = []

    for ts in index:

        window_start = ts - timedelta(hours=lookback_hours)

        relevant = news_df[
            (news_df["timestamp"] < ts) &
            (news_df["timestamp"] >= window_start)
        ]

        if relevant.empty:
            sentiments.append(0.0)
        else:
            sentiments.append(relevant["sentiment"].mean())

    return pd.Series(sentiments, index=index)


# -------------------------------------------------------------------
# REAL-TIME SENTIMENT
# -------------------------------------------------------------------

def get_news_sentiment(symbol: str) -> float:

    if not newsapi:
        return 0.0

    now = datetime.utcnow()
    from_dt = now - timedelta(hours=1)

    records = _fetch_newsapi_bulk(symbol, from_dt, now)

    if not records:
        return 0.0

    return sum([r["sentiment"] for r in records]) / len(records)
