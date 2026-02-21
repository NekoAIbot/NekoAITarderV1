#!/usr/bin/env python3
# File: app/news.py

import os
import time
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv

from newsapi import NewsApiClient, newsapi_exception

# Import fallback VADER sentiment
from app.news_sentiment import get_news_sentiment as get_vader_sentiment

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

load_dotenv()
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
newsapi = NewsApiClient(api_key=NEWSAPI_KEY) if NEWSAPI_KEY else None

# -------------------------------------------------------------------
# FINBERT MODEL
# -------------------------------------------------------------------
FINBERT_MODEL = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
finbert_model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
finbert_model.eval()

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
# FINBERT SCORING
# -------------------------------------------------------------------
def finbert_score(text: str) -> float:
    if not text or text.strip() == "":
        return 0.0
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = finbert_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
    # 0=neg,1=neutral,2=pos
    return probs[2].item() - probs[0].item()


# -------------------------------------------------------------------
# BULK NEWS FETCH
# -------------------------------------------------------------------
def _fetch_newsapi_bulk(symbol: str, from_dt: datetime, to_dt: datetime) -> list[dict]:
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
        records.append({"timestamp": timestamp, "symbol": symbol, "sentiment": sentiment})
    return records


# -------------------------------------------------------------------
# FETCH HISTORICAL NEWS FOR TRAINING
# -------------------------------------------------------------------
def fetch_news() -> pd.DataFrame:
    all_records = []
    now = datetime.utcnow()
    start = now - timedelta(days=LOOKBACK_DAYS)

    symbols = list(QUERY_MAP.keys())
    for symbol in symbols:
        records = _fetch_newsapi_bulk(symbol, start, now)
        if not records:
            # fallback to VADER simulated data
            # we generate placeholder sentiment for each day
            for i in range(LOOKBACK_DAYS):
                ts = start + timedelta(days=i)
                all_records.append({
                    "timestamp": ts,
                    "symbol": symbol,
                    "sentiment": get_vader_sentiment(symbol)
                })
        else:
            all_records.extend(records)
        time.sleep(REQUEST_THROTTLE)

    if not all_records:
        raise ValueError("No news data available for training. Please check API keys or queries.")

    df = pd.DataFrame(all_records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


# -------------------------------------------------------------------
# GET NEWS TIME SERIES FOR ALIGNMENT
# -------------------------------------------------------------------
def get_news_series(symbol: str, index: pd.DatetimeIndex, lookback_hours: int = 24) -> pd.Series:
    news_df = fetch_news()
    if news_df.empty:
        # fallback: zero sentiment
        return pd.Series([0.0]*len(index), index=index)

    news_df = news_df[news_df["symbol"] == symbol]
    sentiments = []
    for ts in index:
        window_start = ts - timedelta(hours=lookback_hours)
        relevant = news_df[(news_df["timestamp"] < ts) & (news_df["timestamp"] >= window_start)]
        if relevant.empty:
            sentiments.append(0.0)
        else:
            sentiments.append(relevant["sentiment"].mean())
    return pd.Series(sentiments, index=index)


# -------------------------------------------------------------------
# REAL-TIME SENTIMENT
# -------------------------------------------------------------------
def get_news_sentiment(symbol: str) -> float:
    # Try FinBERT + NewsAPI first
    now = datetime.utcnow()
    from_dt = now - timedelta(hours=1)
    records = _fetch_newsapi_bulk(symbol, from_dt, now)
    if records:
        return sum([r["sentiment"] for r in records]) / len(records)
    # fallback to VADER
    return get_vader_sentiment(symbol)