import time
import random
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from config import NEWSAPI_KEY

analyzer = SentimentIntensityAnalyzer()

def fetch_headlines(symbol: str) -> list[str]:
    """
    Fetch latest 3 Forex headlines from NewsAPI mentioning our symbol.
    """
    pair = f"{symbol[:3]}/{symbol[3:]}"
    url = "https://newsapi.org/v2/everything"
    params = {
        "q":           f"forex OR {pair}",
        "pageSize":    3,
        "apiKey":      NEWSAPI_KEY,
        "language":    "en",
        "sortBy":      "publishedAt",
    }
    resp = requests.get(url, params=params, timeout=10).json()
    articles = resp.get("articles", [])
    return [a["title"] for a in articles]

def compute_sentiment(headlines: list[str]) -> float:
    """
    Run VADER on each headline, then average the compound scores.
    Returns a percentage [-100…+100].
    """
    if not headlines:
        return 0.0
    scores = []
    for h in headlines:
        vs = analyzer.polarity_scores(h)
        scores.append(vs["compound"])
    # scale compound [-1..1] to [-100..+100]
    return sum(scores) / len(scores) * 100

def get_news_sentiment(symbol: str) -> float:
    """Fetch + analyze, with 1–2s delay to be gentle on API."""
    try:
        heads = fetch_headlines(symbol)
        time.sleep(random.uniform(1, 2))
        return compute_sentiment(heads)
    except Exception:
        return 0.0
