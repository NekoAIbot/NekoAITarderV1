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
        "q":        f"forex OR {pair}",
        "pageSize": 3,
        "apiKey":   NEWSAPI_KEY,
        "language": "en",
        "sortBy":   "publishedAt",
    }
    resp = requests.get(url, params=params, timeout=10).json()
    articles = resp.get("articles", [])
    titles = [a["title"] for a in articles]
    print(f"[Debug][News] {symbol} fetched {len(titles)} headlines.")
    return titles

def compute_sentiment(headlines: list[str]) -> float:
    """
    Run VADER on each headline, then average the compound scores.
    Returns a percentage [-100…+100].
    """
    if not headlines:
        print("[Debug][News]   no headlines → sentiment=0.0")
        return 0.0
    scores = []
    for h in headlines:
        vs = analyzer.polarity_scores(h)
        print(f"[Debug][News]   \"{h}\" → compound={vs['compound']}")
        scores.append(vs["compound"])
    avg = sum(scores) / len(scores) * 100
    print(f"[Debug][News]   avg compound→ {avg:.1f}%")
    return avg

def get_news_sentiment(symbol: str) -> float:
    """Fetch + analyze, with 1–2s delay to be gentle on API."""
    try:
        headlines = fetch_headlines(symbol)
        time.sleep(random.uniform(1, 2))
        return compute_sentiment(headlines)
    except Exception as e:
        print(f"[Warning][News] sentiment fetch failed for {symbol}: {e}")
        return 0.0
