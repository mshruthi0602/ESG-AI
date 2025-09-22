# src/news_fetcher.py
import random
from typing import Dict, List

def get_headlines_for_universe(tickers: List[str], max_items: int = 5, use_cache_minutes: int = 60) -> Dict[str, dict]:
    """
    Mock news fetcher used by the app for demo/offline runs.
    Returns a dict per ticker with label, confidence, top_headline, and a Bing search URL.
    NOTE: This intentionally does NOT call external APIs so that tests are reproducible.

    Example return:
    {
        "AAPL": {
            "label": "positive",
            "confidence": 0.91,
            "top_headline": "AAPL stock outlook improves as market reacts to earnings",
            "url": "https://www.bing.com/news/search?q=AAPL+stock"
        },
        ...
    }
    """
    results: Dict[str, dict] = {}
    for t in tickers:
        try:
            # Simple mock headline pool per ticker
            headlines = [
                f"{t} stock outlook improves as market reacts to earnings",
                f"{t} faces regulatory scrutiny in overseas markets",
                f"{t} expands into renewable energy investments",
            ]

            if not headlines:
                # Fallback when no headlines (shouldn't happen in this mock)
                results[t] = {
                    "label": "neutral",
                    "confidence": 0.5,
                    "top_headline": "No recent news found",
                    "url": f"https://www.bing.com/news/search?q={t}+stock",
                }
            else:
                # Pick a top headline and a mock sentiment
                top = random.choice(headlines)
                sentiment = random.choice(["positive", "neutral", "negative"])
                results[t] = {
                    "label": sentiment,
                    "confidence": round(random.uniform(0.70, 0.95), 2),
                    "top_headline": top,
                    "url": f"https://www.bing.com/news/search?q={t}+stock",
                }
        except Exception:
            # Defensive fallback
            results[t] = {
                "label": "neutral",
                "confidence": 0.5,
                "top_headline": "No recent news found",
                "url": f"https://www.bing.com/news/search?q={t}+stock",
            }
    return results


# --- Helpers for training/evaluation (used by smoke test and model_trainer) ---

def build_sentiment_map(results: Dict[str, dict]) -> Dict[str, str]:
    """
    Convert fetch results into {ticker: 'positive'|'neutral'|'negative'}.
    Safe for feature engineering; ignores confidence/headline/url.

    This also defends against accidental shapes like:
      {"label": {"label": "positive", "confidence": 0.88}, ...}
    """
    out: Dict[str, str] = {}
    for t, v in (results or {}).items():
        lbl = v.get("label", "neutral")
        # If someone accidentally passed {'label': {...}}, unwrap
        if isinstance(lbl, dict):
            lbl = lbl.get("label", "neutral")
        lbl = str(lbl).strip().lower()
        if lbl not in ("positive", "neutral", "negative"):
            lbl = "neutral"
        out[t] = lbl
    return out


def get_latest_news_for_ticker(ticker: str) -> str:
    """
    Return a single headline string for a ticker (used by the trainer CLI example).
    This uses the same mock pool as get_headlines_for_universe().
    """
    res = get_headlines_for_universe([ticker])
    info = res.get(ticker, {})
    return info.get("top_headline", "")
