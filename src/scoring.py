# src/scoring.py
def aggregate_sentiment(headlines_by_ticker, max_items=5):
    results = {}
    for ticker, news in headlines_by_ticker.items():
        try:
            # Ensure we’re always working with a list
            headlines = news if isinstance(news, list) else [news]

            # Slice safely
            selected = headlines[:max_items]

            # Handle case where no headlines exist
            if not selected:
                results[ticker] = {
                    "label": "neutral",
                    "confidence": 0.5,
                    "counts": {"positive": 0, "neutral": 1, "negative": 0},
                    "top_headline": "No recent news",
                    "url": f"https://www.bing.com/news/search?q={ticker}+stock"
                }
                continue

            # Take first headline’s sentiment as representative
            sample = selected[0]
            results[ticker] = {
                "label": sample.get("label", "neutral"),
                "confidence": sample.get("confidence", 0.5),
                "counts": {
                    "positive": 1 if sample.get("label") == "positive" else 0,
                    "neutral": 1 if sample.get("label") == "neutral" else 0,
                    "negative": 1 if sample.get("label") == "negative" else 0
                },
                "top_headline": sample.get("top_headline", ""),
                "url": sample.get("url", f"https://www.bing.com/news/search?q={ticker}+stock")
            }
        except Exception as e:
            print(f"[ERROR] aggregate_sentiment {ticker}: {e}")
            results[ticker] = {
                "label": "neutral",
                "confidence": 0.5,
                "counts": {"positive": 0, "neutral": 1, "negative": 0},
                "top_headline": "Error parsing news",
                "url": f"https://www.bing.com/news/search?q={ticker}+stock"
            }
    return results
