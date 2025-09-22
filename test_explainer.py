from src.explanation_generator import generate_explanations

sample = [
    {"Ticker": "NVDA", "ESG_Score": 28, "ESG_Category": "High", "Risk_Category": "Medium", "Sentiment": "positive", "News": "AI chip demand grows", "NewsURL": "https://example.com"},
    {"Ticker": "TXN", "ESG_Score": 22, "ESG_Category": "Medium", "Risk_Category": "Medium", "Sentiment": "neutral", "News": "Steady performance", "NewsURL": "https://example.com"},
    {"Ticker": "INTC", "ESG_Score": 15, "ESG_Category": "Low", "Risk_Category": "High", "Sentiment": "negative", "News": "Cost cutting news", "NewsURL": "https://example.com"}
]

for line in generate_explanations(sample):
    print(line, "\n")
