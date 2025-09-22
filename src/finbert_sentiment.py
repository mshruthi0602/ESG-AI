# src/finbert_sentiment.py
from __future__ import annotations
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

MODEL_NAME = "ProsusAI/finbert"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
except Exception as e:
    print(f"[FINBERT] Pipeline load failed: {e}")
    sentiment_pipeline = None  # evaluator will fall back to lexicon if needed

# Normalise raw labels from the model
LABEL_MAP = {
    "label_0": "negative", "label_1": "neutral", "label_2": "positive",
    "LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive",
    "negative": "negative", "neutral": "neutral", "positive": "positive",
}

def label_text(text: str) -> str:
    """
    Return canonical class 'positive' | 'neutral' | 'negative' for a single headline.
    Used by tests/test_finbert_eval.py.
    """
    if not text or sentiment_pipeline is None:
        return "neutral"
    try:
        out = sentiment_pipeline(text[:512])[0]
        return LABEL_MAP.get(out["label"], out["label"].lower())
    except Exception as e:
        print(f"[FINBERT] Error(label_text): {e}")
        return "neutral"

def get_sentiment(text: str) -> dict:
    """
    Return both label and confidence, e.g. {"label": "positive", "confidence": 0.87}.
    Useful in the app for applying guardrails with majority share.
    """
    if not text or sentiment_pipeline is None:
        return {"label": "neutral", "confidence": 0.0}
    try:
        out = sentiment_pipeline(text[:512])[0]
        label = LABEL_MAP.get(out["label"], out["label"].lower())
        return {"label": label, "confidence": float(out.get("score", 0.0))}
    except Exception as e:
        print(f"[FINBERT] Error(get_sentiment): {e}")
        return {"label": "neutral", "confidence": 0.0}


