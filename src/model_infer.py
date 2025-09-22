# src/model_infer.py
import os, joblib
import numpy as np

MODEL_PATH = os.path.join("models", "predictor.pkl")

_model = None
_features = None

def _ensure_loaded():
    global _model, _features
    if _model is None and os.path.exists(MODEL_PATH):
        blob = joblib.load(MODEL_PATH)
        _model = blob["model"]
        _features = blob["features"]
    return _model is not None

def predict_proba(feature_frame):
    """
    feature_frame: DataFrame from make_panel_features (or subset for today's rows).
    Returns dict[ticker] -> probability (0..1) of positive next-month return.
    """
    if not _ensure_loaded():
        return {}
    last = feature_frame.sort_values(["Ticker", "Date"]).groupby("Ticker").tail(1)
    X = last[_features].values
    probs = _model.predict_proba(X)[:, 1]
    return dict(zip(last["Ticker"].tolist(), probs.tolist()))
