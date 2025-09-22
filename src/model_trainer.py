# src/model_trainer.py
from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Optional boosters
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

from src.price_data import get_price_history
from src.features import make_panel_features

# -----------------------------------------------------------------------------
# config
# -----------------------------------------------------------------------------
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = ["mom5", "mom20", "vol60", "ESG_Quality", "SentimentNum"]


# -----------------------------------------------------------------------------
# model factory
# -----------------------------------------------------------------------------
def build_model(kind: str):
    k = kind.lower()

    if k in {"m0", "logreg", "lr"}:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")),
        ]), "logreg"

    if k in {"m1", "rf", "randomforest"}:
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ), "random_forest"

    if k in {"m2", "xgb", "xgboost"} and HAS_XGB:
        return XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=0,                    # avoids thread oversubscription on Windows
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
        ), "xgboost"

    if k in {"m2", "lgbm", "lightgbm"} and HAS_LGBM:
        return LGBMClassifier(
            n_estimators=600,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            objective="binary",
        ), "lightgbm"

    raise ValueError(f"Unknown or unavailable model kind: {kind}")


# -----------------------------------------------------------------------------
# dataset assembly
# -----------------------------------------------------------------------------
def _make_Xy(esg_df: pd.DataFrame, sentiment_map: dict[str, str],
             tickers: list[str], period: str = "2y"):
    prices = get_price_history(tickers, period=period, interval="1d")
    Xy, y = make_panel_features(prices, esg_df, sentiment_map)
    if Xy is None or y is None or Xy.empty:
        raise RuntimeError("Insufficient data to train.")
    for col in FEATURES:
        if col not in Xy.columns:
            raise RuntimeError(f"Missing feature: {col}")
    Xy = Xy.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return Xy, np.asarray(y)


# -----------------------------------------------------------------------------
# training (robust to xgboost version differences)
# -----------------------------------------------------------------------------
def train_predictor(esg_df: pd.DataFrame,
                    sentiment_map: dict[str, str],
                    tickers: list[str],
                    period: str = "2y",
                    kind: str = "rf",
                    shap_out: str | None = None,
                    tune_xgb: bool = False,
                    activate: bool = False):
    Xy, y = _make_Xy(esg_df, sentiment_map, tickers, period=period)

    # chronological 80/20 split
    split = int(len(Xy) * 0.80)
    X_tr, X_va = Xy.iloc[:split], Xy.iloc[split:]
    y_tr, y_va = y[:split], y[split:]

    model, model_type = build_model(kind)

    if model_type == "xgboost" and tune_xgb:
        fitted = False
        # Try old API (early_stopping_rounds)
        try:
            model.fit(
                X_tr[FEATURES].values, y_tr,
                eval_set=[(X_va[FEATURES].values, y_va)],
                early_stopping_rounds=75,
                verbose=False,
            )
            fitted = True
        except TypeError:
            # Try new API (callbacks)
            try:
                import xgboost as xgb
                cb = xgb.callback.EarlyStopping(rounds=75, save_best=True)
                model.fit(
                    X_tr[FEATURES].values, y_tr,
                    eval_set=[(X_va[FEATURES].values, y_va)],
                    callbacks=[cb],
                    verbose=False,
                )
                fitted = True
            except TypeError:
                fitted = False
        if not fitted:
            # Fallback: no early stopping
            model.fit(X_tr[FEATURES].values, y_tr)
    else:
        model.fit(X_tr[FEATURES].values, y_tr)

    proba_va = model.predict_proba(X_va[FEATURES].values)[:, 1]
    va_auc = roc_auc_score(y_va, proba_va) if len(np.unique(y_va)) > 1 else float("nan")
    print(f"[TRAIN] {model_type} | Validation AUC: {va_auc:.3f}")

    short = {"logreg": "logreg", "random_forest": "rf", "xgboost": "xgb", "lightgbm": "lgbm"}[model_type]
    out_path = MODELS_DIR / f"predictor_{short}.pkl"
    blob = {"model": model, "features": FEATURES, "model_type": model_type, "va_auc": float(va_auc)}
    joblib.dump(blob, out_path)
    print(f"[SAVE] {out_path}")

    if shap_out and model_type in {"random_forest", "xgboost", "lightgbm"}:
        try:
            from src.shap_utils import save_shap_summary
            save_shap_summary(model, X_va[FEATURES].values, FEATURES, shap_out)
            print(f"[SHAP] summary saved → {shap_out}")
        except Exception as e:
            print(f"[SHAP] skipped: {e}")

    if activate:
        joblib.dump(blob, MODELS_DIR / "predictor.pkl")
        print("[activate] models/predictor.pkl updated")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    from src.data_loader import load_esg_data
    from src.news_fetcher import get_headlines_for_universe, build_sentiment_map

    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="rf", help="logreg|rf|xgb|lgbm")
    ap.add_argument("--period", default="2y")
    ap.add_argument("--tickers", nargs="*", default=None)
    ap.add_argument("--shap", default=None)
    ap.add_argument("--tune-xgb", action="store_true", help="use early stopping if supported by your xgboost")
    ap.add_argument("--activate", action="store_true", help="copy trained model to models/predictor.pkl")
    args = ap.parse_args()

    esg = load_esg_data()
    tickers = args.tickers or esg["Ticker"].astype(str).str.upper().tolist()[:120]
    news = get_headlines_for_universe(tickers, max_items=5)
    sentiment_map = build_sentiment_map(news)

    train_predictor(
        esg_df=esg,
        sentiment_map=sentiment_map,
        tickers=tickers,
        period=args.period,
        kind=args.kind,
        shap_out=args.shap,
        tune_xgb=args.tune_xgb,
        activate=args.activate,
    )
