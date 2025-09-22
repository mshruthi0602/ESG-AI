import numpy as np
import pandas as pd
from src.price_data import get_prices_matrix


def categorize_esg(esg_score: float) -> str:
    if esg_score < 20:
        return "Low"
    elif esg_score < 30:
        return "Medium"
    return "High"


def categorize_risk(volatility: float, thresholds: dict) -> str:
    v = float(volatility)  # ensure scalar, not Series
    if v <= thresholds["low"]:
        return "Low"
    elif v <= thresholds["high"]:
        return "Medium"
    return "High"


def optimize_portfolio(esg_df, sentiment_scores, esg_pref, risk_pref, industry_label=None):
    tickers = esg_df["Ticker"].tolist()
    prices = get_prices_matrix(tickers, lookback_days=252)

    # daily returns
    rets = prices.pct_change().dropna(how="all")
    vol = rets.std() * np.sqrt(252)  # annualized volatility

    # thresholds from distribution
    vols = vol.dropna().sort_values()
    low_thr = float(vols.quantile(0.33)) if not vols.empty else 0.2
    high_thr = float(vols.quantile(0.66)) if not vols.empty else 0.4
    thresholds = {"low": low_thr, "high": high_thr}

    print(f"[OPTIMIZER] Risk thresholds → Low ≤ {low_thr:.3f}, Medium ≤ {high_thr:.3f}")

    recommendations = []
    for _, row in esg_df.iterrows():
        t = row["Ticker"]
        esg_score = row.get("ESG_Score", np.nan)
        if np.isnan(esg_score) or t not in vol:
            continue

        esg_cat = categorize_esg(esg_score)
        risk_cat = categorize_risk(vol[t], thresholds)
        sentiment = sentiment_scores.get(t, "neutral")
        industry = row.get("Industry", "")

        # STRICT TIER LOGIC (improved)
        if esg_cat == esg_pref and risk_cat == risk_pref:
            if not industry_label or industry_label.lower() in industry.lower():
                tier = "Green"   # exact ESG, Risk, and Industry
            else:
                tier = "Green"   # still Green if ESG + Risk match, even if industry mismatch
        elif (esg_cat == esg_pref or risk_cat == risk_pref) and sentiment != "negative":
            tier = "Yellow"
        else:
            tier = "Red"

        rec = {
            "Ticker": t,
            "ESG_Score": float(esg_score),
            "ESG_Category": esg_cat,
            "Risk_Category": risk_cat,
            "Volatility": float(vol[t]),
            "Sentiment": sentiment,
            "Industry": industry,
            "Tier": tier,
        }
        recommendations.append(rec)

        print(
            f"[DEBUG] {t}: ESG={float(esg_score):.1f} ({esg_cat}), "
            f"Vol={float(vol[t]):.3f} ({risk_cat}), "
            f"Sentiment={sentiment}, Tier={tier}"
        )

    if not recommendations:
        return [{"Ticker": "No suitable match"}]

    return recommendations
