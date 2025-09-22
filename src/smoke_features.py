# src/smoke_features.py
from __future__ import annotations
import pandas as pd
from collections import defaultdict

from src.data_loader import load_esg_data
from src.price_data import get_price_history
from src.news_fetcher import get_headlines_for_universe, build_sentiment_map
from src.features import make_panel_features

NEED_ROWS = max(20, 60) + 21 + 5  # = 86 rows required per ticker

def _has_cols(df, cols):
    return all(c in df.columns for c in cols)

def diagnose(prices: dict[str, pd.DataFrame], esg: pd.DataFrame, tickers: list[str]) -> None:
    print(f"[diag] NEED_ROWS per ticker: {NEED_ROWS}")
    esg_set = set(esg["Ticker"].astype(str).str.upper().unique())
    issues = defaultdict(list)
    for t in tickers:
        df = prices.get(t)
        if df is None or df.empty:
            issues["no_prices"].append(t); continue
        cols = list(df.columns)
        if not _has_cols(df, ["Adj Close"]):
            issues["missing_adj_close"].append(f"{t} cols={cols}"); continue
        if not _has_cols(df, ["Date"]):
            # allow Date as index
            if df.index.name and df.index.name.lower() in ("date","datetime"):
                pass
            else:
                issues["missing_date_col"].append(f"{t} cols={cols}")
        n = len(df)
        if n < NEED_ROWS:
            issues["too_few_rows"].append(f"{t} rows={n}")
        if t not in esg_set:
            issues["not_in_esg"].append(t)
    for k, vals in issues.items():
        print(f"[diag] {k}: {len(vals)} → {vals[:8]}{' ...' if len(vals)>8 else ''}")
    if not issues:
        print("[diag] No structural issues detected; proceeding.")

def main():
    # 1) Load ESG and pick tickers from ESG to ensure a match
    esg = load_esg_data()
    esg_tickers = esg["Ticker"].astype(str).str.upper().unique().tolist()

    # Prefer a small mega-cap subset that usually has data
    preferred = [t for t in ["AAPL","MSFT","NVDA","AMZN","TSLA","GOOGL","META","NFLX"] if t in esg_tickers]
    tickers = preferred if len(preferred) >= 5 else esg_tickers[:10]
    print("[smoke] Using tickers:", tickers)

    # 2) Try 3y to be safe
    prices = get_price_history(tickers, period="3y", interval="1d")
    diagnose(prices, esg, tickers)

    # 3) Build sentiment labels (labels-only map)
    news = get_headlines_for_universe(tickers)
    sent_map = build_sentiment_map(news)

    # 4) Attempt feature build
    X, y = make_panel_features(prices, esg, sent_map)
    print(f"[smoke] try#1 rows: {0 if X is None else len(X)} | has_y: {y is not None}")
    if X is not None and len(X) > 0:
        print(X.head().to_string(index=False))
        return

    # 5) Second attempt: extend universe a bit more
    more = esg_tickers[:30]
    if set(more) != set(tickers):
        print("[smoke] Extending tickers to 30 from ESG for a second attempt.")
        prices2 = get_price_history(more, period="3y", interval="1d")
        news2 = get_headlines_for_universe(more)
        sent_map2 = build_sentiment_map(news2)
        X2, y2 = make_panel_features(prices2, esg, sent_map2)
        print(f"[smoke] try#2 rows: {0 if X2 is None else len(X2)} | has_y: {y2 is not None}")
        if X2 is not None and len(X2) > 0:
            print(X2.head().to_string(index=False))
            return

    # 6) Synthetic fallback (proves the pipeline, rules out code bugs)
    print("[smoke] Still no rows — showing a minimal synthetic pass to prove the pipeline.")
    import numpy as np, datetime as dt
    dates = pd.date_range(dt.date.today() - pd.Timedelta(days=180), periods=120, freq="B")
    syn_prices = {}
    for t in ["SYN1","SYN2","SYN3","SYN4","SYN5"]:
        base = 100 + np.cumsum(np.random.normal(0, 0.8, size=len(dates)))
        syn_prices[t] = pd.DataFrame({"Date": dates, "Adj Close": base})
    esg_syn = pd.DataFrame({"Ticker":["SYN1","SYN2","SYN3","SYN4","SYN5"],
                            "ESG_Score":[20, 30, 25, 18, 40]})
    sent_syn = {t: "neutral" for t in ["SYN1","SYN2","SYN3","SYN4","SYN5"]}
    Xs, ys = make_panel_features(syn_prices, esg_syn, sent_syn)
    print(f"[smoke] synthetic rows: {0 if Xs is None else len(Xs)} | has_y: {ys is not None}")
    if Xs is not None:
        print(Xs.head().to_string(index=False))

if __name__ == "__main__":
    main()
