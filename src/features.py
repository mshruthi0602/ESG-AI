# src/features.py
import numpy as np
import pandas as pd

def _need_rows(lookback_slow=20, vol_win=60, fwd_horizon=21):
    return max(lookback_slow, vol_win) + fwd_horizon + 5  # = 86 by default

def _find_col(df: pd.DataFrame, candidates):
    """Case-insensitive column resolver; returns the actual column name or None."""
    lookup = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lookup:
            return lookup[cand.lower()]
    return None

def _normalize_price_df(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Ensure we have Date + Adj Close (or a safe fallback from Close).
    Returns a cleaned/sorted frame or None if impossible.
    """
    if df is None or df.empty:
        return None
    out = df.copy()

    # Date column (allow datetime index or variants)
    date_col = _find_col(out, ["Date", "Datetime", "timestamp"])
    if date_col is None:
        # try index
        if out.index.name and str(out.index.name).lower() in ("date", "datetime", "timestamp"):
            out = out.reset_index().rename(columns={out.index.name: "Date"})
        else:
            # last resort – assume current index is datetime-like
            out = out.reset_index().rename(columns={"index": "Date"})
    else:
        if date_col != "Date":
            out = out.rename(columns={date_col: "Date"})

    # Adj Close (with generous fallbacks)
    adj_col = _find_col(out, [
        "Adj Close", "AdjClose", "Adjusted Close", "Adjusted_Close",
        "adj close", "adjclose", "adjusted close", "adjusted_close",
        "close_adj", "close_adjusted"
    ])
    if adj_col is None:
        close_col = _find_col(out, ["Close", "close", "Last", "last", "Price", "price"])
        if close_col is None:
            return None
        out["Adj Close"] = pd.to_numeric(out[close_col], errors="coerce")
    else:
        out["Adj Close"] = pd.to_numeric(out[adj_col], errors="coerce")

    # Coerce Date, drop bad rows, sort
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date", "Adj Close"]).sort_values("Date")
    # keep only essentials to avoid surprises
    return out[["Date", "Adj Close"]]

def _daily_returns(price_df):
    p = pd.to_numeric(price_df["Adj Close"], errors="coerce")
    return p.pct_change().fillna(0.0)

def make_panel_features(price_dict, esg_df, sentiment_map,
                        lookback_fast=5, lookback_slow=20, vol_win=60, fwd_horizon=21):
    """
    Build a training/inference frame from prices + ESG + sentiment.
    Returns (X, y) for training or (X, None) for inference if no forward returns available.
    Expected output columns (in order):
      ['Ticker','Date','mom5','mom20','vol60','ESG_Quality','SentimentNum'] and optionally 'y'
    """
    rows = []
    need = _need_rows(lookback_slow, vol_win, fwd_horizon)

    # ESG sanity
    esg_df = esg_df.copy()
    if "Ticker" not in esg_df.columns:
        raise ValueError("ESG frame must have a 'Ticker' column")
    if "ESG_Score" not in esg_df.columns:
        esg_df["ESG_Score"] = np.nan
    esg_df["Ticker"] = esg_df["Ticker"].astype(str).str.upper()

    for ticker, df in (price_dict or {}).items():
        # Normalize price columns/format
        df = _normalize_price_df(df)
        if df is None or df.shape[0] < need:
            continue  # skip if we can't reliably compute features

        r = _daily_returns(df)
        mom5  = (df["Adj Close"] / df["Adj Close"].shift(lookback_fast) - 1.0).rename("mom5")
        mom20 = (df["Adj Close"] / df["Adj Close"].shift(lookback_slow) - 1.0).rename("mom20")
        vol60 = r.rolling(vol_win).std().rename("vol60")
        fwd   = (df["Adj Close"].shift(-fwd_horizon) / df["Adj Close"] - 1.0).rename("fwd21")

        tmp = pd.concat([df["Date"], mom5, mom20, vol60, fwd], axis=1).dropna().copy()
        if tmp.empty:
            continue
        tkr = str(ticker).upper()
        tmp["Ticker"] = tkr

        # ESG features
        esg_row = esg_df.loc[esg_df["Ticker"] == tkr]
        esg_val = float(esg_row["ESG_Score"].iloc[0]) if not esg_row.empty else np.nan
        tmp["ESG_Score"] = esg_val
        # Sustainalytics: lower risk score is better → flip sign for "quality"
        tmp["ESG_Quality"] = -tmp["ESG_Score"]

        # sentiment (label → numeric)
        sent = (sentiment_map.get(tkr) or "neutral")
        if isinstance(sent, dict):  # defensive unwrap
            sent = sent.get("label", "neutral")
        sent = str(sent).strip().lower()
        tmp["SentimentNum"] = 1 if sent == "positive" else (-1 if sent == "negative" else 0)

        rows.append(tmp)

    full = pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame()
    if full.empty:
        return None, None

    # Deterministic feature order
    feats = full[["Ticker", "Date", "mom5", "mom20", "vol60", "ESG_Quality", "SentimentNum"]].dropna().copy()

    # Labels if available
    if "fwd21" in full.columns and full["fwd21"].notna().any():
        y_series = (full.loc[feats.index, "fwd21"] > 0.0).astype(int)
        feats["y"] = y_series.values
        return feats, feats["y"].values

    return feats, None
