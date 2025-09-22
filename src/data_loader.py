# src/data_loader.py
import pandas as pd

def load_esg_data(valid_tickers=None):
    df = pd.read_csv("data/sp500_esg_risk.csv")
    df.columns = [c.strip() for c in df.columns]

    # Standardise names if present
    if "Symbol" in df.columns:
        df = df.rename(columns={"Symbol": "Ticker"})
    if "Total ESG Risk score" in df.columns:
        df = df.rename(columns={"Total ESG Risk score": "ESG_Score"})

    # Ensure columns exist
    for c in ["Ticker", "Sector", "Industry", "ESG_Score"]:
        if c not in df.columns:
            df[c] = pd.NA

    # Clean types
    df["Ticker"] = df["Ticker"].astype(str).str.upper()
    df["Sector"] = df["Sector"].astype(str)
    df["Industry"] = df["Industry"].astype(str)
    df["ESG_Score"] = pd.to_numeric(df["ESG_Score"], errors="coerce")

    # Keep rows with ticker, industry, and numeric ESG
    df = df.dropna(subset=["Ticker", "Industry", "ESG_Score"])

    if valid_tickers is not None:
        valid = [t.upper() for t in valid_tickers]
        df = df[df["Ticker"].isin(valid)]

    return df.reset_index(drop=True)

def get_unique_industries(valid_tickers=None):
    return sorted(load_esg_data(valid_tickers)["Industry"].dropna().astype(str).unique().tolist())

def get_unique_sectors(valid_tickers=None):
    return sorted(load_esg_data(valid_tickers)["Sector"].dropna().astype(str).unique().tolist())
