# src/price_data.py
from __future__ import annotations
from pathlib import Path
import re
import pandas as pd

# -------- robust date parsing --------
def _to_datetime_robust(s: pd.Series) -> pd.Series:
    # Try ISO fast path; then fall back to parser for any stragglers
    dt = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    if dt.isna().any():
        dt2 = pd.to_datetime(s, errors="coerce")
        dt = dt.fillna(dt2)
    return dt

# -------- optional live sources --------
try:
    import yfinance as yf
    HAS_YF = True
except Exception:
    HAS_YF = False

try:
    # Optional fallback (Stooq)
    from pandas_datareader import data as pdr
    HAS_PDR = True
except Exception:
    HAS_PDR = False

# Search both dirs; prefer saving into data/prices
CACHE_DIRS = [Path("data/prices"), Path("data/cache_prices")]
for d in CACHE_DIRS:
    d.mkdir(parents=True, exist_ok=True)

# ---------------- helpers ----------------
def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with columns ['Date','Adj Close'] sorted ascending."""
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join([str(c) for c in tup if str(c) != ""]) for tup in df.columns]

    out = df.copy()

    # ---- Date ----
    date_col = next((c for c in ["Date", "Datetime", "date", "datetime", "timestamp"] if c in out.columns), None)
    if date_col is None:
        if out.index.name and str(out.index.name).lower() in ("date", "datetime", "timestamp"):
            out = out.reset_index().rename(columns={out.index.name: "Date"})
        else:
            out = out.reset_index().rename(columns={"index": "Date"})
    elif date_col != "Date":
        out = out.rename(columns={date_col: "Date"})

    # ---- Adj Close (fallbacks) ----
    adj_col = next(
        (c for c in [
            "Adj Close", "AdjClose", "Adjusted Close", "adjusted close", "adj close",
            "Close", "close", "Last", "Price", "price"
        ] if c in out.columns),
        None
    )
    if adj_col is None:
        return pd.DataFrame()

    out["Adj Close"] = pd.to_numeric(out[adj_col], errors="coerce")
    out["Date"] = _to_datetime_robust(out["Date"])
    out = out.dropna(subset=["Date", "Adj Close"]).sort_values("Date")
    return out[["Date", "Adj Close"]]

def _load_csv_from(dirpath: Path, ticker: str) -> pd.DataFrame:
    f = dirpath / f"{ticker}.csv"
    if not f.exists():
        return pd.DataFrame()

    # Detect your custom cache layout:
    # row0 'Price,<T>' ; row1 'Ticker,<T>' ; row2 'Date,' ; rows: '<YYYY-MM-DD>,<price>'
    try:
        raw = pd.read_csv(f, header=None)
        if raw.shape[0] >= 4 and raw.shape[1] >= 2:
            v00 = str(raw.iloc[0, 0]).strip().lower()
            v10 = str(raw.iloc[1, 0]).strip().lower()
            v20 = str(raw.iloc[2, 0]).strip().lower()
            if v00 == "price" and v10 == "ticker" and v20.startswith("date"):
                df = raw.iloc[2:].copy()
                df.columns = ["Date", "Adj Close"]
                return _normalize_df(df)
    except Exception:
        pass

    # Normal CSVs (Yahoo/2-col Date+Price/etc.)
    try:
        df = pd.read_csv(f)
        if "Date" in df.columns and "Adj Close" not in df.columns and len(df.columns) == 2:
            other = [c for c in df.columns if c != "Date"][0]
            df = df.rename(columns={other: "Adj Close"})
        return _normalize_df(df)
    except Exception:
        return pd.DataFrame()

def _load_csv(ticker: str) -> pd.DataFrame:
    for d in CACHE_DIRS:
        df = _load_csv_from(d, ticker)
        if not df.empty:
            return df
    return pd.DataFrame()

def _save_csv(ticker: str, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    out = _normalize_df(df)
    if out.empty:
        return
    (CACHE_DIRS[0] / f"{ticker}.csv").parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(CACHE_DIRS[0] / f"{ticker}.csv", index=False)

# --- period helpers for Stooq (approx) ---
def _period_to_days(period: str) -> int:
    if not isinstance(period, str):
        return 365
    m = re.match(r"^\s*(\d+)\s*([dwmy])\s*$", period.lower())
    if not m:
        return 365
    n, u = int(m.group(1)), m.group(2)
    return {
        "d": n,
        "w": n * 7,
        "m": n * 30,
        "y": n * 365,
    }[u]

# ---------------- live downloaders ----------------
def _download_yf(ticker: str, period="3y", interval="1d") -> pd.DataFrame:
    if not HAS_YF:
        return pd.DataFrame()
    try:
        raw = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
        if isinstance(raw, pd.DataFrame) and not raw.empty:
            df = raw.reset_index()
            if "Adj Close" not in df.columns and "Close" in df.columns:
                df["Adj Close"] = df["Close"]
            return _normalize_df(df)
    except Exception:
        pass
    return pd.DataFrame()

def _download_stooq(ticker: str, period="3y") -> pd.DataFrame:
    """Optional fallback via pandas-datareader (Stooq). Requires `pip install pandas-datareader`."""
    if not HAS_PDR:
        return pd.DataFrame()
    try:
        # Stooq returns reverse chronological by default
        df = pdr.DataReader(ticker, "stooq")
        if isinstance(df, pd.DataFrame) and not df.empty:
            df = df.reset_index()  # Date column
            # Cut roughly to desired age
            days = _period_to_days(period)
            cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=days)
            df = df[df["Date"] >= cutoff]
            if "Adj Close" not in df.columns and "Close" in df.columns:
                df["Adj Close"] = df["Close"]
            return _normalize_df(df)
    except Exception:
        pass
    return pd.DataFrame()

# ---------------- public API ----------------
def get_price_history(
    tickers,
    period: str = "3y",
    interval: str = "1d",
    prefer_live: bool = True,
    min_rows: int = 120,
    use_cache: bool = True,
    allow_stooq_fallback: bool = True,  # <— optional live fallback
) -> dict[str, pd.DataFrame]:
    """
    Return {ticker: DataFrame('Date','Adj Close')}.
    - prefer_live=True → try live first (Yahoo → Stooq* if enabled), then cache
    - prefer_live=False → try cache first, then live
    - Successful live fetches are cached to data/prices/
    """
    out: dict[str, pd.DataFrame] = {}
    tickers = [str(t).upper() for t in tickers]

    for t in tickers:
        df = pd.DataFrame()

        if prefer_live:
            # 1) Yahoo
            df = _download_yf(t, period, interval)
            # 2) Stooq fallback (optional)
            if (df.empty or len(df) < min_rows) and allow_stooq_fallback:
                df = _download_stooq(t, period)
            # 3) Cache
            if (df.empty or len(df) < min_rows) and use_cache:
                df = _load_csv(t)
        else:
            # 1) Cache
            df = _load_csv(t) if use_cache else pd.DataFrame()
            # 2) Live
            if df.empty or len(df) < min_rows:
                live = _download_yf(t, period, interval)
                if (live.empty or len(live) < min_rows) and allow_stooq_fallback:
                    live = _download_stooq(t, period)
                if not live.empty:
                    df = live

        if not df.empty:
            _save_csv(t, df)

        out[t] = df

    return out

def list_cached_tickers() -> list[str]:
    seen = []
    for d in CACHE_DIRS:
        for p in d.glob("*.csv"):
            name = p.stem.upper()
            if name not in seen:
                seen.append(name)
    return seen
# === Matrix helpers for app/portfolio optimizer ===
def get_prices_matrix(tickers, period="3y", interval="1d",
                      how: str = "inner", prefer_live: bool = True) -> pd.DataFrame:
    """
    Date-indexed wide matrix (Adj Close) with one column per ticker.
    Joins on dates across tickers, forward-fills small gaps.
    """
    from .price_data import get_price_history
    tickers = [str(t).upper() for t in tickers]
    hist = get_price_history(tickers, period=period, interval=interval, prefer_live=prefer_live)
    frames = []
    for t, df in hist.items():
        if df is None or df.empty:
            continue
        d = df[["Date", "Adj Close"]].copy()
        d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
        d = d.dropna(subset=["Date"]).rename(columns={"Adj Close": t}).set_index("Date")
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    mat = pd.concat(frames, axis=1, join=how).sort_index()
    mat = mat.ffill().dropna(how="all")
    return mat

def get_returns_matrix(tickers, period="3y", interval="1d",
                       how: str = "inner", prefer_live: bool = True) -> pd.DataFrame:
    px = get_prices_matrix(tickers, period=period, interval=interval, how=how, prefer_live=prefer_live)
    return px.pct_change().dropna(how="all") if not px.empty else px

# --- helpers to infer a period from lookback ---
def _infer_period_from_lookback(lookback_days: int) -> str:
    # Overshoot the request so we have headroom for rolling features
    if lookback_days <= 260:
        return "2y"
    if lookback_days <= 520:
        return "3y"
    return "5y"


def get_prices_matrix(
    tickers,
    lookback_days: int = 252,
    period: str | None = None,
    interval: str = "1d",
    prefer_live: bool = False,          # use cache-first by default for stability
    min_rows: int = 120,
    use_cache: bool = True,
    allow_stooq_fallback: bool = True,
) -> pd.DataFrame:
    """
    Build a wide prices matrix with Date index and one column per ticker.
    Returns the last `lookback_days` aligned rows (inner-joined on common dates).
    """
    tickers = [str(t).upper() for t in tickers]
    period = period or _infer_period_from_lookback(lookback_days)

    bundle = get_price_history(
        tickers,
        period=period,
        interval=interval,
        prefer_live=prefer_live,
        min_rows=min_rows,
        use_cache=use_cache,
        allow_stooq_fallback=allow_stooq_fallback,
    )

    series = []
    for t, df in bundle.items():
        if df is None or df.empty:
            continue
        s = (df.set_index("Date")["Adj Close"]
                .astype(float)
                .rename(t))
        series.append(s)

    if not series:
        return pd.DataFrame()

    # align on common dates so optimizers see a consistent panel
    wide = pd.concat(series, axis=1, join="inner").sort_index()

    # keep the most recent N rows; if fewer exist, return what we have
    if lookback_days and lookback_days > 0:
        wide = wide.tail(lookback_days)

    return wide


def get_returns_matrix(price_matrix: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns from a price matrix."""
    if price_matrix is None or price_matrix.empty:
        return pd.DataFrame()
    return price_matrix.pct_change().dropna(how="all")
