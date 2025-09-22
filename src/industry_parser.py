# src/industry_parser.py
import re, difflib
from typing import List, Tuple

def _norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

# Sector synonyms → canonical sector label
SECTOR_SYNONYMS = {
    "Technology": {
        "tech","technology","software","saas","cloud","semiconductor","semiconductors",
        "chip","chips","it","it services","computer","computers","internet","electronics",
        "electronic","communication equipment","data processing"
    },
    "Healthcare": {"health","healthcare","health care","biotech","biotechnology","pharma","pharmaceutical","medical"},
    "Energy": {"energy","oil","gas","o&g","renewable","renewables","solar","wind"},
    "Consumer Cyclical": {"retail","ecommerce","e commerce","automobile","auto","apparel","hotels","resorts","restaurants","leisure","travel"},
    "Consumer Staples": {"staples","beverages","food","tobacco","household","personal products","grocery"},
    "Industrials": {"industrial","industrials","aerospace","defense","transportation","machinery","construction","electrical equipment"},
    "Materials": {"materials","basic materials","chemicals","metals","mining","paper","packaging"},
    "Real Estate": {"real estate","reit","reits","property","properties"},
    "Utilities": {"utility","utilities","regulated electric","regulated gas"},
    "Communication Services": {"communication","communications","telecom","telecommunications","media","entertainment","advertising"},
    "Financial Services": {"finance","financial","financials","bank","banks","banking","insurance","asset management","brokerage","capital markets","wealth"}
}

# Extra patterns to catch Industry names 
IT_INDUSTRY_PATTERNS = {
    "software","application software","software application","systems software",
    "semiconductor","semiconductors","it services","computer","internet",
    "electronic equipment","communication equipment","data processing","technology hardware"
}

def detect_filters(message: str, dataset_sectors: List[str], dataset_industries: List[str]) -> Tuple[str, List[str], List[str]]:
    """
    Returns (display_label, sectors_to_keep, industries_to_keep).
    Empty lists mean 'no filter' for that dimension.
    """
    msg = _norm(message)

    # Explicit 'all' -- no filter
    if any(w in msg for w in ["any industry", "all industry", "all industries", "any sector", "all sector", "all"]):
        return "", [], []

    # 1) If the message directly contains an industry name
    inds_norm = [_norm(x) for x in dataset_industries]
    for ind, indn in zip(dataset_industries, inds_norm):
        if indn and indn in msg:
            return ind, [], [ind]

    # 2) Sector synonyms
    ds_sectors_norm = { _norm(s): s for s in dataset_sectors }
    for canonical_sector, keys in SECTOR_SYNONYMS.items():
        if any(k in msg for k in keys):
            # Only apply if that sector actually exists in dataset
            for dsn, original in ds_sectors_norm.items():
                if _norm(canonical_sector) == dsn:
                    # Also collect industries that look tech-like if Technology
                    if canonical_sector == "Technology":
                        inds = [orig for orig, n in zip(dataset_industries, inds_norm)
                                if any(p in n for p in IT_INDUSTRY_PATTERNS)]
                        return canonical_sector, [original], inds
                    return canonical_sector, [original], []
    # 3) Fuzzy fallback against sector names
    sectors_norm_list = list(ds_sectors_norm.keys())
    match = difflib.get_close_matches(msg, sectors_norm_list, n=1, cutoff=0.6)
    if match:
        return ds_sectors_norm[match[0]], [ds_sectors_norm[match[0]]], []

    # 4) Fuzzy against industries
    match = difflib.get_close_matches(msg, inds_norm, n=1, cutoff=0.6)
    if match:
        k = inds_norm.index(match[0])
        return dataset_industries[k], [], [dataset_industries[k]]

    return "", [], []
