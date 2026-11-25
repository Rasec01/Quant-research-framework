"""
Fama-French five-factor data from Kenneth French's data library.

The monthly file holds, in percent: Mkt-RF (market minus risk-free), SMB
(size), HML (value), RMW (profitability), CMA (investment) and RF (one-month
T-bill). https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
"""
import io
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd

FF5_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
           "F-F_Research_Data_5_Factors_2x3_CSV.zip")
FACTORS = ["SMB", "HML", "RMW", "CMA"]


def parse_ff_csv(text: str) -> pd.DataFrame:
    """Read the monthly block of a Ken French CSV; returns decimals indexed by month end."""
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(",") and "Mkt-RF" in l)
    cols = [c.strip() for c in lines[start].split(",")[1:]]
    rows = []
    for line in lines[start + 1:]:
        parts = [p.strip() for p in line.split(",")]
        if not parts[0].isdigit() or len(parts[0]) != 6:
            break  # the annual block starts after the monthly one
        rows.append([parts[0]] + [float(p) for p in parts[1:]])
    df = pd.DataFrame(rows, columns=["month"] + cols)
    df.index = pd.to_datetime(df.pop("month"), format="%Y%m") + pd.offsets.MonthEnd(0)
    return df / 100


def load_ff5(cache: Optional[str] = "data/ff5_monthly.csv") -> pd.DataFrame:
    """Download (or read from cache) the monthly five-factor file."""
    if cache and Path(cache).exists():
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    import urllib.request

    with urllib.request.urlopen(FF5_URL, timeout=60) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    text = z.read(z.namelist()[0]).decode("latin-1")
    df = parse_ff_csv(text)
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache)
    print(f"[data] Fama-French 5 factors, {df.index[0]:%Y-%m} → {df.index[-1]:%Y-%m}")
    return df
