#!/usr/bin/env python
# coding: utf-8

# In[1]:


# File: data_loader.py

from __future__ import annotations
from pathlib import Path
from typing import Iterable, Callable, Union, Optional
import hashlib
import pandas as pd
from pandas.errors import EmptyDataError, ParserError

# ── internals ──────────────────────────────────────────────────────────
def _hash(parts: Iterable[str | int | float]) -> str:
    key = "_".join(str(p) for p in parts if p is not None)
    return hashlib.md5(key.encode()).hexdigest()[:8]

def _safe_load(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_parquet(path)
    except (EmptyDataError, ParserError, OSError, ValueError) as e:
        print(f"⚠️  Cache file corrupted, deleting: {path.name} ({type(e).__name__}: {e})")
        path.unlink(missing_ok=True)          # drop corrupt file
        return None

# ── public API ────────────────────────────────────────────────────────
def cache_read(
    *,
    key: Iterable[str | int | float],
    loader: Callable[[], Union[pd.Series, pd.DataFrame]],
    cache_dir: Union[str, Path] = "cache",
    prefix: Optional[str] = None,
) -> Union[pd.Series, pd.DataFrame]:
    """
    Returns cached object if present, else computes via `loader()` and caches.

    Example
    -------
    series = cache_read(
        key     = ["SPY", "2020-01", "2024-06"],
        loader  = lambda: expensive_fetch(...),
        cache_dir = "cache_prices",
        prefix  = "SPY",
    )
    """
    cache_dir = Path(cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    fname = f"{prefix or key[0]}_{_hash(key)}.parquet"
    path  = cache_dir / fname

    if path.is_file():
        df = _safe_load(path)
        if df is not None:
            return df.iloc[:, 0] if df.shape[1] == 1 else df

    obj = loader()                                    # cache miss → compute
    df  = obj.to_frame(name=obj.name or "value") if isinstance(obj, pd.Series) else obj
    df.to_parquet(path, engine="pyarrow", compression="zstd", index=True)
    return obj


def cache_write(
    obj: Union[pd.Series, pd.DataFrame],
    *,
    key: Iterable[str | int | float],
    cache_dir: Union[str, Path] = "cache",
    prefix: Optional[str] = None,
) -> Path:
    """
    Force-write `obj` under a key.  Returns the Path written.
    """
    cache_dir = Path(cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    fname = f"{prefix or key[0]}_{_hash(key)}.parquet"
    path  = cache_dir / fname
    df    = obj.to_frame(name=obj.name or "value") if isinstance(obj, pd.Series) else obj
    df.to_parquet(path, engine="pyarrow", compression="zstd", index=True)
    return path


# In[2]:


# File: data_loader.py

import requests
import pandas as pd
import numpy as np
import statsmodels.api as sm
from datetime import datetime
from typing import Optional, Union, List, Dict
from dotenv import load_dotenv
import os

# Load .env file before accessing environment variables
load_dotenv()

# Configuration
FMP_API_KEY = os.getenv("FMP_API_KEY")
API_KEY  = FMP_API_KEY
BASE_URL = "https://financialmodelingprep.com/stable"


def fetch_monthly_close(
    ticker: str,
    start_date: Optional[Union[str, datetime]] = None,
    end_date:   Optional[Union[str, datetime]] = None
) -> pd.Series:
    """
    Fetch month-end closing prices for a given ticker from FMP.

    Uses the `/stable/historical-price-eod/full` endpoint with optional
    `from` and `to` parameters, then resamples to month-end.

    Args:
        ticker (str):       Stock or ETF symbol.
        start_date (str|datetime, optional): Earliest date (inclusive).
        end_date   (str|datetime, optional): Latest date (inclusive).

    Returns:
        pd.Series: Month-end close prices indexed by date.
    """
    # ----- loader (runs only on cache miss) ------------------------------
    def _api_pull() -> pd.Series:
        params = {"symbol": ticker, "apikey": API_KEY, "serietype": "line"}
        if start_date:
            params["from"] = pd.to_datetime(start_date).date().isoformat()
        if end_date:
            params["to"]   = pd.to_datetime(end_date).date().isoformat()
    
        resp = requests.get(f"{BASE_URL}/historical-price-eod/full", params=params, timeout=30)
        resp.raise_for_status()
        raw  = resp.json()
        data = raw if isinstance(raw, list) else raw.get("historical", [])
    
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        monthly = df.sort_index().resample("ME")["close"].last()
        return monthly

    # ----- call cache layer ---------------------------------------------
    return cache_read(
        key=[ticker, start_date or "none", end_date or "none"],
        loader=_api_pull,
        cache_dir="cache_prices",
        prefix=ticker,
    )


def fetch_monthly_treasury_rates(
    maturity: str = "month3",
    start_date: Optional[Union[str, datetime]] = None,
    end_date:   Optional[Union[str, datetime]] = None
) -> pd.Series:
    """
    Fetch month-end Treasury rates for a given maturity from FMP.

    Uses the `/stable/treasury-rates` endpoint to get Treasury rates,
    then resamples to month-end to align with stock price data.

    Args:
        maturity (str): Treasury maturity ("month3", "month6", "year1", etc.)
        start_date (str|datetime, optional): Earliest date (inclusive).
        end_date   (str|datetime, optional): Latest date (inclusive).

    Returns:
        pd.Series: Month-end Treasury rates (as percentages) indexed by date.
    """
    # ----- loader (runs only on cache miss) ------------------------------
    def _api_pull() -> pd.Series:
        params = {"apikey": API_KEY}
        if start_date:
            params["from"] = pd.to_datetime(start_date).date().isoformat()
        if end_date:
            params["to"] = pd.to_datetime(end_date).date().isoformat()
        
        resp = requests.get(f"{BASE_URL}/treasury-rates", params=params, timeout=30)
        resp.raise_for_status()
        raw = resp.json()
        
        # Create DataFrame from API response
        df = pd.DataFrame(raw)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        
        # Extract the specified maturity column
        if maturity not in df.columns:
            available = list(df.columns)
            raise ValueError(f"Maturity '{maturity}' not available. Available: {available}")
        
        # Sort by date (API already filtered by date range)
        df_sorted = df.sort_index()
        
        # Resample to month-end (align with stock prices)
        monthly = df_sorted.resample("ME")[maturity].last()
        monthly.name = f"treasury_{maturity}"
        return monthly

    # ----- call cache layer ---------------------------------------------
    return cache_read(
        key=["treasury", maturity, start_date or "none", end_date or "none"],
        loader=_api_pull,
        cache_dir="cache_prices",
        prefix=f"treasury_{maturity}",
    )

# ----------------------------------------------------------------------
#  RAM-cache wrapper  (add this at the very bottom of data_loader.py)
# ----------------------------------------------------------------------
from functools import lru_cache
import pandas as pd                                 # already imported above

# 1) private handle to the disk-cached version
_fetch_monthly_close_disk = fetch_monthly_close     
_fetch_monthly_treasury_rates_disk = fetch_monthly_treasury_rates

# 2) re-export the public name with an LRU layer
@lru_cache(maxsize=256)          # tune size to taste
def fetch_monthly_close(         # ← same name seen by callers
    ticker: str,
    start_date: str | None = None,
    end_date:   str | None = None,
) -> pd.Series:
    """
    RAM-cached → disk-cached → network price fetch.
    Same signature and behaviour as the original function.
    """
    return _fetch_monthly_close_disk(ticker, start_date, end_date)


@lru_cache(maxsize=64)          # smaller cache for Treasury rates
def fetch_monthly_treasury_rates(
    maturity: str = "month3",
    start_date: str | None = None,
    end_date:   str | None = None,
) -> pd.Series:
    """
    RAM-cached → disk-cached → network Treasury rate fetch.
    Same signature and behaviour as the original function.
    """
    return _fetch_monthly_treasury_rates_disk(maturity, start_date, end_date)



# In[ ]:




