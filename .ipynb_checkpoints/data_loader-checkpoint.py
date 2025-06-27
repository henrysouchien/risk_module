#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# File: data_loader.py

import requests
import pandas as pd
import numpy as np
import statsmodels.api as sm
from datetime import datetime
from typing import Optional, Union, List, Dict

# Configuration
API_KEY  = "0ZDOD7zoxCPQLDOyDw5e1tE8bwEFfKWk"
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


