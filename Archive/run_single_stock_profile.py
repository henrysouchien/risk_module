#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from pprint import pprint

from risk_summary import (
    get_stock_risk_profile,
    get_detailed_stock_factor_profile
)

from factor_utils import (
    fetch_excess_return,
    fetch_peer_median_monthly_returns,
    calc_monthly_returns,
    compute_volatility,
    compute_regression_metrics,
    compute_factor_metrics
)

from data_loader import fetch_monthly_close


# In[ ]:


# File: run_single_stock_profile.py

#from risk_summary import get_stock_risk_profile  # only if it's in a separate .py file

# Example: Get 5-year risk profile for PCTY vs SPY
result = get_stock_risk_profile("PCTY", benchmark="SPY", years=5)

print("=== Volatility Metrics ===")
print(result["vol_metrics"])

print("\n=== Regression Risk Metrics ===")
print(result["risk_metrics"])


# In[ ]:


# File: run_single_stock_profile.py

#from risk_summary import get_detailed_stock_factor_profile  # only if it's in a separate .py file

start = "2019-04-30"
end   = "2024-03-31"

profile = get_detailed_stock_factor_profile(
    ticker="PCTY",
    start_date=start,
    end_date=end,
    factor_proxies={
        "market": "SPY",
        "momentum": "MTUM",
        "value": "IWD",
        "industry": "XSW",
        "subindustry": ["PAYC", "PYCR", "CDAY", "ADP", "PAYX", "WDAY"]
    }
)

print("=== Volatility ===")
print(profile["vol_metrics"])

print("\n=== Market Regression ===")
print(profile["regression_metrics"])

print("\n=== Factor Summary ===")
print(profile["factor_summary"])

