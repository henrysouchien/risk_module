#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ─── CELL: Consolidated Helpers & Test with Docstrings ─────────────────────────

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


def calc_monthly_returns(prices: pd.Series) -> pd.Series:
    """
    Compute percent-change monthly returns from price series.

    Args:
        prices (pd.Series): Month-end price series.

    Returns:
        pd.Series: Monthly % change returns, NaNs dropped.
    """
    return prices.pct_change().dropna()


def compute_volatility(returns: pd.Series) -> Dict[str, float]:
    """
    Calculate monthly and annualized volatility from a returns series.

    Args:
        returns (pd.Series): Series of periodic returns.

    Returns:
        dict: {
            "monthly_vol": float,  # standard deviation of returns
            "annual_vol":  float   # scaled by sqrt(12)
        }
    """
    vol_m = float(returns.std())
    vol_a = vol_m * np.sqrt(12)
    return {"monthly_vol": vol_m, "annual_vol": vol_a}


def compute_regression_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Run OLS regression of stock returns vs. market returns.

    Args:
        df (pd.DataFrame): DataFrame with columns ["stock", "market"].

    Returns:
        dict: {
            "beta":      float,  # slope coefficient
            "alpha":     float,  # intercept
            "r_squared": float,  # model R²
            "idio_vol":  float   # std deviation of residuals
        }
    """
    X     = sm.add_constant(df["market"])
    model = sm.OLS(df["stock"], X).fit()
    return {
        "beta":      float(model.params["market"]),
        "alpha":     float(model.params["const"]),
        "r_squared": float(model.rsquared),
        "idio_vol_m":  float(model.resid.std())
    }


def fetch_peer_median_monthly_returns(
    tickers: List[str],
    start_date: Optional[Union[str, datetime]] = None,
    end_date:   Optional[Union[str, datetime]] = None
) -> pd.Series:
    """
    Compute the cross-sectional median of peer tickers' monthly returns.

    Args:
        tickers (List[str]): List of peer ticker symbols.
        start_date (str|datetime, optional): Earliest date for fetch.
        end_date   (str|datetime, optional): Latest date for fetch.

    Returns:
        pd.Series: Median of monthly returns across peers.
    """
    series_list = []
    for t in tickers:
        prices = fetch_monthly_close(t, start_date=start_date, end_date=end_date)
        rets   = calc_monthly_returns(prices).rename(t)
        series_list.append(rets)
    df_peers = pd.concat(series_list, axis=1).dropna()
    return df_peers.median(axis=1)


def fetch_excess_return(
    etf_ticker: str,
    market_ticker: str = "SPY",
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None
) -> pd.Series:
    """
    Compute style-factor excess returns: ETF minus market, aligned by index.

    Returns:
        pd.Series: Excess monthly returns (etf - market), aligned on date.
    """
    etf_ret    = calc_monthly_returns(fetch_monthly_close(etf_ticker,    start_date, end_date))
    market_ret = calc_monthly_returns(fetch_monthly_close(market_ticker, start_date, end_date))

    # Force strict index alignment before subtraction
    common_idx = etf_ret.index.intersection(market_ret.index)
    etf_aligned    = etf_ret.loc[common_idx]
    market_aligned = market_ret.loc[common_idx]

    return etf_aligned - market_aligned

def compute_factor_metrics(
    stock_returns: pd.Series,
    factor_dict: Dict[str, pd.Series]
) -> pd.DataFrame:
    """
    Runs independent single-factor regressions of stock returns vs. each factor.

    For each factor, calculates:
      • beta (cov(stock, factor) / var(factor))
      • R² (correlation squared)
      • idiosyncratic volatility (monthly residual std deviation)

    Args:
        stock_returns (pd.Series): Monthly stock returns (datetime index).
        factor_dict (Dict[str, pd.Series]): Dict of factor name to factor return series.

    Returns:
        pd.DataFrame: One row per factor, columns: beta, r_squared, idio_vol_m
    """
    results = {}

    for name, factor_series in factor_dict.items():
        # Force exact alignment
        common_idx = stock_returns.index.intersection(factor_series.index)
        stock = stock_returns.loc[common_idx]
        factor = factor_series.loc[common_idx]

        if len(stock) < 2:
            continue  # Skip if not enough data

        # Calculate regression statistics
        cov = stock.cov(factor)
        var = factor.var()
        beta = cov / var
        alpha = stock.mean() - beta * factor.mean()
        resid = stock - (alpha + beta * factor)
        idio_vol_m = resid.std(ddof=1)
        r_squared = stock.corr(factor) ** 2

        results[name] = {
            "beta":        float(beta),
            "r_squared":   float(r_squared),
            "idio_vol_m":  float(idio_vol_m)
        }

    return pd.DataFrame(results).T  # One row per factor


# In[2]:


import pandas as pd
from datetime import datetime
#from data_loader import fetch_monthly_close  # or wherever you defined this

def verify_monthly_returns(
    prices: pd.Series
) -> pd.DataFrame:
    """
    Construct a table showing month-end prices and returns for verification.

    Args:
        prices (pd.Series): Month-end closing prices indexed by date.

    Returns:
        pd.DataFrame: Columns:
            - price: Month-end closing price
            - pct_change: pandas pct_change() result
            - manual_return: (price_t / price_{t-1}) - 1
    """
    df = prices.to_frame(name="price")
    df["pct_change"]    = df["price"].pct_change()
    df["manual_return"] = df["price"] / df["price"].shift(1) - 1
    return df.dropna()

# Usage example:

# Define exact window
start_date = "2019-04-30"
end_date   = "2024-03-31"

# Fetch month-end prices
prices = fetch_monthly_close("PCTY", start_date=start_date, end_date=end_date)

# Build verification table
df_verify = verify_monthly_returns(prices)

# Display to inspect
print(df_verify)


# In[3]:


import pandas as pd
from datetime import datetime

def verify_excess_return(
    etf_prices: pd.Series,
    market_prices: pd.Series
) -> pd.DataFrame:
    """
    Create a table showing ETF returns, market returns, and excess returns.

    Args:
        etf_prices (pd.Series): Month-end prices of the ETF.
        market_prices (pd.Series): Month-end prices of the market index.

    Returns:
        pd.DataFrame: Columns:
            - etf_price
            - market_price
            - etf_return
            - market_return
            - excess_return (etf - market)
    """
    etf_df = etf_prices.to_frame(name="etf_price")
    market_df = market_prices.to_frame(name="market_price")

    df = etf_df.join(market_df, how="inner")
    df["etf_return"] = df["etf_price"].pct_change()
    df["market_return"] = df["market_price"].pct_change()
    df["excess_return"] = df["etf_return"] - df["market_return"]

    return df.dropna()

# Define exact window
start_date = "2019-04-30"
end_date   = "2024-03-31"

# Fetch price series
etf_prices    = fetch_monthly_close("IWD", start_date=start_date, end_date=end_date)
market_prices = fetch_monthly_close("SPY", start_date=start_date, end_date=end_date)

# Build verification table
df_check = verify_excess_return(etf_prices, market_prices)

# Display to inspect
print(df_check)


# In[4]:


# ─── Test Block: Fixed Risk Calculations ─────────────────────────────────────

# Define exact window
start = "2019-04-30"
end   = "2024-03-31"

# Fetch and compute stock returns
stock_prices  = fetch_monthly_close("PCTY", start_date=start, end_date=end)
stock_returns = calc_monthly_returns(stock_prices)

# Helper: align any factor return series to stock_returns
def stable_align(series: pd.Series) -> pd.Series:
    return series.loc[stock_returns.index.intersection(series.index)]

# Fetch and persist factor return series over the same window
spy_prices      = fetch_monthly_close("SPY", start_date=start, end_date=end)
spy_returns     = calc_monthly_returns(spy_prices)

momentum_raw    = fetch_excess_return(
    "MTUM", "SPY", start_date=start, end_date=end
)
value_raw       = fetch_excess_return(
    "IWD",  "SPY", start_date=start, end_date=end
)
industry_raw    = fetch_peer_median_monthly_returns(
    ["XSW"], start_date=start, end_date=end
)
subindustry_raw = fetch_peer_median_monthly_returns(
    ["PAYC","CDAY","ADP","PYCR"], start_date=start, end_date=end
)

# Align all series to stock_returns
market_returns    = stable_align(spy_returns)
momentum_returns  = stable_align(momentum_raw)
value_returns     = stable_align(value_raw)
industry_returns  = stable_align(industry_raw)
subind_returns    = stable_align(subindustry_raw)

# Build DataFrame for base market regression
df_ret = pd.DataFrame({
    "stock":  stock_returns,
    "market": market_returns
}).dropna()

# Run and print base risk stats
print("n_obs:", len(df_ret))
print("Volatility Metrics:", compute_volatility(df_ret["stock"]))
print("Risk Metrics:",       compute_regression_metrics(df_ret))

# Build factor dictionary and run single-factor regressions
factor_dict = {
    "market":      market_returns,
    "momentum":    momentum_returns,
    "value":       value_returns,
    "industry":    industry_returns,
    "subindustry": subind_returns
}

df_factors_summary = compute_factor_metrics(stock_returns, factor_dict)
print("Single-Factor Regression Summary:\n", df_factors_summary)


# In[5]:


# Define exact window
start = "2019-04-30"
end = "2024-03-31"

value = fetch_excess_return("IWD", "SPY", start_date=start)
common_idx = stock_returns.index.intersection(value.index)

# Confirm sizes
print("Length of stock_returns:", len(stock_returns))
print("Length of value factor:", len(value))
print("Length after intersection:", len(common_idx))

# Manual beta calc (Excel-style)
s = stock_returns.loc[common_idx]
v = value.loc[common_idx]

cov = s.cov(v)
var = v.var()
beta = cov / var

print("Manual value beta:", beta)
print("Manual R²:", s.corr(v)**2)


# In[6]:


import pandas as pd

# Define exact window
start = "2019-04-30"
end = "2024-03-31"

# Fetch prices and returns
stock_prices  = fetch_monthly_close("PCTY", start_date=start, end_date=end)
value_prices  = fetch_monthly_close("IWD",  start_date=start, end_date=end)
market_prices = fetch_monthly_close("SPY",  start_date=start, end_date=end)

stock_returns = calc_monthly_returns(stock_prices)
value_factor  = calc_monthly_returns(value_prices) - calc_monthly_returns(market_prices)

# Align data
common_idx = stock_returns.index.intersection(value_factor.index)
stock_aligned = stock_returns.loc[common_idx]
value_aligned = value_factor.loc[common_idx]

# Raw beta components
cov = stock_aligned.cov(value_aligned)
var = value_aligned.var()
beta = cov / var

# Output
print("n_obs:", len(common_idx))
print("cov:", cov)
print("var:", var)
print("beta:", beta)


# In[8]:


def main():
    # 5-year trailing window
    start = pd.Timestamp.today() - pd.DateOffset(years=5)

    # Fetch data
    stock_prices  = fetch_monthly_close("PCTY", start_date=start)
    market_prices = fetch_monthly_close("SPY",  start_date=start)

    # Calculate returns
    df_ret = pd.DataFrame({
        "stock":  calc_monthly_returns(stock_prices),
        "market": calc_monthly_returns(market_prices)
    }).dropna()

    # Compute metrics
    vol_metrics = compute_volatility(df_ret["stock"])
    risk_metrics = compute_regression_metrics(df_ret)

    # Output
    print("Volatility Metrics:", vol_metrics)
    print("Risk Metrics:", risk_metrics)

if __name__ == "__main__":
    main()


# In[9]:


# ─── CELL: Corrected Beta Diagnostic for Exact Date Window ─────────────────────

import numpy as np
import pandas as pd

# 1) Define the exact window you want to analyze
start_date = "2019-05-31"
end_date   = "2024-03-31"

# 2) Rebuild df_ret on that window
stock_prices  = fetch_monthly_close("PCTY", start_date=start_date, end_date=end_date)
market_prices = fetch_monthly_close("SPY",  start_date=start_date, end_date=end_date)

stock_ret  = calc_monthly_returns(stock_prices)
market_ret = calc_monthly_returns(market_prices)

df_ret = pd.DataFrame({
    "stock":  stock_ret,
    "market": market_ret
}).dropna()

# 3) Compute sample (n−1) covariance & variance to match Excel’s COVARIANCE.S / VAR.S
cov_sample = df_ret["stock"].cov(df_ret["market"])   # ddof=1 by default
var_sample = df_ret["market"].var()                  # ddof=1 by default

# 4) Manual beta calculation (sample)
beta_manual = cov_sample / var_sample

# 5) Recompute regression metrics on this exact df_ret
risk_metrics_exact = compute_regression_metrics(df_ret)

# 6) Correlation and R²
corr = df_ret["stock"].corr(df_ret["market"])
r_squared = risk_metrics_exact["r_squared"]

# 7) Print out all diagnostics
print(f"Window: {start_date} → {end_date}")
print(f"Observations (n):            {len(df_ret)}")
print(f"Sample Covariance (COV.S):   {cov_sample:.6f}")
print(f"Sample Variance (VAR.S):     {var_sample:.6f}")
print(f"Manual β (cov/var):          {beta_manual:.4f}")
print(f"Reported β (from regression): {risk_metrics_exact['beta']:.4f}")
print(f"Correlation:                 {corr:.4f}")
print(f"R² from regression:          {r_squared:.4f}")


# In[10]:


start = pd.Timestamp.today() - pd.DateOffset(years=5)

# no need to pass end_date unless you want to cap earlier
stock_prices  = fetch_monthly_close("PCTY", start_date=start)
market_prices = fetch_monthly_close("SPY",  start_date=start)

print(stock_prices.head())
print(stock_prices.tail())
print(f"Fetched {len(stock_prices)} months: {stock_prices.index.min().date()} → {stock_prices.index.max().date()}")


# In[11]:


# Cell 2 – Fetch, compute returns, volatility & regression

# 5-year trailing window
start = pd.Timestamp.today() - pd.DateOffset(years=5)

# 1) Fetch month-end closes
stock_prices  = fetch_monthly_close("PCTY", start_date=start)
market_prices = fetch_monthly_close("SPY",  start_date=start)

# 2) Calculate % monthly returns
stock_ret  = calc_monthly_returns(stock_prices)
market_ret = calc_monthly_returns(market_prices)

# 3) Combine into DataFrame
df_ret = pd.DataFrame({
    "stock":  stock_ret,
    "market": market_ret
}).dropna()

# 4) Compute volatility metrics
vol_metrics = compute_volatility(df_ret["stock"])
print("Volatility Metrics:", vol_metrics)

# 5) Compute regression risk metrics
risk_metrics = compute_regression_metrics(df_ret)
print("Risk Metrics:", risk_metrics)


# In[48]:


# File: portfolio_risk.py

import pandas as pd
import numpy as np
from typing import Dict

def normalize_weights(weights: Dict[str, float], normalize: bool = True) -> Dict[str, float]:
    """
    Ensure weights sum to 1. If normalize is False, returns as-is.
    """
    if not normalize:
        return weights
    total = sum(weights.values())
    if total == 0:
        raise ValueError("Sum of weights is zero, cannot normalize.")
    return {t: w / total for t, w in weights.items()}

def compute_portfolio_returns(
    returns: pd.DataFrame,
    weights: Dict[str, float]
) -> pd.Series:
    """
    Given a DataFrame of individual asset returns (columns = tickers)
    and a dict of weights, compute the weighted portfolio return series.
    """
    w = normalize_weights(weights)
    # align columns and weights
    aligned = returns[list(w.keys())].dropna()
    weight_vec = np.array([w[t] for t in aligned.columns])
    # dot product row-wise
    port_ret = aligned.values.dot(weight_vec)
    return pd.Series(port_ret, index=aligned.index, name="portfolio")

def compute_covariance_matrix(
    returns: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute the sample covariance matrix of asset returns.
    """
    return returns.cov()

def compute_correlation_matrix(
    returns: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute the sample correlation matrix of asset returns.

    Args:
        returns (pd.DataFrame): DataFrame where each column is an asset's return series.

    Returns:
        pd.DataFrame: Correlation matrix between assets.
    """
    return returns.corr()

def compute_portfolio_volatility(
    weights: Dict[str, float],
    cov_matrix: pd.DataFrame
) -> float:
    """
    Compute portfolio volatility = sqrt(w^T Σ w).
    """
    w = normalize_weights(weights)
    w_vec = np.array([w[t] for t in cov_matrix.index])
    var_p = float(w_vec.T.dot(cov_matrix.values).dot(w_vec))
    return np.sqrt(var_p)

def compute_risk_contributions(
    weights: Dict[str, float],
    cov_matrix: pd.DataFrame
) -> pd.Series:
    """
    Compute each asset’s risk contribution to total portfolio volatility.
    RC_i = w_i * (Σ w)_i / σ_p
    Returns a Series indexed by ticker.
    """
    w = normalize_weights(weights)
    w_vec = np.array([w[t] for t in cov_matrix.index])
    sigma_p = compute_portfolio_volatility(weights, cov_matrix)
    # marginal contributions = (Σ w)_i
    marg = cov_matrix.values.dot(w_vec)
    rc = w_vec * marg / sigma_p
    return pd.Series(rc, index=cov_matrix.index, name="risk_contrib")

def compute_herfindahl(
    weights: Dict[str, float]
) -> float:
    """
    Compute the Herfindahl index = sum(w_i^2).
    Indicates portfolio concentration (0 = fully diversified, 1 = single asset).
    """
    w = normalize_weights(weights)
    return float(sum([w_i ** 2 for w_i in w.values()]))

# Example usage snippet (to paste in your notebook):
#
# import pandas as pd
# from portfolio_risk import (
#     compute_portfolio_returns,
#     compute_covariance_matrix,
#     compute_portfolio_volatility,
#     compute_risk_contributions,
#     compute_herfindahl
# )
#
# # assume df_ret is a DataFrame of monthly returns for your universe
# weights = {"PCTY": 0.4, "AAPL": 0.6}
#
# # 1) Portfolio returns
# port_ret = compute_portfolio_returns(df_ret, weights)
#
# # 2) Covariance
# cov = compute_covariance_matrix(df_ret)
#
# # 3) Portfolio volatility
# vol = compute_portfolio_volatility(weights, cov)
# print("Portfolio Volatility:", vol)
#
# # 4) Risk contributions
# rc = compute_risk_contributions(weights, cov)
# print("Risk Contributions:\n", rc)
#
# # 5) Concentration (Herfindahl)
# h = compute_herfindahl(weights)
# print("Herfindahl Index:", h)


# In[77]:


import pandas as pd
from typing import Dict

def compute_stock_factor_betas(
    stock_ret: pd.Series,
    factor_rets: Dict[str, pd.Series]
) -> Dict[str, float]:
    """
    Wrapper that re-uses compute_factor_metrics to get betas for a single stock.

    Args
    ----
    stock_ret : pd.Series
        The stock’s return series, indexed by date.
    factor_rets : Dict[str, pd.Series]
        Mapping {factor_name: factor_return_series}.

    Returns
    -------
    Dict[str, float]
        {factor_name: beta} pulled straight from compute_factor_metrics.
    """
    # call the existing helper (must already be imported / defined in scope)
    df_metrics = compute_factor_metrics(stock_ret, factor_rets)

    # return only the β column as a plain dict
    return df_metrics["beta"].to_dict()

def calc_factor_vols(
    factor_dict: Dict[str, pd.Series],
    annualize: bool = True
) -> Dict[str, float]:
    """
    Return annualised σ for every factor series supplied.

    factor_dict  – {"market": Series, "momentum": Series, ...}
    """
    k = 12 ** 0.5 if annualize else 1.0
    return {name: float(series.std(ddof=1) * k) for name, series in factor_dict.items()}


def calc_weighted_factor_variance(
    weights: Dict[str, float],
    betas_df: pd.DataFrame,
    factor_vols: Dict[str, float]
) -> pd.DataFrame:
    """
    Weighted factor variance for each (asset, factor):

       w_i² · β_i,f² · σ_f²

    weights      – {"PCTY": 0.15, ...}
    betas_df     – DataFrame index=tickers, columns=factors, values=β
    factor_vols  – {"market": 0.18, ...}  (annual σ, *not* %)

    Returns DataFrame same shape as betas_df.
    """
    w2 = pd.Series(weights).pow(2)
    sigma2 = {f: v ** 2 for f, v in factor_vols.items()}
    var_df = betas_df.pow(2) * pd.DataFrame(sigma2, index=["__tmp__"]).T   # β² σ²
    return var_df.mul(w2, axis=0)   # multiply each row by w_i²


# In[104]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Dict, List, Optional, Any, Union

def get_returns_dataframe(
    weights: Dict[str, float],
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    rets = {}
    for t in weights:
        prices = fetch_monthly_close(t, start_date=start_date, end_date=end_date)
        rets[t] = calc_monthly_returns(prices)
    return pd.DataFrame(rets).dropna()

def compute_target_allocations(
    weights: Dict[str, float],
    expected_returns: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    df = pd.DataFrame({
        "Portfolio Weight": pd.Series(weights),
        "Equal Weight":     pd.Series({t: 1/len(weights) for t in weights})
    })
    if expected_returns:
        total = sum(expected_returns.values())
        df["Prop Target"] = pd.Series({t: expected_returns[t]/total for t in expected_returns})
        df["Prop Diff"]   = df["Portfolio Weight"] - df["Prop Target"]
    df["Eq Diff"] = df["Portfolio Weight"] - df["Equal Weight"]
    return df
    
def build_portfolio_view(
    weights: Dict[str, float],
    start_date: str,
    end_date:   str,
    expected_returns: Optional[Dict[str, float]] = None,
    stock_factor_proxies: Optional[Dict[str, Dict[str, Union[str, List[str]]]]] = None
) -> Dict[str, Any]:
    
    # 0) portfolio-level data using intersection frame -----------------
    df_ret   = get_returns_dataframe(weights, start_date, end_date)
    df_alloc = compute_target_allocations(weights, expected_returns)

    port_ret = compute_portfolio_returns(df_ret, weights)
    cov_mat  = compute_covariance_matrix(df_ret)
    corr_mat = compute_correlation_matrix(df_ret)

    vol_m = compute_portfolio_volatility(weights, cov_mat)
    vol_a = vol_m * np.sqrt(12)
    rc    = compute_risk_contributions(weights, cov_mat)
    hhi   = compute_herfindahl(weights)

    # 1) per-stock factor betas on each stock’s *own* window -----------
    df_stock_betas = pd.DataFrame(index=weights.keys())
    idio_var_dict  = {}

    if stock_factor_proxies:
        for ticker, proxies in stock_factor_proxies.items():
            # fetch this stock’s complete return series
            prices    = fetch_monthly_close(ticker, start_date=start_date, end_date=end_date)
            stock_ret = calc_monthly_returns(prices)
            idx       = stock_ret.index

            # build factor dictionary aligned to idx
            fac_dict: Dict[str, pd.Series] = {}

            mkt_t = proxies.get("market")
            if mkt_t:
                mkt_ret = calc_monthly_returns(
                    fetch_monthly_close(mkt_t, start_date=start_date, end_date=end_date)
                ).reindex(idx).dropna()
                fac_dict["market"] = mkt_ret

            mom_t = proxies.get("momentum")
            if mom_t and mkt_t:
                mom_ret = fetch_excess_return(mom_t, mkt_t, start_date, end_date).reindex(idx).dropna()
                fac_dict["momentum"] = mom_ret

            val_t = proxies.get("value")
            if val_t and mkt_t:
                val_ret = fetch_excess_return(val_t, mkt_t, start_date, end_date).reindex(idx).dropna()
                fac_dict["value"] = val_ret

            for facname in ("industry", "subindustry"):
                proxy = proxies.get(facname)
                if proxy:
                    if isinstance(proxy, list):
                        ser = fetch_peer_median_monthly_returns(proxy, start_date, end_date)
                    else:
                        ser = calc_monthly_returns(
                            fetch_monthly_close(proxy, start_date=start_date, end_date=end_date)
                        )
                    fac_dict[facname] = ser.reindex(idx).dropna()

            # drop rows with any NaN
            factor_df  = pd.DataFrame(fac_dict).dropna(how="any")
            if factor_df.empty:
                # cannot compute betas / idio var with no data
                continue
        
            aligned_s = stock_ret.reindex(factor_df.index)
                    
            # betas
            betas = compute_stock_factor_betas(
                aligned_s,                               # stock on same dates
                {c: factor_df[c] for c in factor_df}     # factors on same dates
            )

            df_stock_betas.loc[ticker, betas.keys()] = pd.Series(betas)

            # idiosyncratic variance (σ² of residuals vs same factors)
            X      = sm.add_constant(factor_df)
            resid  = aligned_s - sm.OLS(aligned_s, X).fit().fittedvalues
            idio_var_dict[ticker] = float(resid.var(ddof=1))

    # ─────────────────────────  FACTOR-LEVEL σ & VAR (stock-specific)  ────────────
    df_factor_vols   = pd.DataFrame(index=df_stock_betas.index,
                                    columns=df_stock_betas.columns)   # σ_i,f (annual)
    weighted_factor_var = pd.DataFrame(index=df_stock_betas.index,
                                       columns=df_stock_betas.columns) # w_i² β² σ²
    
    if stock_factor_proxies:                                           # ← guard
        w2 = pd.Series(weights).pow(2)                                 # w_i²
    
        for tkr, proxies in stock_factor_proxies.items():
    
            # ----- rebuild this stock’s factor-return dict (same logic as above) --
            idx_stock = calc_monthly_returns(
                fetch_monthly_close(tkr, start_date, end_date)
            ).index
    
            fac_ret: Dict[str, pd.Series] = {}
    
            mkt = proxies.get("market")
            if mkt:
                fac_ret["market"] = calc_monthly_returns(
                    fetch_monthly_close(mkt, start_date, end_date)
                ).reindex(idx_stock).dropna()
    
            def _excess(etf: str) -> pd.Series:
                return fetch_excess_return(etf, mkt, start_date, end_date
                       ).reindex(idx_stock).dropna()
    
            if proxies.get("momentum"):
                fac_ret["momentum"] = _excess(proxies["momentum"])
            if proxies.get("value"):
                fac_ret["value"]    = _excess(proxies["value"])
    
            for fac in ("industry", "subindustry"):
                proxy = proxies.get(fac)
                if proxy:
                    ser = ( fetch_peer_median_monthly_returns(proxy, start_date, end_date)
                            if isinstance(proxy, list)
                            else calc_monthly_returns(
                                    fetch_monthly_close(proxy, start_date, end_date) ) )
                    fac_ret[fac] = ser.reindex(idx_stock).dropna()
    
            if not fac_ret:         # nothing to measure
                continue
    
            # ----- annual σ_i,f ----------------------------------------------------
            sigmas = pd.Series({f: r.std(ddof=1) * np.sqrt(12) for f, r in fac_ret.items()})
            df_factor_vols.loc[tkr, sigmas.index] = sigmas
    

            # df_factor_vols  : σ-table (annual factor vols by stock)
            df_factor_vols = (
                df_factor_vols
                    .apply(pd.to_numeric, errors="coerce")  # force numeric, NaNs where bad
                    .astype("float64", copy=False)          # ensure float dtype, no copy if already
                    .fillna(0.0)                           # now safe – no warning
            )
            
            # betas_filled   : β-table with NaNs → 0.0
            betas_filled = (
                df_stock_betas
                    .apply(pd.to_numeric, errors="coerce")
                    .astype("float64", copy=False)
                    .fillna(0.0)
            )
    
        # ----- weighted factor variance  w_i² β_i,f² σ_i,f² -----------------------
        weighted_factor_var = betas_filled.pow(2) * df_factor_vols.pow(2)
        weighted_factor_var = weighted_factor_var.mul(w2, axis=0)
    
    # ──────────────────────────────────────────────────────────────────────────────


    # 2) portfolio-level factor betas ---------------------------------
    w_series                = pd.Series(weights)
    portfolio_factor_betas  = df_stock_betas.mul(w_series, axis=0).sum(skipna=True)

    # 3) per-asset annualised stats ----------------------------------------
    asset_vol_a = df_ret.std(ddof=1) * np.sqrt(12)               # total σ_annual
    asset_var_m = df_ret.var(ddof=1)                             # monthly σ²
    w_series    = pd.Series(weights)
    
    # idiosyncratic
    idio_var_m  = pd.Series(idio_var_dict).reindex(w_series.index)
    idio_vol_a  = idio_var_m.pow(0.5) * np.sqrt(12)              # σ_idio annual
    
    df_asset = pd.DataFrame({
        "Vol A":              asset_vol_a,                       # total annual σ
        "Weighted Vol A":     asset_vol_a * w_series,
        #"Var M":              asset_var_m,                       # monthly total σ² (for reference)
        #"Weighted Var M":     asset_var_m * (w_series ** 2),
        "Idio Vol A":         idio_vol_a,                        # idio annual σ
        "Weighted IdioVol A": idio_vol_a * w_series,
        #"Idio Var M":         idio_var_m,                        # monthly idio σ²
        #"Weighted IdioVar M": idio_var_m * (w_series ** 2),
    })

    # 4) assemble output ----------------------------------------------
    return {
        "allocations":            df_alloc,
        "covariance_matrix":      cov_mat,
        "correlation_matrix":     corr_mat,
        "volatility_monthly":     vol_m,
        "volatility_annual":      vol_a,
        "risk_contributions":     rc,
        "herfindahl":             hhi,
        "df_stock_betas":         df_stock_betas,
        "portfolio_factor_betas": portfolio_factor_betas,
        "factor_vols":            df_factor_vols,         
        "weighted_factor_var":    weighted_factor_var, 
        "asset_vol_summary":      df_asset,
        "portfolio_returns":      port_ret
    }


# In[105]:


# ─── CELL: Run Portfolio View ─────────────────────────────────────────────────

import pandas as pd

# 1) Define window, weights, expected returns, and per-stock factor proxies
start_date = "2019-05-31"
end_date   = "2024-03-31"
weights    = {"TW":0.15, "MSCI":0.15, "NVDA":0.17, "PCTY":0.15, "AT.L":0.28}
expected_returns = {"TW":0.15,"MSCI":0.16,"NVDA":0.20,"PCTY":0.17,"AT.L":0.25}

# stock_factor_proxies maps each ticker to the ETFs/peer‐lists you want to use:
stock_factor_proxies = {
    "TW":   {"market":"SPY","momentum":"MTUM","value":"IWD","industry":"KCE","subindustry":["TW","MSCI","NVDA"]},
    "MSCI": {"market":"SPY","momentum":"MTUM","value":"IWD","industry":"KCE","subindustry":["TW","MSCI","NVDA"]},
    "NVDA": {"market":"SPY","momentum":"MTUM","value":"IWD","industry":"SOXX","subindustry":["SOXX","XSW","IXC"]},
    "PCTY": {"market":"SPY","momentum":"MTUM","value":"IWD","industry":"XSW","subindustry":["PAYC","CDAY","ADP"]},
    "AT.L": {"market":"ACWX","momentum":"IMTM","value":"IVLU","industry":"IXC","subindustry":["IXC"]}
}

# 2) Call the high-level summary builder
summary = build_portfolio_view(
    weights,
    start_date,
    end_date,
    expected_returns=expected_returns,
    stock_factor_proxies=stock_factor_proxies
)

# 3) Unpack and print what you need
print("=== Target Allocations ===")
print(summary["allocations"], "\n")

print("=== Portfolio Returns (head) ===")
print(summary["portfolio_returns"].head(), "\n")

print("=== Covariance Matrix ===")
print(summary["covariance_matrix"], "\n")

print("=== Correlation Matrix ===")
print(summary["correlation_matrix"], "\n")

print(f"Monthly Volatility:  {summary['volatility_monthly']:.4%}")
print(f"Annual Volatility:   {summary['volatility_annual']:.4%}\n")

print("=== Risk Contributions ===")
print(summary["risk_contributions"], "\n")

print("Herfindahl Index:", summary["herfindahl"], "\n")

print("=== Per-Stock Factor Betas ===")
print(summary["df_stock_betas"], "\n")

print("=== Portfolio-Level Factor Betas ===")
print(summary["portfolio_factor_betas"], "\n")

print("=== Per-Asset Vol & Var ===")
print(summary["asset_vol_summary"], "\n")

# ── NEW: factor-level diagnostics ─────────────────────────────────────────
print("=== Factor Annual Volatilities (σ_i,f) ===")
print(summary["factor_vols"].round(4))          # pretty table in Jupyter

print("\n=== Weighted Factor Variance   w_i² · β_i,f² · σ_i,f² ===")
print(summary["weighted_factor_var"].round(6))


# In[ ]:





# In[ ]:




