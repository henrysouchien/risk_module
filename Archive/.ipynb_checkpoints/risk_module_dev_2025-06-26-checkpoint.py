#!/usr/bin/env python
# coding: utf-8

# In[1]:


#  File: data_loader.py

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


# In[2]:


#  File: factor_utils.py

import requests
import pandas as pd
import numpy as np
import statsmodels.api as sm
from datetime import datetime
from typing import Optional, Union, List, Dict

# Configuration
API_KEY  = "0ZDOD7zoxCPQLDOyDw5e1tE8bwEFfKWk"
BASE_URL = "https://financialmodelingprep.com/stable"


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


# In[3]:


# File: risk_summary.py

from datetime import datetime
from typing import Dict, Union
import pandas as pd

def get_stock_risk_profile(
    ticker: str,
    start_date: Union[str, pd.Timestamp],
    end_date: Union[str, pd.Timestamp],
    benchmark: str = "SPY"
) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Pulls monthly prices between given dates, computes returns, vol, and regression metrics.
    Returns a dict:
      {
        "vol_metrics": {...},
        "risk_metrics": {...}
      }

    Args:
        ticker (str): Stock symbol.
        start_date (str or pd.Timestamp): Start of analysis window.
        end_date (str or pd.Timestamp): End of analysis window.
        benchmark (str): Benchmark ticker for regression (default: "SPY").
    """
    stock_prices  = fetch_monthly_close(ticker,    start_date=start_date, end_date=end_date)
    market_prices = fetch_monthly_close(benchmark, start_date=start_date, end_date=end_date)

    stock_ret  = calc_monthly_returns(stock_prices)
    market_ret = calc_monthly_returns(market_prices)

    df_ret = pd.DataFrame({
        "stock":  stock_ret,
        "market": market_ret
    }).dropna()

    vol_metrics  = compute_volatility(df_ret["stock"])
    risk_metrics = compute_regression_metrics(df_ret)

    return {
        "vol_metrics":  vol_metrics,
        "risk_metrics": risk_metrics
    }


# In[4]:


# File: run_single_stock_profile.py

#from risk_summary import get_stock_risk_profile  # only if it's in a separate .py file
start = "2019-04-30"
end   = "2024-03-31"

# Example: Get 5-year risk profile for PCTY vs SPY
result = get_stock_risk_profile("PCTY", start_date=start, end_date=end, benchmark="SPY")

print("=== Volatility Metrics ===")
print(result["vol_metrics"])

print("\n=== Regression Risk Metrics ===")
print(result["risk_metrics"])


# In[5]:


# File: risk_summary.py

from typing import List, Dict, Optional, Union
import pandas as pd

def get_detailed_stock_factor_profile(
    ticker: str,
    start_date: Union[str, pd.Timestamp],
    end_date: Union[str, pd.Timestamp],
    factor_proxies: Dict[str, Union[str, List[str]]],
    market_ticker: str = "SPY"
) -> Dict[str, Union[pd.DataFrame, Dict[str, float]]]:
    """
    Computes full factor risk diagnostics for a stock over a given window,
    using specified ETF proxies and peer sets.

    Args:
        ticker (str): Stock ticker to analyze.
        start_date (str or Timestamp): Start date for analysis window.
        end_date (str or Timestamp): End date for analysis window.
        factor_proxies (dict): Mapping of factor name → ETF or peer list.
            Required keys: market, momentum, value, industry, subindustry
        market_ticker (str): Market benchmark ticker (for excess returns).

    Returns:
        dict:
            - vol_metrics: volatility stats
            - regression_metrics: beta, alpha, R², idio vol (market only)
            - factor_summary: DataFrame of beta / R² / idio vol per factor
    """
    stock_prices  = fetch_monthly_close(ticker, start_date=start_date, end_date=end_date)
    stock_returns = calc_monthly_returns(stock_prices)

    def align(series: pd.Series) -> pd.Series:
        return series.loc[stock_returns.index.intersection(series.index)]

    # Fetch and align all factor return series
    market_ret   = align(calc_monthly_returns(fetch_monthly_close(factor_proxies["market"], start_date, end_date)))
    momentum_ret = align(fetch_excess_return(factor_proxies["momentum"], market_ticker, start_date, end_date))
    value_ret    = align(fetch_excess_return(factor_proxies["value"], market_ticker, start_date, end_date))

    industry_ret = align(fetch_peer_median_monthly_returns(
        factor_proxies["industry"] if isinstance(factor_proxies["industry"], list)
        else [factor_proxies["industry"]],
        start_date, end_date
    ))

    subind_ret = align(fetch_peer_median_monthly_returns(
        factor_proxies["subindustry"] if isinstance(factor_proxies["subindustry"], list)
        else [factor_proxies["subindustry"]],
        start_date, end_date
    ))

    # Build combined factor dictionary
    factor_dict = {
        "market":      market_ret,
        "momentum":    momentum_ret,
        "value":       value_ret,
        "industry":    industry_ret,
        "subindustry": subind_ret
    }

    # Regression vs market only
    df_reg = pd.DataFrame({"stock": stock_returns, "market": market_ret}).dropna()

    return {
        "vol_metrics": compute_volatility(df_reg["stock"]),
        "regression_metrics": compute_regression_metrics(df_reg),
        "factor_summary": compute_factor_metrics(stock_returns, factor_dict)
    }


# In[6]:


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


# In[7]:


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


# In[8]:


# File: factor_utils.py

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


# In[9]:


# File: portfolio_risk.py

from typing import Any

def compute_portfolio_variance_breakdown(
    weights: Dict[str, float],
    idio_var_dict: Dict[str, float],
    weighted_factor_var: pd.DataFrame,
    vol_m: float
) -> Dict[str, Any]:
    """
    Returns a structured variance decomposition:
      - total variance
      - idiosyncratic variance + %
      - factor variance + %
      - per-factor variance + %
    """
    w = pd.Series(weights)
    w2 = w.pow(2)

    # Idiosyncratic variance (sum of w_i² * σ²_idio_i)
    idio_var_series = pd.Series(idio_var_dict).reindex(w.index).fillna(0.0)
    idio_var = float((w2 * idio_var_series).sum())

    # Factor variance (sum of weighted factor variance matrix)
    factor_var_matrix = (
        weighted_factor_var
        .drop(columns=["industry", "subindustry"], errors="ignore")  # REMOVE
        .fillna(0.0)
    )
    
    per_factor_var = factor_var_matrix.sum(axis=0)
    factor_var = float(per_factor_var.sum())

    # Total portfolio variance
    port_var = factor_var + idio_var

    # % shares
    idio_pct   = idio_var   / port_var if port_var else 0.0
    factor_pct = factor_var / port_var if port_var else 0.0

    # Breakdown of factor variance by factor
    per_factor_var = factor_var_matrix.sum(axis=0)
    per_factor_pct = per_factor_var / port_var

    return {
        "portfolio_variance":      port_var,
        "idiosyncratic_variance":  idio_var,
        "idiosyncratic_pct":       idio_pct,
        "factor_variance":         factor_var,
        "factor_pct":              factor_pct,
        "factor_breakdown_var":    per_factor_var.to_dict(),
        "factor_breakdown_pct":    per_factor_pct.to_dict()
    }


# In[10]:


# File: portfolio_risk.py

import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Dict, List, Optional, Any, Union

def get_returns_dataframe(
    weights: Dict[str, float],
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Fetch and compute monthly returns for all tickers in the weights dictionary.

    Args:
        weights (Dict[str, float]): Portfolio weights (tickers as keys).
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: Monthly return series for all tickers, aligned and cleaned.
    """
    rets = {}
    for t in weights:
        prices = fetch_monthly_close(t, start_date=start_date, end_date=end_date)
        rets[t] = calc_monthly_returns(prices)
    return pd.DataFrame(rets).dropna()

def compute_target_allocations(
    weights: Dict[str, float],
    expected_returns: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Compute target allocations based on expected returns and equal weight comparison.

    Args:
        weights (Dict[str, float]): Current portfolio weights.
        expected_returns (Optional[Dict[str, float]]): Expected returns for tickers.

    Returns:
        pd.DataFrame: Allocation table with portfolio weight, equal weight, and proportional return targets.
    """
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
    """
    Builds a complete portfolio risk profile using historical returns, factor regressions,
    and variance decomposition.

    Performs:
    - Aggregates returns, volatility, and correlation for the portfolio.
    - Runs per-stock single-factor regressions to compute betas (market, momentum, value, industry, subindustry).
    - Calculates idiosyncratic volatilities and annualized variances.
    - Computes per-stock factor volatilities (σ_i,f) and weighted factor variance (w² · β² · σ²).
    - Decomposes portfolio variance into idiosyncratic vs factor-driven.
    - Aggregates per-industry ETF variance contributions (based on industry proxies).
    - Computes portfolio-level factor betas and Herfindahl concentration.
    - Summarizes per-industry group betas from weighted contributions of individual stock betas.

    Args:
        weights (Dict[str, float]):
            Portfolio weights by ticker (not required to sum to 1).
        start_date (str):
            Historical window start date (format: YYYY-MM-DD).
        end_date (str):
            Historical window end date (format: YYYY-MM-DD).
        expected_returns (Optional[Dict[str, float]]):
            Optional target returns per ticker for allocation gap display.
        stock_factor_proxies (Optional[Dict]):
            Mapping of each stock to its factor proxies:
                - "market": ETF ticker (e.g., SPY)
                - "momentum": ETF ticker (e.g., MTUM)
                - "value": ETF ticker (e.g., IWD)
                - "industry": ETF ticker (e.g., SOXX)
                - "subindustry": list of tickers (e.g., ["PAYC", "CDAY"])

    Returns:
        Dict[str, Any]: Portfolio diagnostics including:
            - 'allocations': target vs actual vs expected returns
            - 'portfolio_returns': aggregated monthly returns
            - 'covariance_matrix': asset return covariances
            - 'correlation_matrix': asset return correlations
            - 'volatility_monthly': annualized volatility from monthly returns
            - 'volatility_annual': total annual portfolio volatility
            - 'risk_contributions': risk contribution by asset
            - 'herfindahl': portfolio concentration score
            - 'df_stock_betas': per-stock factor betas from regressions
            - 'portfolio_factor_betas': weighted sum of factor exposures
            - 'factor_vols': per-stock annualized factor volatilities
            - 'weighted_factor_var': w² · β² · σ² contributions
            - 'asset_vol_summary': asset-level volatility and idio stats
            - 'variance_decomposition': total vs idio vs factor variance
            - 'industry_variance': {
                'absolute': variance by industry proxy,
                'percent_of_portfolio': % variance per industry,
                'per_industry_group_beta': weighted betas per industry ETF
              }
    """
    # ─── 0. Portfolio Return Setup ──────────────────────────────────────────────
    df_ret   = get_returns_dataframe(weights, start_date, end_date)
    df_alloc = compute_target_allocations(weights, expected_returns)

    port_ret = compute_portfolio_returns(df_ret, weights)
    cov_mat  = compute_covariance_matrix(df_ret)
    corr_mat = compute_correlation_matrix(df_ret)

    vol_m = compute_portfolio_volatility(weights, cov_mat)
    vol_a = vol_m * np.sqrt(12)
    rc    = compute_risk_contributions(weights, cov_mat)
    hhi   = compute_herfindahl(weights)

    w_series                = pd.Series(weights)

    # ─── 1. Stock-Level Factor Exposures ────────────────────────────────────────
    df_stock_betas = pd.DataFrame(index=weights.keys())
    idio_var_dict  = {}

    if stock_factor_proxies:
        for ticker, proxies in stock_factor_proxies.items():
            # Fetch stock returns
            prices    = fetch_monthly_close(ticker, start_date=start_date, end_date=end_date)
            stock_ret = calc_monthly_returns(prices)
            idx       = stock_ret.index

            # Build aligned factor series
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
                continue # Skip if no usable data
        
            aligned_s = stock_ret.reindex(factor_df.index)
                    
            # Run single-factor regression to get betas
            betas = compute_stock_factor_betas(
                aligned_s,                               # stock on same dates
                {c: factor_df[c] for c in factor_df}     # factors on same dates
            )
            df_stock_betas.loc[ticker, betas.keys()] = pd.Series(betas)

            # Idiosyncratic variance (monthly → annual)
            X      = sm.add_constant(factor_df)
            resid  = aligned_s - sm.OLS(aligned_s, X).fit().fittedvalues
        
            # Convert monthly residual variance to annual variance
            monthly_idio_var = resid.var(ddof=1)
            annual_idio_var = monthly_idio_var * 12
            idio_var_dict[ticker] = float(annual_idio_var)

    # ─── 2. Compute Factor Volatility & Weighted Variance ───────────────────────
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


    # ─── 3a. Aggregate Industry-Level Variance ───────────────────────────────────
    industry_var_dict = {}
    
    # Step: reverse-map which stock maps to which industry ETF
    for tkr, proxies in stock_factor_proxies.items():
        ind = proxies.get("industry")
        if ind:
            v = weighted_factor_var.loc[tkr, "industry"] if "industry" in weighted_factor_var.columns else 0.0
            industry_var_dict[ind] = industry_var_dict.get(ind, 0.0) + v

    # ─── 3b. Compute Per-Industry Group Beta (and max weighted exposure) ──────────────
    industry_groups: Dict[str, float] = {}

    for ticker in w_series.index:
        proxy = stock_factor_proxies.get(ticker, {}).get("industry")
        beta = df_stock_betas.get("industry", {}).get(ticker, 0.0)
        weight = w_series[ticker]
        if proxy:
            industry_groups[proxy] = industry_groups.get(proxy, 0.0) + (weight * beta)
    
    # ─── 4. Final Portfolio Stats (Volatility, Idio, Betas) ─────────────────────
    portfolio_factor_betas  = df_stock_betas.mul(w_series, axis=0).sum(skipna=True)

    # 4a) per-asset annualised stats ----------------------------------------
    asset_vol_a = df_ret.std(ddof=1) * np.sqrt(12)               # total σ_annual
    asset_var_m = df_ret.var(ddof=1)                             # monthly σ²
    w_series    = pd.Series(weights)
    
    # idiosyncratic
    idio_var_a  = pd.Series(idio_var_dict).reindex(w_series.index)         # already annual
    idio_vol_a  = idio_var_a.pow(0.5)                                       # √(annual var)
    weighted_idio_var_model = w_series.pow(2) * idio_var_a  # w² · σ²_idio

    # Manually compute (w × σ_idio)² for comparison
    weighted_idio_vol = idio_vol_a * w_series
    weighted_idio_var_manual = (weighted_idio_vol) ** 2
    
    df_asset = pd.DataFrame({
        "Vol A":              asset_vol_a,                       # total annual σ
        "Weighted Vol A":     asset_vol_a * w_series,
        #"Var M":              asset_var_m,                       # monthly total σ² (for reference)
        #"Weighted Var M":     asset_var_m * (w_series ** 2),
        "Idio Vol A":         idio_vol_a,                        # idio annual σ
        "Weighted Idio Vol A": weighted_idio_vol,
        "Weighted Idio Var": weighted_idio_var_model,
        #"Manual Weighted Idio Var": weighted_idio_var_manual
        #"Weighted IdioVar A": idio_var_a * (w_series ** 2),
    })

    # ─── 5. Industry Variance % Contribution ────────────────────────────────────
    total_port_var = (
        compute_portfolio_variance_breakdown(
            weights, idio_var_dict, weighted_factor_var, vol_m
        )["portfolio_variance"]
    )
    
    industry_pct_dict = {
        k: v / total_port_var if total_port_var else 0.0
        for k, v in industry_var_dict.items()
    }

    # ─── 6. Assemble Final Output ───────────────────────────────────────────────
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
        "portfolio_returns":      port_ret,
        "variance_decomposition": compute_portfolio_variance_breakdown(
        weights, idio_var_dict, weighted_factor_var, vol_m),
        "industry_variance": {
        "absolute": industry_var_dict,
        "percent_of_portfolio": industry_pct_dict,
        "per_industry_group_beta": industry_groups,
    }
    }


# In[11]:


# File: run_portfolio_risk.py

from typing import Dict, Callable, Union

def standardize_portfolio_input(
    raw_input: Dict[str, Dict[str, Union[float, int]]],
    price_fetcher: Callable[[str], float]
) -> Dict[str, Union[Dict[str, float], float]]:
    """
    Normalize portfolio input into weights using shares, dollar value, or direct weight.

    Args:
        raw_input (dict): Dict of ticker → {"shares": int}, {"dollars": float}, or {"weight": float}
        price_fetcher (callable): Function to fetch latest price for a given ticker

    Returns:
        dict: {
            "weights": Dict[ticker, normalized weight],
            "dollar_exposure": Dict[ticker, dollar amount],
            "total_value": float,
            "net_exposure": float,
            "gross_exposure": float,
            "leverage": float
        }
    """
    dollar_exposure = {}

    for ticker, entry in raw_input.items():
        if "weight" in entry:
            # Will normalize weights separately
            continue
        elif "dollars" in entry:
            dollar_exposure[ticker] = float(entry["dollars"])
        elif "shares" in entry:
            price = price_fetcher(ticker)
            dollar_exposure[ticker] = float(entry["shares"]) * price
        else:
            raise ValueError(f"Invalid input for {ticker}: must provide 'shares', 'dollars', or 'weight'.")

    # If any weights were specified, override dollar_exposure logic
    if all("weight" in entry for entry in raw_input.values()):
        weights = {t: float(v["weight"]) for t, v in raw_input.items()}
        normalized_weights = normalize_weights(weights)

        net_exposure = sum(weights.values())
        gross_exposure = sum(abs(w) for w in weights.values())
        leverage = gross_exposure / net_exposure if net_exposure != 0 else np.inf
        
        return {
            "weights": normalized_weights,
            "dollar_exposure": None,
            "total_value": None,
            "net_exposure": net_exposure,
            "gross_exposure": gross_exposure,
            "leverage": leverage
        }

    total_value = sum(dollar_exposure.values())
    weights = {t: v / total_value for t, v in dollar_exposure.items()}

    net_exposure = sum(weights.values())
    gross_exposure = sum(abs(w) for w in weights.values())
    leverage = gross_exposure / net_exposure if net_exposure else np.inf

    return {
        "weights": weights,
        "dollar_exposure": dollar_exposure,
        "total_value": total_value,
        "net_exposure": net_exposure,
        "gross_exposure": gross_exposure,
        "leverage": leverage
    }


# In[12]:


# File: run_portfolio_risk.py

def latest_price(ticker: str) -> float:
    """
    Fetches the latest available month-end closing price for a given ticker.

    Args:
        ticker (str): Ticker symbol of the stock or ETF.

    Returns:
        float: Most recent non-NaN month-end closing price.
    """
    prices = fetch_monthly_close(ticker)
    return prices.dropna().iloc[-1]


# In[13]:


# File: run_portfolio_risk.py

import yaml
from pprint import pprint
from typing import Optional

def load_and_display_portfolio_config(filepath: str = "portfolio.yaml") -> Optional[dict]:
    """
    Loads the portfolio YAML file, parses weights, and prints input diagnostics.

    Args:
        filepath (str): Path to the YAML portfolio config file.

    Returns:
        dict: Parsed config dictionary (useful for testing or further calls).
    """
    with open(filepath, "r") as f:
        config = yaml.safe_load(f)

    # Extract components
    start_date            = config["start_date"]
    end_date              = config["end_date"]
    portfolio_input       = config["portfolio_input"]
    expected_returns      = config["expected_returns"]
    stock_factor_proxies  = config["stock_factor_proxies"]

    parsed  = standardize_portfolio_input(portfolio_input, latest_price)
    weights = parsed["weights"]

    # ─── Print outputs ─────────────────────────────────────
    print("=== Normalized Weights ===")
    print(weights)

    print("\n=== Dollar Exposure ===")
    print(parsed["dollar_exposure"])

    print("\n=== Total Portfolio Value ===")
    print(parsed["total_value"])

    print("\n=== Net Exposure (sum of weights) ===")
    print(parsed["net_exposure"])

    print("\n=== Gross Exposure (sum of abs(weights)) ===")
    print(parsed["gross_exposure"])

    print("\n=== Leverage (gross / net) ===")
    print(parsed["leverage"])

    print("\n=== Expected Returns ===")
    pprint(expected_returns)

    print("\n=== Stock Factor Proxies ===")
    for ticker, proxies in stock_factor_proxies.items():
        print(f"\n→ {ticker}")
        pprint(proxies)

    return config


# In[14]:


# Load portfolio and display configs

config = load_and_display_portfolio_config("portfolio.yaml")
start_date = config["start_date"]
end_date = config["end_date"]
expected_returns = config["expected_returns"]
stock_factor_proxies = config["stock_factor_proxies"]
portfolio_input = config["portfolio_input"]
weights = standardize_portfolio_input(portfolio_input, latest_price)["weights"]


# In[15]:


# File: run_portfolio_risk.py

def display_portfolio_summary(summary: dict):
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

    print("=== Factor Annual Volatilities (σ_i,f) ===")
    print(summary["factor_vols"].round(4))

    print("\n=== Weighted Factor Variance   w_i² · β_i,f² · σ_i,f² ===")
    print(summary["weighted_factor_var"].round(6), "\n")

    print("=== Portfolio Variance Decomposition ===")
    var_dec = summary["variance_decomposition"]
    print(f"Portfolio Variance:          {var_dec['portfolio_variance']:.4f}")
    print(f"Idiosyncratic Variance:      {var_dec['idiosyncratic_variance']:.4f}  ({var_dec['idiosyncratic_pct']:.0%})")
    print(f"Factor Variance:             {var_dec['factor_variance']:.4f}  ({var_dec['factor_pct']:.0%})\n")

    print("=== Factor Variance (absolute) ===")
    for k, v in var_dec["factor_breakdown_var"].items():
        print(f"{k.title():<10} : {v:.5f}")

    filtered = {
        k: v for k, v in var_dec["factor_breakdown_pct"].items()
        if k not in ("industry", "subindustry")
    }

    print("\n=== Factor Variance (% of Portfolio, excluding industry) ===")
    for k, v in filtered.items():
        print(f"{k.title():<10} : {v:.0%}")

    print("\n=== Industry Variance (absolute) ===")
    for k, v in summary["industry_variance"]["absolute"].items():
        print(f"{k:<10} : {v:.6f}")

    print("\n=== Industry Variance (% of Portfolio) ===")
    for k, v in summary["industry_variance"]["percent_of_portfolio"].items():
        print(f"{k:<10} : {v:.1%}")

    print("\n=== Per-Industry Group Betas ===")
    per_group = summary["industry_variance"].get("per_industry_group_beta", {})
    for k, v in sorted(per_group.items(), key=lambda kv: -abs(kv[1])):
        print(f"{k:<12} : {v:.4f}")


# In[16]:


# File: risk_runner.py

# 2) Call the high-level summary builder
summary = build_portfolio_view(
    weights,
    start_date,
    end_date,
    expected_returns=expected_returns,
    stock_factor_proxies=stock_factor_proxies
)
display_portfolio_summary(summary)


# In[17]:


# File: risk_helpers.py

from typing import Dict, Union, List
import pandas as pd

def get_worst_monthly_factor_losses(
    stock_factor_proxies: Dict[str, Dict[str, Union[str, List[str]]]],
    start_date: str,
    end_date: str
) -> Dict[str, float]:
    """
    For each unique factor proxy (ETF or peer group), fetch monthly returns over a historical window,
    and compute the worst single-month return (min).

    Only includes factor types: market, momentum, value, and industry.

    Args:
        stock_factor_proxies (Dict): From portfolio.yaml — maps tickers to their factor proxies.
        start_date (str): Start date for return window (YYYY-MM-DD).
        end_date (str): End date for return window (YYYY-MM-DD).

    Returns:
        Dict[str, float]: {proxy: worst 1-month return}
    """
    from data_loader import fetch_monthly_close
    from factor_utils import calc_monthly_returns

    allowed_factors = {"market", "momentum", "value", "industry"}
    unique_proxies = set()

    for proxy_map in stock_factor_proxies.values():
        for k, v in proxy_map.items():
            if k not in allowed_factors:
                continue  # skip subindustry and others
            if isinstance(v, list):
                unique_proxies.update(v)
            else:
                unique_proxies.add(v)

    worst_losses = {}

    for proxy in sorted(unique_proxies):
        try:
            prices = fetch_monthly_close(proxy, start_date, end_date)
            returns = calc_monthly_returns(prices)
            if not returns.empty:
                worst_losses[proxy] = float(returns.min())
        except Exception as e:
            print(f"⚠️ Failed for proxy {proxy}: {e}")

    return worst_losses


# In[18]:


# File: risk_helpers.py

from typing import Dict, Union, List, Tuple

def aggregate_worst_losses_by_factor_type(
    stock_factor_proxies: Dict[str, Dict[str, Union[str, List[str]]]],
    worst_losses: Dict[str, float]
) -> Dict[str, Tuple[str, float]]:
    """
    Aggregate the worst 1-month return per factor type by scanning all proxies
    assigned to each factor type across the portfolio and selecting the worst-performing one.

    Args:
        stock_factor_proxies (Dict): Mapping from tickers to their factor proxy assignments.
        worst_losses (Dict): Precomputed worst monthly return per ETF or peer.

    Returns:
        Dict[str, Tuple[str, float]]: {factor_type: (proxy, worst_return)}
    """
    factor_types = ["market", "momentum", "value", "industry"]
    factor_to_proxies: Dict[str, set] = {ftype: set() for ftype in factor_types}

    for proxy_map in stock_factor_proxies.values():
        for ftype in factor_types:
            proxy = proxy_map.get(ftype)
            if isinstance(proxy, list):
                factor_to_proxies[ftype].update(proxy)
            elif proxy:
                factor_to_proxies[ftype].add(proxy)

    factor_worst: Dict[str, Tuple[str, float]] = {}
    for ftype, proxies in factor_to_proxies.items():
        worst_proxy = None
        worst_val = float("inf")
        for proxy in proxies:
            val = worst_losses.get(proxy)
            if val is not None and val < worst_val:
                worst_val = val
                worst_proxy = proxy
        if worst_proxy is not None:
            factor_worst[ftype] = (worst_proxy, worst_val)

    return factor_worst


# In[19]:


# ─── risk_helpers.py ──────────────────────────────────────────────
from typing import Dict, Tuple, List
import pandas as pd

def compute_max_betas(
    proxies: Dict[str, Dict[str, List[str] | str]],
    start_date: str,
    end_date:   str,
    loss_limit_pct: float,
) -> Dict[str, float]:
    """
    Pure function – NO YAML reads, NO printing.

    Parameters
    ----------
    proxies : dict          # stock_factor_proxies section
    start_date, end_date : str  # analysis window (YYYY-MM-DD)
    loss_limit_pct : float      # e.g. -0.10

    Returns
    -------
    {factor_type: max_beta}
    """
    from risk_helpers import (
        get_worst_monthly_factor_losses,
        aggregate_worst_losses_by_factor_type,
    )

    worst_losses   = get_worst_monthly_factor_losses(proxies, start_date, end_date)
    worst_by_type  = aggregate_worst_losses_by_factor_type(proxies, worst_losses)

    return {
        ftype: float("inf") if worst >= 0 else loss_limit_pct / worst
        for ftype, (_, worst) in worst_by_type.items()
    }


# In[20]:


# ─── risk_helpers.py ─────────────────────────────────────────────────────────

from typing import Dict, Tuple, List
from datetime import datetime
import yaml
import pandas as pd

def calc_max_factor_betas(
    portfolio_yaml: str = "portfolio.yaml",
    risk_yaml: str = "risk_limits.yaml",
    lookback_years: int = 10,
    echo: bool = True,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Derive max-allowable portfolio betas for each factor type and industry
    from historical worst 1-month factor proxy returns.

    Parameters
    ----------
    portfolio_yaml : str
        Path to the YAML file containing `stock_factor_proxies`.
    risk_yaml : str
        Path to YAML containing `max_single_factor_loss`.
    lookback_years : int
        Historical window length to scan (ending today).
    echo : bool
        If True, pretty-prints the intermediate tables to stdout.

    Returns
    -------
    Tuple[Dict[str, float], Dict[str, float]]
        - max_betas:         {factor_type: max_beta}
        - max_betas_by_proxy: {industry_proxy: max_beta}
    """
    # 1. --- load configs -----------------------------------------------------
    with open(portfolio_yaml, "r") as f:
        port_cfg = yaml.safe_load(f)
    with open(risk_yaml, "r") as f:
        risk_cfg = yaml.safe_load(f)

    proxies = port_cfg["stock_factor_proxies"]
    loss_limit = risk_cfg["max_single_factor_loss"]  # e.g. -0.10

    # 2. --- date window ------------------------------------------------------
    end_dt = datetime.today()
    start_dt = end_dt - pd.DateOffset(years=lookback_years)
    end_str, start_str = end_dt.strftime("%Y-%m-%d"), start_dt.strftime("%Y-%m-%d")

    # 3. --- worst per-proxy --------------------------------------------------
    worst_per_proxy = get_worst_monthly_factor_losses(
        proxies, start_str, end_str
    )

    # 4. --- worst per factor-type -------------------------------------------
    worst_by_factor = aggregate_worst_losses_by_factor_type(
        proxies, worst_per_proxy
    )

    # 5. --- max beta per factor type ----------------------------------------------------
    max_betas = compute_max_betas(
        proxies, start_str, end_str, loss_limit
    )

    # 6. Compute per-industry-proxy max betas
    industry_proxies = set()
    for proxy_map in proxies.values():
        proxy = proxy_map.get("industry")
        if isinstance(proxy, list):
            industry_proxies.update(proxy)
        elif proxy:
            industry_proxies.add(proxy)

    max_betas_by_proxy = {}
    for proxy in sorted(industry_proxies):
        worst = worst_per_proxy.get(proxy)
        if worst is None or worst >= 0:
            max_betas_by_proxy[proxy] = float("inf")
        else:
            max_betas_by_proxy[proxy] = loss_limit / worst

    # --- pretty print block --------------------------------------------------
    if echo:
        print("\n=== Worst Monthly Losses per Proxy ===")
        for p, v in sorted(worst_per_proxy.items(), key=lambda kv: kv[1]):
            print(f"{p:<12} : {v:.2%}")

        print("\n=== Worst Monthly Losses per Factor Type ===")
        for ftype, (p, v) in worst_by_factor.items():
            print(f"{ftype:<10} → {p:<12} : {v:.2%}")

        print(f"\n=== Max Allowable Beta per Factor "
              f"(Loss Limit = {loss_limit:.0%}) ===")
        for ftype, beta in max_betas.items():
            print(f"{ftype:<10} → β ≤ {beta:.2f}")

        print("\n=== Max Beta per Industry Proxy ===")
        for p, b in sorted(max_betas_by_proxy.items()):
            print(f"{p:<12} → β ≤ {b:.2f}")

    return max_betas, max_betas_by_proxy


# In[21]:


# File: risk_runner.py

max_betas, max_betas_by_proxy = calc_max_factor_betas(
    portfolio_yaml="portfolio.yaml",
    risk_yaml="risk_limits.yaml",
    lookback_years=10,
    echo=True   # turn off if you only need the dict
)


# In[22]:


# File: run_portfolio_risk.py

import pandas as pd
from typing import Dict, Optional

def evaluate_portfolio_beta_limits(
    portfolio_factor_betas: pd.Series,
    max_betas: Dict[str, float],
    proxy_betas: Optional[Dict[str, float]] = None,
    max_proxy_betas: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Compares each factor's actual portfolio beta to the allowable max beta.
    Also supports proxy-level checks like individual industry ETFs.
    
    If proxy-level data (e.g. per-industry ETF) is available, it skips
    the aggregate 'industry' row to avoid double counting.

    Parameters
    ----------
    portfolio_factor_betas : pd.Series
        e.g. {"market": 0.74, "momentum": 1.1, ...}
    max_betas : dict
        e.g. {"market": 0.80, "momentum": 1.56, ...}
    proxy_betas : dict, optional
        e.g. {"SOXX": 0.218, "KCE": 0.287}
    max_proxy_betas : dict, optional
        e.g. {"SOXX": 0.56, "KCE": 0.49}

    Returns
    -------
    pd.DataFrame
        Rows: factors and proxies. Columns: portfolio_beta, max_allowed_beta, pass, buffer.
    """
    rows = []

    skip_industry = proxy_betas is not None and max_proxy_betas is not None

    # ─── Factor-level checks ─────────────────────────────────
    for factor, max_b in max_betas.items():
        if skip_industry and factor == "industry":
            continue  # skip aggregate industry if per-proxy provided

        actual = portfolio_factor_betas.get(factor, 0.0)
        rows.append({
            "factor": factor,
            "portfolio_beta": actual,
            "max_allowed_beta": max_b,
            "pass": abs(actual) <= max_b,
            "buffer": max_b - abs(actual),
        })

    # ─── Proxy-level checks (e.g. SOXX, XSW) ─────────────────
    if proxy_betas and max_proxy_betas:
        for proxy, actual in proxy_betas.items():
            max_b = max_proxy_betas.get(proxy, float("inf"))
            label = f"industry_proxy::{proxy}"
            rows.append({
                "factor": label,
                "portfolio_beta": actual,
                "max_allowed_beta": max_b,
                "pass": abs(actual) <= max_b,
                "buffer": max_b - abs(actual),
            })

    df = pd.DataFrame(rows).set_index("factor")
    return df[["portfolio_beta", "max_allowed_beta", "pass", "buffer"]]


# In[23]:


# File: risk_runner.py
# Check portfolio beta limit checks

df_beta_check = evaluate_portfolio_beta_limits(
    portfolio_factor_betas = summary["portfolio_factor_betas"],
    max_betas              = max_betas,
    proxy_betas            = summary["industry_variance"].get("per_industry_group_beta"),
    max_proxy_betas        = max_betas_by_proxy
)
# Separate into two sections
df_factors = df_beta_check[~df_beta_check.index.str.startswith("industry_proxy::")]
df_proxies = df_beta_check[df_beta_check.index.str.startswith("industry_proxy::")].copy()

# Clean up proxy labels
df_proxies.index = df_proxies.index.str.replace("industry_proxy::", "")

# Print factor-level
print("=== Portfolio Factor Exposure Checks ===\n")
print(df_factors.to_string(
    index_names=False,
    formatters={
        "portfolio_beta":   "{:+.2f}".format,
        "max_allowed_beta": "{:.2f}".format,
        "buffer":           "{:+.2f}".format,
        "pass":             lambda x: "PASS" if x else "FAIL"
    }
))

# Add spacing between sections
print("\n=== Industry Exposure Checks ===\n")
print(df_proxies.to_string(
    index_names=False,
    formatters={
        "portfolio_beta":   "{:+.2f}".format,
        "max_allowed_beta": "{:.2f}".format,
        "buffer":           "{:+.2f}".format,
        "pass":             lambda x: "PASS" if x else "FAIL"
    }
))


# In[24]:


# File: run_portfolio_risk.py

from typing import Dict, Any
import pandas as pd

def evaluate_portfolio_risk_limits(
    summary: Dict[str, Any],
    portfolio_limits: Dict[str, float],
    concentration_limits: Dict[str, float],
    variance_limits: Dict[str, float]
) -> pd.DataFrame:
    """
    Evaluates portfolio risk metrics against configured limits.

    Args:
        summary (dict): Output from build_portfolio_view().
        portfolio_limits (dict): {"max_volatility": float, "max_loss": float}
        concentration_limits (dict): {"max_single_stock_weight": float}
        variance_limits (dict): Keys include "max_factor_contribution", etc.

    Returns:
        pd.DataFrame: One row per check with actual, limit, and pass/fail.
    """
    results = []

    # 1. Volatility Check
    actual_vol = summary["volatility_annual"]
    vol_limit = portfolio_limits["max_volatility"]
    results.append({
        "Metric": "Volatility",
        "Actual": actual_vol,
        "Limit": vol_limit,
        "Pass": actual_vol <= vol_limit
    })

    # 2. Concentration Check
    weights = summary["allocations"]["Portfolio Weight"]
    max_weight = weights.abs().max()
    weight_limit = concentration_limits["max_single_stock_weight"]
    results.append({
        "Metric": "Max Weight",
        "Actual": max_weight,
        "Limit": weight_limit,
        "Pass": max_weight <= weight_limit
    })

    # 3. Factor Variance Contribution
    var_decomp = summary["variance_decomposition"]
    factor_pct = var_decomp["factor_pct"]
    factor_limit = variance_limits["max_factor_contribution"]
    results.append({
        "Metric": "Factor Var %",
        "Actual": factor_pct,
        "Limit": factor_limit,
        "Pass": factor_pct <= factor_limit
    })

    # 4. Market Variance Contribution
    market_pct = var_decomp["factor_breakdown_pct"].get("market", 0.0)
    market_limit = variance_limits["max_market_contribution"]
    results.append({
        "Metric": "Market Var %",
        "Actual": market_pct,
        "Limit": market_limit,
        "Pass": market_pct <= market_limit
    })

    # 5. Top Industry Exposure
    industry_pct_dict = summary["industry_variance"].get("percent_of_portfolio", {})
    max_industry_pct = max(industry_pct_dict.values()) if industry_pct_dict else 0.0
    industry_limit = variance_limits["max_industry_contribution"]
    results.append({
        "Metric": "Max Industry Var %",
        "Actual": max_industry_pct,
        "Limit": industry_limit,
        "Pass": max_industry_pct <= industry_limit
    })

    return pd.DataFrame(results)


# In[25]:


# === Evaluate Portfolio vs. Limits ===

# Step 0: Load Portfolio + Risk Config ────────────────────────────────────────
with open("portfolio.yaml", "r") as f:
    portfolio_config = yaml.safe_load(f)

with open("risk_limits.yaml", "r") as f:
    risk_config = yaml.safe_load(f)

stock_factor_proxies = portfolio_config["stock_factor_proxies"]
LOSS_LIMIT = risk_config["max_single_factor_loss"]  # e.g. -0.10

# Step 1: Run risk limit evaluation
df_risk = evaluate_portfolio_risk_limits(
    summary,
    risk_config["portfolio_limits"],
    risk_config["concentration_limits"],
    risk_config["variance_limits"]
)

# Step 2: Pretty print results
print("=== Portfolio Risk Limit Checks ===")
for _, row in df_risk.iterrows():
    status = "→ PASS" if row["Pass"] else "→ FAIL"
    print(f"{row['Metric']:<22} {row['Actual']:.2%}  ≤ {row['Limit']:.2%}  {status}")


# In[48]:


# ─── File: portfolio_optimizer.py ──────────────────────────────────────────
"""
Light-weight optimisation helpers that bolt onto the existing risk-runner.

Requires:
    pip install cvxpy

Functions
---------
simulate_portfolio_change(weights, edits, risk_cfg, start, end, proxies)
    → returns (summary, df_risk, df_beta)

solve_min_variance_with_risk_limits(weights, risk_cfg, start, end, proxies)
    → returns new_weights OR raises ValueError if infeasible
"""
from typing import Dict, Any
import cvxpy as cp
import pandas as pd
from copy import deepcopy

# ────────────────────────────────────────────────────────────────────────────
def simulate_portfolio_change(
    weights: Dict[str, float],
    edits: Dict[str, float],
    risk_cfg: Dict[str, Any],
    start: str,
    end: str,
    proxies: Dict[str, Dict[str, Any]],
):
    """
    Build a *new* summary after applying `edits` (delta-weights).
    `edits` can add new tickers or override existing weights.

    Example:
        new_summary, df_risk, df_beta = simulate_portfolio_change(
            weights,
            edits={"MSFT": +0.05, "AAPL": -0.02},
            ...
        )
    """
    # --- 1. rebuild weights -------------------------------------------------
    new_w = deepcopy(weights)
    for tkr, w in edits.items():
        new_w[tkr] = new_w.get(tkr, 0.0) + w

    # normalize
    tot = sum(new_w.values())
    new_w = {k: v / tot for k, v in new_w.items()}

    # --- 2. fresh risk summary ---------------------------------------------
    summary = build_portfolio_view(
        new_w, start, end, expected_returns=None, stock_factor_proxies=proxies
    )

    # --- 3. risk-limit checker ---------------------------------------------
    df_risk = evaluate_portfolio_risk_limits(
        summary,
        risk_cfg["portfolio_limits"],
        risk_cfg["concentration_limits"],
        risk_cfg["variance_limits"],
    )

    # --- 4. dynamic β caps (factor + proxy) --------------------------------    
    max_betas = compute_max_betas(
        proxies, 
        start, 
        end, 
        loss_limit_pct=risk_cfg["max_single_factor_loss"]
    )
    
    df_beta = evaluate_portfolio_beta_limits(summary["portfolio_factor_betas"], max_betas)

    return summary, df_risk, df_beta


# ────────────────────────────────────────────────────────────────────────────
def solve_min_variance_with_risk_limits(
    weights: Dict[str, float],
    risk_cfg: Dict[str, Any],
    start: str,
    end: str,
    proxies: Dict[str, Dict[str, Any]],
    allow_short: bool = False,
):
    """
    Finds the *smallest-variance* weights that satisfy **all** limits.
    Keeps the current universe (no new tickers). If infeasible, raises.

    Returns
    -------
    Dict[str, float] : optimised weights (sum to 1)
    """
    tickers = list(weights)
    n       = len(tickers)

    # Pre-compute covariance
    base_summary = build_portfolio_view(weights, start, end, None, proxies)
    Σ = base_summary["covariance_matrix"].loc[tickers, tickers].values

    # Limits for betas
    max_betas = compute_max_betas(
        proxies, 
        start, 
        end, 
        loss_limit_pct=risk_cfg["max_single_factor_loss"]
    )

    # Variables
    w = cp.Variable(n)

    # Objective: minimise portfolio variance wᵀ Σ w
    obj = cp.Minimize(cp.quad_form(w, Σ))

    cons = []

    # 1. Weights sum to 1 (fully invested)
    cons += [cp.sum(w) == 1]

    # 2. Concentration limit
    max_weight = risk_cfg["concentration_limits"]["max_single_stock_weight"]
    cons += [cp.abs(w) <= max_weight]

    if not allow_short:
        cons += [w >= 0]

    # 3. Factor beta limits
    beta_mat = base_summary["df_stock_betas"].fillna(0.0).loc[tickers]  # shape n × factors
    for fac, max_b in max_betas.items():
        if fac not in beta_mat:
            continue
        cons += [
            cp.abs(beta_mat[fac].values @ w) <= max_b
        ]

    # 4. Gross volatility limit
    max_vol = risk_cfg["portfolio_limits"]["max_volatility"]
    cons += [cp.quad_form(w, Σ) <= max_vol**2]

    prob = cp.Problem(obj, cons)
    prob.solve(solver=cp.ECOS, verbose=False)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise ValueError(f"Infeasible under current limits (status={prob.status})")

    new_w = {t: float(w.value[i]) for i, t in enumerate(tickers)}
    return new_w


# In[49]:


# ─── File: portfolio_optimizer.py ──────────────────────────────────────────

# ---------------------------------------------------------------
#  WHAT-IF helper
# ---------------------------------------------------------------

from typing import Dict, Tuple
import pandas as pd

def run_what_if(
    base_weights: pd.Series,
    delta: Dict[str, float],
    risk_cfg: Dict,
    start_date: str,
    end_date: str,
    factor_proxies: Dict[str, Dict],
) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    Apply absolute weight shifts (`delta`) to `base_weights`, evaluate the
    resulting portfolio, and pretty-print a compact risk report.

    Parameters
    ----------
    base_weights : pd.Series
        Current portfolio weights (summing to 1.0).
    delta : dict
        {ticker: +shift or –shift}.  Shifts are *absolute* (e.g. +0.05 = +5 ppts).
    risk_cfg : dict
        Parsed risk-limits YAML (needs `portfolio_limits`, `concentration_limits`,
        `variance_limits`, `max_single_factor_loss`).
    start_date, end_date : str
        Analysis window (YYYY-MM-DD).
    factor_proxies : dict
        Mapping used by `simulate_portfolio_change`.

    Returns
    -------
    summary : dict              # build_portfolio_view output
    risk_df : pd.DataFrame      # risk-limit check table
    beta_df : pd.DataFrame      # factor-beta check table
    """
    # --- run change ---------------------------------------------------------
    from run_portfolio_risk import (
        evaluate_portfolio_risk_limits,
        evaluate_portfolio_beta_limits,
    )
    summary, risk_df, beta_df = simulate_portfolio_change(
        base_weights, delta, risk_cfg,
        start_date, end_date, factor_proxies
    )

    # --- fancy title --------------------------------------------------------
    delta_str = " / ".join(f"{v:+.0%} {k}" for k, v in delta.items())
    print(f"\n📐  What-if Risk Checks ({delta_str})\n")

    # --- risk table ---------------------------------------------------------
    pct = lambda x: f"{x:.1%}"
    print(risk_df.to_string(index=False,
                            formatters={"Actual": pct, "Limit": pct}))

    # --- beta table ---------------------------------------------------------
    print("\n📊  What-if Factor Betas\n")
    print(beta_df.to_string(formatters={
        "portfolio_beta":    "{:.2f}".format,     # or "{:.2f}" if you prefer two decimals
        "max_allowed_beta":  "{:.2f}".format,
        "buffer":            "{:.2f}".format,
        "pass":              lambda x: "PASS" if x else "FAIL"
    }))

    return summary, risk_df, beta_df


# In[50]:


# ─── File: portfolio_optimizer.py ──────────────────────────────────────────

import pandas as pd

def _fmt_pct(x: float) -> str:
    return f"{x:.1%}"

# ────────────────────────────────────────────────────────────────────
def compare_risk_tables(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Side-by-side diff for the risk-limit checker."""
    left  = old.rename(columns={"Actual": "Old",  "Pass": "Old Pass"})
    right = new.rename(columns={"Actual": "New",  "Pass": "New Pass"})
    out   = (
        left.merge(right, on=["Metric", "Limit"], how="outer", sort=False)
            .assign(Δ=lambda d: d["New"] - d["Old"])
            .loc[:, ["Metric", "Old", "New", "Δ", "Limit", "Old Pass", "New Pass"]]
    )
    return out

# ────────────────────────────────────────────────────────────────────
def compare_beta_tables(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """
    Diff for the factor-beta checker.
      • Accepts either camel- or snake-case column names.
      • Fills missing Max-Beta / Pass columns with sensible defaults.
      • Index must be Factor for both inputs.
    """
    def _clean(df: pd.DataFrame, tag: str) -> pd.DataFrame:
        colmap = {
            "portfolio_beta": "Beta",
            "max_allowed_beta": "Max Beta",
            "max_beta": "Max Beta",
            "pass": "Pass",
        }
        df = df.rename(columns=colmap)
        if "Max Beta" not in df.columns:
            df["Max Beta"] = 0.0
        if "Pass" not in df.columns:
            df["Pass"] = False
        df = df.rename(columns={"Beta": tag, "Pass": f"{tag} Pass"})
        return df[[tag, "Max Beta", f"{tag} Pass"]]

    left  = _clean(old.copy(), "Old")
    right = _clean(new.copy(), "New")

    merged = left.merge(
        right,
        left_index=True,
        right_index=True,
        how="outer",
        sort=False
    )

    # unify the duplicated Max Beta columns
    merged["Max Beta"] = merged["Max Beta_x"].combine_first(merged["Max Beta_y"])
    merged = merged.drop(columns=["Max Beta_x", "Max Beta_y"])

    out = (
        merged
        .assign(Δ=lambda d: d["New"] - d["Old"])
        .loc[:, ["Old", "New", "Δ", "Max Beta", "Old Pass", "New Pass"]]
    )
    return out


# In[51]:


# ─── portfolio_optimizer.py ───────────────────────

# ---------------------------------------------------------------
#  Risk evaluation portfolio helper
# ---------------------------------------------------------------

import pandas as pd
from typing import Dict, Any, Tuple

def evaluate_weights(
    weights: Dict[str, float],
    risk_cfg: Dict[str, Any],
    start_date: str,
    end_date: str,
    proxies: Dict[str, Dict[str, Any]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs the standard risk + beta limit checks on a given weight dict.
    Returns (df_risk, df_beta) – no printing.
    """
    from portfolio_risk import build_portfolio_view
    from run_portfolio_risk import (
        evaluate_portfolio_risk_limits,
        evaluate_portfolio_beta_limits,
    )
    from risk_helpers import compute_max_betas

    summary = build_portfolio_view(
        weights, start_date, end_date,
        expected_returns=None, stock_factor_proxies=proxies
    )

    df_risk = evaluate_portfolio_risk_limits(
        summary,
        risk_cfg["portfolio_limits"],
        risk_cfg["concentration_limits"],
        risk_cfg["variance_limits"],
    )

    max_betas = compute_max_betas(
        proxies=proxies,
        start_date=start_date,
        end_date=end_date,
        loss_limit_pct=risk_cfg["max_single_factor_loss"],
    )

    df_beta = evaluate_portfolio_beta_limits(
        summary["portfolio_factor_betas"],
        max_betas,
    )
    return df_risk, df_beta


# In[52]:


# ─── File: helpers_input.py ──────────────────────────────────
"""
Helpers for ingesting *what-if* portfolio changes.

parse_delta(...)
    • Accepts a YAML file path (optional) and/or an in-memory shift dict.
    • Returns a tuple: (delta_dict, new_weights_dict_or_None).

Precedence rules
----------------
1. If YAML contains `new_weights:` → treat as full replacement; shift_dict ignored.
2. Else, build a *delta* dict:     YAML `delta:` first, then merge/override
   any overlapping keys from `shift_dict`.
3. YAML missing or empty           → use shift_dict alone.
"""

import yaml
from pathlib import Path
from typing import Dict, Tuple, Optional

def _parse_shift(txt: str) -> float:
    """
    Convert a human-friendly shift string to decimal.

    "+200bp", "-75bps", "1.5%", "-0.01"  →  0.02, -0.0075, 0.015, -0.01
    """
    t = txt.strip().lower().replace(" ", "")
    if t.endswith("%"):
        return float(t[:-1]) / 100
    if t.endswith(("bp", "bps")):
        return float(t.rstrip("ps").rstrip("bp")) / 10_000
    return float(t)                       # already decimal

def parse_delta(
    yaml_path: Optional[str] = None,
    literal_shift: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, float], Optional[Dict[str, float]]]:
    """
    Parse a what-if scenario.

    Parameters
    ----------
    yaml_path : str | None
        Path to a YAML file that may contain `new_weights:` or `delta:`.
    literal_shift : dict | None
        In-memory dict of {ticker: shift_string}.  Overrides YAML deltas.

    Returns
    -------
    (delta_dict, new_weights_dict_or_None)
    """
    delta: Dict[str, float] = {}
    new_w: Optional[Dict[str, float]] = None

    # ── YAML branch (only if file is present) ─────────────────────────
    if yaml_path and Path(yaml_path).is_file():
        cfg = yaml.safe_load(Path(yaml_path).read_text()) or {}
        
        # 1) full-replacement portfolio
        if "new_weights" in cfg:               
            w = {k: float(v) for k, v in cfg["new_weights"].items()}
            s = sum(w.values()) or 1.0
            new_w = {k: v / s for k, v in w.items()}
            return {}, new_w

        # 2) incremental tweaks
        if "delta" in cfg:                     
            delta.update({k: _parse_shift(v) for k, v in cfg["delta"].items()})

    # ── literal shift branch (CLI / notebook) ────────────────────────
    if literal_shift:
        delta.update({k: _parse_shift(v) for k, v in literal_shift.items()})

    # ── sanity check -------------------------------------------------------
    if not delta and new_w is None:
        raise ValueError(
            "No delta or new_weights provided (YAML empty and literal_shift is None)"
        )

    return delta, new_w


# In[53]:


# helpers_display.py  

def _print_single_portfolio(risk_df, beta_df, title: str = "What-if") -> None:
    """
    Pretty-print risk-limit and factor-beta tables for a *single* portfolio
    (new weights or what-if scenario).

    Parameters
    ----------
    risk_df : pd.DataFrame
        Output from `evaluate_portfolio_risk_limits` (or risk_new in run_what_if)
        with columns ["Metric", "Actual", "Limit", "Pass"].
    beta_df : pd.DataFrame
        Output from `evaluate_portfolio_beta_limits` with columns
        ["portfolio_beta", "max_allowed_beta", "pass", "buffer"] and the
        factor name as index.
    title : str, default "What-if"
        Heading prefix used in the console output.

    Notes
    -----
    • Percentages (`Actual`, `Limit`) are rendered with **one-decimal** precision.
    • Betas, max-betas, and buffer columns are rendered to **four** decimals.
    • Pass/fail booleans are mapped to the strings ``PASS`` / ``FAIL``.
    • Prints directly to stdout; returns None.
    """
    pct = lambda x: f"{x:.1%}"                # 1-decimal percentage

    print(f"\n📐  {title} Risk Checks\n")
    print(
        risk_df.to_string(
            index=False,
            formatters={"Actual": pct, "Limit": pct}
        )
    )

    print(f"\n📊  {title} Factor Betas\n")
    print(
        beta_df.to_string(
            formatters={
                "portfolio_beta":   "{:.4f}".format,
                "max_allowed_beta": "{:.4f}".format,
                "buffer":           "{:.4f}".format,
                "pass":             lambda x: "PASS" if x else "FAIL",
            }
        )
    )


# In[54]:


# === What-If Risk Calculations using Shift Inputs ===

shift_dict = {"TW": "+500bp", "PCTY": "-200bp"}


# In[55]:


# === What-If Risk Calculations using Inputs for New Weights with Before / After Risk ===

# ──────────────────────────────────────────────────────────────────────────────
# WHAT-IF DRIVER
#
# Input precedence
# ----------------
# 1. If `what_if_portfolio.yaml` contains a top-level `new_weights:` section
#    → treat as a full-replacement portfolio.
#      • `shift_dict` is ignored in this case.
#
# 2. Otherwise, build an incremental *delta* dict:
#      • YAML `delta:` values are parsed first.
#      • Any overlapping keys in `shift_dict` overwrite the YAML values.
#
# 3. Branch logic
#      • full-replacement  → evaluate_weights(new_weights_yaml)
#      • incremental tweak → run_what_if(base_weights, delta)
#
# 4. After computing the new portfolio’s risk/beta tables once,
#    we also compute the baseline (unchanged) tables once, then
#    show before-vs-after diffs.
#
# Note: No function ever writes back to the YAML file; all merges happen
#       in memory.
# ──────────────────────────────────────────────────────────────────────────────

# 1) Parse input ────────────────────────────────────────────────
delta, new_weights_yaml = parse_delta(
    yaml_path="what_if_portfolio.yaml",      # or None
    literal_shift=shift_dict       # None or {"TW": "+500bp", …}
)

# 2) Build the *new* portfolio once ─────────────────────────────
if new_weights_yaml:                # full-replacement supplied in YAML
    risk_new, beta_new = evaluate_weights(
        new_weights_yaml, risk_config,
        config["start_date"], config["end_date"],
        stock_factor_proxies
    )
    _print_single_portfolio(risk_new, beta_new, title="New Portfolio What-if")   
    
else:                              # incremental tweak path (delta shift)
    summary_new, risk_new, beta_new = run_what_if(
        base_weights=weights,
        delta=delta,
        risk_cfg=risk_config,
        start_date=config["start_date"],
        end_date=config["end_date"],
        factor_proxies=stock_factor_proxies
    )

# 3) Baseline portfolio (unchanged) ────────────────────────────
risk_base, beta_base = evaluate_weights(
    weights, risk_config,
    config["start_date"], config["end_date"],
    stock_factor_proxies
)

# 4) Diff & pretty-print ───────────────────────────────────────
cmp_risk = (compare_risk_tables(risk_base, risk_new)
              .set_index("Metric")
              .loc[risk_base["Metric"].tolist()]        # keep original row order
              .reset_index()
           )

cmp_beta = (compare_beta_tables(beta_base, beta_new)
                .reindex(beta_base.index)  
           )
            
print("\n📐  Risk Limits — Before vs After\n")
print(cmp_risk.to_string(index=False,
                         formatters={"Old": _fmt_pct, "New": _fmt_pct,
                                     "Δ": _fmt_pct, "Limit": _fmt_pct}))

print("\n📊  Factor Betas — Before vs After\n")
print(
    cmp_beta.to_string(
        formatters={
            "Old":       "{:.2f}".format,   # two-decimals, e.g. 0.5830
            "New":       "{:.2f}".format,
            "Δ":         "{:.2f}".format,
            "Max Beta":  "{:.2f}".format,   # keep two-decimals on Max Beta if you like
            "Old Pass":  lambda x: "PASS" if x else "FAIL",
            "New Pass":  lambda x: "PASS" if x else "FAIL",
        }
    )
)


# In[56]:


# ─── File: portfolio_optimizer.py ──────────────────────────────────

# ---------------------------------------------------------------
#  Minimum variance portfolio helper
# ---------------------------------------------------------------

from typing import Dict, Any
import pandas as pd

def run_min_var_optimiser(
    weights: Dict[str, float],
    risk_cfg: Dict[str, Any],
    start_date: str,
    end_date:   str,
    proxies: Dict[str, Dict[str, Any]],
    echo: bool = True,
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame]:
    """
    Minimum-variance portfolio under firm-wide limits
    ------------------------------------------------

    Objective
    ---------
    **min wᵀ Σ w**  
    Σ = monthly covariance estimated over *start_date*→*end_date*.

    Constraints
    -----------
    1. ∑ wᵢ = 1  (fully invested)  
    2. wᵢ ≥ 0  (long-only; see lower-level solver for shorts)  
    3. |wᵢ| ≤ single-name cap from *risk_cfg*  
    4. √12 · √(wᵀ Σ w) ≤ σ_cap  
    5. |β_port,f| ≤ dynamic β_max,f (via `compute_max_betas`)

    Convex QP solved with CVXPY + ECOS.  
    Returns only the optimised weights; use `evaluate_weights(...)`
    if you need PASS/FAIL tables.

    Parameters
    ----------
    weights : {ticker: weight} (sums ≈ 1)  
    risk_cfg : parsed *risk_limits.yaml*  
    start_date, end_date : YYYY-MM-DD window for Σ & betas  
    proxies : `stock_factor_proxies` from portfolio YAML  
    echo : print weights ≥ 0.01 % when True

    Returns
    -------
    Dict[str, float] – optimised weights (summing to 1)
    """
    
    # 1. ---------- solve ----------------------------------------------------
    new_w = solve_min_variance_with_risk_limits(
        weights,
        risk_cfg,
        start_date,
        end_date,
        proxies,
    )

    # 2. ---------- optional console output ---------------------------------
    if echo:
        # 3a. pretty-print weights ≥ 0.01 %
        print("\n🎯  Target minimum-variance weights:\n")
        (pd.Series(new_w, name="Weight")
           .loc[lambda s: s.abs() > 0.0001]
           .sort_values(ascending=False)
           .apply(lambda x: f"{x:.2%}")
           .pipe(lambda s: print(s.to_string()))
        )

    return new_w


# In[35]:


# === Optimizer: Lowest Variance Weights within Risk Limits ===

# 1. get the weights
w_min = run_min_var_optimiser(
    weights,
    risk_config,
    config["start_date"], config["end_date"],
    stock_factor_proxies,
    echo=True          # or False if you don’t want the quick print-out
)

# 2. when you actually need the risk tables ↓
risk_tbl, beta_tbl = evaluate_weights(
    w_min, risk_config,
    config["start_date"], config["end_date"],
    stock_factor_proxies
)

# 3a. risk table
print("\n📐  Optimised Portfolio – Risk Checks\n")
pct = lambda x: f"{x:.2%}"
print(risk_tbl.to_string(index=False,
                        formatters={"Actual": pct, "Limit": pct}))

# 3b. beta table
print("\n📊  Optimised Portfolio – Factor Betas\n")
print(beta_tbl.to_string(formatters={
    "Beta":      "{:.2f}".format,
    "Max Beta":  "{:.2f}".format,
    "Buffer":    "{:.2f}".format,
    "Pass":      lambda x: "PASS" if x else "FAIL",
}))
# …then display / log those tables however you like


# In[36]:


# File: portfolio_optimizer.py

# ---------------------------------------------------------------------
# Max-return subject to risk limits
# ---------------------------------------------------------------------

import cvxpy as cp
import numpy as np
import pandas as pd
from typing import Dict, Union, List

def solve_max_return_with_risk_limits(
    init_weights: Dict[str, float],
    risk_cfg: Dict[str, Dict[str, float]],
    start_date: str,
    end_date: str,
    stock_factor_proxies: Dict[str, Dict[str, Union[str, List[str]]]],
    expected_returns: Dict[str, float],
    allow_short: bool = False
) -> Dict[str, float]:
    """
    Max-return portfolio subject to firm-wide limits.

    Objective
    ---------
    Maximise Σ w_i·μ_i   (μ = expected annual return in decimals)
    subject to ALL risk limits defined in `risk_cfg`.

    Risk Constraints
    -----------
    1. Σ wᵢ = 1                               (fully invested)  
    2. wᵢ ≥ 0  if ``allow_short`` is False    (long-only)  
    3. wᵢ ≤ single-name cap                   (concentration limit)  
    4. √(12 · wᵀΣw) ≤ σ_cap                   (annual vol cap)  
       – Σ is the monthly covariance from *start_date*–*end_date*.

    This is a convex quadratic program solved with CVXPY + ECOS.

    Parameters
    ----------
    init_weights          : current {ticker: weight}.
    risk_cfg              : parsed risk_limits.yaml (needs the three sub-dicts).
    start_date / end_date : window used to build covariance & factor tables.
    stock_factor_proxies  : same structure used elsewhere in your toolkit.
    expected_returns      : {ticker: annual exp return}.  **Required**.
    allow_short           : set True if negatives are allowed.

    Returns
    -------
    dict {ticker: new_weight}.  Sums to 1 by construction.
    """
    # ---- 1. Build Σ (monthly) with your existing helper ----------------
    from portfolio_risk import build_portfolio_view   # re-use, keeps code DRY

    tickers = list(init_weights.keys())
    view = build_portfolio_view(
        init_weights, start_date, end_date,
        expected_returns=None,           # we only need cov matrix
        stock_factor_proxies=stock_factor_proxies
    )
    Σ_m = view["covariance_matrix"].loc[tickers, tickers].values
    μ    = np.array([expected_returns.get(t, 0.0) for t in tickers])

    if np.allclose(μ, 0):
        raise ValueError("expected_returns is empty or zeros – nothing to maximise")

    # ---- 2. CVXPY variables & objective --------------------------------
    w = cp.Variable(len(tickers))
    objective = cp.Maximize(μ @ w)     # linear objective

    # ---- 3. Constraints -------------------------------------------------
    cons = []

    # fully invested
    cons += [cp.sum(w) == 1]

    # long-only?
    if not allow_short:
        cons += [w >= 0]

    # single-name cap
    w_cap = risk_cfg["concentration_limits"]["max_single_stock_weight"]
    cons += [w <= w_cap]

    # volatility cap  (monthly Σ → annual σ = √12·sqrt(wᵀΣw))
    σ_cap = risk_cfg["portfolio_limits"]["max_volatility"]
    cons += [cp.sqrt(cp.quad_form(w, Σ_m)) * np.sqrt(12) <= σ_cap]

    # You can add more factor / industry caps here if you want them
    # (reuse values already in risk_cfg).

    # ---- 4. Solve -------------------------------------------------------
    prob = cp.Problem(objective, cons)
    prob.solve(solver=cp.ECOS, qcp=True, verbose=False)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise ValueError(f"Solver returned status {prob.status}")

    return {t: float(v) for t, v in zip(tickers, w.value)}


# In[37]:


# === Optimizer: Highest Return Weights within Risk Limits ===

# 1. run optimiser → weights only
w_opt = solve_max_return_with_risk_limits(
    weights,                    # current allocation
    risk_config,                # parsed risk_limits.yaml
    config["start_date"],
    config["end_date"],
    stock_factor_proxies,
    expected_returns=config["expected_returns"]  # MUST be present
)

print("\n🎯  Target max-return, risk-constrained weights:\n")
for k,v in sorted(w_opt.items(), key=lambda kv: -kv[1]):
    if abs(v) > 1e-4:
        print(f"{k:<8} : {v:.2%}")

# 2. whenever you want: evaluate & pretty-print
risk_tbl, beta_tbl = evaluate_weights(
    w_opt, risk_config,
    config["start_date"], config["end_date"],
    stock_factor_proxies
)

print("\n📐  Max-return Portfolio – Risk Checks\n")
pct = lambda x: f"{x:.2%}"
print(risk_tbl.to_string(index=False, formatters={"Actual": pct, "Limit": pct}))

print("\n📊  Max-return Portfolio – Factor Betas\n")
print(beta_tbl.to_string(formatters={
    "Beta": "{:.2f}".format, "Max Beta": "{:.2f}".format,
    "Buffer": "{:.2f}".format, "Pass": lambda x: "PASS" if x else "FAIL"
}))

