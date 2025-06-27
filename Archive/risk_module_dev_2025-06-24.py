#!/usr/bin/env python
# coding: utf-8

# In[25]:


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


# In[26]:


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


# In[47]:


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


# In[48]:


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


# In[29]:


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


# In[46]:


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


# In[31]:


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


# In[32]:


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


# In[33]:


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


# In[34]:


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
    Builds a full risk profile of a portfolio using historical return data,
    factor regressions, and variance decomposition.

    Performs:
    - Return aggregation and portfolio-level volatility
    - Covariance, correlation, and risk contribution breakdown
    - Single-factor regressions for each stock to compute:
        • Betas (market, momentum, value, industry, subindustry)
        • Idiosyncratic volatility and variance
    - Factor volatilities and weighted factor variance: w² · β² · σ²
    - Decomposition of portfolio variance into:
        • Idiosyncratic component
        • Factor-driven component
        • Per-factor attribution
    - Aggregation of industry-level variance contributions

    Args:
        weights (Dict[str, float]):
            Portfolio weights by ticker (can be raw and leveraged).
        start_date (str):
            Start of historical analysis window (YYYY-MM-DD).
        end_date (str):
            End of analysis window (YYYY-MM-DD).
        expected_returns (Optional[Dict[str, float]]):
            Optional expected return input to display allocation gaps.
        stock_factor_proxies (Optional[Dict]):
            Per-stock mapping to proxies used for regression:
                - "market": ETF ticker (e.g., SPY)
                - "momentum": ETF ticker (e.g., MTUM)
                - "value": ETF ticker (e.g., IWD)
                - "industry": ETF ticker (e.g., SOXX)
                - "subindustry": list of tickers (e.g., ["PAYC", "CDAY"])

    Returns:
        Dict[str, Any]: Full portfolio diagnostics including:
            - 'allocations': Target vs. actual vs. expected return weights
            - 'portfolio_returns': Monthly return series
            - 'covariance_matrix': Asset covariance
            - 'correlation_matrix': Asset correlations
            - 'volatility_monthly' and 'volatility_annual'
            - 'risk_contributions': Contribution to total vol by asset
            - 'herfindahl': Concentration index
            - 'df_stock_betas': Betas per asset per factor
            - 'portfolio_factor_betas': Weighted portfolio betas
            - 'factor_vols': Annual σ per factor per stock
            - 'weighted_factor_var': w² · β² · σ² table
            - 'asset_vol_summary': Total and idiosyncratic σ and var
            - 'variance_decomposition': Total, idio, factor, per-factor breakdown
            - 'industry_variance': Per-industry absolute and % attribution
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


    # ─── 3. Aggregate Industry-Level Variance ───────────────────────────────────
    industry_var_dict = {}
    
    # Step: reverse-map which stock maps to which industry ETF
    for tkr, proxies in stock_factor_proxies.items():
        ind = proxies.get("industry")
        if ind:
            v = weighted_factor_var.loc[tkr, "industry"] if "industry" in weighted_factor_var.columns else 0.0
            industry_var_dict[ind] = industry_var_dict.get(ind, 0.0) + v
    

    # ─── 4. Final Portfolio Stats (Volatility, Idio, Betas) ─────────────────────
    w_series                = pd.Series(weights)
    portfolio_factor_betas  = df_stock_betas.mul(w_series, axis=0).sum(skipna=True)

    # 3) per-asset annualised stats ----------------------------------------
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
        "percent_of_portfolio": industry_pct_dict
    }
    }


# In[35]:


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


# In[36]:


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


# In[44]:


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


# In[45]:


# Load portfolio and display configs

config = load_and_display_portfolio_config("portfolio.yaml")
start_date = config["start_date"]
end_date = config["end_date"]
expected_returns = config["expected_returns"]
stock_factor_proxies = config["stock_factor_proxies"]
portfolio_input = config["portfolio_input"]
weights = standardize_portfolio_input(portfolio_input, latest_price)["weights"]


# In[42]:


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


# In[43]:


# 2) Call the high-level summary builder
summary = build_portfolio_view(
    weights,
    start_date,
    end_date,
    expected_returns=expected_returns,
    stock_factor_proxies=stock_factor_proxies
)
display_portfolio_summary(summary)


# In[ ]:




