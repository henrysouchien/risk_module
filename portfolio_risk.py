#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# File: portfolio_risk.py

import pandas as pd
import numpy as np
from typing import Dict

from data_loader import fetch_monthly_close
from factor_utils import (
    calc_monthly_returns,
    fetch_excess_return,
    fetch_peer_median_monthly_returns,
    compute_stock_factor_betas,
)

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


# In[ ]:


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


# In[ ]:


# File: portfolio_risk.py

import pandas as pd
import numpy as np
from typing import Dict

def compute_euler_variance_percent(
    *,                       # force keyword args for clarity
    weights: Dict[str, float],
    cov_matrix: pd.DataFrame,
) -> pd.Series:
    """
    Euler (marginal) variance decomposition.

    Returns each asset’s share of **total portfolio variance** as a %
    (values sum exactly to 1.0).

    Parameters
    ----------
    weights     : {ticker: weight}
    cov_matrix  : Σ, index/cols = same tickers
    """
    w = pd.Series(weights, dtype=float).loc[cov_matrix.index]
    # marginal contributions Σ·w
    sigma_w = cov_matrix.values @ w.values
    # component (Euler) contributions w_i · (Σ·w)_i
    contrib = pd.Series(w.values * sigma_w, index=cov_matrix.index)
    return contrib / contrib.sum()          # normalise to 1.0


# In[1]:


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
    - Computes Euler (marginal) variance contributions for every stock.
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
            - 'euler_variance_pct': per-stock share of total variance (Series, sums to 1.0)
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

    # ─── 2a. Compute Factor Volatility & Weighted Variance ───────────────────────
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
            
        # ---------- after loop: clean tables & build w²·β²·σ² -------------
        
        # df_factor_vols  : σ-table (annual factor vols by stock)
        df_factor_vols = (
            df_factor_vols
                .apply(pd.to_numeric, errors="coerce")  # force numeric, NaNs where bad
                .astype("float64", copy=False)          # ensure float dtype, no copy if already
                .fillna(0.0)                           # now safe – no warning
        )
        
        # betas_filled β-table with NaNs → 0.0
        betas_filled = (
            df_stock_betas
                .apply(pd.to_numeric, errors="coerce")
                .astype("float64", copy=False)
                .fillna(0.0)
        )

        # ----- weighted factor variance  w_i² β_i,f² σ_i,f² -----------------------
        weighted_factor_var = betas_filled.pow(2) * df_factor_vols.pow(2)
        weighted_factor_var = weighted_factor_var.mul(w2, axis=0)

    # ─── 2b. Euler variance attribution  -------------------------------
    cov_annual = cov_mat * 12                       # annualise Σ (12× monthly)
    
    euler_var_pct = compute_euler_variance_percent(
        weights       = weights,
        cov_matrix    = cov_annual,                 # use annual Σ
    )

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
        "euler_variance_pct":  euler_var_pct,
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


# In[ ]:




