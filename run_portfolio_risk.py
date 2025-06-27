#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import yaml
from pprint import pprint
from typing import Optional
import statsmodels.api as sm

from typing import Dict, Callable, Union, Optional, Any, List
from data_loader import fetch_monthly_close
from factor_utils import (
    calc_monthly_returns,
    fetch_excess_return,
    fetch_peer_median_monthly_returns,
    compute_volatility,
    compute_regression_metrics,
    compute_factor_metrics,
    compute_stock_factor_betas,
    calc_factor_vols,
    calc_weighted_factor_variance
)
from portfolio_risk import (
    compute_portfolio_returns,
    compute_covariance_matrix,
    compute_correlation_matrix,
    compute_portfolio_volatility,
    compute_risk_contributions,
    compute_herfindahl,
    compute_portfolio_variance_breakdown,
    get_returns_dataframe,
    compute_target_allocations,
    normalize_weights,
    build_portfolio_view  # if reused, or remove if it's redefined in this file
)


# In[ ]:


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


# In[ ]:


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


# In[ ]:


# ── run_portfolio_risk.py ────────────────────────────────────────────
import yaml
from pprint import pprint
from typing import Dict, Callable, Optional, Any

# --------------------------------------------------------------------
# 1) Pure-data loader  → returns a dict you can reuse programmatically
# --------------------------------------------------------------------
def load_portfolio_config(
    filepath: str = "portfolio.yaml",
    price_fetcher: Callable[[str], float] | None = None,
) -> Dict[str, Any]:
    """
    Load the YAML and return a dict with parsed + normalised fields.
    No printing, no side effects.
    """
    from run_portfolio_risk import standardize_portfolio_input, latest_price  # local imports

    price_fetcher = price_fetcher or latest_price

    with open(filepath, "r") as f:
        cfg_raw = yaml.safe_load(f)

    # • Keep the original keys for downstream code
    cfg: Dict[str, Any] = dict(cfg_raw)          # shallow copy
    parsed = standardize_portfolio_input(cfg["portfolio_input"], price_fetcher)

    cfg.update(
        weights           = parsed["weights"],
        dollar_exposure   = parsed["dollar_exposure"],
        total_value       = parsed["total_value"],
        net_exposure      = parsed["net_exposure"],
        gross_exposure    = parsed["gross_exposure"],
        leverage          = parsed["leverage"],
    )
    return cfg


# --------------------------------------------------------------------
# 2) Pretty-printer  → consumes the dict returned by loader
# --------------------------------------------------------------------
def display_portfolio_config(cfg: Dict[str, Any]) -> None:
    """
    Nicely print the fields produced by load_portfolio_config().
    """
    print("=== Normalized Weights ===")
    print(cfg["weights"])

    print("\n=== Dollar Exposure ===")
    print(cfg["dollar_exposure"])

    print("\n=== Total Portfolio Value ===")
    print(cfg["total_value"])

    print("\n=== Net Exposure (sum of weights) ===")
    print(cfg["net_exposure"])

    print("\n=== Gross Exposure (sum of abs(weights)) ===")
    print(cfg["gross_exposure"])

    print("\n=== Leverage (gross / net) ===")
    print(cfg["leverage"])

    print("\n=== Expected Returns ===")
    pprint(cfg["expected_returns"])

    print("\n=== Stock Factor Proxies ===")
    for ticker, proxies in cfg["stock_factor_proxies"].items():
        print(f"\n→ {ticker}")
        pprint(proxies)


# --------------------------------------------------------------------
# 3) Convenience shim for legacy calls  (optional but zero-cost)
# --------------------------------------------------------------------
def load_and_display_portfolio_config(
    filepath: str = "portfolio.yaml",
    price_fetcher: Callable[[str], float] | None = None,
) -> Dict[str, Any]:
    """
    Drop-in replacement for the old monolithic helper.
    Returns the same dict loader now provides.
    """
    cfg = load_portfolio_config(filepath, price_fetcher)
    display_portfolio_config(cfg)
    return cfg


# In[ ]:


# File: run_portfolio_risk.py

def display_portfolio_summary(summary: dict):
    print("\n=== Target Allocations ===")
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


# In[ ]:


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


# In[ ]:


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

