#!/usr/bin/env python
# coding: utf-8

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


# File: run_portfolio_risk.py

import yaml
from pprint import pprint

# ─── Load Portfolio Config ─────────────────────────────────────────────
with open("portfolio.yaml", "r") as f:
    config = yaml.safe_load(f)

# Extract inputs
start_date            = config["start_date"]
end_date              = config["end_date"]
portfolio_input       = config["portfolio_input"]
expected_returns      = config["expected_returns"]
stock_factor_proxies  = config["stock_factor_proxies"]

# ─── Standardize Portfolio Weights ─────────────────────────────────────
parsed  = standardize_portfolio_input(portfolio_input, latest_price)
weights = parsed["weights"]

# ─── Display Inputs and Exposures ──────────────────────────────────────
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

# ─── Display Expectations and Factor Proxies ───────────────────────────
print("\n=== Expected Returns ===")
pprint(expected_returns)

print("\n=== Stock Factor Proxies ===")
for ticker, proxies in stock_factor_proxies.items():
    print(f"\n→ {ticker}")
    pprint(proxies)


# In[ ]:


# File: run_portfolio_risk.py

# ─── Run Portfolio View ─────────────────────────────────────────────────

import pandas as pd

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
print(summary["weighted_factor_var"].round(6), "\n")

print("=== Portfolio Variance Decomposition ===")
var_dec = summary["variance_decomposition"]

print(f"Portfolio Variance:          {var_dec['portfolio_variance']:.4f}")
print(f"Idiosyncratic Variance:      {var_dec['idiosyncratic_variance']:.4f}  ({var_dec['idiosyncratic_pct']:.0%})")
print(f"Factor Variance:             {var_dec['factor_variance']:.4f}  ({var_dec['factor_pct']:.0%})\n")

print("=== Factor Variance (absolute) ===")
for k, v in var_dec["factor_breakdown_var"].items():
    print(f"{k.title():<10} : {v:.5f}")

# Optional: exclude 'industry' and 'subindustry' from factor breakdown
filtered = {
    k: v for k, v in var_dec["factor_breakdown_pct"].items()
    if k not in ("industry", "subindustry")
    }

#print("\n=== Factor Variance (% of Portfolio) ===")
#for k, v in var_dec["factor_breakdown_pct"].items():
    #print(f"{k.title():<10} : {v:.0%}")

print("\n=== Factor Variance (% of Portfolio, excluding industry) ===")
for k, v in filtered.items():
    print(f"{k.title():<10} : {v:.0%}")

print("\n=== Industry Variance (absolute) ===")
for k, v in summary["industry_variance"]["absolute"].items():
    print(f"{k:<10} : {v:.6f}")

print("\n=== Industry Variance (% of Portfolio) ===")
for k, v in summary["industry_variance"]["percent_of_portfolio"].items():
    print(f"{k:<10} : {v:.1%}")

