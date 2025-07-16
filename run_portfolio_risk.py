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

# Auto-detect cash positions from database (with YAML fallback)
def get_cash_positions():
    # LOGGING: Add cash position detection logging with data source and timing
    # LOGGING: Add resource usage monitoring for cache initialization here
    # LOGGING: Add critical alert for database connection failures here
    try:
        # Try database first
        from inputs.database_client import DatabaseClient
        db_client = DatabaseClient()
        cash_map = db_client.get_cash_mappings()
        return set(cash_map.get("proxy_by_currency", {}).values())
    except Exception as e:
        # Fallback to YAML
        print(f"⚠️ Database unavailable ({e}), using cash_map.yaml fallback")
        try:
            with open("cash_map.yaml", "r") as f:
                cash_map = yaml.safe_load(f)
                return set(cash_map.get("proxy_by_currency", {}).values())
        except FileNotFoundError:
            # Fallback to common cash proxies
            print("⚠️ cash_map.yaml not found, using default cash proxies")
            return {"SGOV", "ESTR", "IB01", "CASH", "USD"}

cash_positions = get_cash_positions()

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
    # LOGGING: Add portfolio standardization start logging with input size and format
    # LOGGING: Add workflow state logging for portfolio standardization workflow here
    # LOGGING: Add resource usage monitoring for portfolio processing here

    # LOGGING: Add portfolio processing logging with ticker count and validation
    # LOGGING: Add workflow state logging for portfolio processing completion here
    dollar_exposure = {}

    # LOGGING: Add portfolio processing logging with ticker count and validation
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

        # Calculate exposure excluding only POSITIVE cash positions
        # Negative cash positions (margin debt) should be included
        risky_weights = {
            t: w for t, w in weights.items() 
            if t not in cash_positions or w < 0  # Include negative cash positions (margin debt)
        }
        net_exposure = sum(risky_weights.values())
        gross_exposure = sum(abs(w) for w in risky_weights.values())

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

    # Calculate exposure excluding only POSITIVE cash positions
    # Negative cash positions (margin debt) should be included
    risky_weights = {
        t: w for t, w in weights.items() 
        if t not in cash_positions or w < 0  # Include negative cash positions (margin debt)
    }
    net_exposure = sum(risky_weights.values())
    gross_exposure = sum(abs(w) for w in risky_weights.values())

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
    from settings import PORTFOLIO_DEFAULTS
    
    weights = cfg["weights"]
    
    # Simple check based on the setting
    normalize_setting = PORTFOLIO_DEFAULTS.get("normalize_weights", True)
    title = "=== PORTFOLIO ALLOCATIONS BEING ANALYZED (Normalized Weights) ===" if normalize_setting else "=== PORTFOLIO ALLOCATIONS BEING ANALYZED (Raw Weights) ==="
    
    print(title)
    
    # Sort by weight (descending) for better readability
    sorted_weights = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)
    
    total_weight = 0
    for ticker, weight in sorted_weights:
        print(f"{ticker:<8} {weight:>7.2%}")
        total_weight += weight
    
    print("─" * 18)
    print(f"{'Total':<8} {total_weight:>7.2%}")
    print()

    print("=== DOLLAR EXPOSURE BY POSITION ===")
    dollar_exp = cfg["dollar_exposure"]
    # Sort by absolute dollar exposure (descending) for better readability
    sorted_dollar = sorted(dollar_exp.items(), key=lambda x: abs(x[1]), reverse=True)
    
    total_dollar = 0
    for ticker, amount in sorted_dollar:
        # Format with appropriate precision based on amount size
        if abs(amount) >= 1000:
            print(f"{ticker:<8} ${amount:>12,.0f}")
        else:
            print(f"{ticker:<8} ${amount:>12,.2f}")
        total_dollar += amount
    
    print("─" * 22)
    if abs(total_dollar) >= 1000:
        print(f"{'Total':<8} ${total_dollar:>12,.0f}")
    else:
        print(f"{'Total':<8} ${total_dollar:>12,.2f}")
    print()

    print("=== TOTAL PORTFOLIO VALUE ===")
    total_val = cfg["total_value"]
    if abs(total_val) >= 1000:
        print(f"${total_val:,.0f}")
    else:
        print(f"${total_val:,.2f}")
    
    print("\n=== NET EXPOSURE (sum of weights) ===")
    print(f"{cfg['net_exposure']:.2f}")
    
    print("\n=== GROSS EXPOSURE (sum of abs(weights)) ===")
    print(f"{cfg['gross_exposure']:.2f}")
    
    print("\n=== LEVERAGE (gross / net) ===")
    print(f"{cfg['leverage']:.2f}x")

    print("\n=== EXPECTED RETURNS ===")
    expected_returns = cfg["expected_returns"]
    # Sort by expected return (descending) for better readability
    sorted_returns = sorted(expected_returns.items(), key=lambda x: x[1], reverse=True)
    
    for ticker, return_val in sorted_returns:
        print(f"{ticker:<8} {return_val:>7.2%}")

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

    print("\n=== Top Stock Variance (Euler %) ===")
    euler = summary["euler_variance_pct"]
    top   = dict(sorted(euler.items(), key=lambda kv: -kv[1])[:10])  # top-10
    for tkr, pct in top.items():
        print(f"{tkr:<10} : {pct:6.1%}")
        
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
    # LOGGING: Add risk limits evaluation start logging with limit types and values
    results = []

    # LOGGING: Add volatility check logging with actual vs limit values
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


# In[ ]:


# File: run_portfolio_risk.py

def display_portfolio_performance_metrics(performance_metrics: Dict[str, Any]) -> None:
    """
    Display portfolio performance metrics in a formatted, professional layout.
    
    Parameters
    ----------
    performance_metrics : Dict[str, Any]
        Output from calculate_portfolio_performance_metrics()
    """
    
    # Check for errors
    if "error" in performance_metrics:
        print(f"❌ Error: {performance_metrics['error']}")
        return
    
    # Extract data
    period = performance_metrics["analysis_period"]
    returns = performance_metrics["returns"]
    risk = performance_metrics["risk_metrics"]
    ratios = performance_metrics["risk_adjusted_returns"]
    benchmark = performance_metrics["benchmark_analysis"]
    comparison = performance_metrics["benchmark_comparison"]
    monthly = performance_metrics["monthly_stats"]
    
    # Header
    print("\n" + "="*60)
    print("📊 PORTFOLIO PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Analysis period
    print(f"📅 Analysis Period: {period['start_date']} to {period['end_date']}")
    print(f"📊 Total Months: {period['total_months']} ({period['years']} years)")
    
    # Core performance metrics
    print(f"\n📈 RETURN METRICS")
    print("─" * 40)
    print(f"📊 Total Return:        {returns['total_return']:>8.2f}%")
    print(f"📈 Annualized Return:   {returns['annualized_return']:>8.2f}%")
    print(f"🟢 Best Month:          {returns['best_month']:>8.2f}%")
    print(f"🔴 Worst Month:         {returns['worst_month']:>8.2f}%")
    print(f"🎯 Win Rate:            {returns['win_rate']:>8.1f}%")
    
    # Risk metrics
    print(f"\n⚡ RISK METRICS")
    print("─" * 40)
    print(f"📉 Volatility:          {risk['volatility']:>8.2f}%")
    print(f"📉 Max Drawdown:        {risk['maximum_drawdown']:>8.2f}%")
    print(f"📊 Downside Deviation:  {risk['downside_deviation']:>8.2f}%")
    print(f"📈 Tracking Error:      {risk['tracking_error']:>8.2f}%")
    
    # Risk-adjusted returns
    print(f"\n🎯 RISK-ADJUSTED RETURNS")
    print("─" * 40)
    print(f"⚡ Sharpe Ratio:        {ratios['sharpe_ratio']:>8.3f}")
    print(f"📊 Sortino Ratio:       {ratios['sortino_ratio']:>8.3f}")
    print(f"📈 Information Ratio:   {ratios['information_ratio']:>8.3f}")
    print(f"📉 Calmar Ratio:        {ratios['calmar_ratio']:>8.3f}")
    
    # Benchmark analysis
    print(f"\n🔍 BENCHMARK ANALYSIS ({benchmark['benchmark_ticker']})")
    print("─" * 40)
    print(f"🎯 Alpha (Annual):      {benchmark['alpha_annual']:>8.2f}%")
    print(f"📊 Beta:                {benchmark['beta']:>8.3f}")
    print(f"📈 R-Squared:           {benchmark['r_squared']:>8.3f}")
    print(f"📊 Excess Return:       {benchmark['excess_return']:>8.2f}%")
    
    # Portfolio vs Benchmark comparison
    print(f"\n📊 PORTFOLIO vs {benchmark['benchmark_ticker']} COMPARISON")
    print("─" * 40)
    print(f"{'Metric':<20} {'Portfolio':<12} {'Benchmark':<12}")
    print("─" * 40)
    print(f"{'Return':<20} {comparison['portfolio_return']:>8.2f}%    {comparison['benchmark_return']:>8.2f}%")
    print(f"{'Volatility':<20} {comparison['portfolio_volatility']:>8.2f}%    {comparison['benchmark_volatility']:>8.2f}%")
    print(f"{'Sharpe Ratio':<20} {comparison['portfolio_sharpe']:>8.3f}     {comparison['benchmark_sharpe']:>8.3f}")
    
    # Monthly statistics
    print(f"\n📅 MONTHLY STATISTICS")
    print("─" * 40)
    print(f"📊 Avg Monthly Return:  {monthly['average_monthly_return']:>8.2f}%")
    print(f"🟢 Average Win:         {monthly['average_win']:>8.2f}%")
    print(f"🔴 Average Loss:        {monthly['average_loss']:>8.2f}%")
    print(f"⚖️  Win/Loss Ratio:     {monthly['win_loss_ratio']:>8.2f}")
    print(f"📊 Positive Months:     {returns['positive_months']:>8.0f}")
    print(f"📉 Negative Months:     {returns['negative_months']:>8.0f}")
    
    # Risk-free rate
    print(f"\n🏦 RISK-FREE RATE")
    print("─" * 40)
    print(f"📊 3-Month Treasury:    {performance_metrics['risk_free_rate']:>8.2f}%")
    
    # Performance summary
    print(f"\n✅ PERFORMANCE SUMMARY")
    print("─" * 40)
    
    # Determine performance quality
    annual_return = returns['annualized_return']
    sharpe = ratios['sharpe_ratio']
    max_dd = abs(risk['maximum_drawdown'])
    
    if annual_return > 15 and sharpe > 1.0 and max_dd < 20:
        quality = "🟢 EXCELLENT"
        summary = "Strong returns with good risk management"
    elif annual_return > 10 and sharpe > 0.8 and max_dd < 30:
        quality = "🟡 GOOD"
        summary = "Solid performance with acceptable risk"
    elif annual_return > 5 and sharpe > 0.5 and max_dd < 40:
        quality = "🟠 FAIR"
        summary = "Moderate performance with some risk concerns"
    else:
        quality = "🔴 POOR"
        summary = "Underperforming with significant risk issues"
    
    print(f"{quality}: {summary}")
    
    # Key insights
    print(f"\n💡 KEY INSIGHTS")
    print("─" * 40)
    
    # Alpha analysis
    if benchmark['alpha_annual'] > 3:
        print(f"   • Strong alpha generation (+{benchmark['alpha_annual']:.1f}% vs {benchmark['benchmark_ticker']})")
    elif benchmark['alpha_annual'] > 0:
        print(f"   • Modest alpha generation (+{benchmark['alpha_annual']:.1f}% vs {benchmark['benchmark_ticker']})")
    else:
        print(f"   • Underperforming benchmark ({benchmark['alpha_annual']:+.1f}% vs {benchmark['benchmark_ticker']})")
    
    # Beta analysis
    if benchmark['beta'] > 1.2:
        print(f"   • High market sensitivity (β = {benchmark['beta']:.2f})")
    elif benchmark['beta'] > 0.8:
        print(f"   • Moderate market sensitivity (β = {benchmark['beta']:.2f})")
    else:
        print(f"   • Low market sensitivity (β = {benchmark['beta']:.2f})")
    
    # Sharpe analysis
    if ratios['sharpe_ratio'] > 1.0:
        print(f"   • Good risk-adjusted returns (Sharpe = {ratios['sharpe_ratio']:.2f})")
    elif ratios['sharpe_ratio'] > 0.5:
        print(f"   • Acceptable risk-adjusted returns (Sharpe = {ratios['sharpe_ratio']:.2f})")
    else:
        print(f"   • Poor risk-adjusted returns (Sharpe = {ratios['sharpe_ratio']:.2f})")
    
    # Win rate analysis
    if returns['win_rate'] > 60:
        print(f"   • High consistency ({returns['win_rate']:.0f}% win rate)")
    elif returns['win_rate'] > 50:
        print(f"   • Moderate consistency ({returns['win_rate']:.0f}% win rate)")
    else:
        print(f"   • Low consistency ({returns['win_rate']:.0f}% win rate)")
    
    print("="*60)


