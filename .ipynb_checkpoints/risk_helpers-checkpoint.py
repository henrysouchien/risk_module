#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


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


# In[ ]:


from typing import Dict, Tuple

def infer_max_betas_from_losses(
    worst_factor_losses: Dict[str, Tuple[str, float]],
    loss_limit_pct: float
) -> Dict[str, float]:
    """
    Infers max allowable beta per factor to stay within portfolio loss limit.

    Args:
        worst_factor_losses (Dict[str, Tuple[str, float]]): 
            {factor_type: (proxy, worst_loss)}, e.g. {'market': ('SPY', -0.13)}
        loss_limit_pct (float): Target portfolio loss limit in decimal (e.g., 0.10 for -10%)

    Returns:
        Dict[str, float]: {factor_type: max_beta}
    """
    max_betas = {}

    for factor, (_, worst_return) in worst_factor_losses.items():
        if worst_return == 0:
            continue
        max_beta = loss_limit_pct / abs(worst_return)
        max_betas[factor] = round(max_beta, 3)

    return max_betas

