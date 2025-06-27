#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# File: run_risk.py

import argparse
from risk_summary import get_detailed_stock_factor_profile
from run_portfolio_risk import latest_price, standardize_portfolio_input
from portfolio_risk import build_portfolio_view
import yaml
from pprint import pprint

def run_portfolio(filepath: str):
    with open(filepath, "r") as f:
        config = yaml.safe_load(f)

    weights = standardize_portfolio_input(config["portfolio_input"], latest_price)["weights"]
    summary = build_portfolio_view(
        weights,
        config["start_date"],
        config["end_date"],
        config.get("expected_returns"),
        config.get("stock_factor_proxies")
    )
    pprint(summary["variance_decomposition"])  # Swap with anything you'd want to print

def run_stock(ticker: str, start: str, end: str):
    # NOTE: expects hardcoded factor proxies for now — later can add YAML option
    factor_proxies = {
        "market": "SPY",
        "momentum": "MTUM",
        "value": "IWD",
        "industry": "XSW",
        "subindustry": ["PAYC", "PYCR", "CDAY", "ADP", "PAYX", "WDAY"]
    }
    profile = get_detailed_stock_factor_profile(ticker, start, end, factor_proxies)
    pprint(profile["factor_summary"])  # Swap with whatever you'd want to inspect

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--portfolio", type=str, help="Path to YAML portfolio file")
    parser.add_argument("--stock", type=str, help="Ticker symbol")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")

    args = parser.parse_args()

    if args.portfolio:
        run_portfolio(args.portfolio)
    elif args.stock and args.start and args.end:
        run_stock(args.stock, args.start, args.end)
    else:
        parser.print_help()

