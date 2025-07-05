#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# File: run_risk.py

import argparse
import yaml
from contextlib import redirect_stdout 
from typing import Optional, Dict, Union, List
import pandas as pd
from io import StringIO

from risk_summary import (
    get_detailed_stock_factor_profile,
    get_stock_risk_profile
)
from run_portfolio_risk import (
    latest_price,
    load_portfolio_config,
    display_portfolio_config,
    standardize_portfolio_input,
    display_portfolio_summary,
    evaluate_portfolio_beta_limits,
    evaluate_portfolio_risk_limits,
)
from portfolio_risk import build_portfolio_view
from portfolio_optimizer import (
    run_what_if_scenario,
    print_what_if_report,
    run_min_var,
    print_min_var_report,
    run_max_return_portfolio,
    print_max_return_report,
)  
from risk_helpers import (
    calc_max_factor_betas
)
from proxy_builder import inject_all_proxies
from gpt_helpers import (
    interpret_portfolio_risk,
    generate_subindustry_peers,
)

def run_and_interpret(portfolio_yaml: str):
    """
    Convenience wrapper:

        1. runs `run_portfolio(portfolio_yaml)`
        2. captures everything it prints
        3. feeds that text to GPT for a summary
        4. prints **both** the GPT summary *and* the raw diagnostics
        5. returns the GPT summary string

    Parameters
    ----------
    portfolio_yaml : str
        Path to the portfolio configuration YAML.

    Returns
    -------
    str
        The GPT interpretation of the risk report.
    """
    buf = StringIO()
    with redirect_stdout(buf):
        run_portfolio(portfolio_yaml)

    diagnostics = buf.getvalue()
    summary_txt = interpret_portfolio_risk(diagnostics)

    # --- show results ----------------------------------------------------
    print("\n=== GPT Portfolio Interpretation ===\n")
    print(summary_txt)
    print("\n=== Full Diagnostics ===\n")
    print(diagnostics)

    #return summary_txt    #Optional: return GPT summary text

def run_portfolio(filepath: str):
    """
    High-level “one-click” entry-point for a full portfolio risk run.

    It ties together **all** of the moving pieces you’ve built so far:

        1.  Loads the portfolio YAML file (positions, dates, factor proxies).
        2.  Loads the firm-wide risk-limits YAML.
        3.  Standardises the raw position inputs into weights, then calls
            `build_portfolio_view` to produce the master `summary` dictionary
            (returns, vol, correlation, factor betas, variance decomposition, …).
        4.  Pretty-prints the standard risk summary via `display_portfolio_summary`.
        5.  Derives *dynamic* max-beta limits:
                • looks back over the analysis window,
                • finds worst 1-month drawdowns for every unique factor proxy,
                • converts those losses into a per-factor β ceiling
                  using the global `max_single_factor_loss`.
        6.  Runs two rule-checkers
                • `evaluate_portfolio_risk_limits`   →   Vol, concentration, factor %
                • `evaluate_portfolio_beta_limits`   →   Actual β vs. max β
        7.  Prints both tables in a compact “PASS/FAIL” console report.

    Parameters
    ----------
    filepath : str
        Path to the *portfolio* YAML ( **not** the risk-limits file ).
        The function expects the YAML schema
        (`start_date`, `end_date`, `portfolio_input`, `stock_factor_proxies`, …).

    Side-effects
    ------------
    • Prints a formatted risk report to stdout.
    • Does **not** return anything expect prints; everything is handled inline.
      (If you need the raw DataFrames, simply return `summary`, `df_risk`,
      and `df_beta` at the end.)

    Example
    -------
    >>> run_portfolio("portfolio.yaml")
    === Target Allocations ===
    …                                 # summary table
    === Portfolio Risk Limit Checks ===
    Volatility:             21.65%  ≤ 40.00%     → PASS
    …
    === Beta Exposure Checks ===
    market       β = 0.74  ≤ 0.80  → PASS
    …
    """
        
    # ─── 1. Load YAML Inputs ─────────────────────────────────
     
    config = load_portfolio_config(filepath)
        
    with open("risk_limits.yaml", "r") as f:
        risk_config = yaml.safe_load(f)

    weights = standardize_portfolio_input(config["portfolio_input"], latest_price)["weights"]
    summary = build_portfolio_view(
        weights,
        config["start_date"],
        config["end_date"],
        config.get("expected_returns"),
        config.get("stock_factor_proxies")
    )

    # ─── 2. Display Summary ─────────────────────────────────
    display_portfolio_config(config)
    display_portfolio_summary(summary)
    
    # ─── 3. Compute Beta Limits ─────────────────────────────
    max_betas, max_betas_by_proxy = calc_max_factor_betas(
        portfolio_yaml = filepath,
        risk_yaml      = "risk_limits.yaml",
        lookback_years = 10,
        echo           = True,     # show helper tables
    )

    # ─── 4. Evaluate Portfolio Risk Rules ───────────────────
    print("\n=== Portfolio Risk Limit Checks ===")
    df_risk = evaluate_portfolio_risk_limits(
        summary,
        risk_config["portfolio_limits"],
        risk_config["concentration_limits"],
        risk_config["variance_limits"]
    )
    for _, row in df_risk.iterrows():
        status = "→ PASS" if row["Pass"] else "→ FAIL"
        print(f"{row['Metric']:<22} {row['Actual']:.2%}  ≤ {row['Limit']:.2%}  {status}")

    # ─── 5. Evaluate Beta Limits ────────────────────────────
    print("\n=== Beta Exposure Checks ===")
    df_beta = evaluate_portfolio_beta_limits(
        portfolio_factor_betas = summary["portfolio_factor_betas"],
        max_betas              = max_betas,
        proxy_betas            = summary["industry_variance"].get("per_industry_group_beta"),
        max_proxy_betas        = max_betas_by_proxy
    )
    
    for factor, row in df_beta.iterrows():
        status = "→ PASS" if row["pass"] else "→ FAIL"
        print(f"{factor:<20} β = {row['portfolio_beta']:+.2f}  ≤ {row['max_allowed_beta']:.2f}  {status}")

    # ─────────────────────────────────────────────────────────────────────

def run_what_if(
    filepath: str, 
    scenario_yaml: Optional[str] = None, 
    delta: Optional[str] = None
):
    """
    Execute a single *what-if* scenario on an existing portfolio.

    The function loads the base portfolio & firm-wide risk limits,
    applies either a YAML-defined scenario **or** an inline `delta`
    string, and prints a full before/after risk report.

    Parameters
    ----------
    filepath : str
        Path to the primary portfolio YAML file (same schema as
        ``run_portfolio``).
    scenario_yaml : str, optional
        Path to a YAML file that contains a ``new_weights`` or ``delta``
        section (see ``helpers_input.parse_delta`` for precedence rules).
        If supplied, this file overrides any `delta` string.
    delta : str, optional
        Comma-separated inline weight shifts, e.g.
        ``"TW:+500bp,PCTY:-200bp"``.  Ignored when `scenario_yaml`
        contains a ``new_weights`` block.

    Notes
    -----
    ▸ Does *not* return anything; all output is printed via
      ``print_what_if_report``.  
    ▸ Raises ``ValueError`` if neither YAML nor `delta`
      provide a valid change set.
    """
    # --- load configs ------------------------------------------------------
    config = load_portfolio_config(filepath)
    with open("risk_limits.yaml", "r") as f:
        risk_config = yaml.safe_load(f)

    weights = standardize_portfolio_input(config["portfolio_input"], latest_price)["weights"]

    # parse CLI delta string
    shift_dict = None
    if delta:
        shift_dict = {k.strip(): v.strip() for k, v in (pair.split(":") for pair in delta.split(","))}

    # --- run the engine ----------------------------------------------------
    summary, risk_new, beta_new, cmp_risk, cmp_beta = run_what_if_scenario(
        base_weights = weights,
        config       = config,
        risk_config  = risk_config,
        proxies      = config["stock_factor_proxies"],
        scenario_yaml = scenario_yaml,
        shift_dict   = shift_dict,
    )
    
    # split beta table between factors and industry
    beta_f_new = beta_new[~beta_new.index.str.startswith("industry_proxy::")]
    beta_p_new = beta_new[ beta_new.index.str.startswith("industry_proxy::")].copy()
    beta_p_new.index = beta_p_new.index.str.replace("industry_proxy::", "")
    
    # Print report
    print_what_if_report(
        summary_new=summary,
        risk_new=risk_new,
        beta_f_new=beta_f_new,
        beta_p_new=beta_p_new,
        cmp_risk=cmp_risk,
        cmp_beta=cmp_beta,
    )

def run_min_variance(filepath: str):
    """
    Run the minimum-variance optimiser under current risk limits.

    Steps
    -----
    1. Load portfolio & risk-limit YAML files.
    2. Convert raw position input into normalised weights.
    3. Call :pyfunc:`portfolio_optimizer.run_min_var` to solve for the
       lowest-variance weight vector that satisfies **all** firm-wide
       constraints.
    4. Pretty-print the resulting weights plus risk & beta check tables
       via :pyfunc:`portfolio_optimizer.print_min_var_report`.

    Parameters
    ----------
    filepath : str
        Path to the portfolio YAML file (``start_date``, ``end_date``,
        ``portfolio_input``, etc.).

    Raises
    ------
    ValueError
        Propagated from the optimiser if the constraints are infeasible.

    Side Effects
    ------------
    Prints the optimised weight allocation and PASS/FAIL tables to
    stdout; nothing is returned.
    """
    # --- load configs ------------------------------------------------------
    config = load_portfolio_config(filepath)
    with open("risk_limits.yaml", "r") as f:
        risk_config = yaml.safe_load(f)

    weights = standardize_portfolio_input(config["portfolio_input"], latest_price)["weights"]

    # --- run the engine ----------------------------------------------------
    w, r, b = run_min_var(
        base_weights = weights,
        config       = config,
        risk_config  = risk_config,
        proxies      = config["stock_factor_proxies"],
    )
    print_min_var_report(weights=w, risk_tbl=r, beta_tbl=b)

def run_max_return(filepath: str):
    """
    Solve for the highest-return portfolio that still passes all
    volatility, concentration, and beta limits.

    Workflow
    --------
    * Parse the portfolio and risk-limit YAMLs.
    * Standardise the raw positions → weights.
    * Call :pyfunc:`portfolio_optimizer.run_max_return_portfolio` to
      perform a convex QP that maximises expected return subject to:
        – portfolio σ cap  
        – single-name weight cap  
        – factor & industry beta caps
    * Print the final weight vector and the associated risk / beta
      check tables via :pyfunc:`portfolio_optimizer.print_max_return_report`.

    Parameters
    ----------
    filepath : str
        Path to the portfolio YAML file.

    Notes
    -----
    * Uses the **expected_returns** section inside the portfolio YAML for
      the objective function.  Missing tickers default to 0 % expected
      return.
    * All output is written to stdout; the function does not return
      anything.
    """
    # --- load configs ------------------------------------------------------
    config = load_portfolio_config(filepath)
    with open("risk_limits.yaml", "r") as f:
        risk_config = yaml.safe_load(f)

    weights = standardize_portfolio_input(config["portfolio_input"], latest_price)["weights"]
    
    # --- run the engine ----------------------------------------------------
    w, summary, r, f_b, p_b = run_max_return_portfolio(
        weights     = weights,
        config      = config,
        risk_config = risk_config,
        proxies     = config["stock_factor_proxies"],
    )
    print_max_return_report(weights=w, risk_tbl=r, df_factors=f_b, df_proxies=p_b)
    # ─────────────────────────────────────────────────────────────────────

def run_stock(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    factor_proxies: Optional[Dict[str, Union[str, List[str]]]] = None,
    yaml_path: Optional[str] = None, 
):
    """
    Runs stock risk diagnostics. If factor_proxies are provided, runs detailed multi-factor profile.
    Otherwise, if ``yaml_path`` is supplied, the function looks for the
    ticker under ``stock_factor_proxies`` in that YAML.
    If neither is found, falls back to simple regression vs. market benchmark.

    Args:
        ticker (str): Stock symbol.
        start (Optional[str]): Start date in YYYY-MM-DD format. Defaults to 5 years ago.
        end (Optional[str]): End date in YYYY-MM-DD format. Defaults to today.
        factor_proxies (Optional[Dict[str, Union[str, List[str]]]]): Optional factor mapping.
    """
    ticker = ticker.upper()

    # ─── 1. Resolve date window ─────────────────────────────────────────
    today = pd.Timestamp.today().normalize()
    start = pd.to_datetime(start) if start else today - pd.DateOffset(years=5)
    end   = pd.to_datetime(end)   if end   else today

    # ─── 2. Auto-lookup proxy block from YAML (if requested) ────────────
    if factor_proxies is None and yaml_path:
        try:
            cfg = load_portfolio_config(yaml_path)       # already handles safe_load + validation
            proxies = cfg.get("stock_factor_proxies", {})
            factor_proxies = proxies.get(ticker.upper())
        except Exception as e:
            print(f"⚠️  Could not load proxies from YAML: {e}")

    # ─── 3. Diagnostics path A: multi-factor profile ────────────────────
    if factor_proxies:
        profile = get_detailed_stock_factor_profile(
            ticker, start, end, factor_proxies
        )
        print("=== Volatility ===")
        print(profile["vol_metrics"])

        print("\n=== Market Regression ===")
        print(profile["regression_metrics"])

        print("\n=== Factor Summary ===")
        print(profile["factor_summary"])

    # ─── 4. Diagnostics path B: simple market regression ────────────────
    else:
        result = get_stock_risk_profile(
            ticker,
            start_date=start,
            end_date=end,
            benchmark="SPY"
        )
        print("=== Volatility Metrics ===")
        print(result["vol_metrics"])

        print("\n=== Market Regression (SPY) ===")
        print(result["risk_metrics"])

def run_portfolio_performance(filepath: str):
    """
    Calculate and display comprehensive portfolio performance metrics.

    Workflow
    --------
    * Parse the portfolio YAML file.
    * Standardise the raw positions → weights.
    * Call :func:`portfolio_risk.calculate_portfolio_performance_metrics` to
      compute returns, volatility, Sharpe ratio, alpha, beta, max drawdown,
      and other risk-adjusted performance metrics vs benchmark.
    * Display results using :func:`run_portfolio_risk.display_portfolio_performance_metrics`.

    Parameters
    ----------
    filepath : str
        Path to the portfolio YAML file.

    Notes
    -----
    * Uses the **start_date** and **end_date** from the portfolio YAML for
      the performance analysis window.
    * Defaults to SPY as benchmark but can be customized.
    * Uses 3-month Treasury rates from FMP API as risk-free rate.
    * All output is written to stdout; the function does not return anything.
    """
    from portfolio_risk import calculate_portfolio_performance_metrics
    from run_portfolio_risk import display_portfolio_performance_metrics
    
    print("📊 Portfolio Performance Analysis")
    print("=" * 50)
    
    try:
        # Load portfolio configuration
        config = load_portfolio_config(filepath)
        
        # Standardize portfolio weights
        weights = standardize_portfolio_input(config["portfolio_input"], latest_price)["weights"]
        
        print(f"📁 Portfolio file: {filepath}")
        print(f"📅 Analysis period: {config['start_date']} to {config['end_date']}")
        print(f"📊 Positions: {len(weights)}")
        print()
        
        # Calculate performance metrics
        print("🔄 Calculating performance metrics...")
        performance_metrics = calculate_portfolio_performance_metrics(
            weights=weights,
            start_date=config["start_date"],
            end_date=config["end_date"],
            benchmark_ticker="SPY"  # Could make this configurable later
        )
        
        # Check for calculation errors
        if "error" in performance_metrics:
            print(f"❌ Performance calculation failed: {performance_metrics['error']}")
            return
        
        print("✅ Performance calculation successful!")
        
        # Display the results
        display_portfolio_performance_metrics(performance_metrics)
        
    except FileNotFoundError:
        print(f"❌ Error: Portfolio file '{filepath}' not found")
    except Exception as e:
        print(f"❌ Error during performance analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--portfolio", type=str, help="Path to YAML portfolio file")
    parser.add_argument("--stock", type=str, help="Ticker symbol")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--whatif", action="store_true", help="Run what-if scenario")
    parser.add_argument("--minvar", action="store_true", help="Run min-variance optimization")
    parser.add_argument("--maxreturn", action="store_true", help="Run max-return optimization")
    parser.add_argument("--performance", action="store_true", help="Run portfolio performance analysis")
    parser.add_argument("--scenario", type=str, help="Path to what-if scenario YAML file")
    parser.add_argument("--delta", type=str, help='Inline weight shifts, e.g. "TW:+500bp,PCTY:-200bp"')
    parser.add_argument("--inject_proxies", action="store_true", help="Inject market, industry, and optional subindustry proxies")
    parser.add_argument("--use_gpt", action="store_true", help="Enable GPT-generated subindustry peers (used with --inject_proxies)")
    parser.add_argument("--gpt", action="store_true", help="Run the portfolio report and send the output to GPT for a plain-English summary")
    args = parser.parse_args()

    if args.portfolio and args.inject_proxies:
        inject_all_proxies(args.portfolio, use_gpt_subindustry=args.use_gpt)
    
    elif args.portfolio and args.whatif:
        run_what_if(args.portfolio, scenario_yaml=args.scenario, delta=args.delta)
    
    elif args.portfolio and args.minvar:
        run_min_variance(args.portfolio)
    
    elif args.portfolio and args.maxreturn:
        run_max_return(args.portfolio)

    elif args.portfolio and args.performance:
        run_portfolio_performance(args.portfolio)
        
    elif args.portfolio and args.gpt:
        run_and_interpret(args.portfolio)
    
    elif args.portfolio:
        run_portfolio(args.portfolio)
    
    elif args.stock and args.start and args.end:
        run_stock(args.stock, args.start, args.end)
    
    else:
        parser.print_help()


# In[ ]:




