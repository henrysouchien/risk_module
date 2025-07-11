#!/usr/bin/env python3
# coding: utf-8

# In[ ]:


# File: run_risk.py

import argparse
import yaml
from contextlib import redirect_stdout 
from typing import Optional, Dict, Union, List, Any
import pandas as pd
from io import StringIO
import numpy as np
from datetime import datetime

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
from helpers_display import format_stock_metrics

def make_json_safe(obj):
    """
    Recursively convert any object to JSON-serializable format.
    """
    if isinstance(obj, dict):
        # Convert dictionary, ensuring keys are JSON-serializable
        safe_dict = {}
        for k, v in obj.items():
            # Convert keys to strings if they're not JSON-serializable
            if isinstance(k, (pd.Timestamp, datetime)):
                safe_key = k.strftime("%Y-%m-%d %H:%M:%S") if hasattr(k, 'strftime') else str(k)
            elif isinstance(k, (int, float, str, bool, type(None))):
                safe_key = k
            else:
                safe_key = str(k)
            safe_dict[safe_key] = make_json_safe(v)
        return safe_dict
    elif isinstance(obj, list):
        return [make_json_safe(item) for item in obj]
    elif isinstance(obj, pd.DataFrame):
        # Convert DataFrame to records with safe keys
        return obj.to_dict('records')
    elif isinstance(obj, pd.Series):
        # Convert Series to dict with safe keys
        return {str(k): make_json_safe(v) for k, v in obj.to_dict().items()}
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return obj.strftime("%Y-%m-%d %H:%M:%S") if hasattr(obj, 'strftime') else str(obj)
    elif pd.isna(obj):
        return None
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        # For any other object, try to convert to string
        return str(obj)

def run_and_interpret(portfolio_yaml: str, *, return_data: bool = False):
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
    return_data : bool, default False
        If True, returns structured data instead of printing.
        If False, prints formatted output to stdout (existing behavior).

    Returns
    -------
    str or Dict[str, Any]
        If return_data=False: Returns GPT interpretation string (existing behavior)
        If return_data=True: Returns structured data dictionary with:
            - ai_interpretation: GPT interpretation of the analysis
            - full_diagnostics: Complete analysis output text
            - analysis_metadata: Analysis configuration and timestamps
    """
    buf = StringIO()
    with redirect_stdout(buf):
        run_portfolio(portfolio_yaml)

    diagnostics = buf.getvalue()
    summary_txt = interpret_portfolio_risk(diagnostics)

    if return_data:
        # Return structured data with raw objects (for service layer)
        return {
            "ai_interpretation": summary_txt,
            "full_diagnostics": diagnostics,
            "analysis_metadata": {
                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "portfolio_file": portfolio_yaml,
                "interpretation_service": "gpt",
                "diagnostics_length": len(diagnostics),
                "interpretation_length": len(summary_txt)
            }
        }
    else:
        # CLI MODE: Print formatted output (existing behavior)
        print("\n=== GPT Portfolio Interpretation ===\n")
        print(summary_txt)
        print("\n=== Full Diagnostics ===\n")
        print(diagnostics)
        
        return summary_txt  # Return GPT summary text (existing behavior)

def interpret_portfolio_output(portfolio_output: Dict[str, Any], *, 
                              portfolio_name: Optional[str] = None,
                              return_data: bool = False):
    """
    Add AI interpretation to existing portfolio analysis output.
    
    This function enables two-level caching optimization:
    1. run_portfolio() output can be cached by PortfolioService
    2. AI interpretation can be cached separately
    
    Parameters
    ----------
    portfolio_output : Dict[str, Any]
        Structured output from run_portfolio(return_data=True)
    portfolio_name : Optional[str]
        Name/identifier for the portfolio (for metadata)
    return_data : bool, default False
        If True, returns structured data instead of printing.
        If False, prints formatted output to stdout (existing behavior).
    
    Returns
    -------
    str or Dict[str, Any]
        If return_data=False: Returns GPT interpretation string (existing behavior)
        If return_data=True: Returns structured data dictionary with:
            - ai_interpretation: GPT interpretation of the analysis
            - full_diagnostics: Complete analysis output text
            - analysis_metadata: Analysis configuration and timestamps
    """
    # Generate formatted diagnostics text from structured output
    diagnostics = _format_portfolio_output_as_text(portfolio_output)
    
    # Get AI interpretation
    summary_txt = interpret_portfolio_risk(diagnostics)
    
    if return_data:
        # Return structured data with raw objects (for service layer)
        return {
            "ai_interpretation": summary_txt,
            "full_diagnostics": diagnostics,
            "analysis_metadata": {
                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "portfolio_file": portfolio_name or "portfolio_output",
                "interpretation_service": "gpt",
                "diagnostics_length": len(diagnostics),
                "interpretation_length": len(summary_txt)
            }
        }
    else:
        # CLI MODE: Print formatted output (existing behavior)
        print("\n=== GPT Portfolio Interpretation ===\n")
        print(summary_txt)
        print("\n=== Full Diagnostics ===\n")
        print(diagnostics)
        
        return summary_txt  # Return GPT summary text (existing behavior)

def _format_portfolio_output_as_text(portfolio_output: Dict[str, Any]) -> str:
    """
    Convert structured portfolio output back to formatted text.
    
    This recreates the text that would have been printed by run_portfolio()
    when called without return_data=True.
    """
    from io import StringIO
    
    # Create a buffer to capture formatted output
    buf = StringIO()
    
    # Use the existing formatted report from portfolio output
    formatted_report = portfolio_output.get("formatted_report", "")
    if formatted_report:
        buf.write(formatted_report)
    
    # Add risk analysis tables
    risk_analysis = portfolio_output.get("risk_analysis", {})
    beta_analysis = portfolio_output.get("beta_analysis", {})
    
    # Add risk checks table
    buf.write("\n=== Portfolio Risk Limit Checks ===\n")
    for check in risk_analysis.get("risk_checks", []):
        status = "PASS" if check.get("Pass", False) else "FAIL"
        buf.write(f"{check.get('check_name', 'Unknown')}: {status}\n")
    
    # Add beta checks table
    buf.write("\n=== Beta Exposure Checks ===\n")
    for check in beta_analysis.get("beta_checks", []):
        status = "PASS" if check.get("pass", False) else "FAIL"
        buf.write(f"{check.get('factor', 'Unknown')}: {status}\n")
    
    return buf.getvalue()

def run_portfolio(filepath: str, *, return_data: bool = False):
    """
    High-level "one-click" entry-point for a full portfolio risk run.

    It ties together **all** of the moving pieces you've built so far:

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
        7.  Prints both tables in a compact "PASS/FAIL" console report.

    Parameters
    ----------
    filepath : str
        Path to the *portfolio* YAML ( **not** the risk-limits file ).
        The function expects the YAML schema
        (`start_date`, `end_date`, `portfolio_input`, `stock_factor_proxies`, …).
    return_data : bool, default False
        If True, returns structured data instead of printing.
        If False, prints formatted output to stdout (existing behavior).

    Returns
    -------
    None or Dict[str, Any]
        If return_data=False: Returns None, prints formatted output (existing behavior)
        If return_data=True: Returns structured data dictionary with:
            - portfolio_summary: Complete portfolio view from build_portfolio_view
            - risk_analysis: Risk limit checks and violations
            - beta_analysis: Factor beta checks and violations
            - analysis_metadata: Analysis configuration and timestamps
            - formatted_report: Captured CLI output text

    Side-effects
    ------------
    • When return_data=False: Prints a formatted risk report to stdout.
    • When return_data=True: No console output, returns structured data.

    Example
    -------
    # CLI usage (existing behavior)
    >>> run_portfolio("portfolio.yaml")
    === Target Allocations ===
    …                                 # summary table
    === Portfolio Risk Limit Checks ===
    Volatility:             21.65%  ≤ 40.00%     → PASS
    …
    === Beta Exposure Checks ===
    market       β = 0.74  ≤ 0.80  → PASS
    …
    
    # API usage (new behavior)
    >>> result = run_portfolio("portfolio.yaml", return_data=True)
    >>> print(result["portfolio_summary"]["annual_volatility"])
    0.2165
    >>> print(result["risk_analysis"]["risk_passes"])
    True
    """
    
    # ─── 1. Load YAML Inputs ─────────────────────────────────
    config = load_portfolio_config(filepath)
    
    with open("risk_limits.yaml", "r") as f:
        risk_config = yaml.safe_load(f)

    weights = standardize_portfolio_input(config["portfolio_input"], latest_price)["weights"]
    
    # ─── 2. Build Portfolio View ─────────────────────────────
    summary = build_portfolio_view(
        weights,
        config["start_date"],
        config["end_date"],
        config.get("expected_returns"),
        config.get("stock_factor_proxies")
    )
    
    # ─── 3. Calculate Beta Limits ────────────────────────────
    from settings import PORTFOLIO_DEFAULTS
    lookback_years = PORTFOLIO_DEFAULTS.get('worst_case_lookback_years', 10)
    max_betas, max_betas_by_proxy = calc_max_factor_betas(
        portfolio_yaml=filepath,
        risk_yaml="risk_limits.yaml",
        lookback_years=lookback_years,
        echo=False  # Don't print helper tables when capturing output
    )
    
    # ─── 4. Run Risk Checks ──────────────────────────────────
    df_risk = evaluate_portfolio_risk_limits(
        summary,
        risk_config["portfolio_limits"],
        risk_config["concentration_limits"],
        risk_config["variance_limits"]
    )
    
    df_beta = evaluate_portfolio_beta_limits(
        portfolio_factor_betas=summary["portfolio_factor_betas"],
        max_betas=max_betas,
        proxy_betas=summary["industry_variance"].get("per_industry_group_beta"),
        max_proxy_betas=max_betas_by_proxy
    )

    # ─── 5. Dual-Mode Logic ─────────────────────────────────
    if return_data:
        # API MODE: Capture CLI output and return structured data
        from io import StringIO
        import contextlib
        
        # Capture the CLI output
        buf = StringIO()
        with contextlib.redirect_stdout(buf):
            display_portfolio_config(config)
            display_portfolio_summary(summary)
            
            # Display Risk Rules
            print("\n=== Portfolio Risk Limit Checks ===")
            for _, row in df_risk.iterrows():
                status = "→ PASS" if row["Pass"] else "→ FAIL"
                print(f"{row['Metric']:<22} {row['Actual']:.2%}  ≤ {row['Limit']:.2%}  {status}")

            # Display Beta Limits
            print("\n=== Beta Exposure Checks ===")
            for factor, row in df_beta.iterrows():
                status = "→ PASS" if row["pass"] else "→ FAIL"
                print(f"{factor:<20} β = {row['portfolio_beta']:+.2f}  ≤ {row['max_allowed_beta']:.2f}  {status}")
        
        # Return structured data with captured CLI output
        return {
            "portfolio_summary": summary,
            "risk_analysis": {
                "risk_checks": df_risk.to_dict('records'),
                "risk_passes": bool(df_risk['Pass'].all()),
                "risk_violations": df_risk[~df_risk['Pass']].to_dict('records'),
                "risk_limits": {
                    "portfolio_limits": risk_config["portfolio_limits"],
                    "concentration_limits": risk_config["concentration_limits"],
                    "variance_limits": risk_config["variance_limits"]
                }
            },
            "beta_analysis": {
                "beta_checks": df_beta.to_dict('records'),
                "beta_passes": bool(df_beta['pass'].all()),
                "beta_violations": df_beta[~df_beta['pass']].to_dict('records'),
                "max_betas": max_betas,
                "max_betas_by_proxy": max_betas_by_proxy
            },
            "analysis_metadata": {
                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "portfolio_file": filepath,
                "lookback_years": lookback_years,
                "weights": weights,
                "total_positions": len(weights),
                "active_positions": len([v for v in weights.values() if abs(v) > 0.001])
            },
            "formatted_report": buf.getvalue()
        }
    else:
        # CLI MODE: Print formatted output
        display_portfolio_config(config)
        display_portfolio_summary(summary)
        
        # Display Risk Rules
        print("\n=== Portfolio Risk Limit Checks ===")
        for _, row in df_risk.iterrows():
            status = "→ PASS" if row["Pass"] else "→ FAIL"
            print(f"{row['Metric']:<22} {row['Actual']:.2%}  ≤ {row['Limit']:.2%}  {status}")

        # Display Beta Limits
        print("\n=== Beta Exposure Checks ===")
        for factor, row in df_beta.iterrows():
            status = "→ PASS" if row["pass"] else "→ FAIL"
            print(f"{factor:<20} β = {row['portfolio_beta']:+.2f}  ≤ {row['max_allowed_beta']:.2f}  {status}")

def run_what_if(
    filepath: str, 
    scenario_yaml: Optional[str] = None, 
    delta: Optional[str] = None,
    *,
    return_data: bool = False
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
    return_data : bool, default False
        If True, returns structured data instead of printing.
        If False, prints formatted output to stdout (existing behavior).

    Returns
    -------
    None or Dict[str, Any]
        If return_data=False: Returns None, prints formatted output (existing behavior)
        If return_data=True: Returns structured data dictionary with:
            - scenario_summary: Portfolio view after scenario changes
            - risk_analysis: Risk checks for scenario portfolio
            - beta_analysis: Beta checks for scenario portfolio
            - comparison_analysis: Before/after comparison data
            - scenario_metadata: Scenario configuration and metadata

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
    # First, create base portfolio summary for comparison
    summary_base = build_portfolio_view(
        weights,
        config["start_date"],
        config["end_date"],
        config.get("expected_returns"),
        config.get("stock_factor_proxies")
    )
    
    # Then run the scenario
    summary, risk_new, beta_new, cmp_risk, cmp_beta = run_what_if_scenario(
        base_weights = weights,
        config       = config,
        risk_config  = risk_config,
        proxies      = config["stock_factor_proxies"],
        scenario_yaml = scenario_yaml,
        shift_dict   = shift_dict,
    )
    
    # split beta table between factors and industry
    beta_f_new = beta_new.copy()
    beta_p_new = pd.DataFrame()
    
    # Only try to split if we have a proper index
    if hasattr(beta_new.index, 'str') and len(beta_new) > 0:
        try:
            industry_mask = beta_new.index.str.startswith("industry_proxy::")
            beta_f_new = beta_new[~industry_mask]
            beta_p_new = beta_new[industry_mask].copy()
            if not beta_p_new.empty:
                beta_p_new.index = beta_p_new.index.str.replace("industry_proxy::", "")
        except Exception as e:
            # Fallback: use the original beta table as factor table
            print(f"Warning: Could not split beta table: {e}")
            beta_f_new = beta_new.copy()
            beta_p_new = pd.DataFrame()
    
    # ─── Dual-Mode Logic ─────────────────────────────────────
    if return_data:
        # API MODE: Return structured data (JSON-serializable)
        from core.result_objects import WhatIfResult
        
        # Create formatted report by capturing print output
        from io import StringIO
        from contextlib import redirect_stdout
        
        report_buffer = StringIO()
        with redirect_stdout(report_buffer):
            print_what_if_report(
                summary_new=summary,
                risk_new=risk_new,
                beta_f_new=beta_f_new,
                beta_p_new=beta_p_new,
                cmp_risk=cmp_risk,
                cmp_beta=cmp_beta,
            )
        formatted_report = report_buffer.getvalue()
        
        return make_json_safe({
            "scenario_summary": summary,
            "risk_analysis": {
                "risk_checks": risk_new.to_dict('records') if not risk_new.empty else [],
                "risk_passes": bool(risk_new['Pass'].all()) if not risk_new.empty and 'Pass' in risk_new.columns else True,
                "risk_violations": risk_new[~risk_new['Pass']].to_dict('records') if not risk_new.empty and 'Pass' in risk_new.columns else [],
                "risk_limits": {
                    "portfolio_limits": risk_config["portfolio_limits"],
                    "concentration_limits": risk_config["concentration_limits"],
                    "variance_limits": risk_config["variance_limits"]
                }
            },
            "beta_analysis": {
                "factor_beta_checks": beta_f_new.to_dict('records') if not beta_f_new.empty else [],
                "proxy_beta_checks": beta_p_new.to_dict('records') if not beta_p_new.empty else [],
                "beta_passes": bool(beta_new['pass'].all()) if not beta_new.empty and 'pass' in beta_new.columns else True,
                "beta_violations": beta_new[~beta_new['pass']].to_dict('records') if not beta_new.empty and 'pass' in beta_new.columns else [],
            },
            "comparison_analysis": {
                "risk_comparison": cmp_risk.to_dict('records'),
                "beta_comparison": cmp_beta.to_dict('records'),
            },
            "delta_change": {
                "volatility_delta": summary_base["volatility_annual"] - summary["volatility_annual"],
                "base_volatility": summary_base["volatility_annual"],
                "scenario_volatility": summary["volatility_annual"]
            },
            "scenario_metadata": {
                "scenario_yaml": scenario_yaml,
                "delta_string": delta,
                "shift_dict": shift_dict,
                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "portfolio_file": filepath,
                "base_weights": weights
            },
            "formatted_report": formatted_report
        })
    else:
        # CLI MODE: Print formatted output
        print_what_if_report(
            summary_new=summary,
            risk_new=risk_new,
            beta_f_new=beta_f_new,
            beta_p_new=beta_p_new,
            cmp_risk=cmp_risk,
            cmp_beta=cmp_beta,
        )

def run_min_variance(filepath: str, *, return_data: bool = False):
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
    return_data : bool, default False
        If True, returns structured data instead of printing.
        If False, prints formatted output to stdout (existing behavior).

    Returns
    -------
    None or Dict[str, Any]
        If return_data=False: Returns None, prints formatted output (existing behavior)
        If return_data=True: Returns structured data dictionary with:
            - optimized_weights: Optimized portfolio weights
            - risk_analysis: Risk checks for optimized portfolio
            - beta_analysis: Beta checks for optimized portfolio
            - optimization_metadata: Optimization configuration and results

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
    
    # ─── Dual-Mode Logic ─────────────────────────────────────
    if return_data:
        # API MODE: Return structured data (JSON-serializable)
        from core.result_objects import OptimizationResult
        
        # Create OptimizationResult object for formatted report
        optimization_result = OptimizationResult.from_min_variance_output(
            optimized_weights=w,
            risk_table=r,
            beta_table=b
        )
        
        return make_json_safe({
            "optimized_weights": w,
            "risk_analysis": {
                "risk_checks": r.to_dict('records'),
                "risk_passes": bool(r['Pass'].all()),
                "risk_violations": r[~r['Pass']].to_dict('records'),
                "risk_limits": {
                    "portfolio_limits": risk_config["portfolio_limits"],
                    "concentration_limits": risk_config["concentration_limits"],
                    "variance_limits": risk_config["variance_limits"]
                }
            },
            "beta_analysis": {
                "beta_checks": b.to_dict('records'),
                "beta_passes": bool(b['pass'].all()),
                "beta_violations": b[~b['pass']].to_dict('records'),
            },
            "optimization_metadata": {
                "optimization_type": "minimum_variance",
                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "portfolio_file": filepath,
                "original_weights": weights,
                "total_positions": len(w),
                "active_positions": len([v for v in w.values() if abs(v) > 0.001])
            },
            "formatted_report": optimization_result.to_formatted_report()
        })
    else:
        # CLI MODE: Print formatted output
        print_min_var_report(weights=w, risk_tbl=r, beta_tbl=b)

def run_max_return(filepath: str, *, return_data: bool = False):
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
    return_data : bool, default False
        If True, returns structured data instead of printing.
        If False, prints formatted output to stdout (existing behavior).

    Returns
    -------
    None or Dict[str, Any]
        If return_data=False: Returns None, prints formatted output (existing behavior)
        If return_data=True: Returns structured data dictionary with:
            - optimized_weights: Optimized portfolio weights
            - portfolio_summary: Portfolio view of optimized weights
            - risk_analysis: Risk checks for optimized portfolio
            - beta_analysis: Factor and proxy beta checks for optimized portfolio
            - optimization_metadata: Optimization configuration and results

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
    
    # ─── Dual-Mode Logic ─────────────────────────────────────
    if return_data:
        # API MODE: Return structured data + capture formatted report + create object
        from core.result_objects import OptimizationResult
        from io import StringIO
        import contextlib
        
        # Create result object for structured data
        optimization_result = OptimizationResult.from_max_return_output(
            optimized_weights=w,
            portfolio_summary=summary,
            risk_table=r,
            factor_table=f_b,
            proxy_table=p_b
        )
        
        # Capture the formatted report by running the CLI logic
        report_buffer = StringIO()
        with contextlib.redirect_stdout(report_buffer):
            print_max_return_report(weights=w, risk_tbl=r, df_factors=f_b, df_proxies=p_b)
        
        formatted_report = report_buffer.getvalue()
        
        # Combine object data with additional metadata
        result_data = optimization_result.to_dict()
        result_data.update({
            "optimized_weights": w,
            "portfolio_summary": summary,
            "risk_analysis": {
                "risk_checks": r.to_dict('records'),
                "risk_passes": bool(r['Pass'].all()),
                "risk_violations": r[~r['Pass']].to_dict('records'),
                "risk_limits": {
                    "portfolio_limits": risk_config["portfolio_limits"],
                    "concentration_limits": risk_config["concentration_limits"],
                    "variance_limits": risk_config["variance_limits"]
                }
            },
            "beta_analysis": {
                "factor_beta_checks": f_b.to_dict('records'),
                "proxy_beta_checks": p_b.to_dict('records'),
                "factor_beta_passes": bool(f_b['pass'].all()),
                "proxy_beta_passes": bool(p_b['pass'].all()),
                "factor_beta_violations": f_b[~f_b['pass']].to_dict('records'),
                "proxy_beta_violations": p_b[~p_b['pass']].to_dict('records'),
            },
            "optimization_metadata": {
                "optimization_type": "maximum_return",
                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "portfolio_file": filepath,
                "original_weights": weights,
                "total_positions": len(w),
                "active_positions": len([v for v in w.values() if abs(v) > 0.001]),
                "expected_returns_used": config.get("expected_returns", {})
            },
            "formatted_report": formatted_report
        })
        
        return make_json_safe(result_data)
    else:
        # CLI MODE: Print formatted output
        print_max_return_report(weights=w, risk_tbl=r, df_factors=f_b, df_proxies=p_b)

def run_stock(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    factor_proxies: Optional[Dict[str, Union[str, List[str]]]] = None,
    yaml_path: Optional[str] = None, 
    *,
    return_data: bool = False
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
        yaml_path (Optional[str]): Path to YAML file for factor proxy lookup.
        return_data (bool): If True, returns structured data instead of printing.

    Returns:
        None or Dict[str, Any]: If return_data=False, returns None and prints formatted output.
                                If return_data=True, returns structured data dictionary.
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
            if not return_data:
                print(f"⚠️  Could not load proxies from YAML: {e}")

    # ─── 3. Diagnostics path A: multi-factor profile ────────────────────
    if factor_proxies:
        profile = get_detailed_stock_factor_profile(
            ticker, start, end, factor_proxies
        )
        
        if return_data:
            # API MODE: Return structured data (JSON-serializable)
            from core.result_objects import StockAnalysisResult
            
            # Create StockAnalysisResult object for formatted report
            stock_result = StockAnalysisResult.from_stock_analysis(
                ticker=ticker,
                vol_metrics=profile["vol_metrics"],
                regression_metrics=profile["regression_metrics"],
                factor_summary=profile["factor_summary"]
            )
            
            return make_json_safe({
                "ticker": ticker,
                "analysis_period": {
                    "start_date": start.strftime("%Y-%m-%d"),
                    "end_date": end.strftime("%Y-%m-%d")
                },
                "analysis_type": "multi_factor",
                "volatility_metrics": profile["vol_metrics"],
                "regression_metrics": profile["regression_metrics"],
                "factor_summary": profile["factor_summary"],
                "factor_proxies": factor_proxies,
                "analysis_metadata": {
                    "has_factor_analysis": True,
                    "num_factors": len(factor_proxies) if factor_proxies else 0,
                    "analysis_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                },
                "formatted_report": stock_result.to_formatted_report()
            })
        else:
            # CLI MODE: Print formatted output
            format_stock_metrics(profile["vol_metrics"], "Volatility Metrics")
            format_stock_metrics(profile["regression_metrics"], "Market Regression")
            
            print("=== Factor Summary ===")
            print(profile["factor_summary"])
        
    # ─── 4. Diagnostics path B: simple market regression ────────────────
    else:
        result = get_stock_risk_profile(
            ticker,
            start_date=start,
            end_date=end,
            benchmark="SPY"
        )
        
        if return_data:
            # API MODE: Return structured data (JSON-serializable)
            from core.result_objects import StockAnalysisResult
            
            # Create StockAnalysisResult object for formatted report  
            stock_result = StockAnalysisResult.from_stock_analysis(
                ticker=ticker,
                vol_metrics=result["vol_metrics"],
                regression_metrics=result["risk_metrics"],
                factor_summary=None
            )
            
            return make_json_safe({
                "ticker": ticker,
                "analysis_period": {
                    "start_date": start.strftime("%Y-%m-%d"),
                    "end_date": end.strftime("%Y-%m-%d")
                },
                "analysis_type": "simple_market_regression",
                "volatility_metrics": result["vol_metrics"],
                "risk_metrics": result["risk_metrics"],
                "benchmark": "SPY",
                "analysis_metadata": {
                    "has_factor_analysis": False,
                    "num_factors": 0,
                    "analysis_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                },
                "formatted_report": stock_result.to_formatted_report()
            })
        else:
            # CLI MODE: Print formatted output
            format_stock_metrics(result["vol_metrics"], "Volatility Metrics")
            format_stock_metrics(result["risk_metrics"], "Market Regression (SPY)")

def run_portfolio_performance(filepath: str, *, return_data: bool = False):
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
    return_data : bool, optional
        If True, return structured data for API usage. If False, print formatted output.

    Returns
    -------
    dict or None
        If return_data=True, returns structured performance metrics and metadata.
        If return_data=False, prints results and returns None.

    Notes
    -----
    * Uses the **start_date** and **end_date** from the portfolio YAML for
      the performance analysis window.
    * Defaults to SPY as benchmark but can be customized.
    * Uses 3-month Treasury rates from FMP API as risk-free rate.
    """
    from portfolio_risk import calculate_portfolio_performance_metrics
    from run_portfolio_risk import display_portfolio_performance_metrics
    
    try:
        # Load portfolio configuration
        config = load_portfolio_config(filepath)
        
        # Standardize portfolio weights  
        weights = standardize_portfolio_input(config["portfolio_input"], latest_price)["weights"]
        
        # Calculate performance metrics
        performance_metrics = calculate_portfolio_performance_metrics(
            weights=weights,
            start_date=config["start_date"],
            end_date=config["end_date"],
            benchmark_ticker="SPY"  # Could make this configurable later
        )
        
        # Check for calculation errors
        if "error" in performance_metrics:
            if return_data:
                # API MODE: Return error in structured format
                return make_json_safe({
                    "error": performance_metrics["error"],
                    "analysis_period": {
                        "start_date": config["start_date"],
                        "end_date": config["end_date"]
                    },
                    "portfolio_file": filepath,
                    "analysis_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                })
            else:
                # CLI MODE: Print error and return
                print(f"❌ Performance calculation failed: {performance_metrics['error']}")
                return
        
        if return_data:
            # API MODE: Return structured data (JSON-serializable)
            import io
            import sys
            
            # Capture CLI-style formatted output as string
            original_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()
            
            try:
                # Generate the formatted CLI output
                display_portfolio_performance_metrics(performance_metrics)
                formatted_report_string = captured_output.getvalue()
            finally:
                sys.stdout = original_stdout
            
            return make_json_safe({
                "performance_metrics": performance_metrics,
                "analysis_period": {
                    "start_date": config["start_date"],
                    "end_date": config["end_date"],
                    "years": performance_metrics["analysis_period"]["years"]
                },
                "portfolio_summary": {
                    "file": filepath,
                    "positions": len(weights),
                    "benchmark": "SPY"
                },
                "analysis_metadata": {
                    "analysis_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "calculation_successful": True
                },
                "formatted_report": formatted_report_string
            })
        else:
            # CLI MODE: Print formatted output (original behavior)
            print("📊 Portfolio Performance Analysis")
            print("=" * 50)
            
            print(f"📁 Portfolio file: {filepath}")
            print(f"📅 Analysis period: {config['start_date']} to {config['end_date']}")
            print(f"📊 Positions: {len(weights)}")
            print()
            
            print("🔄 Calculating performance metrics...")
            print("✅ Performance calculation successful!")
            
            # Display the results
            display_portfolio_performance_metrics(performance_metrics)
        
    except FileNotFoundError:
        if return_data:
            # API MODE: Return error in structured format
            return make_json_safe({
                "error": f"Portfolio file '{filepath}' not found",
                "portfolio_file": filepath,
                "analysis_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        else:
            # CLI MODE: Print error
            print(f"❌ Error: Portfolio file '{filepath}' not found")
    except Exception as e:
        if return_data:
            # API MODE: Return error in structured format
            return make_json_safe({
                "error": f"Error during performance analysis: {str(e)}",
                "portfolio_file": filepath,
                "analysis_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        else:
            # CLI MODE: Print error with traceback
            print(f"❌ Error during performance analysis: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--portfolio", type=str, help="Path to YAML portfolio file")
    parser.add_argument("--stock", type=str, help="Ticker symbol")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--yaml-path", type=str, help="Path to YAML file for factor proxy lookup (for detailed stock analysis)")
    parser.add_argument("--factor-proxies", type=str, help='JSON string of factor proxies, e.g. \'{"market": "SPY", "momentum": "MTUM"}\'')
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
        from proxy_builder import inject_all_proxies
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
        # Parse factor proxies from JSON string if provided
        factor_proxies = None
        if args.factor_proxies:
            try:
                import json
                factor_proxies = json.loads(args.factor_proxies)
            except json.JSONDecodeError as e:
                print(f"❌ Error parsing factor proxies JSON: {e}")
                print("   Example format: '{\"market\": \"SPY\", \"momentum\": \"MTUM\", \"value\": \"IWD\"}'")
                parser.print_help()
                exit(1)
        
        # Run stock analysis with all parameters
        run_stock(
            ticker=args.stock,
            start=args.start,
            end=args.end,
            factor_proxies=factor_proxies,
            yaml_path=args.yaml_path
        )
    
    else:
        parser.print_help()


# In[ ]:




