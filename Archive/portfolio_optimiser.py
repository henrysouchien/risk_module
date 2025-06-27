#!/usr/bin/env python
# coding: utf-8

# In[9]:


from typing import Dict, Any, Tuple
import cvxpy as cp
import pandas as pd
from copy import deepcopy

import pandas as pd

from portfolio_risk      import build_portfolio_view
from run_portfolio_risk  import (
    evaluate_portfolio_risk_limits,
    evaluate_portfolio_beta_limits,
)

from risk_helpers import (
    compute_max_betas,
    get_worst_monthly_factor_losses,
    aggregate_worst_losses_by_factor_type,
)

from helpers_display import _drop_factors


# In[10]:


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
    new_w = deepcopy(weights)
    for tkr, w in edits.items():
        new_w[tkr] = new_w.get(tkr, 0.0) + w

    # normalize
    tot = sum(new_w.values())
    new_w = {k: v / tot for k, v in new_w.items()}

    summary = build_portfolio_view(
        new_w, start, end, expected_returns=None, stock_factor_proxies=proxies
    )

    df_risk = evaluate_portfolio_risk_limits(
        summary,
        risk_cfg["portfolio_limits"],
        risk_cfg["concentration_limits"],
        risk_cfg["variance_limits"],
    )

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
    


# In[11]:


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
    *,           
    verbose: bool = True,
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
    verbose : bool, default **True**
    • **True**  → pretty-prints risk / beta tables (old behaviour).  
    • **False** → no console output; function only returns data frames.

    Returns
    -------
    summary : dict              # build_portfolio_view output
    risk_df : pd.DataFrame      # risk-limit check table
    beta_df : pd.DataFrame      # factor-beta check table
    """
    
    # 1) build new portfolio + tables
    summary, risk_df, beta_df = simulate_portfolio_change(
        base_weights, delta, risk_cfg,
        start_date, end_date, factor_proxies
    )

    # 2) optionally pretty-print
    if verbose:
        
        # --- fancy title --------------------------------------------------------
        delta_str = " / ".join(f"{v:+.0%} {k}" for k, v in delta.items())
        print(f"\n📐  What-if Risk Checks ({delta_str})\n")
    
        # --- risk table ---------------------------------------------------------
        pct = lambda x: f"{x:.1%}"
        print(risk_df.to_string(index=False,
                                formatters={"Actual": pct, "Limit": pct}))
    
        # --- beta table ---------------------------------------------------------
        print("\n📊  What-if Factor Betas\n")
        beta_df_disp = _drop_factors(beta_df)
    
        print(beta_df_disp.to_string(formatters={
            "portfolio_beta":    "{:.2f}".format,     # or "{:.2f}" if you prefer two decimals
            "max_allowed_beta":  "{:.2f}".format,
            "buffer":            "{:.2f}".format,
            "pass":              lambda x: "PASS" if x else "FAIL"
        }))

    return summary, risk_df, beta_df


# In[12]:


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


# In[15]:


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


# In[13]:


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


# In[14]:


# File: portfolio_optimizer.py

# ---------------------------------------------------------------------
# Max-return portfolio subject to risk limits
#   • Aggregate β caps: market, momentum, value
#   • Per-proxy β caps: one for each industry ETF / peer basket
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
    allow_short: bool = False,
) -> Dict[str, float]:
    r"""Return the weight vector *w* that maximises expected portfolio return
    subject to **all** firm-wide risk limits.

    The problem is formulated as a convex quadratic programme (QP)::

        maximise   \sum_i w_i * µ_i
        subject to  w  \in  ℝⁿ
                    \sum_i w_i                = 1                 (fully-invested)
                    0 ≤ w_i ≤ w_cap           ∀ i  (long-only + concentration cap)
                    σ_p(w)                    ≤ σ_cap             (annual vol cap)
                    |β_port,f(w)|             ≤ β_cap,f           f ∈ {market,momentum,value}
                    |β_port,proxy(w)|         ≤ β_cap,proxy       ∀ industry proxies

    where ::

        µ_i       expected annual return of ticker *i* (``expected_returns``)
        σ_p(w)    √(12 * wᵀ Σ_m w)   – annualised portfolio volatility
        β_port,f  ∑_i w_i β_{i,f}    – portfolio beta to factor *f*
        β_port,proxy  constructed similarly using each industry ETF / peer basket

    Parameters
    ----------
    init_weights
        Current portfolio weights (need not sum to 1 – they’ll be re-normalised).
    risk_cfg
        Parsed ``risk_limits.yaml`` containing the three sub-dicts:
        ``portfolio_limits``, ``concentration_limits``, ``variance_limits`` and
        the scalar ``max_single_factor_loss``.
    start_date, end_date
        Historical window (YYYY-MM-DD) used for covariance and beta estimation.
    stock_factor_proxies
        Mapping ``{ticker: {market: 'SPY', momentum: 'MTUM', value: 'IWD',
        industry: 'SOXX', …}}``.  Only *industry* proxies are used for the
        per-proxy caps, but the full dict is passed for completeness.
    expected_returns
        Dict of expected **annual** returns (in decimals, eg 0.12 = 12 %).
        Missing tickers default to 0.
    allow_short
        If ``True`` the lower-bound on *w* is removed (long/short optimisation).

    Returns
    -------
    Dict[str, float]
        Optimised weight vector summing exactly to 1.  Keys match the order of
        ``init_weights``.

    Raises
    ------
    ValueError
        * If *expected_returns* is empty / all zeros.
        * If the optimisation is infeasible under the supplied risk limits.

    Notes
    -----
    *Factor & proxy beta caps*
        The aggregate caps for **market**, **momentum** and **value** factors are
        taken from :pyfunc:`risk_helpers.compute_max_betas`.  Per-industry caps
        are derived by dividing the firm-wide *max_single_factor_loss* by each
        proxy’s historical worst 1-month return (see
        :pyfunc:`risk_helpers.get_worst_monthly_factor_losses`).
    """
    from portfolio_risk import build_portfolio_view          # reuse: get Σ & betas
    from risk_helpers   import compute_max_betas, get_worst_monthly_factor_losses

    # ---------- 0. Pre-compute Σ (monthly) & stock-level betas -------------
    tickers = list(init_weights)
    view = build_portfolio_view(
        init_weights, start_date, end_date,
        expected_returns=None,
        stock_factor_proxies=stock_factor_proxies,
    )

    Σ_m = view["covariance_matrix"].loc[tickers, tickers].values          # Σ (monthly)
    β_tbl = view["df_stock_betas"].fillna(0.0).loc[tickers]               # n × factors

    μ = np.array([expected_returns.get(t, 0.0) for t in tickers])
    if np.allclose(μ, 0):
        raise ValueError("expected_returns is empty or zeros – nothing to maximise")

    # ---------- 1. Build β caps -------------------------------------------
    # 1a) Aggregate factors
    all_caps = compute_max_betas(
        stock_factor_proxies,
        start_date, end_date,
        loss_limit_pct=risk_cfg["max_single_factor_loss"],
    )
    agg_caps = {k: all_caps[k] for k in ("market", "momentum", "value")}

    # 1b) Per-industry proxy caps
    loss_lim = risk_cfg["max_single_factor_loss"]            # e.g. -0.10
    worst_proxy_loss = get_worst_monthly_factor_losses(
        stock_factor_proxies, start_date, end_date
    )
    proxy_caps = {
        proxy: (np.inf if loss >= 0 else loss_lim / loss)
        for proxy, loss in worst_proxy_loss.items()
    }

    # Build coefficient vectors c_proxy (length n) such that
    #   β_port,proxy = Σ_i c_proxy[i] · w_i
    coeff_proxy: Dict[str, np.ndarray] = {}
    for proxy in proxy_caps:
        coeff = []
        for t in tickers:
            this_proxy = stock_factor_proxies[t].get("industry")
            coeff.append(β_tbl.loc[t, "industry"] if this_proxy == proxy else 0.0)
        coeff_proxy[proxy] = np.array(coeff)

    # ---------- 2. CVXPY variables & objective ----------------------------
    w = cp.Variable(len(tickers))
    objective = cp.Maximize(μ @ w)

    # ---------- 3. Constraints -------------------------------------------
    cons = [cp.sum(w) == 1]                               # fully invested
    if not allow_short:
        cons += [w >= 0]

    # single-name cap
    cons += [w <= risk_cfg["concentration_limits"]["max_single_stock_weight"]]

    # portfolio vol cap (monthly Σ → annual σ)
    σ_cap = risk_cfg["portfolio_limits"]["max_volatility"]
    cons += [cp.sqrt(cp.quad_form(w, Σ_m)) * np.sqrt(12) <= σ_cap]

    # 3a) Aggregate β caps
    for fac, cap in agg_caps.items():
        if fac in β_tbl.columns:
            cons += [cp.abs(β_tbl[fac].values @ w) <= cap]

    # 3b) Per-proxy β caps
    for proxy, cap in proxy_caps.items():
        cons += [cp.abs(coeff_proxy[proxy] @ w) <= cap]

    # ---------- 4. Solve --------------------------------------------------
    prob = cp.Problem(objective, cons)
    prob.solve(solver=cp.ECOS, qcp=True, verbose=False)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise ValueError(f"Solver status = {prob.status} (infeasible)")

    return {t: float(w.value[i]) for i, t in enumerate(tickers)}

