#!/usr/bin/env python3
"""
Portfolio Risk Score Module

A standalone module that calculates a "credit score" for portfolios (0-100)
based on multiple risk metrics including volatility, concentration, factor exposures,
and variance decomposition.

This module is independent and doesn't modify the existing codebase.
"""

import pandas as pd
import numpy as np
import yaml
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

# Import existing modules without modifying them
try:
    from portfolio_risk import build_portfolio_view
    from risk_helpers import calc_max_factor_betas
    from run_portfolio_risk import standardize_portfolio_input, latest_price
except ImportError:
    print("Warning: Could not import existing modules. Make sure you're in the risk_module directory.")
    build_portfolio_view = None
    calc_max_factor_betas = None
    standardize_portfolio_input = None
    latest_price = None


def calculate_portfolio_risk_score(
    summary: Dict[str, Any],
    portfolio_limits: Dict[str, float],
    concentration_limits: Dict[str, float],
    variance_limits: Dict[str, float],
    max_betas: Dict[str, float],
    portfolio_factor_betas: pd.Series,
    proxy_betas: Optional[Dict[str, float]] = None,
    max_proxy_betas: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Calculate a comprehensive risk score (0-100) for a portfolio, similar to a credit score.
    
    The score combines multiple risk metrics:
    - Portfolio volatility (30% weight)
    - Concentration risk (20% weight) 
    - Factor exposure risk (25% weight)
    - Variance decomposition risk (25% weight)
    
    Parameters
    ----------
    summary : Dict[str, Any]
        Output from build_portfolio_view()
    portfolio_limits : Dict[str, float]
        Portfolio-level risk limits
    concentration_limits : Dict[str, float]
        Concentration risk limits
    variance_limits : Dict[str, float]
        Variance decomposition limits
    max_betas : Dict[str, float]
        Maximum allowed factor betas
    portfolio_factor_betas : pd.Series
        Current portfolio factor betas
    proxy_betas : Optional[Dict[str, float]]
        Industry proxy betas
    max_proxy_betas : Optional[Dict[str, float]]
        Maximum allowed proxy betas
        
    Returns
    -------
    Dict[str, Any]
        Risk score details including:
        - 'score': Overall risk score (0-100)
        - 'category': Risk category (Excellent, Good, Fair, Poor, Very Poor)
        - 'component_scores': Individual component scores
        - 'risk_factors': Specific risk issues identified
        - 'recommendations': Suggested improvements
    """
    
    # Initialize component scores
    component_scores = {}
    risk_factors = []
    recommendations = []
    
    # ─── 1. Portfolio Volatility Score (30% weight) ────────────────────────────
    actual_vol = summary["volatility_annual"]
    vol_limit = portfolio_limits["max_volatility"]
    
    # Score based on how close to limit (0% = at limit, 100% = very low risk)
    vol_ratio = actual_vol / vol_limit
    if vol_ratio <= 0.5:
        vol_score = 100  # Very low risk
    elif vol_ratio <= 0.75:
        vol_score = 80   # Low risk
    elif vol_ratio <= 0.9:
        vol_score = 60   # Moderate risk
    elif vol_ratio <= 1.0:
        vol_score = 40   # High risk
    else:
        vol_score = 0    # Over limit
    
    component_scores["volatility"] = vol_score
    
    if vol_ratio > 0.9:
        risk_factors.append(f"High volatility ({actual_vol:.1%} vs {vol_limit:.1%} limit)")
        if vol_ratio > 1.0:
            recommendations.append("Reduce portfolio volatility by diversifying or reducing risky positions")
    
    # ─── 2. Concentration Risk Score (20% weight) ──────────────────────────────
    weights = summary["allocations"]["Portfolio Weight"]
    max_weight = weights.abs().max()
    weight_limit = concentration_limits["max_single_stock_weight"]
    herfindahl = summary["herfindahl"]
    
    # Score based on max weight and Herfindahl index
    weight_ratio = max_weight / weight_limit
    hhi_score = max(0, 100 - (herfindahl * 100))  # HHI of 0.1 = 90 score
    
    if weight_ratio <= 0.5:
        conc_score = min(100, hhi_score + 20)  # Bonus for good concentration
    elif weight_ratio <= 0.75:
        conc_score = hhi_score
    elif weight_ratio <= 0.9:
        conc_score = max(0, hhi_score - 20)
    elif weight_ratio <= 1.0:
        conc_score = max(0, hhi_score - 40)
    else:
        conc_score = 0
    
    component_scores["concentration"] = conc_score
    
    if weight_ratio > 0.75:
        risk_factors.append(f"High concentration: {max_weight:.1%} in single position")
        recommendations.append("Reduce position size in largest holdings")
    
    if herfindahl > 0.15:
        risk_factors.append(f"Low diversification (HHI: {herfindahl:.3f})")
        recommendations.append("Add more positions to improve diversification")
    
    # ─── 3. Factor Exposure Risk Score (25% weight) ────────────────────────────
    factor_scores = []
    skip_industry = proxy_betas is not None and max_proxy_betas is not None
    
    for factor, max_beta in max_betas.items():
        if skip_industry and factor == "industry":
            continue
            
        actual_beta = portfolio_factor_betas.get(factor, 0.0)
        beta_ratio = abs(actual_beta) / max_beta if max_beta > 0 else 0
        
        if beta_ratio <= 0.5:
            factor_score = 100
        elif beta_ratio <= 0.75:
            factor_score = 80
        elif beta_ratio <= 0.9:
            factor_score = 60
        elif beta_ratio <= 1.0:
            factor_score = 40
        else:
            factor_score = 0
            
        factor_scores.append(factor_score)
        
        if beta_ratio > 0.75:
            risk_factors.append(f"High {factor} exposure: β={actual_beta:.2f} vs {max_beta:.2f} limit")
            recommendations.append(f"Reduce {factor} factor exposure")
    
    # Check proxy-level exposures if available
    if proxy_betas and max_proxy_betas:
        for proxy, max_beta in max_proxy_betas.items():
            actual_beta = proxy_betas.get(proxy, 0.0)
            beta_ratio = abs(actual_beta) / max_beta if max_beta > 0 else 0
            
            if beta_ratio > 0.75:
                risk_factors.append(f"High {proxy} exposure: β={actual_beta:.2f} vs {max_beta:.2f} limit")
                recommendations.append(f"Reduce exposure to {proxy} sector")
    
    factor_score = np.mean(factor_scores) if factor_scores else 100
    component_scores["factor_exposure"] = factor_score
    
    # ─── 4. Systematic Risk Score (25% weight) ────────────────────────────────
    var_decomp = summary["variance_decomposition"]
    factor_pct = var_decomp["factor_pct"]
    market_pct = var_decomp["factor_breakdown_pct"].get("market", 0.0)
    
    # Score based on factor variance contribution
    factor_var_ratio = factor_pct / variance_limits["max_factor_contribution"]
    market_var_ratio = market_pct / variance_limits["max_market_contribution"]
    
    if factor_var_ratio <= 0.5:
        var_score = 100
    elif factor_var_ratio <= 0.75:
        var_score = 80
    elif factor_var_ratio <= 0.9:
        var_score = 60
    elif factor_var_ratio <= 1.0:
        var_score = 40
    else:
        # Linear penalty: 40 - 80 * (factor_var_ratio - 1.0)
        var_score = max(0, 40 - 80 * (factor_var_ratio - 1.0))
    
    # Adjust for market concentration
    if market_var_ratio > 0.75:
        var_score = max(0, var_score - 20)
        risk_factors.append(f"High market variance contribution: {market_pct:.1%}")
        recommendations.append("Reduce market factor exposure")
    
    component_scores["systematic_risk"] = var_score
    
    # ─── 5. Calculate Overall Score ────────────────────────────────────────────
    weights = {
        "volatility": 0.30,
        "concentration": 0.20,
        "factor_exposure": 0.25,
        "systematic_risk": 0.25
    }
    
    overall_score = sum(
        component_scores[component] * weight 
        for component, weight in weights.items()
    )
    
    # ─── 6. Determine Risk Category ────────────────────────────────────────────
    if overall_score >= 90:
        category = "Excellent"
    elif overall_score >= 80:
        category = "Good"
    elif overall_score >= 70:
        category = "Fair"
    elif overall_score >= 60:
        category = "Poor"
    else:
        category = "Very Poor"
    
    # ─── 6. Generate Score Interpretation ──────────────────────────────────────
    if overall_score >= 90:
        interpretation_summary = "Portfolio has very low risk profile"
        interpretation_details = [
            "Well diversified across multiple factors",
            "Conservative volatility and concentration levels", 
            "Suitable for risk-averse investors"
        ]
    elif overall_score >= 80:
        interpretation_summary = "Portfolio has acceptable risk profile"
        interpretation_details = [
            "Generally well-managed risk exposures",
            "Minor areas for improvement identified",
            "Suitable for most investors"
        ]
    elif overall_score >= 70:
        interpretation_summary = "Portfolio has moderate risk concerns"
        interpretation_details = [
            "Some risk factors need attention",
            "Consider implementing recommendations",
            "Monitor closely for changes"
        ]
    elif overall_score >= 60:
        interpretation_summary = "Portfolio has significant risk issues"
        interpretation_details = [
            "Multiple risk factors identified",
            "Immediate action recommended",
            "Consider reducing positions or rebalancing"
        ]
    else:
        interpretation_summary = "Portfolio has severe risk issues"
        interpretation_details = [
            "Critical risk factors present",
            "Immediate portfolio restructuring needed",
            "Consider professional risk management"
        ]

    return {
        "score": round(overall_score, 1),
        "category": category,
        "component_scores": {k: round(v, 1) for k, v in component_scores.items()},
        "risk_factors": risk_factors,
        "recommendations": recommendations,
        "interpretation": {
            "summary": interpretation_summary,
            "details": interpretation_details
        },
        "details": {
            "volatility": {
                "actual": actual_vol,
                "limit": vol_limit,
                "ratio": vol_ratio
            },
            "concentration": {
                "max_weight": max_weight,
                "weight_limit": weight_limit,
                "herfindahl": herfindahl
            },
            "factor_exposures": {
                "portfolio_betas": portfolio_factor_betas.to_dict(),
                "max_betas": max_betas
            }
        }
    }


def display_portfolio_risk_score(risk_score: Dict[str, Any]) -> None:
    """
    Display portfolio risk score in a user-friendly format similar to a credit score report.
    
    Parameters
    ----------
    risk_score : Dict[str, Any]
        Output from calculate_portfolio_risk_score()
    """
    score = risk_score["score"]
    category = risk_score["category"]
    component_scores = risk_score["component_scores"]
    risk_factors = risk_score["risk_factors"]
    recommendations = risk_score["recommendations"]
    
    # Color coding based on score
    if score >= 90:
        color = "🟢"  # Green
    elif score >= 80:
        color = "🟡"  # Yellow
    elif score >= 70:
        color = "🟠"  # Orange
    elif score >= 60:
        color = "🔴"  # Red
    else:
        color = "⚫"  # Black
    
    print(f"\n{'='*60}")
    print(f"📊 PORTFOLIO RISK SCORE")
    print(f"{'='*60}")
    print(f"{color} Overall Score: {score}/100 ({category})")
    print(f"{'='*60}")
    
    # Component breakdown
    print(f"\n📈 Component Scores:")
    print(f"{'─'*40}")
    for component, comp_score in component_scores.items():
        comp_color = "🟢" if comp_score >= 80 else "🟡" if comp_score >= 60 else "🔴"
        print(f"{comp_color} {component.replace('_', ' ').title():<25} {comp_score:>5.1f}/100")
    
    # Risk factors
    if risk_factors:
        print(f"\n⚠️  Risk Factors Identified:")
        print(f"{'─'*40}")
        for factor in risk_factors:
            print(f"   • {factor}")
    
    # Recommendations
    if recommendations:
        print(f"\n💡 Recommendations:")
        print(f"{'─'*40}")
        for rec in recommendations:
            print(f"   • {rec}")
    
    # Score interpretation
    print(f"\n📋 Score Interpretation:")
    print(f"{'─'*40}")
    if score >= 90:
        print("   🟢 EXCELLENT: Portfolio has very low risk profile")
        print("      • Well diversified across multiple factors")
        print("      • Conservative volatility and concentration levels")
        print("      • Suitable for risk-averse investors")
    elif score >= 80:
        print("   🟡 GOOD: Portfolio has acceptable risk profile")
        print("      • Generally well-managed risk exposures")
        print("      • Minor areas for improvement identified")
        print("      • Suitable for most investors")
    elif score >= 70:
        print("   🟠 FAIR: Portfolio has moderate risk concerns")
        print("      • Some risk factors need attention")
        print("      • Consider implementing recommendations")
        print("      • Monitor closely for changes")
    elif score >= 60:
        print("   🔴 POOR: Portfolio has significant risk issues")
        print("      • Multiple risk factors identified")
        print("      • Immediate action recommended")
        print("      • Consider reducing positions or rebalancing")
    else:
        print("   ⚫ VERY POOR: Portfolio has severe risk issues")
        print("      • Critical risk factors present")
        print("      • Immediate portfolio restructuring needed")
        print("      • Consider professional risk management")
    
    print(f"\n{'='*60}")


def run_risk_score_analysis(portfolio_yaml: str = "portfolio.yaml", risk_yaml: str = "risk_limits.yaml"):
    """
    Run a complete risk score analysis on a portfolio.
    
    Parameters
    ----------
    portfolio_yaml : str
        Path to portfolio configuration file
    risk_yaml : str
        Path to risk limits configuration file
    """
    if build_portfolio_view is None or calc_max_factor_betas is None or standardize_portfolio_input is None:
        print("Error: Required modules not available. Make sure you're in the risk_module directory.")
        return
    
    try:
        # Load configuration
        with open(portfolio_yaml, "r") as f:
            config = yaml.safe_load(f)
        
        with open(risk_yaml, "r") as f:
            risk_config = yaml.safe_load(f)
        
        print(f"Analyzing portfolio from {portfolio_yaml}...")
        
        # Standardize portfolio weights first
        raw_weights = config["portfolio_input"]
        standardized = standardize_portfolio_input(raw_weights, latest_price)
        weights = standardized["weights"]
        
        print(f"Portfolio summary:")
        print(f"  Total value: ${standardized['total_value']:,.2f}")
        print(f"  Net exposure: {standardized['net_exposure']:.3f}")
        print(f"  Leverage: {standardized['leverage']:.2f}x")
        
        # Build portfolio view with standardized weights
        summary = build_portfolio_view(
            weights=weights,
            start_date=config["start_date"],
            end_date=config["end_date"],
            expected_returns=config.get("expected_returns"),
            stock_factor_proxies=config.get("stock_factor_proxies")
        )
        
        # Calculate max betas
        max_betas, max_betas_by_proxy = calc_max_factor_betas(
            portfolio_yaml=portfolio_yaml,
            risk_yaml=risk_yaml,
            lookback_years=10,
            echo=False
        )
        
        # Calculate risk score
        risk_score = calculate_portfolio_risk_score(
            summary=summary,
            portfolio_limits=risk_config["portfolio_limits"],
            concentration_limits=risk_config["concentration_limits"],
            variance_limits=risk_config["variance_limits"],
            max_betas=max_betas,
            portfolio_factor_betas=summary["portfolio_factor_betas"],
            proxy_betas=summary["industry_variance"].get("per_industry_group_beta"),
            max_proxy_betas=max_betas_by_proxy
        )
        
        # Display results
        display_portfolio_risk_score(risk_score)
        
        # Return both risk score and raw portfolio analysis data
        return {
            "risk_score": risk_score,
            "portfolio_analysis": summary
        }
        
    except Exception as e:
        print(f"Error running risk score analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run the risk score analysis
    risk_score = run_risk_score_analysis()
    
    if risk_score:
        print(f"\n✅ Risk score analysis completed successfully!")
        print(f"   Score: {risk_score['risk_score']['score']}/100")
        print(f"   Category: {risk_score['risk_score']['category']}")
    else:
        print("\n❌ Risk score analysis failed. Check your configuration files.") 