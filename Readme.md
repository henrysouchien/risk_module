Stock Risk Engine

A modular Python framework for calculating stock-level and portfolio-level risk metrics, factor exposures, and variance decomposition.

⸻

Contents
	•	data_loader.py – Pulls monthly price data from FMP API
	•	factor_utils.py – Factor construction, regressions, variance calcs
	•	portfolio_risk.py – Volatility, risk contributions, portfolio stats
	•	risk_summary.py – Single-stock and factor profile calculators
	•	run_single_stock_profile.py – Test script for stock-level analysis
	•	run_portfolio_risk.py – Main script to run portfolio risk diagnostics
	•	inputs.yaml – User-editable config for expected returns and factor proxies

⸻

Setup Instructions

1. Install Requirements

pip install pandas numpy statsmodels pyyaml requests

2. API Access

Register at FinancialModelingPrep and set your API key inside data_loader.py:

API_KEY = "<your_key_here>"


⸻

Input Format – inputs.yaml

A structured YAML file to store expected returns and factor mappings.

expected_returns:
  TW: 0.15
  MSCI: 0.16
  NVDA: 0.20
  PCTY: 0.17
  AT.L: 0.25
  SHV: 0.02

stock_factor_proxies:
  TW:
    market: SPY
    momentum: MTUM
    value: IWD
    industry: KCE
    subindustry: [TW, MSCI, NVDA]
  MSCI:
    market: SPY
    momentum: MTUM
    value: IWD
    industry: KCE
    subindustry: [TW, MSCI, NVDA]
  NVDA:
    market: SPY
    momentum: MTUM
    value: IWD
    industry: SOXX
    subindustry: [SOXX, XSW, IXC]
  PCTY:
    market: SPY
    momentum: MTUM
    value: IWD
    industry: XSW
    subindustry: [PAYC, CDAY, ADP]
  AT.L:
    market: ACWX
    momentum: IMTM
    value: IVLU
    industry: IXC
    subindustry: [IXC]
  SHV:
    market: SPY
    momentum: IMTM
    value: IWD
    industry: AGG
    subindustry: [SHY]


⸻

Example Output

📉 Portfolio Summary:

Portfolio Variance:          0.0198
Idiosyncratic Variance:      0.0118  (60%)
Factor Variance:             0.0080  (40%)

📈 Portfolio Factor Betas:

market         0.833
momentum      -0.213
value         -1.227
industry       0.583
subindustry    0.605

🧮 Per-Asset Risk Table:

Ticker	Vol A	Idio Vol A	Weighted Idio Var
TW	0.337	0.224	0.00113
MSCI	0.317	0.168	0.00064
NVDA	0.621	0.249	0.00179
PCTY	0.328	0.221	0.00110
AT.L	0.313	0.301	0.00711


⸻

Diagrams & Architecture

See docs/architecture.md for high-level design and modular breakdown.