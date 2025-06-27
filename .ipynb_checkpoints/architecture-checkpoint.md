🧠 Risk Engine Documentation

This folder contains the modular backend components for a portfolio and single-stock risk analysis system. It uses FMP data, runs multi-factor regression diagnostics, and computes full risk decomposition for a portfolio or stock.

⸻

📂 System Architecture

Each layer is modular, stateless, and fully scriptable. Data flows top-down and recomposes bottom-up for full analysis output.

1. 🔌 Data Layer
├── fetch_monthly_close() – from FMP
├── fetch_excess_return(), fetch_peer_median_monthly_returns()
└── calc_monthly_returns() – % monthly returns

2. 📊 Factor Utils (all analytics)
├── compute_volatility()
├── compute_regression_metrics()
├── compute_factor_metrics()
├── compute_stock_factor_betas()
├── calc_factor_vols()
└── calc_weighted_factor_variance()

3. 📈 Stock-Level Risk Engine
├── get_stock_risk_profile() – regression + vol vs. market
└── get_detailed_stock_factor_profile() – full multi-factor regression

4. 💼 Portfolio Risk Engine
├── normalize_weights(), compute_portfolio_returns()
├── compute_covariance_matrix(), compute_correlation_matrix()
├── compute_portfolio_volatility(), compute_risk_contributions()
├── compute_herfindahl()
├── compute_portfolio_variance_breakdown()
├── get_returns_dataframe()
├── compute_target_allocations()
└── build_portfolio_view() – full top-level summary

5. 🎯 Input Parsing / Execution
├── standardize_portfolio_input() – shares/dollars/weights
├── latest_price()
├── inputs.yaml – expected returns + proxies
├── run_portfolio_risk.py – main execution
└── run_single_stock_profile.py – one-off diagnostics


⸻

✅ Status by Module

Layer	File/Function	Status
Data Fetch	fetch_monthly_close	✅ working
Return Calc + Volatility	calc_monthly_returns, compute_volatility	✅ merged into factor_utils
Single-Factor Regression	compute_regression_metrics	✅ merged
Multi-Factor Betas	compute_factor_metrics	✅ working
Factor Variance	calc_factor_vols, calc_weighted_factor_variance	✅ complete
Portfolio Diagnostics	build_portfolio_view	✅ working
Portfolio Input Parsing	standardize_portfolio_input	✅ working
Single Stock Profile	get_detailed_stock_factor_profile	✅ working
YAML Config Support	inputs.yaml	✅ in use


⸻

🧪 Test Entry Points

1. Portfolio Analysis

python run_portfolio_risk.py

2. Single Stock Profile

python run_single_stock_profile.py

Each prints detailed diagnostics including volatility, factor exposure, idiosyncratic risk, and factor variance decomposition.

⸻

💡 Next Opportunities
	•	Extract build_portfolio_view internals into sub-functions
	•	Add Streamlit dashboard (optional)
	•	Hook up GPT/OpenAI to auto-suggest peers for subindustry
	•	Add support for cash exposure and short positions as risk flags