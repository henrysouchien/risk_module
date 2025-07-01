# Risk Module 🧠

A comprehensive portfolio and single-stock risk analysis system that provides multi-factor regression diagnostics, risk decomposition, and portfolio optimization capabilities.

## 🚀 Features

- **Multi-Factor Risk Analysis**: Comprehensive factor modeling with market, momentum, value, and industry factors
- **Portfolio Risk Decomposition**: Detailed variance attribution and risk contribution analysis
- **Single-Stock Risk Profiles**: Individual stock factor exposure and risk metrics
- **Data Caching**: Intelligent caching system for efficient data retrieval from Financial Modeling Prep (FMP)
- **YAML Configuration**: Flexible portfolio and risk limit configuration
- **Risk Limit Monitoring**: Automated risk limit checking and alerts
- **Centralized Settings**: Default configuration management

## 📊 What It Does

This risk module provides:

1. **Portfolio Analysis**: Complete risk decomposition including volatility, factor exposures, and variance attribution
2. **Single-Stock Diagnostics**: Detailed factor regression analysis for individual securities
3. **Risk Monitoring**: Automated checking against configurable risk limits
4. **Data Management**: Efficient caching and retrieval of market data from FMP API
5. **Configuration Management**: Centralized default settings and configuration

## 🏗️ Architecture

The system is built with a modular, layered architecture:

```
risk_module/
├── data_loader.py          # Data fetching and caching layer
├── factor_utils.py         # Factor analysis and regression utilities
├── portfolio_risk.py       # Portfolio-level risk calculations
├── risk_summary.py         # Single-stock risk profiling
├── run_portfolio_risk.py   # Main portfolio analysis execution
├── run_risk.py            # Risk analysis runner
├── settings.py            # Default configuration settings
├── portfolio.yaml         # Portfolio configuration
├── risk_limits.yaml       # Risk limit definitions
└── helpers_*.py           # Utility modules
```

### Core Components

- **Data Layer** (`data_loader.py`): FMP API integration with intelligent caching
- **Factor Utils** (`factor_utils.py`): Multi-factor regression and volatility calculations
- **Portfolio Engine** (`portfolio_risk.py`): Portfolio risk decomposition and analysis
- **Stock Profiler** (`risk_summary.py`): Individual stock factor exposure analysis
- **Settings Manager** (`settings.py`): Centralized default configuration

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- Financial Modeling Prep API key

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/henrysouchien/risk_module.git
   cd risk_module
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Or install manually:
   ```bash
   pip install pandas numpy statsmodels requests python-dotenv pyarrow streamlit pyyaml flask flask-limiter redis
   ```

3. **Configure API key**:
   Create a `.env` file in the project root:
   ```bash
   FMP_API_KEY=your_fmp_api_key_here
   ```

## 📖 Usage

### Portfolio Analysis

Run a complete portfolio risk analysis:

```bash
python run_portfolio_risk.py
```

This will:
- Load portfolio configuration from `portfolio.yaml`
- Fetch market data for all securities
- Perform multi-factor regression analysis
- Calculate portfolio risk metrics
- Generate comprehensive risk report

### Single Stock Analysis

Analyze individual stock risk profile:

```bash
python run_single_stock_profile.py
```

### Configuration

#### Default Settings (`settings.py`)

Centralized default configuration for the risk module:

```python
PORTFOLIO_DEFAULTS = {
    "start_date": "2019-01-31",
    "end_date": "2025-06-27"
}
```

These defaults are used when specific dates aren't provided in portfolio configurations.

#### Date Logic and Calculation Windows

The risk module uses a consistent date system across all calculations:

**Primary Portfolio System:**
- **Source**: Dates are read from `portfolio.yaml` (`start_date` and `end_date`)
- **Usage**: All portfolio risk calculations, factor regressions, and optimizations use this window
- **Consistency**: Ensures all calculations use the same historical period for accurate comparisons

**Fallback System:**
- **Default**: `PORTFOLIO_DEFAULTS` from `settings.py` when portfolio dates aren't specified
- **Proxy Generation**: GPT peer generation and validation use portfolio dates for consistency
- **Data Quality**: Peer filtering ensures all tickers have sufficient observations within the date window

**Independent Analysis:**
- **Single Stock**: `run_stock()` uses flexible dates (5-year default or explicit parameters)
- **Historical Analysis**: `calc_max_factor_betas()` uses 10-year lookback for worst-case scenarios
- **Purpose**: These functions serve different use cases and appropriately use different date logic

**Calculation Alignment:**
- All factor calculations (market, momentum, value, industry) use the same date window
- Peer median returns are calculated over the same period as the target ticker
- Regression windows are consistent across all securities in the portfolio
- Data quality validation ensures stable factor betas by preventing insufficient observation windows

#### Portfolio Configuration (`portfolio.yaml`)

```yaml
start_date: "2019-05-31"
end_date: "2024-03-31"

portfolio_input:
  TW:   {weight: 0.15}
  MSCI: {weight: 0.15}
  NVDA: {weight: 0.17}
  # ... more positions

expected_returns:
  TW: 0.15
  MSCI: 0.16
  # ... expected returns for each position

stock_factor_proxies:
  TW:
    market: SPY
    momentum: MTUM
    value: IWD
    industry: KCE
    subindustry: [TW, MSCI, NVDA]
  # ... factor proxies for each position
```

#### Risk Limits (`risk_limits.yaml`)

```yaml
portfolio_limits:
  max_volatility: 0.40
  max_loss: -0.25

concentration_limits:
  max_single_stock_weight: 0.40

variance_limits:
  max_factor_contribution: 0.30
  max_market_contribution: 0.30
  max_industry_contribution: 0.30
```

## 📈 Output Examples

### Portfolio Risk Summary

The system generates comprehensive risk reports including:

- **Volatility Analysis**: Portfolio and individual position volatility
- **Factor Exposures**: Market, momentum, value, and industry factor betas
- **Risk Decomposition**: Variance attribution by factor and position
- **Concentration Analysis**: Herfindahl index and position concentration
- **Risk Limit Monitoring**: Automated checking against defined limits

### Single Stock Profile

Individual stock analysis provides:

- **Factor Regression**: Multi-factor regression diagnostics
- **Risk Metrics**: Volatility, beta, and idiosyncratic risk
- **Factor Contributions**: Detailed factor exposure breakdown
- **Peer Comparison**: Relative performance vs. factor proxies

## 🔧 Advanced Usage

### Custom Factor Models

You can customize factor models by modifying the `stock_factor_proxies` section in `portfolio.yaml`:

```yaml
stock_factor_proxies:
  YOUR_STOCK:
    market: SPY          # Market factor proxy
    momentum: MTUM       # Momentum factor proxy
    value: IWD          # Value factor proxy
    industry: KCE       # Industry factor proxy
    subindustry: [PEER1, PEER2, PEER3]  # Sub-industry peers
```

### Data Quality Validation

The system includes robust data quality validation to ensure stable factor calculations:

- **Individual Ticker Validation**: Each ticker must have ≥3 price observations for returns calculation
- **Peer Group Validation**: Subindustry peers must have ≥ target ticker's observations to prevent regression window limitations
- **Automatic Filtering**: Problematic peers are automatically filtered out during proxy generation
- **Stable Factor Betas**: Prevents extreme factor betas caused by insufficient peer data

This validation ensures that factor regressions use the full available data window and produce stable, reliable results.

### Risk Limit Customization

Adjust risk limits in `risk_limits.yaml` to match your risk tolerance:

```yaml
portfolio_limits:
  max_volatility: 0.35    # Maximum portfolio volatility
  max_loss: -0.20         # Maximum expected loss

concentration_limits:
  max_single_stock_weight: 0.25  # Maximum single position weight
```

## 📐 Mathematical Reference

### Portfolio Volatility & Risk

**Portfolio Volatility**
```
σ_p = √(w^T Σ w)
```
*Function: `compute_portfolio_volatility()`*
Total portfolio risk measured as the square root of weighted covariance matrix.

**Risk Contributions**
```
RC_i = w_i × (Σw)_i / σ_p
```
*Function: `compute_risk_contributions()`*
Each asset's contribution to total portfolio volatility, showing which positions drive risk.

**Herfindahl Index (Concentration)**
```
H = Σ(w_i²)
```
*Function: `compute_herfindahl()`*
Portfolio concentration measure where 0 = fully diversified, 1 = single asset.

### Factor Analysis

**Factor Beta**
```
β_i,f = Cov(r_i, r_f) / Var(r_f)
```
*Function: `compute_stock_factor_betas()`*
Sensitivity of stock returns to factor returns, measured via linear regression.

**Portfolio Factor Beta**
```
β_p,f = Σ(w_i × β_i,f)
```
*Function: `build_portfolio_view()`*
Weighted average of individual stock betas, showing portfolio's factor exposure.

**Excess Return**
```
r_excess = r_etf - r_market
```
*Function: `fetch_excess_return()`*
Style factor returns relative to market benchmark, used for momentum and value factors.

### Variance Decomposition

**Total Portfolio Variance**
```
σ²_p = σ²_factor + σ²_idiosyncratic
```
Portfolio variance decomposed into systematic (factor) and unsystematic (idiosyncratic) components.

**Factor Variance**
```
σ²_factor = Σ(w_i² × β_i,f² × σ_f²)
```
*Function: `compute_portfolio_variance_breakdown()`*
Systematic risk contribution from factor exposures, weighted by position sizes and factor volatilities.

**Idiosyncratic Variance**
```
σ²_idio = Σ(w_i² × σ²_idio,i)
```
Unsystematic risk from individual stock-specific factors, diversifiable through position sizing.

**Euler Variance Contribution**
```
VC_i = w_i × (Σw)_i / Σ(w_i × (Σw)_i)
```
*Function: `compute_euler_variance_percent()`*
Marginal contribution of each asset to total portfolio variance, summing to 100%.

### Volatility Calculations

**Annualized Volatility**
```
σ_annual = σ_monthly × √12
```
*Function: `compute_volatility()`*
Monthly volatility scaled to annual basis using square root of time rule.

**Monthly Returns**
```
r_t = (P_t - P_{t-1}) / P_{t-1}
```
*Function: `calc_monthly_returns()`*
Percentage change in price from one month-end to the next.

### Optimization Constraints

**Portfolio Variance Constraint**
```
w^T Σ w ≤ σ_max²
```
Maximum allowable portfolio volatility constraint for risk management.

**Factor Beta Constraint**
```
|Σ(w_i × β_i,f)| ≤ β_max,f
```
Maximum allowable exposure to each factor, preventing excessive systematic risk.

**Weight Constraint**
```
0 ≤ w_i ≤ w_max
```
Individual position size limits for concentration risk management.

## 🌐 Web Application

### Flask Web App (`app.py`)

The risk module includes a production-ready Flask web application with:

**Features:**
- **Portfolio Configuration**: Web-based YAML editor for portfolio setup
- **Risk Analysis**: Execute portfolio risk analysis through web interface
- **Rate Limiting**: Tiered access with public/registered/paid user limits
- **API Key Management**: Secure key generation and validation
- **Usage Tracking**: Comprehensive logging and analytics
- **Export Functionality**: Download analysis results

**Access Tiers:**
- **Public**: Limited daily usage (5 analyses/day)
- **Registered**: Enhanced limits (15 analyses/day)
- **Paid**: Full access (30 analyses/day)

**Usage:**
```bash
# Start the web server
python app.py

# Access via browser
http://localhost:5000
```

## 🔗 Additional Integrations

### Plaid Financial Data Integration (`plaid_loader.py`)

Automated portfolio data import from financial institutions:

**Features:**
- **Multi-Institution Support**: Connect to multiple brokerage accounts
- **Automatic Holdings Import**: Fetch current positions and balances
- **Cash Position Mapping**: Convert cash to appropriate ETF proxies
- **AWS Secrets Management**: Secure storage of access tokens
- **Portfolio YAML Generation**: Automatic conversion to risk module format

**Supported Institutions:**
- Interactive Brokers
- Other Plaid-supported brokerages

**Usage:**
```python
from plaid_loader import convert_plaid_df_to_yaml_input

# Convert Plaid holdings to portfolio.yaml
convert_plaid_df_to_yaml_input(
    holdings_df,
    output_path="portfolio.yaml",
    dates={"start_date": "2020-01-01", "end_date": "2024-12-31"}
)
```

### Cash Position Mapping (`cash_map.yaml`)

Configuration for mapping cash positions to appropriate ETF proxies:

```yaml
proxy_by_currency:        # ETF proxy for each currency
  USD: SGOV
  EUR: ESTR
  GBP: IB01

alias_to_currency:        # Broker cash tickers → currency
  CUR:USD: USD            # Interactive Brokers
  USD CASH: USD
  CASH: USD               # Generic fallback
```

### Factor Proxy Configuration

**Industry Mapping (`industry_to_etf.yaml`):**
```yaml
"Technology": "XLK"
"Healthcare": "XLV"
"Financial Services": "XLF"
# ... more industry mappings
```

**Exchange-Specific Proxies (`exchange_etf_proxies.yaml`):**
```yaml
NASDAQ:
  market: "SPY"
  momentum: "MTUM"
  value: "IWD"
DEFAULT:
  market: "ACWX"
  momentum: "IMTM"
  value: "EFV"
```

## 🧪 Testing

The system includes several test entry points:

1. **Portfolio Analysis**: `python run_portfolio_risk.py`
2. **Single Stock Profile**: `python run_single_stock_profile.py`
3. **Risk Runner**: `python run_risk.py`

Each provides detailed diagnostics and risk metrics.

## 📁 Project Structure

```
risk_module/
├── README.md                 # This file
├── architecture.md          # Detailed architecture documentation
├── settings.py              # Default configuration settings
├── portfolio.yaml           # Portfolio configuration
├── risk_limits.yaml         # Risk limit definitions
├── data_loader.py           # Data fetching and caching
├── factor_utils.py          # Factor analysis utilities
├── portfolio_risk.py        # Portfolio risk calculations
├── risk_summary.py          # Single-stock risk profiling
├── run_portfolio_risk.py    # Portfolio analysis runner
├── run_risk.py             # Risk analysis runner
├── helpers_display.py       # Display utilities
├── helpers_input.py         # Input processing utilities
├── risk_helpers.py          # Risk calculation helpers
├── portfolio_optimizer.py   # Portfolio optimization
├── proxy_builder.py         # Factor proxy generation and GPT peer integration
├── gpt_helpers.py           # GPT integration and peer generation
├── plaid_loader.py          # Plaid financial data integration
├── app.py                   # Flask web application
├── cash_map.yaml            # Cash position mapping configuration
├── industry_to_etf.yaml     # Industry to ETF mapping
├── exchange_etf_proxies.yaml # Exchange-specific ETF proxies
├── what_if_portfolio.yaml   # What-if scenario configuration
├── run_risk_summary_to_gpt_dev.py # GPT interpretation runner
└── cache_prices/           # Cached price data (gitignored)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **statsmodels**: Statistical modeling and regression
- **requests**: HTTP library for API calls
- **python-dotenv**: Environment variable management
- **pyarrow**: Parquet file handling for caching
- **streamlit**: Web dashboard framework
- **pyyaml**: YAML configuration file handling
- **flask**: Web application framework
- **flask-limiter**: Rate limiting for web API
- **redis**: Caching and session management

## 🆘 Support

For questions or issues:

1. Check the `architecture.md` file for detailed technical documentation
2. Review the example configurations in `portfolio.yaml` and `risk_limits.yaml`
3. Open an issue on GitHub with detailed error information

## 🚀 Future Enhancements

- [ ] Streamlit dashboard integration
- [X] GPT-powered peer suggestion system ✅ **Implemented**
- [X] Support for cash exposure and short positions ✅ **Implemented**
- [ ] Web dashboard interface (Flask app) 🔄 **In Development** - Figma UI design in progress
- [X] Plaid financial data integration ✅ **Implemented**
- [ ] Real-time risk monitoring
- [ ] GPT-powered suggestions connected to what-if analysis
- [ ] Additional factor models (quality, size, etc.)
- [ ] Backtesting capabilities
- [ ] Risk attribution visualization
- [ ] Visualization to compare current vs. suggested vs. historical portfolios
- [ ] Advanced portfolio optimization features

---

**Built with ❤️ for quantitative risk analysis**