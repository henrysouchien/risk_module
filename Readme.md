# Risk Module 🧠

**Purpose**: To help people make better investment decisions by making portfolio risk understandable and actionable through AI-powered analysis and guidance.

A comprehensive portfolio and single-stock risk analysis system that provides multi-factor regression diagnostics, risk decomposition, and portfolio optimization capabilities.

## 🚀 Features

- **Multi-Factor Risk Analysis**: Understand how market forces affect your portfolio to make better allocation decisions
- **Portfolio Risk Decomposition**: See which positions drive your risk to know where to focus your risk management
- **Single-Stock Risk Profiles**: Analyze individual stocks to make informed buy/sell decisions
- **Data Caching**: Fast, reliable data access for consistent analysis
- **YAML Configuration**: Easy portfolio setup and risk limit management
- **Risk Limit Monitoring**: Get alerts when your portfolio exceeds your risk tolerance
- **Centralized Settings**: Consistent analysis across different portfolios

## 📊 What It Does

This risk module helps you make better investment decisions by:

1. **Portfolio Analysis**: Understanding your overall risk profile to make informed allocation decisions
2. **Single-Stock Diagnostics**: Evaluating individual stocks to make better buy/sell choices
3. **Risk Monitoring**: Staying within your risk tolerance to avoid unpleasant surprises
4. **Data Management**: Ensuring reliable, consistent analysis for confident decision-making
5. **Configuration Management**: Maintaining consistent risk parameters across your portfolios

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

The system generates actionable risk insights including:

- **Volatility Analysis**: Understand your portfolio's risk level to make informed allocation decisions
- **Factor Exposures**: See your market bets to decide if you want to hedge or adjust exposures
- **Risk Decomposition**: Identify what's driving your risk to focus your management efforts
- **Concentration Analysis**: Check your diversification to decide if you need more positions
- **Risk Limit Monitoring**: Stay within your comfort zone to avoid unpleasant surprises

### Single Stock Profile

Individual stock analysis helps you make better buy/sell decisions by providing:

- **Factor Regression**: Understand how market forces affect this stock to make informed decisions
- **Risk Metrics**: See the stock's risk profile to decide if it fits your portfolio
- **Factor Contributions**: Identify what's driving the stock's performance to assess its role
- **Peer Comparison**: Compare against similar stocks to make better relative value decisions

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

The system includes robust data quality validation to ensure you get reliable insights for confident decision-making:

- **Individual Ticker Validation**: Ensures each stock has enough data for accurate risk assessment
- **Peer Group Validation**: Prevents unreliable comparisons that could lead to bad decisions
- **Automatic Filtering**: Removes problematic data so you can trust the analysis
- **Stable Factor Betas**: Ensures risk metrics are reliable for making allocation decisions

This validation ensures that your risk analysis is trustworthy, so you can make decisions with confidence.

### Risk Limit Customization

Set your risk tolerance in `risk_limits.yaml` to get alerts when your portfolio exceeds your comfort zone:

```yaml
portfolio_limits:
  max_volatility: 0.35    # Maximum portfolio volatility you're comfortable with
  max_loss: -0.20         # Maximum loss you can tolerate

concentration_limits:
  max_single_stock_weight: 0.25  # Maximum single position size for diversification
```

## 📐 Mathematical Reference

### Portfolio Volatility & Risk

**Portfolio Volatility**
```
σ_p = √(w^T Σ w)
```
*Function: `compute_portfolio_volatility()`*
**Purpose**: Measures how much your portfolio can swing up or down, helping you understand if the risk level matches your comfort zone and timeline.

**Risk Contributions**
```
RC_i = w_i × (Σw)_i / σ_p
```
*Function: `compute_risk_contributions()`*
**Purpose**: Shows which positions are driving your portfolio's risk, helping you decide which stocks to reduce if you need to lower overall risk.

**Herfindahl Index (Concentration)**
```
H = Σ(w_i²)
```
*Function: `compute_herfindahl()`*
**Purpose**: Measures how diversified your portfolio is, helping you decide if you're over-concentrated in too few positions and need to add more holdings.

### Factor Analysis

**Factor Beta**
```
β_i,f = Cov(r_i, r_f) / Var(r_f)
```
*Function: `compute_stock_factor_betas()`*
**Purpose**: Shows how sensitive your stocks are to market forces (like tech sector moves or value vs growth trends), helping you understand if you're taking unintended bets.

**Portfolio Factor Beta**
```
β_p,f = Σ(w_i × β_i,f)
```
*Function: `build_portfolio_view()`*
**Purpose**: Shows your overall exposure to market factors, helping you decide if you want to hedge certain exposures or if you're comfortable with your current market bets.

**Excess Return**
```
r_excess = r_etf - r_market
```
*Function: `fetch_excess_return()`*
**Purpose**: Measures how much a factor (like momentum or value) moves independently of the market, helping you understand if you're getting compensated for taking factor-specific risk.

### Variance Decomposition

**Total Portfolio Variance**
```
σ²_p = σ²_factor + σ²_idiosyncratic
```
**Purpose**: Breaks down your portfolio's risk into what you can control (stock selection) vs. what you can't (market factors), helping you focus your risk management efforts.

**Factor Variance**
```
σ²_factor = Σ(w_i² × β_i,f² × σ_f²)
```
*Function: `compute_portfolio_variance_breakdown()`*
**Purpose**: Shows how much of your risk comes from market factors you can't control, helping you decide if you need to hedge or if you're comfortable with systematic risk exposure.

**Idiosyncratic Variance**
```
σ²_idio = Σ(w_i² × σ²_idio,i)
```
**Purpose**: Shows how much of your risk comes from individual stock choices, helping you decide if you need more diversification or if your stock selection is adding value.

**Euler Variance Contribution**
```
VC_i = w_i × (Σw)_i / Σ(w_i × (Σw)_i)
```
*Function: `compute_euler_variance_percent()`*
**Purpose**: Shows which positions contribute most to your portfolio's ups and downs, helping you identify where to focus your risk management efforts.

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
**Purpose**: Limits your exposure to market factors to prevent taking too much systematic risk.

**Weight Constraint**
```
0 ≤ w_i ≤ w_max
```
**Purpose**: Prevents over-concentration in single positions to maintain proper diversification.

## 🌐 Web Application

### Flask Web App (`app.py`)

The risk module includes a production-ready Flask web application that makes risk analysis accessible to anyone:

**Features:**
- **Portfolio Configuration**: Easy web-based setup for your portfolio
- **Risk Analysis**: Get actionable insights through a simple web interface
- **Rate Limiting**: Fair usage limits to ensure quality service for all users
- **API Key Management**: Secure access to protect your data
- **Usage Tracking**: Monitor your analysis history and trends
- **Export Functionality**: Download your risk reports for record-keeping

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

Automatically import your portfolio data from your brokerage accounts to save time and ensure accuracy:

**Features:**
- **Multi-Institution Support**: Connect all your brokerage accounts in one place
- **Automatic Holdings Import**: Get your current positions without manual entry
- **Cash Position Mapping**: Properly account for cash positions in your risk analysis
- **AWS Secrets Management**: Keep your financial data secure and private
- **Portfolio YAML Generation**: Convert your holdings to the right format automatically

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

The system includes several ways to test your portfolio analysis:

1. **Portfolio Analysis**: `python run_portfolio_risk.py` - Get a complete risk profile of your portfolio
2. **Single Stock Profile**: `python run_single_stock_profile.py` - Analyze individual stocks for buy/sell decisions
3. **Risk Runner**: `python run_risk.py` - Flexible risk analysis with different scenarios

Each provides actionable insights to help you make better investment decisions.

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

- [ ] AI-powered conversational interface for easier decision-making
- [X] GPT-powered peer suggestion system ✅ **Implemented**
- [X] Support for cash exposure and short positions ✅ **Implemented**
- [ ] Web dashboard interface (Flask app) 🔄 **In Development** - Figma UI design in progress
- [X] Plaid financial data integration ✅ **Implemented**
- [ ] Real-time risk monitoring and alerts
- [ ] AI-powered portfolio recommendations and what-if analysis
- [ ] Additional factor models (quality, size, etc.) for more comprehensive analysis
- [ ] Backtesting capabilities to validate investment decisions
- [ ] Interactive risk attribution visualization
- [ ] Portfolio comparison tools (current vs. suggested vs. historical)
- [ ] Advanced portfolio optimization with multiple objectives

---

**Built with ❤️ for quantitative risk analysis**