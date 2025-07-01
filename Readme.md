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
   pip install pandas numpy statsmodels requests python-dotenv pyarrow
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

### Risk Limit Customization

Adjust risk limits in `risk_limits.yaml` to match your risk tolerance:

```yaml
portfolio_limits:
  max_volatility: 0.35    # Maximum portfolio volatility
  max_loss: -0.20         # Maximum expected loss

concentration_limits:
  max_single_stock_weight: 0.25  # Maximum single position weight
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
├── gpt_helpers.py              # GPT integration and peer generation
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

## 🆘 Support

For questions or issues:

1. Check the `architecture.md` file for detailed technical documentation
2. Review the example configurations in `portfolio.yaml` and `risk_limits.yaml`
3. Open an issue on GitHub with detailed error information

## 🚀 Future Enhancements

- [ ] Streamlit dashboard integration
- [X] GPT-powered peer suggestion system ✅ **Implemented**
- [X] Support for cash exposure and short positions ✅ **Implemented**
- [ ] Real-time risk monitoring
- [ ] GPT-powered suggestions connected to what-if analysis
- [ ] Additional factor models (quality, size, etc.)
- [ ] Backtesting capabilities
- [ ] Risk attribution visualization
- [ ] Visuzalization to compare current vs. suggested vs. historical portfolios
- [ ] Web dashboard interface (Flask app in progress)
- [ ] Plaid financial data integration (in development)
- [ ] Advanced portfolio optimization features

---

**Built with ❤️ for quantitative risk analysis**