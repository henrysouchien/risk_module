# 🧠 Risk Module Architecture Documentation

This document provides a comprehensive overview of the Risk Module's architecture, design principles, and technical implementation details.

## 📋 Table of Contents

- [System Overview](#system-overview)
- [Architecture Layers](#architecture-layers)
- [Data Flow](#data-flow)
- [Component Details](#component-details)
- [Configuration Management](#configuration-management)
- [Caching Strategy](#caching-strategy)
- [Risk Calculation Framework](#risk-calculation-framework)
- [API Integration](#api-integration)
- [Performance Considerations](#performance-considerations)
- [Testing Strategy](#testing-strategy)
- [Future Enhancements](#future-enhancements)

## 🎯 System Overview

The Risk Module is a modular, stateless Python framework designed for comprehensive portfolio and single-stock risk analysis. It provides multi-factor regression diagnostics, risk decomposition, and portfolio optimization capabilities through a layered architecture that promotes maintainability, testability, and extensibility.

### Data Quality Assurance

The system includes robust data quality validation to prevent unstable factor calculations. A key improvement addresses the issue where insufficient peer data could cause extreme factor betas (e.g., -58.22 momentum beta) by limiting regression windows to only 2 observations instead of the full available data.

**Problem Solved**: The `filter_valid_tickers()` function now ensures that subindustry peers have ≥ target ticker's observations, preventing regression window limitations and ensuring stable factor betas.

### Core Design Principles

- **Modularity**: Each component has a single responsibility and clear interfaces
- **Statelessness**: Functions are pure and don't maintain internal state
- **Caching**: Intelligent caching at multiple levels for performance
- **Configuration-Driven**: YAML-based configuration for flexibility
- **Extensible**: Easy to add new factors, risk metrics, and data sources

## 🏗️ Architecture Layers

The system follows a layered architecture pattern with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    Presentation Layer                        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ run_portfolio_  │  │ run_single_     │  │ run_risk.py  │ │
│  │ risk.py         │  │ stock_profile.py│  │              │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                    Business Logic Layer                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ portfolio_risk. │  │ risk_summary.py │  │ factor_utils.│ │
│  │ py              │  │                 │  │ py           │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                    Data Access Layer                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ data_loader.py  │  │ helpers_input.  │  │ helpers_     │ │
│  │                 │  │ py              │  │ display.py   │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                    Configuration Layer                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ portfolio.yaml  │  │ risk_limits.    │  │ .env         │ │
│  │                 │  │ yaml            │  │              │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 📂 File Structure

```
risk_module/
├── 📄 README.md                    # Project documentation
├── 📄 architecture.md              # This file
├── ⚙️ settings.py                  # Default configuration settings
├── ⚙️ portfolio.yaml              # Portfolio configuration
├── ⚙️ risk_limits.yaml            # Risk limit definitions
├── 🔌 data_loader.py              # Data fetching and caching
├── 📊 factor_utils.py             # Factor analysis utilities
├── 💼 portfolio_risk.py           # Portfolio risk calculations
├── 📈 risk_summary.py             # Single-stock risk profiling
├── 🚀 run_portfolio_risk.py       # Portfolio analysis runner
├── 🎯 run_risk.py                 # Risk analysis runner
├── 🛠️ helpers_display.py          # Display utilities
├── 🛠️ helpers_input.py            # Input processing utilities
├── 🛠️ risk_helpers.py             # Risk calculation helpers
├── ⚡ portfolio_optimizer.py       # Portfolio optimization
├── 🤖 gpt_helpers.py              # GPT integration and peer generation
└── 📁 cache_prices/               # Cached price data (gitignored)
```

## 🔄 Data Flow

### Portfolio Analysis Flow

```
1. Configuration Loading
   portfolio.yaml → helpers_input.py → standardized portfolio data

2. Data Retrieval
   ticker list → data_loader.py → cached/API price data

3. Data Quality Validation
   peer groups → proxy_builder.py → filtered valid peers

4. Factor Analysis
   price data → factor_utils.py → factor returns and betas

5. Risk Calculation
   factor data + weights → portfolio_risk.py → risk metrics

6. Reporting
   risk metrics → helpers_display.py → formatted output
```

### Data Quality Validation Flow

```
1. Peer Generation
   GPT → generate_subindustry_peers() → candidate peer list

2. Individual Validation
   candidate peers → filter_valid_tickers() → peers with ≥3 observations

3. Peer Group Validation
   valid peers → filter_valid_tickers(target_ticker) → peers with ≥ target observations

4. Factor Calculation
   validated peers → fetch_peer_median_monthly_returns() → stable factor data
```

### Single Stock Analysis Flow

```
1. Stock Configuration
   stock + factor proxies → risk_summary.py → factor setup

2. Data Collection
   stock + proxy tickers → data_loader.py → price series

3. Regression Analysis
   price data → factor_utils.py → multi-factor regression

4. Risk Profiling
   regression results → risk_summary.py → risk profile
```

## 🔧 Component Details

### 1. Data Layer (`data_loader.py`)

**Purpose**: Efficient data retrieval with intelligent caching

**Key Functions**:
- `fetch_monthly_close()`: FMP API integration with caching
- `cache_read()`: Multi-level caching (RAM → Disk → Network)
- `cache_write()`: Force cache updates

**Features**:
- Automatic cache invalidation
- Compressed parquet storage
- MD5-based cache keys
- Error handling and retry logic

**Caching Strategy**:
```
RAM Cache (LRU) → Disk Cache (Parquet) → Network (FMP API)
```

### 2. Factor Analysis (`factor_utils.py`)

**Purpose**: Multi-factor regression and risk calculations

**Key Functions**:
- `compute_volatility()`: Rolling volatility calculations
- `compute_regression_metrics()`: Single-factor regression
- `compute_factor_metrics()`: Multi-factor regression
- `compute_stock_factor_betas()`: Factor exposure calculation
- `calc_factor_vols()`: Factor volatility estimation
- `calc_weighted_factor_variance()`: Portfolio factor variance

**Statistical Methods**:
- Ordinary Least Squares (OLS) regression
- Rolling window calculations
- Robust error handling
- R-squared and significance testing

### 3. Portfolio Risk Engine (`portfolio_risk.py`)

**Purpose**: Portfolio-level risk decomposition and analysis

**Key Functions**:
- `normalize_weights()`: Weight standardization
- `compute_portfolio_returns()`: Portfolio return calculation
- `compute_covariance_matrix()`: Risk matrix construction
- `compute_portfolio_volatility()`: Portfolio volatility
- `compute_risk_contributions()`: Risk attribution
- `compute_herfindahl()`: Concentration analysis
- `build_portfolio_view()`: Comprehensive risk summary

**Risk Metrics**:
- Portfolio volatility
- Factor exposures
- Risk contributions
- Variance decomposition
- Concentration measures

### 4. Single Stock Profiler (`risk_summary.py`)

**Purpose**: Individual stock risk analysis and factor profiling

**Key Functions**:
- `get_stock_risk_profile()`: Basic risk metrics
- `get_detailed_stock_factor_profile()`: Comprehensive analysis
- Factor regression diagnostics
- Peer comparison analysis

**Analysis Components**:
- Multi-factor regression
- Factor beta calculation
- Idiosyncratic risk estimation
- Factor contribution analysis

### 5. Data Quality Validation (`proxy_builder.py`)

**Purpose**: Ensures data quality and prevents unstable factor calculations

**Key Functions**:
- `filter_valid_tickers()`: Validates ticker data quality and peer group consistency
- `get_subindustry_peers_from_ticker()`: GPT-generated peer selection with validation
- `inject_subindustry_peers_into_yaml()`: Peer injection with quality checks

**Validation Criteria**:
- **Individual Ticker**: ≥3 price observations for returns calculation
- **Peer Group**: Each peer must have ≥ target ticker's observations
- **Regression Stability**: Prevents extreme factor betas from insufficient data
- **Automatic Filtering**: Removes problematic peers during proxy generation

**Benefits**:
- Prevents regression window limitations
- Ensures stable factor betas
- Maintains data consistency across factors
- Automatic quality control for GPT-generated peers

### 5. Execution Layer

**Portfolio Runner** (`run_portfolio_risk.py`):
- End-to-end portfolio analysis
- Configuration validation
- Error handling and reporting
- Output formatting

**Single Stock Runner** (`run_single_stock_profile.py`):
- Individual stock diagnostics
- Factor model validation
- Detailed regression analysis

**Risk Runner** (`run_risk.py`):
- Flexible risk analysis entry point
- What-if scenario testing
- Batch processing capabilities

## ⚙️ Configuration Management

### Default Settings (`settings.py`)

**Purpose**: Centralized default configuration management

**Structure**:
```python
PORTFOLIO_DEFAULTS = {
    "start_date": "2019-01-31",
    "end_date": "2025-06-27"
}
```

**Usage**:
- Provides sensible defaults for portfolio analysis
- Used when specific dates aren't provided in YAML configurations
- Centralizes configuration to avoid hardcoded values throughout the codebase
- Easy to modify for different analysis periods

**Integration**:
```python
from settings import PORTFOLIO_DEFAULTS

# Use defaults when not specified
start_date = config.get('start_date', PORTFOLIO_DEFAULTS['start_date'])
end_date = config.get('end_date', PORTFOLIO_DEFAULTS['end_date'])
```

### Portfolio Configuration (`portfolio.yaml`)

**Structure**:
```yaml
# Date Range
start_date: "2019-05-31"
end_date: "2024-03-31"

# Portfolio Positions
portfolio_input:
  TICKER: {weight: 0.XX}

# Expected Returns
expected_returns:
  TICKER: 0.XX

# Factor Proxies
stock_factor_proxies:
  TICKER:
    market: MARKET_PROXY
    momentum: MOMENTUM_PROXY
    value: VALUE_PROXY
    industry: INDUSTRY_PROXY
    subindustry: [PEER1, PEER2, PEER3]
```

**Validation Rules**:
- Weights must sum to 1.0
- All tickers must have factor proxies
- Date ranges must be valid
- Expected returns must be reasonable

### Risk Limits (`risk_limits.yaml`)

**Structure**:
```yaml
# Portfolio-Level Limits
portfolio_limits:
  max_volatility: 0.40
  max_loss: -0.25

# Concentration Limits
concentration_limits:
  max_single_stock_weight: 0.40

# Variance Attribution Limits
variance_limits:
  max_factor_contribution: 0.30
  max_market_contribution: 0.30
  max_industry_contribution: 0.30

# Factor Risk Limits
max_single_factor_loss: -0.10
```

## 💾 Caching Strategy

### Multi-Level Caching

1. **RAM Cache** (LRU):
   - Function-level caching with `@lru_cache`
   - Fastest access for frequently used data
   - Configurable cache size

2. **Disk Cache** (Parquet):
   - Compressed parquet files
   - Persistent across sessions
   - MD5-based cache keys
   - Automatic cleanup of corrupt files

3. **Network Cache** (FMP API):
   - Last resort for data retrieval
   - Rate limiting and error handling
   - Automatic retry logic

### Cache Key Strategy

```python
# Cache key components
key = [ticker, start_date, end_date, factor_type]
fname = f"{prefix}_{hash(key)}.parquet"
```

### Cache Invalidation

- Automatic invalidation on file corruption
- Manual invalidation through cache clearing
- Version-based invalidation for API changes

## 📊 Risk Calculation Framework

### Factor Model Structure

**Standard Factors**:
- Market Factor (SPY, ACWX)
- Momentum Factor (MTUM, IMTM)
- Value Factor (IWD, IVLU)
- Industry Factor (KCE, SOXX, XSW)
- Sub-industry Factor (Peer group)

**Factor Construction**:
1. Proxy selection based on stock characteristics
2. Return calculation and normalization
3. Factor correlation analysis
4. Beta calculation through regression

### Risk Decomposition

**Variance Attribution**:
```
Total Variance = Market Variance + Factor Variance + Idiosyncratic Variance
```

**Risk Contributions**:
```
Position Risk Contribution = Weight × Marginal Risk Contribution
```

**Concentration Measures**:
```
Herfindahl Index = Σ(Weight²)
```

## 🔌 API Integration

### Financial Modeling Prep (FMP)

**Endpoints Used**:
- `/historical-price-eod/full`: End-of-day price data
- Parameters: symbol, from, to, apikey, serietype

**Data Processing**:
- Monthly resampling to month-end
- Return calculation and normalization
- Missing data handling
- Outlier detection and treatment

**Error Handling**:
- Rate limiting compliance
- Network timeout handling
- API error response parsing
- Automatic retry with exponential backoff

## ⚡ Performance Considerations

### Optimization Strategies

1. **Caching**:
   - Multi-level caching reduces API calls
   - Compressed storage reduces disk usage
   - LRU cache optimizes memory usage

2. **Vectorization**:
   - NumPy operations for bulk calculations
   - Pandas vectorized operations
   - Efficient matrix operations

3. **Parallel Processing**:
   - Concurrent API calls where possible
   - Batch processing for multiple securities
   - Async/await for I/O operations

### Memory Management

- Lazy loading of large datasets
- Garbage collection optimization
- Memory-efficient data structures
- Streaming for large files

## 🧪 Testing Strategy

### Test Entry Points

1. **Portfolio Analysis**:
   ```bash
   python run_portfolio_risk.py
   ```

2. **Single Stock Profile**:
   ```bash
   python run_single_stock_profile.py
   ```

3. **Risk Runner**:
   ```bash
   python run_risk.py
   ```

### Validation Checks

- **Data Quality**: Missing data detection
- **Statistical Validity**: Regression diagnostics
- **Risk Limits**: Automated limit checking
- **Configuration**: YAML validation
- **Performance**: Execution time monitoring

### Error Handling

- Graceful degradation on API failures
- Comprehensive error messages
- Fallback strategies for missing data
- Logging for debugging and monitoring

## 🚀 Future Enhancements

### Planned Features

1. **Streamlit Dashboard**:
   - Interactive risk visualization
   - Real-time portfolio monitoring
   - Dynamic configuration updates

2. **GPT Integration**:
   - Automated peer suggestion
   - Natural language risk reports
   - Intelligent factor selection

3. **Advanced Risk Models**:
   - Conditional Value at Risk (CVaR)
   - Expected Shortfall
   - Tail risk measures

4. **Real-time Monitoring**:
   - Live data feeds
   - Alert system
   - Automated rebalancing

5. **Backtesting Framework**:
   - Historical performance analysis
   - Strategy comparison
   - Risk-adjusted returns

### Technical Improvements

1. **Performance**:
   - GPU acceleration for large portfolios
   - Distributed computing support
   - Real-time data streaming

2. **Extensibility**:
   - Plugin architecture
   - Custom factor models
   - Alternative data sources

3. **User Experience**:
   - Web-based interface
   - Mobile app support
   - API endpoints for integration

## 📈 Status by Module

| Layer | File/Function | Status | Notes |
|-------|---------------|--------|-------|
| Data Fetch | `fetch_monthly_close` | ✅ Working | FMP API integration complete |
| Return Calc | `calc_monthly_returns` | ✅ Complete | Merged into factor_utils |
| Volatility | `compute_volatility` | ✅ Complete | Rolling window implementation |
| Single-Factor Regression | `compute_regression_metrics` | ✅ Complete | OLS with diagnostics |
| Multi-Factor Betas | `compute_factor_metrics` | ✅ Working | Multi-factor regression |
| Factor Variance | `calc_factor_vols` | ✅ Complete | Factor volatility calculation |
| Portfolio Diagnostics | `build_portfolio_view` | ✅ Working | Comprehensive risk summary |
| Portfolio Input Parsing | `standardize_portfolio_input` | ✅ Working | YAML configuration support |
| Single Stock Profile | `get_detailed_stock_factor_profile` | ✅ Working | Individual stock analysis |
| YAML Config Support | `portfolio.yaml` | ✅ In Use | Flexible configuration |
| Risk Limits | `risk_limits.yaml` | ✅ Complete | Automated limit checking |
| Caching System | `data_loader.py` | ✅ Complete | Multi-level caching |
| Display Utils | `helpers_display.py` | ✅ Working | Formatted output |
| Input Utils | `helpers_input.py` | ✅ Working | Configuration parsing |

## 🔗 Dependencies

### Core Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **statsmodels**: Statistical modeling and regression
- **requests**: HTTP library for API calls
- **python-dotenv**: Environment variable management
- **pyarrow**: Parquet file handling for caching

### Optional Dependencies

- **cvxpy**: Portfolio optimization (future)
- **streamlit**: Web dashboard (future)
- **plotly**: Interactive visualizations (future)

## 📚 Additional Resources

- [README.md](./README.md): Project overview and usage guide
- [portfolio.yaml](./portfolio.yaml): Example portfolio configuration
- [risk_limits.yaml](./risk_limits.yaml): Risk limit definitions
- [Financial Modeling Prep API](https://financialmodelingprep.com/developer/docs/): API documentation

---

**Architecture Version**: 1.0  
**Last Updated**: 2024  
**Maintainer**: Henry Souchien