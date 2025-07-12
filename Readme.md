# Risk Module 🧠

**Purpose**: To help people make better investment decisions by making portfolio risk understandable and actionable through AI-powered analysis and guidance.

A comprehensive portfolio and single-stock risk analysis system that provides multi-factor regression diagnostics, risk decomposition, and portfolio optimization capabilities through a **clean 3-layer architecture**.

## 🚀 Features

- **Multi-Factor Risk Analysis**: Understand how market forces affect your portfolio to make better allocation decisions
- **Portfolio Risk Decomposition**: See which positions drive your risk to know where to focus your risk management
- **Comprehensive Risk Scoring**: Credit-score-like rating (0-100) with detailed component analysis and historical stress testing
- **Portfolio Performance Analysis**: Calculate comprehensive performance metrics including returns, Sharpe ratio, alpha, beta, and maximum drawdown
- **Single-Stock Risk Profiles**: Analyze individual stocks to make informed buy/sell decisions
- **Risk Limit Monitoring**: Get alerts when your portfolio exceeds your risk tolerance with suggested limits
- **Data Caching**: Fast, reliable data access for consistent analysis
- **YAML Configuration**: Easy portfolio setup and risk limit management
- **Centralized Settings**: Consistent analysis across different portfolios

## 🏗️ Architecture Overview

### **NEW: Clean 3-Layer Architecture**

The Risk Module has been refactored from a monolithic structure into a clean, professional architecture:

```
┌─────────────────────────────────────────────────────┐
│                ROUTES LAYER                         │
│           (User Interface)                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │
│  │ CLI         │ │ API         │ │ AI Chat     │   │
│  │ run_risk.py │ │ routes/api  │ │ routes/claude│   │
│  └─────────────┘ └─────────────┘ └─────────────┘   │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│                CORE LAYER                           │
│         (Pure Business Logic)                       │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │
│  │ Portfolio   │ │ Stock       │ │ Optimization│   │
│  │ Analysis    │ │ Analysis    │ │ & Scenarios │   │
│  │ core/       │ │ core/       │ │ core/       │   │
│  └─────────────┘ └─────────────┘ └─────────────┘   │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│                DATA LAYER                           │
│         (Data Access & Storage)                     │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │
│  │ Risk Engine │ │ Factor      │ │ Data        │   │
│  │ portfolio_  │ │ Calculations│ │ Loading     │   │
│  │ risk.py     │ │ factor_utils│ │ data_loader │   │
│  └─────────────┘ └─────────────┘ └─────────────┘   │
└─────────────────────────────────────────────────────┘
```

### **Key Architectural Benefits**

✅ **Single Source of Truth**: All interfaces use the same core business logic  
✅ **Dual-Mode Functions**: Every function works in both CLI and API modes  
✅ **100% Backward Compatibility**: Existing code works identically  
✅ **Enterprise-Ready**: Professional architecture suitable for production  
✅ **Perfect Test Coverage**: All functionality thoroughly tested  

## 🌐 Current API Status

**✅ Complete API Coverage**: All major analysis functions available via REST API with dual output format:

| Analysis Type | CLI Command | API Endpoint | Core Module |
|---------------|-------------|--------------|-------------|
| **Portfolio Risk** | `run_risk.py --portfolio` | `POST /api/analyze` | `core/portfolio_analysis.py` |
| **Risk Score** | `portfolio_risk_score.py` | `POST /api/risk-score` | Built-in scoring engine |
| **Performance** | `run_risk.py --performance` | `POST /api/performance` | `core/performance_analysis.py` |
| **Stock Analysis** | `run_risk.py --stock` | `POST /api/stock` | `core/stock_analysis.py` |
| **What-If Scenarios** | `run_risk.py --what-if` | `POST /api/what-if` | `core/scenario_analysis.py` |
| **Optimization** | `run_risk.py --optimize` | `POST /api/optimize` | `core/optimization.py` |
| **AI Interpretation** | `run_risk.py --interpret` | `POST /api/interpret` | `core/interpretation.py` |

### **Dual-Mode Architecture**

Every function supports both CLI and API modes seamlessly:

```python
# CLI Mode - prints formatted output
result = run_portfolio("portfolio.yaml")

# API Mode - returns structured data
result = run_portfolio("portfolio.yaml", return_data=True)
```

**Testing Commands:**
```bash
# Start API server
python3 app.py

# Test all endpoints (shows both structured data + formatted reports)
python3 show_api_output.py analyze      # Complete portfolio analysis
python3 show_api_output.py risk-score   # Risk score analysis
python3 show_api_output.py performance  # Performance metrics
python3 show_api_output.py health       # API health check
```

## 🌐 Interface Layer

For web interface, REST API, and Claude AI chat integration, see:
- **[Interface README](docs/interfaces/INTERFACE_README.md)** - User guide for REST API, Claude chat, and web interface
- **[Interface Architecture](docs/interfaces/INTERFACE_ARCHITECTURE.md)** - Technical architecture of the interface layer

## 🤖 AI Assistant Guidelines

**For AI assistants helping users with this risk module:**

**Key Functions to Know:**
- `inject_all_proxies()` - Set up factor proxies for new portfolios (required before analysis)
- `run_portfolio()` - Full portfolio analysis with risk decomposition
- `run_portfolio_performance()` - Calculate comprehensive performance metrics (returns, Sharpe ratio, alpha, beta, max drawdown)
- `run_stock()` - Single stock analysis and factor exposure
- `run_what_if()` - Scenario testing for portfolio changes
- `run_min_variance()` - Lowest risk portfolio optimization
- `run_max_return()` - Maximum return portfolio optimization
- `run_risk_score_analysis()` - Complete risk score analysis with detailed reporting

**Common User Requests:**
- "Set up a new portfolio" → Use `inject_all_proxies()` first, then `run_portfolio()`
- "Analyze my portfolio risk" → Use `run_portfolio()` function
- "Calculate portfolio performance" → Use `run_portfolio_performance()` function
- "What's my portfolio's historical return?" → Use `run_portfolio_performance()` function
- "Show me Sharpe ratio and alpha" → Use `run_portfolio_performance()` function
- "Analyze a single stock" → Use `run_stock()` function
- "What if I reduce position X?" → Use `run_what_if()` function
- "Optimize for minimum risk" → Use `run_min_variance()` function
- "Optimize for maximum return" → Use `run_max_return()` function
- "What's my risk score?" → Use `run_risk_score_analysis()` for complete analysis
- "Why is my risk score low?" → Check component breakdown in risk score analysis
- "How do I improve my portfolio?" → Review risk score recommendations and suggested limits
- "What are my risk limits?" → Use `run_risk_score_analysis()` for suggested limits

**Key Configuration Files:**
- `portfolio.yaml` - Portfolio weights and factor proxies
- `risk_limits.yaml` - Risk tolerance settings
- `settings.py` - Default parameters

**Risk Score Interpretation:**
The risk score measures "disruption risk" - how likely your portfolio is to exceed your maximum acceptable loss in various failure scenarios.

- **90-100 (Excellent)**: Very low disruption risk - all potential losses well within limits
- **80-89 (Good)**: Acceptable disruption risk - minor risk management improvements recommended
- **70-79 (Fair)**: Moderate disruption risk - some potential losses exceed limits, improvements needed
- **60-69 (Poor)**: High disruption risk - multiple losses exceed limits, significant action required
- **0-59 (Very Poor)**: Portfolio needs immediate restructuring to avoid unacceptable losses

**Component Scores:**
- **Factor Risk (35%)**: Market/value/momentum exposure vs. historical worst losses
- **Concentration Risk (30%)**: Position sizes and diversification vs. single-stock failure scenarios
- **Volatility Risk (20%)**: Portfolio volatility level vs. maximum reasonable volatility
- **Sector Risk (15%)**: Sector concentration vs. historical sector crashes

**Common Issues:**
- NaN values → Insufficient data for peer analysis
- High systematic risk → Factor variance > 30% limit
- Low risk score → Check concentration and factor exposures
- Optimization infeasible → Risk limits too restrictive

**Function Parameters:**
- `run_portfolio(filepath)` - Full portfolio analysis from YAML file
- `run_portfolio_performance(filepath)` - Calculate performance metrics from YAML file
- `run_stock(ticker, start=None, end=None, factor_proxies=None, yaml_path=None)` - Analyze single stock
- `run_what_if(filepath, scenario_yaml=None, delta=None)` - Test portfolio modifications
- `run_min_variance(filepath)` - Optimize for minimum risk from YAML file
- `run_max_return(filepath)` - Optimize for maximum return from YAML file
- `run_risk_score_analysis(portfolio_yaml="portfolio.yaml", risk_yaml="risk_limits.yaml")` - Complete risk analysis score
- `inject_all_proxies(yaml_path="portfolio.yaml", use_gpt_subindustry=False)` - Set up factor proxies

## 📊 What It Does

This risk module helps you make better investment decisions by:

1. **Portfolio Analysis**: Understanding your overall risk profile to make informed allocation decisions
2. **Portfolio Performance Analysis**: Evaluating your historical returns, risk-adjusted performance, and alpha generation
3. **Single-Stock Diagnostics**: Evaluating individual stocks to make better buy/sell choices
4. **Risk Monitoring**: Staying within your risk tolerance to avoid unpleasant surprises
5. **Data Management**: Ensuring reliable, consistent analysis for confident decision-making
6. **Configuration Management**: Maintaining consistent risk parameters across your portfolios

## 🏗️ Core Business Logic

The system's business logic has been extracted into dedicated core modules:

### **Core Modules**

- **Portfolio Analysis** (`core/portfolio_analysis.py`): Pure portfolio risk analysis logic
- **Stock Analysis** (`core/stock_analysis.py`): Individual stock factor exposure and risk analysis
- **Scenario Analysis** (`core/scenario_analysis.py`): What-if scenario testing logic
- **Optimization** (`core/optimization.py`): Portfolio optimization algorithms
- **Performance Analysis** (`core/performance_analysis.py`): Performance metrics calculation
- **Interpretation** (`core/interpretation.py`): AI-powered analysis interpretation
- **Utilities** (`utils/serialization.py`): Data serialization and formatting

### **Data Processing Components**

- **Risk Analysis Engine** (`portfolio_risk.py`): Portfolio risk calculations, performance metrics, and factor decomposition
- **Risk Scoring System** (`portfolio_risk_score.py`): Comprehensive 0-100 risk scoring with historical stress testing and disruption risk analysis
- **Factor Analytics** (`factor_utils.py`): Multi-factor regression, volatility calculations, and factor exposure analysis
- **Portfolio Optimizer** (`portfolio_optimizer.py`): Minimum variance and maximum return optimization with risk constraints
- **Stock Profiler** (`risk_summary.py`): Individual stock factor exposure and risk analysis
- **Data Integration** (`data_loader.py`): FMP API integration with intelligent caching and Treasury rate data
- **Plaid Integration** (`plaid_loader.py`): Automatic portfolio import from brokerage accounts
- **Proxy Builder** (`proxy_builder.py`): Automated factor proxy generation with GPT-powered peer analysis
- **Web Application** (`app.py`): Production Flask app with OAuth, API endpoints, and AI chat integration

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

### Quick Start

1. **Set up a new portfolio**:
   ```python
   from proxy_builder import inject_all_proxies
   from run_risk import run_portfolio
   
   # Set up factor proxies for new portfolio
   inject_all_proxies("portfolio.yaml", use_gpt_subindustry=True)
   
   # Run full portfolio analysis
   run_portfolio("portfolio.yaml")
   ```

2. **Analyze existing portfolio**:
   ```python
   from run_risk import run_portfolio
   
   # Run analysis on existing portfolio.yaml
   run_portfolio("portfolio.yaml")
   ```

3. **Calculate portfolio performance**:
   ```python
   from run_risk import run_portfolio_performance
   
   # Calculate comprehensive performance metrics
   run_portfolio_performance("portfolio.yaml")
   ```

4. **API Mode - Get structured data**:
   ```python
   from run_risk import run_portfolio
   
   # Get structured data for API consumption
   result = run_portfolio("portfolio.yaml", return_data=True)
   
   # Access specific metrics
   portfolio_risk = result["portfolio_summary"]["volatility_annual"]
   factor_betas = result["portfolio_summary"]["portfolio_factor_betas"]
   ```

### Command-Line Interface

The system provides a unified command-line interface for all analysis types:

```bash
# Portfolio risk analysis
python run_risk.py --portfolio portfolio.yaml

# Portfolio performance analysis
python run_risk.py --portfolio portfolio.yaml --performance

# Comprehensive risk score analysis (0-100 with detailed reporting)
python -c "from portfolio_risk_score import run_risk_score_analysis; run_risk_score_analysis()"

# Single stock analysis
python run_risk.py --stock AAPL

# What-if scenario analysis
python run_risk.py --portfolio portfolio.yaml --whatif

# Portfolio optimization
python run_risk.py --portfolio portfolio.yaml --minvar    # Minimum variance
python run_risk.py --portfolio portfolio.yaml --maxreturn # Maximum return
```

### Portfolio Analysis

Run a complete portfolio risk analysis:

```python
from run_risk import run_portfolio

# Full portfolio analysis with risk decomposition
run_portfolio("portfolio.yaml")
```

This will:
- Load portfolio configuration from `portfolio.yaml`
- Fetch market data for all securities
- Perform multi-factor regression analysis
- Calculate portfolio risk metrics
- Generate comprehensive risk report

### Comprehensive Risk Score Analysis

Get a credit-score-like rating (0-100) for your portfolio with detailed analysis:

```python
from portfolio_risk_score import run_risk_score_analysis

# Complete risk score analysis with detailed reporting
results = run_risk_score_analysis("portfolio.yaml", "risk_limits.yaml")

# The analysis provides:
# - Overall risk score (0-100) with category (Excellent/Good/Fair/Poor/Very Poor)
# - Component scores for factor, concentration, volatility, and sector risks
# - Detailed risk limit violations and specific recommendations
# - Suggested risk limits based on your loss tolerance
# - Historical worst-case scenario analysis
```

This comprehensive analysis will:
- Calculate your portfolio's disruption risk vs. your maximum acceptable loss
- Break down risk into four components: factor (35%), concentration (30%), volatility (20%), sector (15%)
- Identify specific risk limit violations with actionable recommendations
- Suggest appropriate risk limits based on your risk tolerance
- Use historical worst-case scenarios for realistic stress testing
- Provide color-coded, credit-score-like reporting for easy interpretation

### Portfolio Performance Analysis

Calculate comprehensive performance metrics:

```python
from run_risk import run_portfolio_performance

# Calculate performance metrics (returns, Sharpe ratio, alpha, beta, max drawdown)
run_portfolio_performance("portfolio.yaml")
```

This will:
- Load portfolio configuration from the YAML file
- Calculate historical returns and volatility
- Compute risk-adjusted performance metrics (Sharpe ratio, Sortino ratio)
- Analyze performance vs benchmark (alpha, beta, tracking error)
- Display professional performance report with insights

### Single Stock Analysis

Analyze individual stock risk profile:

```python
from run_risk import run_stock

# Single stock analysis
run_stock("AAPL")  # Analyze Apple stock
```

### New Portfolio Setup

For new portfolios, set up factor proxies first:

```python
from proxy_builder import inject_all_proxies

# Set up factor proxies (required for new portfolios)
inject_all_proxies(use_gpt_subindustry=True)
```

### Configuration

#### Default Settings (`settings.py`)

Centralized default configuration for the risk module:

```python
PORTFOLIO_DEFAULTS = {
    "start_date": "2019-01-31",
    "end_date": "2025-06-27",
    "normalize_weights": False,  # Global default for portfolio weight normalization
    "worst_case_lookback_years": 10  # Historical lookback period for worst-case scenario analysis
}
```

These defaults are used when specific dates aren't provided in portfolio configurations.

### Weight Normalization Behavior

**Default**: `normalize_weights = False` (Raw weights represent true economic exposure)

**Key Behavior**:
- **Risk Analysis**: Uses raw weights to calculate true portfolio risk exposure
- **Portfolio Display**: Shows "Raw Weights" when `False`, "Normalized Weights" when `True`
- **Leverage Calculation**: No double-counting of leverage when using raw weights
- **Optimization Functions**: Always normalize weights internally for mathematical stability

**When to Use Each Setting**:
- **`normalize_weights = False`** (Default): When portfolio weights represent actual dollar exposures, including leverage and short positions
- **`normalize_weights = True`**: When portfolio weights need to be normalized to sum to 1.0 for display or analysis purposes

**Example**:
```python
# Raw weights (default) - represents true economic exposure
portfolio = {
    "NVDA": 0.50,   # 50% long position
    "SGOV": -0.08,  # 8% short position (creates leverage)
    "SPY": 0.30     # 30% long position
}
# Net: 72%, Gross: 88%, Leverage: 1.22x

# Risk calculations use these raw weights directly
# No leverage double-counting in risk calculations
```

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
- **Historical Analysis**: `calc_max_factor_betas()` uses `worst_case_lookback_years` (default: 10 years) for worst-case scenarios
- **Purpose**: These functions serve different use cases and appropriately use different date logic

**Historical Worst-Case Analysis:**
- **Lookback Period**: Uses `worst_case_lookback_years` setting (default: 10 years) ending today
- **Purpose**: Calculates maximum allowable risk exposure based on historical worst-case factor performance
- **Independence**: This analysis window is separate from portfolio analysis dates to ensure consistent risk benchmarks

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

### Comprehensive Risk Score Report

The comprehensive risk score analysis provides a credit-score-like report with:

- **Overall Risk Score (0-100)**: Single number measuring disruption risk with clear category (Excellent/Good/Fair/Poor/Very Poor)
- **Component Breakdown**: Detailed scores for factor risk (35%), concentration risk (30%), volatility risk (20%), and sector risk (15%)
- **Risk Interpretation**: Color-coded assessment with specific explanations of what each score means
- **Actionable Recommendations**: Specific suggestions for improving your portfolio risk profile
- **Limit Violations**: Detailed analysis of which risk limits are being exceeded and by how much
- **Suggested Risk Limits**: Backwards-calculated appropriate limits based on your maximum acceptable loss
- **Historical Context**: Stress testing based on actual historical worst-case scenarios

### Portfolio Risk Summary

The system generates actionable risk insights including:

- **Volatility Analysis**: Understand your portfolio's risk level to make informed allocation decisions
- **Factor Exposures**: See your market bets to decide if you want to hedge or adjust exposures
- **Risk Decomposition**: Identify what's driving your risk to focus your management efforts
- **Concentration Analysis**: Check your diversification to decide if you need more positions
- **Risk Limit Monitoring**: Stay within your comfort zone to avoid unpleasant surprises

### Portfolio Performance Summary

The system generates comprehensive performance metrics to help you evaluate your investment strategy:

- **Return Analysis**: Total returns, annualized returns, and monthly performance tracking
- **Risk-Adjusted Metrics**: Sharpe ratio, Sortino ratio, and Information ratio for risk-adjusted performance evaluation
- **Benchmark Comparison**: Alpha, beta, and tracking error analysis vs SPY benchmark
- **Drawdown Analysis**: Maximum drawdown, recovery periods, and downside risk assessment
- **Win Rate Analysis**: Percentage of positive months and best/worst performance periods
- **Professional Insights**: Automated performance quality assessment and recommendations

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

## 🛡️ Risk Management Framework

### Risk Limits Framework

The risk management system uses a comprehensive framework of limits designed to control portfolio risk across multiple dimensions. These limits are based on fundamental risk management principles and protect against various types of portfolio losses.

#### Core Risk Limit Categories

**1. Overall Portfolio Risk Limits**

- **Volatility Limit (40%)**: Control total portfolio risk and align with risk tolerance
- **Loss Limit (-25%)**: Set maximum acceptable portfolio loss and define risk budget

**2. Concentration Risk Limits**

- **Single-Stock Weight Limit (40%)**: "Limit risk from errors" - prevent single-stock blowups
- **Herfindahl Index Monitoring**: Measure portfolio concentration (target: < 0.15)

**3. Factor Risk Exposure Limits**

- **Factor Variance Contribution (30%)**: Control systematic risk exposure
- **Market Variance Contribution (50%)**: Control market beta exposure
- **Industry Variance Contribution (30%)**: Prevent sector concentration

**4. Factor Beta Limits**

- **Market Beta Limit (0.77)**: Derived from loss limit ÷ worst market loss
- **Momentum Beta Limit (0.79)**: Control momentum factor exposure
- **Value Beta Limit (0.55)**: Control value factor exposure
- **Industry Beta Limits (Varies)**: Each industry has different volatility characteristics

#### Risk Limit Setting Process

1. **Define Risk Tolerance**: Set maximum acceptable portfolio loss and volatility tolerance
2. **Calculate Factor Limits**: Analyze historical factor performance and derive beta limits
3. **Set Variance Limits**: Determine acceptable systematic risk exposure
4. **Monitor and Adjust**: Regularly review limit breaches and adjust as needed

#### Interpreting Limit Breaches

- **PASS**: Portfolio meets risk limit
- **FAIL**: Portfolio exceeds risk limit, action required
- **Risk Score Impact**: Limit breaches reduce overall risk score
- **Action Recommendations**: Specific guidance for each type of breach

### Risk Scoring Framework

The portfolio risk score provides a comprehensive 0-100 rating that evaluates portfolio risk across multiple dimensions. Similar to a credit score, it combines various risk metrics into a single actionable number with clear categories and recommendations.

#### Risk Score Components

The risk score is calculated using four main components:

1. **Portfolio Volatility (30% weight)**: Based on actual volatility vs. limit
2. **Concentration Risk (20% weight)**: Based on max weight and Herfindahl Index
3. **Factor Exposure Risk (25% weight)**: Average score across all factor betas
4. **Systematic Risk (25% weight)**: Based on factor variance contribution vs. limit

#### Risk Score Categories

| Score Range | Category | Description | Action Required |
|-------------|----------|-------------|-----------------|
| **90-100** | **Excellent** | Very low risk portfolio | Monitor and maintain |
| **80-89** | **Good** | Low risk portfolio | Minor adjustments may be needed |
| **70-79** | **Fair** | Moderate risk portfolio | Consider risk reduction |
| **60-69** | **Poor** | High risk portfolio | Significant action recommended |
| **0-59** | **Very Poor** | Very high risk portfolio | Immediate action required |

#### Scoring Methodology

**Portfolio Volatility Score (30%)**
- Score 100: Volatility ≤ 50% of limit (very low risk)
- Score 80: Volatility ≤ 75% of limit (low risk)
- Score 60: Volatility ≤ 90% of limit (moderate risk)
- Score 40: Volatility ≤ 100% of limit (high risk)
- Score 0: Volatility > limit (over limit)

**Concentration Risk Score (20%)**
- **Max Weight Component**: Based on largest position vs. limit
- **Herfindahl Index Component**: Score = max(0, 100 - (HHI × 100))
- **Combined Score**: Weighted average with bonus for good concentration

**Factor Exposure Risk Score (25%)**
- **Calculation**: Average score across all factor betas
- **Factors Evaluated**: Market, momentum, value, and industry betas
- **Scoring**: Based on beta vs. respective limits

**Systematic Risk Score (25%)**
- **Calculation**: Based on factor variance contribution vs. limit
- **Linear Penalty**: For exceeding limits using formula: `max(0, 40 - 80 × (factor_var_ratio - 1.0))`

#### Action Recommendations

**For Poor/Very Poor Scores (0-69)**
- Identify primary risk factors from score breakdown
- Reduce largest positions if concentration is high
- Add defensive positions if volatility is high
- Diversify sectors if systematic risk is high
- Consider portfolio optimization (min variance or max return)

**For Fair Scores (70-79)**
- Target specific weaknesses from component scores
- Gradual position adjustments (don't make dramatic changes)
- Add diversification with new positions in different sectors
- Review risk tolerance to ensure limits match strategy

**For Good/Excellent Scores (80-100)**
- Regular monitoring and monthly reviews
- Rebalancing to maintain target weights
- Risk limit review to ensure limits remain appropriate
- Performance tracking to ensure risk-adjusted returns

#### Integration with Portfolio Management

- **Risk Score Updates**: Monthly or after significant changes
- **Decision Making**: Use risk score to guide new positions, sizing, and rebalancing
- **Reporting**: Include risk score and breakdown in client reports and management reviews
- **Customization**: Adjust component weights and thresholds for different strategies

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

The risk module includes a production-ready Flask web application (3,000+ lines) that makes risk analysis accessible to anyone:

**Features:**
- **Google OAuth Authentication**: Secure user management and session handling
- **Plaid Integration**: Automatic portfolio import from brokerage accounts
- **Claude AI Chat**: Interactive risk analysis assistance and natural language queries
- **RESTful API**: Multiple endpoints for portfolio analysis and risk scoring
- **Rate Limiting**: Tiered access control with fair usage policies
- **Admin Dashboard**: Usage tracking, cache management, and system monitoring
- **Export Functionality**: Download analysis results and portfolio reports

**Access Tiers:**
- **Public**: Limited daily usage (5 analyses/day)
- **Registered**: Enhanced limits (15 analyses/day) + Google OAuth access
- **Paid**: Full access (30 analyses/day) + priority support

**API Endpoints:**

| Endpoint | Returns | Purpose |
|----------|---------|---------|
| `POST /api/analyze` | Structured data + formatted report | Complete portfolio risk analysis |
| `POST /api/risk-score` | Structured data + formatted report | Risk score calculation (0-100) |
| `POST /api/performance` | Structured data + formatted report | Performance metrics (returns, Sharpe, alpha, beta) |
| `POST /api/portfolio-analysis` | GPT interpretation + analysis | AI-powered portfolio analysis |
| `POST /api/claude_chat` | AI conversation | Interactive risk analysis assistant |
| `GET /plaid/holdings` | Portfolio data | Import from brokerage accounts |
| `GET /api/health` | System status | API health check |

**Key Features:**
- **Dual Output Format**: Every analysis endpoint returns both structured JSON data AND human-readable formatted reports
- **Complete Parity**: API provides same analysis as CLI with identical output
- **Real-time Analysis**: Fresh analysis with current market data

**Usage:**
```bash
# Start the web server
python app.py

# Test API endpoints with structured data + formatted reports
python3 show_api_output.py analyze        # Complete portfolio analysis
python3 show_api_output.py risk-score     # Risk score analysis  
python3 show_api_output.py performance    # Performance metrics

# API usage example
curl -X POST http://localhost:5001/api/performance?key=paid_key_789 \
  -H "Content-Type: application/json" \
  -d '{}'
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

```python
from run_risk import run_portfolio, run_portfolio_performance, run_stock, run_what_if, run_min_variance, run_max_return
from portfolio_risk_score import run_risk_score_analysis

# 1. Full portfolio analysis
run_portfolio("portfolio.yaml")

# 2. Portfolio performance analysis
run_portfolio_performance("portfolio.yaml")

# 3. Single stock analysis
run_stock("AAPL", start="2020-01-01", end="2024-12-31")

# 4. What-if scenario testing
run_what_if("portfolio.yaml", delta="AAPL:-500bp,SGOV:+500bp")

# 5. Portfolio optimization
run_min_variance("portfolio.yaml")  # Minimum risk portfolio
run_max_return("portfolio.yaml")    # Maximum return portfolio

# 6. Risk scoring
results = run_risk_score_analysis("portfolio.yaml", "risk_limits.yaml")
```

Each provides actionable insights to help you make better investment decisions.

## 📁 Complete Project Structure

### Core Analysis Engine
```
risk_module/
├── portfolio_risk.py           # Portfolio risk calculations and performance metrics
├── portfolio_risk_score.py     # 0-100 risk scoring with detailed analysis
├── factor_utils.py             # Multi-factor regression and volatility calculations
├── risk_summary.py             # Single-stock risk profiling
├── portfolio_optimizer.py      # Portfolio optimization algorithms
└── risk_helpers.py             # Risk calculation utilities
```

### Data & Integration Layer
```
├── data_loader.py              # API integration with intelligent caching
├── plaid_loader.py             # Plaid brokerage integration (29KB)
├── proxy_builder.py            # Factor proxy generation and GPT peer integration
└── gpt_helpers.py              # AI assistant integration
```

### Entry Points & Runners
```
├── run_risk.py                 # Main command-line interface and entry points
├── run_portfolio_risk.py       # Portfolio analysis runner with display utilities
└── run_risk_summary_to_gpt_dev.py # GPT interpretation runner
```

### Web Application
```
├── app.py                      # Flask web application (3,156 lines)
│                               # • Google OAuth authentication
│                               # • Plaid integration endpoints
│                               # • Claude AI chat interface
│                               # • RESTful API with rate limiting
│                               # • Admin dashboard and monitoring
└── frontend/                   # Frontend assets (if applicable)
```

### Configuration Files
```
├── portfolio.yaml              # Portfolio positions and factor proxies
├── risk_limits.yaml            # Risk tolerance and limit settings
├── settings.py                 # Default system configuration
├── cash_map.yaml               # Cash position mapping to ETF proxies
├── industry_to_etf.yaml        # Industry classification to ETF mapping
├── exchange_etf_proxies.yaml   # Exchange-specific factor proxies
└── what_if_portfolio.yaml      # What-if scenario configurations
```

### Utilities & Helpers
```
├── helpers_display.py          # Output formatting and display utilities
├── helpers_input.py            # Input processing and validation utilities
└── update_secrets.sh           # Secrets synchronization script
```

### 5-Layer Enterprise Architecture
```
├── 📊 Core Risk Engine (Layer 1)
│   ├── portfolio_risk.py              # Portfolio risk calculations (32KB)
│   ├── portfolio_risk_score.py        # Risk scoring system (53KB)
│   ├── factor_utils.py                # Factor analysis utilities (8KB)
│   ├── risk_summary.py                # Single-stock risk profiling (4KB)
│   ├── portfolio_optimizer.py         # Portfolio optimization (36KB)
│   ├── data_loader.py                 # Data fetching and caching (8KB)
│   ├── proxy_builder.py               # Factor proxy generation (19KB)
│   ├── plaid_loader.py                # Plaid brokerage integration (29KB)
│   ├── gpt_helpers.py                 # GPT integration (4KB)
│   └── risk_helpers.py                # Risk calculation helpers (8KB)
│
├── 📁 inputs/ (Layer 2: Data Management)
│   ├── portfolio_manager.py           # Portfolio operations
│   ├── risk_config.py                 # Risk limits management
│   ├── returns_calculator.py          # Returns estimation
│   └── file_manager.py                # File operations
│
├── 📁 services/ (Layer 3: AI Services)
│   ├── claude/
│   │   ├── function_executor.py       # 14 Claude functions (618 lines)
│   │   └── chat_service.py            # Claude conversation orchestration
│   └── portfolio/
│       └── context_service.py         # Portfolio caching (374 lines)
│
├── 📁 routes/ (Layer 4: Web Interface)
│   ├── api.py                         # Core API endpoints
│   ├── claude.py                      # Claude chat endpoint
│   ├── plaid.py                       # Plaid integration endpoints
│   ├── auth.py                        # Authentication endpoints
│   └── admin.py                       # Admin endpoints
│
├── 📁 frontend/ (Layer 5: Frontend)
│   ├── src/
│   │   ├── App.js                     # React SPA (1,477 lines)
│   │   └── components/                # React components
│   └── public/                        # Static assets
```

### Documentation & Development Tools
```
├── README.md                   # Main project documentation (this file)
├── architecture.md             # Detailed technical architecture (1,000+ lines)
├── docs/                       # Comprehensive documentation
│   ├── interfaces/             # Interface alignment documentation
│   ├── planning/               # Architecture and planning docs
│   ├── WEB_APP.md             # Web application API reference
│   └── API_REFERENCE.md       # Detailed API documentation
├── tools/                      # Development utilities
│   ├── view_alignment.py       # Interface alignment viewer
│   ├── check_dependencies.py   # Dependency impact analysis
│   └── test_all_interfaces.py  # Interface testing suite
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
└── .env                       # Environment variables (API keys, secrets)
```

### Data & Cache Directories
```
├── cache_prices/              # Cached price data (gitignored)
├── exports/                   # Analysis export files
├── error_logs/                # System error logs
├── templates/                 # Web application templates
└── Archive/                   # Historical files and backups
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
- [ ] Real-time risk monitoring and alerts based on volatility
- [ ] AI-powered portfolio recommendations and what-if analysis
- [ ] Additional factor models (quality, size, etc.) for more comprehensive analysis
- [ ] Backtesting capabilities to validate investment decisions
- [ ] Interactive risk attribution visualization
- [ ] Portfolio comparison tools (current vs. suggested vs. historical)
- [ ] Advanced portfolio optimization with multiple objectives

---

**Built with ❤️ for quantitative risk analysis**