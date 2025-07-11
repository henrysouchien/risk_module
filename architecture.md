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

### Interface Layer

For web interface, REST API, and Claude AI chat integration, see:
- **[Interface README](docs/interfaces/INTERFACE_README.md)** - User guide for REST API, Claude chat, and web interface
- **[Interface Architecture](docs/interfaces/INTERFACE_ARCHITECTURE.md)** - Technical architecture of the interface layer

## 🏗️ Architecture Layers

The system follows a sophisticated **5-layer enterprise architecture** with clear separation of concerns and comprehensive interface coverage:

```
┌─────────────────────────────────────────────────────────────┐
│                    LAYER 5: FRONTEND                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ frontend/src/   │  │ React SPA       │  │ Interface    │ │
│  │ App.js          │  │ (1,477 lines)   │  │ Alignment    │ │
│  │ (UI Components) │  │                 │  │ Tools        │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                    LAYER 4: WEB INTERFACE                    │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ routes/api.py   │  │ routes/claude.py│  │ routes/      │ │
│  │ (REST API)      │  │ (AI Chat)       │  │ plaid.py     │ │
│  │                 │  │                 │  │ auth.py      │ │
│  │                 │  │                 │  │ admin.py     │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                    LAYER 3: AI SERVICES                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ services/claude/│  │ services/       │  │ 14 Claude    │ │
│  │ function_       │  │ portfolio/      │  │ Functions    │ │
│  │ executor.py     │  │ context_service │  │ (618 lines)  │ │
│  │ (AI Functions)  │  │ (Portfolio Cache│  │              │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                    LAYER 2: DATA MANAGEMENT                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ inputs/         │  │ inputs/         │  │ inputs/      │ │
│  │ portfolio_      │  │ risk_config.py  │  │ returns_     │ │
│  │ manager.py      │  │ (Risk Limits)   │  │ calculator.py│ │
│  │ (Portfolio Ops) │  │                 │  │ inputs/      │ │
│  │                 │  │                 │  │ file_manager │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                    LAYER 1: CORE RISK ENGINE                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ portfolio_risk. │  │ portfolio_risk_ │  │ factor_utils.│ │
│  │ py              │  │ score.py        │  │ py           │ │
│  │ (Risk Analysis) │  │ (Risk Scoring)  │  │ (Factor Calc)│ │
│  │                 │  │                 │  │              │ │
│  │ portfolio_      │  │ risk_summary.py │  │ data_loader. │ │
│  │ optimizer.py    │  │ (Stock Profile) │  │ py           │ │
│  │ (Optimization)  │  │                 │  │ (Data Access)│ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Interface Coverage Analysis

The system provides **4 distinct interfaces** with varying levels of functional coverage:

| Interface | Coverage | Key Components | Status |
|-----------|----------|----------------|--------|
| **CLI** | 21% | `run_risk.py`, `portfolio_risk_score.py`, `proxy_builder.py` | ⚠️ Missing 9 functions |
| **API** | 85% | `routes/api.py`, `routes/claude.py`, `routes/plaid.py` | ✅ Comprehensive |
| **Claude** | 36% | `services/claude/function_executor.py` (14 functions) | ✅ AI-powered |
| **Inputs** | 100% | `inputs/portfolio_manager.py`, `inputs/risk_config.py` | ✅ Foundation layer |

**Priority Gap**: Adding 9 missing CLI functions would increase overall alignment from 21% to 44%.

## 📂 File Structure

### Complete Enterprise Directory Structure

```
risk_module/
├── 📄 Readme.md                    # Main project documentation
├── 📄 architecture.md              # Technical architecture (this file)
├── ⚙️ settings.py                  # Default configuration settings
├── 🔧 app.py                       # Flask web application (13KB)
├── 🔒 update_secrets.sh            # Secrets synchronization script
├── 📋 requirements.txt             # Python dependencies
├── 📜 LICENSE                      # MIT License
│
├── 📊 Core Risk Engine (Layer 1)
│   ├── 💼 portfolio_risk.py           # Portfolio risk calculations (32KB)
│   ├── 📈 portfolio_risk_score.py     # Risk scoring system (53KB)
│   ├── 📊 factor_utils.py             # Factor analysis utilities (8KB)
│   ├── 📋 risk_summary.py             # Single-stock risk profiling (4KB)
│   ├── ⚡ portfolio_optimizer.py       # Portfolio optimization (36KB)
│   ├── 🔌 data_loader.py              # Data fetching and caching (8KB)
│   ├── 🤖 gpt_helpers.py              # GPT integration (4KB)
│   ├── 🔧 proxy_builder.py            # Factor proxy generation (19KB)
│   ├── 🏦 plaid_loader.py             # Plaid brokerage integration (29KB)
│   └── 🛠️ risk_helpers.py             # Risk calculation helpers (8KB)
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
│   │   ├── chat_service.py            # Claude conversation orchestration
│   │   └── claude_utils.py            # Claude utilities
│   └── portfolio/
│       ├── context_service.py         # Portfolio caching (374 lines)
│       └── portfolio_utils.py         # Portfolio utilities
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
│   │   ├── components/                # React components
│   │   └── utils/                     # Frontend utilities
│   └── public/                        # Static assets
│
├── 📁 docs/ (Documentation)
│   ├── interfaces/
│   │   ├── alignment_table.md         # Interface alignment mapping
│   │   ├── INTERFACE_README.md        # Interface documentation
│   │   └── INTERFACE_ARCHITECTURE.md  # Interface architecture
│   ├── planning/
│   │   ├── COMPLETE_IMPLEMENTATION_PLAN.md
│   │   ├── MIGRATION_CHECKLIST.md
│   │   ├── REFACTORING_PLAN.md
│   │   ├── DATA_OBJECTS_DESIGN.md
│   │   └── ARCHITECTURE_DECISIONS.md
│   ├── API_REFERENCE.md               # API documentation
│   ├── WEB_APP.md                     # Web application guide
│   └── README.md                      # Documentation index
│
├── 📁 tools/ (Utilities)
│   ├── view_alignment.py              # Terminal alignment viewer
│   ├── check_dependencies.py          # Dependency impact analysis
│   └── test_all_interfaces.py         # Interface testing suite
│
├── 📁 utils/ (Utilities)
│   ├── helpers_display.py             # Display utilities (5KB)
│   ├── helpers_input.py               # Input processing utilities (2KB)
│   └── various utility modules...
│
├── 📁 templates/ (Web Templates)
│   └── flask templates for web UI
│
├── 📁 Configuration Files
│   ├── ⚙️ portfolio.yaml              # Portfolio configuration
│   ├── ⚙️ risk_limits.yaml            # Risk limit definitions
│   ├── 🗺️ cash_map.yaml               # Cash position mapping
│   ├── 🏭 industry_to_etf.yaml        # Industry classification mapping
│   ├── 📊 exchange_etf_proxies.yaml   # Exchange-specific proxies
│   ├── 🔧 what_if_portfolio.yaml      # What-if scenarios
│   └── 🔑 .env                        # Environment variables
│
├── 📁 Entry Points & Runners
│   ├── 🎯 run_risk.py                 # Main CLI interface (20KB)
│   ├── 🚀 run_portfolio_risk.py       # Portfolio analysis runner (24KB)
│   └── 🤖 run_risk_summary_to_gpt_dev.py # GPT interpretation runner
│
├── 📁 Data & Cache Directories
│   ├── 📁 cache_prices/               # Cached price data (gitignored)
│   ├── 📁 exports/                    # Analysis export files
│   ├── 📁 error_logs/                 # System error logs
│   └── 📁 Archive/                    # Historical files
│
└── 📁 Development & Testing
    ├── Various .ipynb files           # Jupyter notebooks for development
    ├── test_*.py files                # Testing scripts
    └── *_dev.py files                 # Development versions
```

### Key Directory Purposes

| Directory | Purpose | Layer | Key Files |
|-----------|---------|-------|-----------|
| **inputs/** | Data management operations | 2 | `portfolio_manager.py`, `risk_config.py` |
| **services/** | AI services and orchestration | 3 | `function_executor.py`, `context_service.py` |
| **routes/** | Web API endpoints | 4 | `api.py`, `claude.py`, `plaid.py` |
| **frontend/** | React user interface | 5 | `App.js`, components |
| **docs/** | Comprehensive documentation | - | Interface docs, planning docs |
| **tools/** | Development utilities | - | Alignment tools, testing scripts |
| **utils/** | Helper functions | - | Display, input processing |

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

### Portfolio Performance Analysis Flow

```
1. Configuration Loading
   portfolio.yaml → run_portfolio_performance() → portfolio weights + dates

2. Data Collection
   portfolio tickers + benchmark → data_loader.py → historical price data
   Treasury rates → fetch_monthly_treasury_rates() → risk-free rate data

3. Return Calculation
   price data → calculate_portfolio_performance_metrics() → portfolio returns
   benchmark data → calculate_portfolio_performance_metrics() → benchmark returns

4. Performance Metrics
   returns + risk-free rates → performance calculations → comprehensive metrics
   - Annualized returns and volatility
   - Risk-adjusted metrics (Sharpe, Sortino, Information ratios)
   - Benchmark analysis (alpha, beta, tracking error)
   - Drawdown analysis and recovery periods

5. Display & Reporting
   performance metrics → display_portfolio_performance_metrics() → formatted output
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
- Treasury rate integration for risk-free rates

**Caching Strategy**:
```
RAM Cache (LRU) → Disk Cache (Parquet) → Network (FMP API)
```

**Treasury Rate Integration**:
The system now uses professional-grade risk-free rates from the FMP Treasury API instead of ETF price movements:
- `get_treasury_rate_from_fmp()`: Core function to fetch 3-month Treasury rates from FMP API
- `fetch_monthly_treasury_rates()`: Retrieves historical Treasury yields with date filtering
- Proper date range filtering for historical analysis aligned with portfolio periods
- Cache-enabled for performance with monthly resampling
- Eliminates contamination from bond price fluctuations in rate calculations
- Integrated into `calculate_portfolio_performance_metrics()` for accurate Sharpe ratio calculations

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
- `normalize_weights()`: Weight standardization (used only in optimization functions)
- `compute_portfolio_returns()`: Portfolio return calculation
- `compute_covariance_matrix()`: Risk matrix construction
- `compute_portfolio_volatility()`: Portfolio volatility
- `compute_risk_contributions()`: Risk attribution
- `calculate_portfolio_performance_metrics()`: Comprehensive performance analysis

**Weight Normalization Behavior**:
- **Default**: `normalize_weights = False` in `PORTFOLIO_DEFAULTS` (raw weights represent true economic exposure)
- **Risk Analysis**: Uses raw weights to calculate true portfolio risk exposure without leverage double-counting
- **Optimization**: Always normalizes weights internally for mathematical stability
- **Display**: Shows "Raw Weights" vs "Normalized Weights" based on setting

### 4. Portfolio Performance Engine (`portfolio_risk.py`)

**Purpose**: Portfolio performance metrics and risk-adjusted return analysis

**Key Functions**:
- `calculate_portfolio_performance_metrics()`: Calculate returns, Sharpe ratio, alpha, beta, max drawdown
- `get_treasury_rate_from_fmp()`: Fetch 3-month Treasury rates from FMP API with error handling
- `fetch_monthly_treasury_rates()`: Retrieve historical Treasury rates with caching and date filtering

**Features**:
- Historical return analysis with proper compounding
- Risk-adjusted performance metrics (Sharpe, Sortino, Information ratios)
- Benchmark comparison (alpha, beta, tracking error)
- Drawdown analysis and recovery periods
- Professional risk-free rate integration using Treasury yields
- Comprehensive display formatting with automated insights
- Win rate and best/worst month analysis

**Performance Metrics Calculated**:
- Total and annualized returns
- Volatility (annual standard deviation)
- Maximum drawdown and recovery analysis
- Sharpe ratio (excess return per unit of risk)
- Sortino ratio (downside risk-adjusted returns)
- Information ratio (tracking error-adjusted alpha)
- Alpha and beta vs benchmark (SPY)
- Tracking error and correlation analysis
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

### 5. AI Services Layer (`services/`)

**Purpose**: AI-powered portfolio analysis and conversational interface

#### Claude Function Executor (`services/claude/function_executor.py`)
**618 lines of sophisticated AI function integration**

**Core Analysis Functions (5)**:
- `run_portfolio_analysis()`: Complete portfolio risk analysis with GPT interpretation
- `analyze_stock()`: Single stock analysis with factor decomposition
- `get_risk_score()`: Portfolio risk scoring with detailed breakdown
- `calculate_portfolio_performance()`: Performance metrics and benchmarking
- `run_what_if_scenario()`: Portfolio scenario testing

**Scenario & Optimization Functions (4)**:
- `run_what_if_scenario()`: Portfolio modification testing
- `create_portfolio_scenario()`: New portfolio creation from user input
- `optimize_portfolio_min_variance()`: Minimum risk optimization
- `optimize_portfolio_max_return()`: Maximum return optimization

**Portfolio Management Functions (6)**:
- `create_portfolio_scenario()`: Portfolio creation with validation
- `inject_all_proxies()`: Factor proxy setup and peer generation
- `save_portfolio_yaml()`: Portfolio configuration persistence
- `load_portfolio_yaml()`: Portfolio configuration loading
- `update_portfolio_weights()`: Weight modification
- `validate_portfolio_config()`: Configuration validation

**Returns Management Functions (3)**:
- `estimate_expected_returns()`: Historical returns estimation
- `set_expected_returns()`: Manual returns configuration
- `update_portfolio_expected_returns()`: Returns persistence

**Risk Management Functions (5)**:
- `view_current_risk_limits()`: Risk limits inspection
- `update_risk_limits()`: Risk tolerance modification
- `reset_risk_limits()`: Risk limits reset to defaults
- `validate_risk_limits()`: Risk configuration validation
- `get_risk_score()`: Comprehensive risk assessment

**File Management Functions (4)**:
- `list_portfolios()`: Portfolio file listing
- `backup_portfolio()`: Portfolio backup creation
- `restore_portfolio()`: Portfolio restoration
- `delete_portfolio()`: Portfolio file deletion

**Features**:
- Natural language interface for all risk analysis functions
- Automatic parameter validation and error handling
- GPT-powered interpretation of results
- Seamless integration with core risk engine
- Context-aware responses based on portfolio state

#### Portfolio Context Service (`services/portfolio/context_service.py`)
**374 lines of portfolio caching and context management**

**Key Functions**:
- `cache_portfolio_context()`: Portfolio state caching
- `get_portfolio_context()`: Context retrieval for conversations
- `update_portfolio_context()`: Context updates after modifications
- `clear_portfolio_context()`: Context cleanup

**Features**:
- Redis-based portfolio state caching
- Context persistence across conversations
- Automatic context updates after portfolio modifications
- Performance optimization for repeated analysis

### 6. Data Management Layer (`inputs/`)

**Purpose**: Specialized modules for data operations and configuration management (Layer 2)

The inputs layer provides a clean abstraction for all data management operations, serving as the foundation for the entire system.

#### Portfolio Manager (`inputs/portfolio_manager.py`)
**Portfolio configuration and operations**

**Key Functions**:
- `create_portfolio_yaml()`: Create new portfolio configurations
- `load_yaml_config()`: Load and validate portfolio configurations
- `save_yaml_config()`: Persist portfolio configurations
- `update_portfolio_weights()`: Modify portfolio positions
- `create_what_if_yaml()`: Generate scenario configurations
- `validate_portfolio_config()`: Portfolio validation and error checking

**Features**:
- YAML configuration management
- Portfolio weight normalization (optional, default: False)
- Data validation and error handling
- Scenario generation for what-if analysis
- Backup and versioning support

#### Risk Configuration Manager (`inputs/risk_config.py`)
**Risk limits and tolerance management**

**Key Functions**:
- `view_current_risk_limits()`: Display current risk tolerance settings
- `update_risk_limits()`: Modify risk tolerance parameters
- `reset_risk_limits()`: Reset to default risk settings
- `validate_risk_limits()`: Risk configuration validation
- `calculate_risk_metrics()`: Risk calculation utilities

**Features**:
- Risk limit validation and enforcement
- Default risk tolerance management
- Risk metric calculation support
- Configuration backup and recovery
- Integration with risk scoring system

#### Returns Calculator (`inputs/returns_calculator.py`)
**Expected returns estimation and management**

**Key Functions**:
- `estimate_historical_returns()`: Calculate historical expected returns
- `update_portfolio_expected_returns()`: Update portfolio return expectations
- `set_expected_returns()`: Manual return specification
- `validate_return_assumptions()`: Return validation and reasonableness checks
- `calculate_risk_adjusted_returns()`: Risk-adjusted return calculations

**Features**:
- Historical return analysis
- Return assumption validation
- Risk-adjusted return calculations
- Integration with portfolio optimization
- Return forecasting utilities

#### File Manager (`inputs/file_manager.py`)
**File operations and data persistence**

**Key Functions**:
- `load_yaml_config()`: Universal YAML configuration loader
- `save_yaml_config()`: Universal YAML configuration saver
- `backup_portfolio()`: Portfolio backup creation
- `restore_portfolio()`: Portfolio restoration from backup
- `list_portfolios()`: Portfolio file discovery
- `delete_portfolio()`: Safe portfolio deletion

**Features**:
- Universal configuration file handling
- Backup and recovery operations
- File validation and error handling
- Directory management and organization
- Integration with all system components

#### Layer 2 Architecture Benefits

**1. Data Abstraction**:
- Clean separation between data operations and business logic
- Consistent data access patterns across all interfaces
- Centralized data validation and error handling

**2. Configuration Management**:
- Unified approach to YAML configuration handling
- Validation and error checking for all data inputs
- Backup and recovery capabilities

**3. Interface Foundation**:
- Provides consistent data operations for all 4 interfaces
- Ensures data integrity across CLI, API, Claude, and Frontend
- Enables rapid interface development through reusable components

**4. System Integration**:
- Seamless integration with Core Risk Engine (Layer 1)
- Supports AI Services (Layer 3) with clean data access
- Enables Web Interface (Layer 4) and Frontend (Layer 5)

### 7. Execution Layer

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
    "end_date": "2025-06-27",
    "normalize_weights": False,  # Global default for portfolio weight normalization
    "worst_case_lookback_years": 10  # Historical lookback period for worst-case scenario analysis
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
normalize_weights = config.get('normalize_weights', PORTFOLIO_DEFAULTS['normalize_weights'])
```

### Date Logic and Calculation Windows

**System Architecture**:
The risk module implements a three-tier date system for different calculation purposes:

**1. Primary Portfolio System**:
```python
# Source: portfolio.yaml
config = load_portfolio_config("portfolio.yaml")
start_date = config["start_date"]  # e.g., "2019-05-31"
end_date = config["end_date"]      # e.g., "2024-03-31"

# Usage: All portfolio calculations
summary = build_portfolio_view(weights, start_date, end_date, ...)
```

**2. Fallback System**:
```python
# Source: settings.py PORTFOLIO_DEFAULTS
from settings import PORTFOLIO_DEFAULTS

# Used when portfolio dates not specified
start = start or PORTFOLIO_DEFAULTS["start_date"]
end = end or PORTFOLIO_DEFAULTS["end_date"]
normalize_weights = normalize_weights or PORTFOLIO_DEFAULTS["normalize_weights"]
```

**3. Independent Analysis System**:
```python
# Single stock analysis (flexible dates)
today = pd.Timestamp.today().normalize()
start = start or today - pd.DateOffset(years=5)
end = end or today

# Historical worst-case analysis (configurable lookback)
from settings import PORTFOLIO_DEFAULTS
lookback_years = PORTFOLIO_DEFAULTS.get('worst_case_lookback_years', 10)
end_dt = datetime.today()
start_dt = end_dt - pd.DateOffset(years=lookback_years)
```

**Calculation Consistency**:
- **Factor Regressions**: All use same date window for stable betas
- **Peer Validation**: Subindustry peers validated over same period as target
- **Data Quality**: Minimum observation requirements prevent regression window limitations
- **Optimization**: All portfolio optimizations use consistent date windows

**Data Flow**:
```
portfolio.yaml → load_portfolio_config() → build_portfolio_view() → factor calculations
     ↓
PORTFOLIO_DEFAULTS (fallback) → proxy generation → peer validation
     ↓
Independent functions → flexible date logic for specific use cases
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

## 📐 Mathematical Framework

The risk module implements a comprehensive mathematical framework for portfolio risk analysis. For detailed mathematical formulas and their implementations, see the **Mathematical Reference** section in the README.md file.

**Key Mathematical Components**:
- **Portfolio Volatility**: `σ_p = √(w^T Σ w)`
- **Factor Betas**: `β_i,f = Cov(r_i, r_f) / Var(r_f)`
- **Risk Contributions**: `RC_i = w_i × (Σw)_i / σ_p`
- **Variance Decomposition**: Total = Factor + Idiosyncratic
- **Euler Variance**: Marginal variance contributions

**Implementation Functions**:
- `compute_portfolio_volatility()`: Portfolio risk calculation
- `compute_stock_factor_betas()`: Factor exposure analysis
- `compute_risk_contributions()`: Risk attribution
- `compute_portfolio_variance_breakdown()`: Variance decomposition
- `compute_euler_variance_percent()`: Marginal contributions

## 🌐 Web Application Architecture

### Flask Web App (`app.py`)

**Production-Ready Features** (3,156 lines):
- **Google OAuth Authentication**: Secure user management and session handling
- **Multi-Tier Access Control**: Public/Registered/Paid user tiers with rate limiting
- **Plaid Integration**: Real-time portfolio import from brokerage accounts
- **Claude AI Chat**: Interactive risk analysis assistance and natural language queries
- **RESTful API**: Multiple endpoints for portfolio analysis and risk scoring
- **Portfolio Configuration Interface**: Web-based YAML editor and management
- **Risk Analysis Execution**: Server-side portfolio analysis with export functionality
- **Admin Dashboard**: Usage tracking, cache management, and system monitoring
- **API Key Management**: Secure key generation, validation, and Kartra integration

**Rate Limiting Strategy**:
```python
# Tiered rate limits
limits = {
    "public": "5 per day",
    "registered": "15 per day", 
    "paid": "30 per day"
}
```

**Security Features**:
- API key validation
- Rate limiting by user tier
- Error logging and monitoring
- Secure token storage

### Route Documentation (`routes/`)

The web interface is organized into 5 specialized route modules for clean separation of concerns:

#### Core API Routes (`routes/api.py`)
**Primary risk analysis endpoints**

| Endpoint | Method | Purpose | Returns |
|----------|--------|---------|---------|
| `/api/analyze` | POST | Portfolio risk analysis | Structured data + CLI-style formatted report |
| `/api/risk-score` | POST | Risk scoring analysis | Structured data + CLI-style formatted report |
| `/api/performance` | POST | Performance metrics | Structured data + CLI-style formatted report |
| `/api/what-if` | POST | Scenario analysis | Structured data + raw analysis output |
| `/api/optimize` | POST | Portfolio optimization | Structured data + optimization results |

**Response Format**:
```json
{
  "success": true,
  "performance_metrics": {
    "returns": {"annualized_return": 25.98, ...},
    "risk_metrics": {"volatility": 19.80, ...},
    "risk_adjusted_returns": {"sharpe_ratio": 1.18, ...},
    ...
  },
  "formatted_report": "📊 PORTFOLIO PERFORMANCE ANALYSIS\n============...",
  "summary": {"key_metrics": "..."},
  "timestamp": "2024-01-01T12:00:00Z"
}
```

**Features**:
- **Dual Output Format**: Both structured JSON data AND human-readable formatted reports
- Rate limiting by user tier
- Input validation and sanitization
- Comprehensive error handling
- Export functionality for analysis results

#### Claude AI Chat Routes (`routes/claude.py`)
**AI-powered conversational analysis**

| Endpoint | Method | Purpose | Parameters |
|----------|--------|---------|------------|
| `/api/claude_chat` | POST | Interactive AI analysis | `message`, `conversation_history` |
| `/api/claude_functions` | GET | List available functions | - |
| `/api/claude_context` | GET | Get conversation context | `session_id` |

**Features**:
- Integration with 14 Claude functions
- Context-aware conversations
- Function calling and parameter validation
- Natural language result interpretation
- Session management and persistence

#### Plaid Integration Routes (`routes/plaid.py`)
**Brokerage account integration**

| Endpoint | Method | Purpose | Parameters |
|----------|--------|---------|------------|
| `/plaid/link` | POST | Create Plaid link token | `user_id` |
| `/plaid/exchange` | POST | Exchange public token | `public_token`, `user_id` |
| `/plaid/accounts` | GET | List connected accounts | `user_id` |
| `/plaid/holdings` | GET | Get account holdings | `user_id`, `account_id` |
| `/plaid/import` | POST | Import portfolio data | `user_id`, `account_id` |

**Features**:
- Multi-institution support
- Real-time holdings import
- Cash position mapping
- Portfolio YAML generation
- AWS Secrets Manager integration

#### Authentication Routes (`routes/auth.py`)
**User management and security**

| Endpoint | Method | Purpose | Parameters |
|----------|--------|---------|------------|
| `/auth/login` | POST | User login | `email`, `password` |
| `/auth/logout` | POST | User logout | - |
| `/auth/register` | POST | User registration | `email`, `password`, `tier` |
| `/auth/profile` | GET | Get user profile | - |
| `/auth/api-key` | POST | Generate API key | `user_id` |

**Features**:
- Google OAuth integration
- Multi-tier user management (public/registered/paid)
- Secure session handling
- API key generation and validation
- Rate limiting enforcement

#### Admin Routes (`routes/admin.py`)
**System administration and monitoring**

| Endpoint | Method | Purpose | Parameters |
|----------|--------|---------|------------|
| `/admin/usage` | GET | Usage statistics | `date_range` |
| `/admin/cache` | DELETE | Clear system cache | `cache_type` |
| `/admin/users` | GET | User management | `filters` |
| `/admin/logs` | GET | System logs | `level`, `date_range` |
| `/admin/health` | GET | System health check | - |

**Features**:
- Usage tracking and analytics
- Cache management
- User administration
- System monitoring
- Error log analysis

### API Response Format

**Service Layer Endpoints** provide dual output format:

```json
{
  "success": true,
  "risk_results": {
    // Structured data with all metrics
    "volatility_annual": 0.198,
    "factor_exposures": {...},
    // ... comprehensive structured data
  },
  "formatted_report": "📊 PORTFOLIO RISK ANALYSIS\n============...",
  "summary": {
    // Key metrics summary
    "overall_risk": "Medium",
    "key_recommendations": [...]
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

**Direct Endpoints** return raw function output:

```json
{
  "success": true,
  "data": {
    // Raw function output
  },
  "endpoint": "direct/portfolio",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### Error Handling

- **Rate Limiting**: HTTP 429 with retry-after header
- **Authentication**: HTTP 401 for invalid credentials
- **Authorization**: HTTP 403 for insufficient permissions
- **Validation**: HTTP 400 with detailed error messages
- **Server Error**: HTTP 500 with error tracking ID

### Frontend Architecture (`frontend/`)

The frontend is a modern React Single Page Application (SPA) that provides an intuitive interface for portfolio risk analysis.

#### React Application (`frontend/src/App.js`)
**1,477 lines of sophisticated React components**

**Core Features**:
- **Portfolio Management**: Upload, edit, and manage portfolio configurations
- **Risk Analysis Dashboard**: Interactive risk metrics and visualizations
- **Claude AI Chat**: Conversational interface for portfolio analysis
- **Plaid Integration**: Connect and import brokerage accounts
- **Performance Tracking**: Historical performance analysis and benchmarking
- **Risk Scoring**: Visual risk score breakdown and recommendations
- **What-If Analysis**: Interactive scenario testing

**Component Structure**:
```
frontend/src/
├── App.js                     # Main application (1,477 lines)
├── components/
│   ├── Dashboard/             # Risk analysis dashboard
│   ├── Portfolio/             # Portfolio management
│   ├── Chat/                  # Claude AI chat interface
│   ├── Plaid/                 # Brokerage integration
│   ├── Analysis/              # Risk analysis components
│   ├── Performance/           # Performance tracking
│   └── Common/                # Shared components
├── services/
│   ├── api.js                 # API service layer
│   ├── claude.js              # Claude chat service
│   └── plaid.js               # Plaid integration service
├── utils/
│   ├── helpers.js             # Utility functions
│   ├── validation.js          # Input validation
│   └── formatting.js          # Data formatting
└── styles/
    ├── components/            # Component-specific styles
    └── global/                # Global styles
```

**Key Components**:

1. **Portfolio Dashboard**:
   - Real-time risk metrics display
   - Interactive charts and visualizations
   - Portfolio composition breakdown
   - Risk limit monitoring

2. **Claude Chat Interface**:
   - Natural language query processing
   - Context-aware conversations
   - Function calling integration
   - Result visualization

3. **Plaid Integration**:
   - Account linking workflow
   - Holdings import interface
   - Multi-institution support
   - Cash position mapping

4. **Risk Analysis Tools**:
   - Factor exposure analysis
   - Risk decomposition charts
   - Concentration analysis
   - Historical performance tracking

5. **What-If Scenarios**:
   - Interactive portfolio modification
   - Scenario comparison
   - Risk impact analysis
   - Optimization suggestions

**State Management**:
- React hooks for local state
- Context API for global state
- Redux for complex state management
- Local storage for persistence

**API Integration**:
- Axios for HTTP requests
- Error handling and retry logic
- Rate limiting compliance
- Real-time updates

**User Experience Features**:
- Responsive design for mobile/desktop
- Loading states and progress indicators
- Error boundaries for graceful failures
- Accessibility compliance
- Dark/light mode support

#### Frontend Build Process

**Development Setup**:
```bash
cd frontend/
npm install
npm start                    # Development server
npm run build               # Production build
npm test                    # Run tests
```

**Production Build**:
- Webpack bundling and optimization
- CSS/JS minification
- Asset optimization
- Environment variable injection

**Deployment**:
- Served through Flask static files
- CDN integration for assets
- Service worker for offline support
- Progressive Web App (PWA) capabilities

### Frontend-Backend Integration

**Data Flow**:
```
User Input → React Component → API Service → Flask Route → Core Engine → Database
     ↓
React State ← Component Update ← API Response ← Flask Response ← Analysis Results
```

**Real-time Features**:
- WebSocket connections for live updates
- Server-sent events for analysis progress
- Polling for portfolio updates
- Push notifications for risk alerts

**Security**:
- JWT token authentication
- CSRF protection
- Input sanitization
- XSS prevention
- Content Security Policy

### Interface Alignment System (`tools/`)

The system includes sophisticated tools for managing the complexity of the 4-interface architecture and ensuring consistency across all user touchpoints.

#### Interface Alignment Analysis

**Problem**: The risk module provides the same functionality through 4 different interfaces (CLI, API, Claude, Inputs), but maintaining consistency across all interfaces is challenging.

**Solution**: A comprehensive alignment tracking system that maps all functions across interfaces and identifies gaps.

#### Alignment Tools

**1. Interface Alignment Table (`docs/interfaces/alignment_table.md`)**
- **Purpose**: Complete mapping of 39 functions across 4 interfaces
- **Categories**: 9 functional categories (Core Analysis, Portfolio Management, etc.)
- **Status Tracking**: Alignment percentages and gap identification
- **Priority Analysis**: Identifies which missing functions would provide maximum impact

**Current Status**:
- **Overall Alignment**: 21% (8/39 functions fully aligned)
- **Biggest Gap**: Missing 9 CLI functions (would increase alignment to 44%)
- **Best Coverage**: Inputs layer (100%), API layer (85%)
- **Development Priority**: Add CLI wrappers for existing functions

**2. Terminal Alignment Viewer (`tools/view_alignment.py`)**
- **Purpose**: Quick terminal-friendly view of alignment status
- **Features**: Clean formatting, file location reference, priority recommendations
- **Usage**: `python tools/view_alignment.py`

**Output Example**:
```
🔍 CORE ANALYSIS FUNCTIONS
📋 Portfolio Analysis
  CLI:    ✅ run_portfolio()
  API:    ✅ /api/analyze + /api/claude_chat
  Claude: ✅ run_portfolio_analysis()
  Inputs: ✅ load_yaml_config()
  Status: ✅ FULLY ALIGNED
```

**3. Dependency Checker (`tools/check_dependencies.py`)**
- **Purpose**: Impact analysis for function modifications
- **Features**: Dependency mapping, testing chains, impact assessment
- **Usage**: `python tools/check_dependencies.py create_portfolio_yaml`

**Output Example**:
```
🔍 DEPENDENCY CHECK: create_portfolio_yaml
📁 Source File: inputs/portfolio_manager.py
🔗 Used By:
  • Claude: create_portfolio_scenario() → services/claude/function_executor.py
  • API: /api/claude_chat → routes/claude.py
  • CLI: ❌ Missing run_create_portfolio_scenario()
🧪 Testing Chain:
  1. Test inputs/portfolio_manager.py → create_portfolio_yaml()
  2. Test services/claude/function_executor.py → create_portfolio_scenario()
  3. Test /api/claude_chat endpoint → routes/claude.py
  4. Test frontend Claude chat integration
```

**4. Interface Testing Suite (`tools/test_all_interfaces.py`)**
- **Purpose**: Comprehensive testing across all interfaces
- **Features**: End-to-end testing, interface consistency validation
- **Coverage**: All 39 functions across 4 interfaces

#### Interface Architecture Benefits

**1. Consistency Tracking**:
- Ensures all interfaces provide equivalent functionality
- Prevents feature drift between interfaces
- Maintains user experience consistency

**2. Gap Analysis**:
- Identifies missing functions that would improve user experience
- Prioritizes development based on impact
- Tracks alignment progress over time

**3. Development Planning**:
- Guides feature development priorities
- Ensures comprehensive interface coverage
- Supports systematic interface expansion

**4. Quality Assurance**:
- Validates function behavior across interfaces
- Ensures consistent parameter handling
- Maintains interface compatibility

#### Interface Alignment Metrics

**Function Categories & Alignment**:
- **Core Analysis**: 60% aligned (3/5 functions) - Good coverage
- **Scenario & Optimization**: 75% aligned (3/4 functions) - Excellent coverage
- **Portfolio Management**: 17% aligned (1/6 functions) - Needs improvement
- **Returns Management**: 0% aligned (0/3 functions) - Missing CLI functions
- **Risk Limits**: 0% aligned (0/5 functions) - Missing CLI functions
- **Plaid Integration**: 0% aligned (0/5 functions) - Missing CLI functions
- **File Management**: 0% aligned (0/4 functions) - Missing CLI functions
- **Auth & Admin**: 0% aligned (0/4 functions) - Missing CLI functions
- **AI Orchestration**: 0% aligned (0/3 functions) - Missing CLI functions

**Development Impact**:
Adding the 9 missing CLI functions would:
- Increase overall alignment from 21% to 44%
- Provide complete CLI workflow coverage
- Enable consistent behavior across all interfaces
- Support power users who prefer command-line operations

## 🔗 External Integrations

### Plaid Financial Data Integration (`plaid_loader.py`)

**Automated Portfolio Import**:
- **Multi-Institution Support**: Connect to multiple brokerage accounts
- **Real-Time Holdings**: Fetch current positions and balances
- **Cash Position Mapping**: Convert cash to appropriate ETF proxies
- **AWS Secrets Management**: Secure storage of access tokens
- **Portfolio YAML Generation**: Automatic conversion to risk module format

**Data Flow**:
```
Plaid API → Holdings Data → Cash Mapping → Portfolio YAML → Risk Analysis
```

**Supported Features**:
- Interactive Brokers integration
- Multi-currency support
- Automatic cash gap detection
- Portfolio consolidation

### Cash Position Mapping (`cash_map.yaml`)

**Configuration Structure**:
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

**Usage**:
- Maps broker-specific cash tickers to currencies
- Converts cash positions to appropriate ETF proxies
- Supports multi-currency portfolios

### Factor Proxy Configuration

**Industry Mapping (`industry_to_etf.yaml`)**:
- Maps FMP industry classifications to representative ETFs
- Supports custom industry definitions
- Fallback to default proxies for unknown industries

**Exchange-Specific Proxies (`exchange_etf_proxies.yaml`)**:
- Exchange-specific factor proxy selection
- Optimized for different market characteristics
- Fallback to global proxies for international securities

## 🔌 API Integration

### Financial Modeling Prep (FMP)

**Endpoints Used**:
- `/historical-price-eod/full`: End-of-day price data
- `/treasury`: 3-month Treasury yields for risk-free rate calculations
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

2. **Advanced GPT Integration**:
   - Automated peer suggestion ✅ **Implemented**
   - Natural language risk reports ✅ **Implemented**
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
   - Web-based interface 🔄 **In Development** - Figma UI design in progress
   - Mobile app support
   - API endpoints for integration ✅ **Implemented**

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
| Portfolio Optimization | `portfolio_optimizer.py` | ✅ Working | Min variance and max return |
| GPT Integration | `gpt_helpers.py` | ✅ Working | Peer generation and interpretation |
| Proxy Builder | `proxy_builder.py` | ✅ Working | Factor proxy generation |
| Web Application | `app.py` | 🔄 In Development | Flask web interface - Figma UI design in progress |
| Plaid Integration | `plaid_loader.py` | ✅ Working | Financial data import |
| Risk Helpers | `risk_helpers.py` | ✅ Working | Risk calculation utilities |

## 📦 Dependencies

### Core Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **statsmodels**: Statistical modeling and regression
- **requests**: HTTP library for API calls
- **python-dotenv**: Environment variable management
- **pyarrow**: Parquet file handling for caching

### Web Application Dependencies

- **flask**: Web application framework
- **flask-limiter**: Rate limiting for web API
- **redis**: Caching and session management
- **streamlit**: Web dashboard framework (future)

### External API Dependencies

- **plaid**: Financial data integration
- **openai**: GPT integration for peer generation
- **boto3**: AWS Secrets Manager integration

### Configuration Dependencies

- **pyyaml**: YAML configuration file handling

## 🛠️ Helper Utilities

### Display Utilities (`helpers_display.py`)

**Functions**:
- `_drop_factors()`: Remove presentation-only factor rows
- `_print_single_portfolio()`: Pretty-print risk and beta tables
- `compare_risk_tables()`: Side-by-side risk table comparison
- `compare_beta_tables()`: Factor beta table comparison

**Usage**:
```python
from helpers_display import compare_risk_tables

# Compare before/after risk metrics
comparison = compare_risk_tables(old_risk_df, new_risk_df)
```

### Input Processing (`helpers_input.py`)

**Functions**:
- `parse_delta()`: Parse what-if scenario changes
- `_parse_shift()`: Convert human-friendly shift strings to decimals

**Supported Formats**:
- `"+200bp"` → `0.02`
- `"-75bps"` → `-0.0075`
- `"1.5%"` → `0.015`
- `"-0.01"` → `-0.01`

**Precedence Rules**:
1. YAML `new_weights:` → full replacement
2. YAML `delta:` + literal shifts → merged changes
3. Literal shifts only → fallback option

### GPT Integration (`gpt_helpers.py`)

**Functions**:
- `interpret_portfolio_risk()`: GPT-based risk interpretation
- `generate_subindustry_peers()`: GPT-powered peer generation

**Features**:
- Professional risk analysis interpretation
- Automated peer group generation
- Error handling and validation
- Configurable model parameters

## 📚 Additional Resources

- [README.md](./README.md): Project overview and usage guide
- [portfolio.yaml](./portfolio.yaml): Example portfolio configuration
- [risk_limits.yaml](./risk_limits.yaml): Risk limit definitions
- [Financial Modeling Prep API](https://financialmodelingprep.com/developer/docs/): API documentation

---

**Architecture Version**: 1.0  
**Last Updated**: 2024  
**Maintainer**: Henry Souchien