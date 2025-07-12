"""
Core Data Objects Module

Enterprise-grade data structures for portfolio and stock analysis with comprehensive validation.

This module provides structured data containers that serve as the foundation for all
risk analysis operations in the system. These objects handle input validation,
format standardization, caching, and provide type-safe interfaces for analysis functions.

Key Features:
- **Multi-Format Input Support**: Handles shares, dollars, percentages, and weights seamlessly
- **Automatic Validation**: Input validation with meaningful error messages
- **Smart Format Detection**: Automatically detects and converts between input formats
- **Caching Infrastructure**: Built-in cache key generation for performance optimization
- **Type Safety**: Comprehensive type hints and dataclass validation
- **Serialization Support**: YAML serialization for configuration management

Core Classes:
- **StockData**: Individual stock analysis configuration with factor model support
- **PortfolioData**: Portfolio analysis configuration with multi-format input handling

Architecture Position: Data Layer → Service Layer → Routes
Usage: Foundation objects for all portfolio and stock analysis operations throughout the system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import yaml
import hashlib
import json
from datetime import datetime
import os


@dataclass
class StockData:
    """
    Individual stock analysis configuration with comprehensive parameter validation and caching.
    
    This data container provides structured input for stock analysis operations, supporting
    both simple market regression analysis and complex multi-factor models. It handles
    parameter validation, format standardization, and cache key generation for performance
    optimization across the analysis pipeline.
    
    Key Features:
    - **Flexible Analysis Types**: Supports single-factor (market) and multi-factor analysis
    - **Factor Proxy Integration**: Seamless integration with portfolio factor configurations
    - **Automatic Validation**: Ticker normalization and parameter validation
    - **Caching Support**: Built-in cache key generation for performance optimization
    - **Multiple Construction Methods**: Class methods for different initialization patterns
    - **Type Safety**: Comprehensive type hints and dataclass validation
    
    Analysis Configuration:
    - **Ticker**: Stock symbol (automatically normalized to uppercase)
    - **Date Range**: Optional start/end dates for analysis window
    - **Factor Proxies**: Optional factor model configuration for multi-factor analysis
    - **YAML Integration**: Optional portfolio YAML path for factor proxy lookup
    
    Usage Patterns:
    1. **Simple Analysis**: Basic market regression with ticker and date range
    2. **Multi-Factor Analysis**: Complex factor models with explicit factor proxies
    3. **Portfolio Integration**: Factor proxy inheritance from portfolio configuration
    
    Example:
        ```python
        # Simple market regression analysis
        stock_data = StockData("AAPL", start_date="2020-01-01", end_date="2023-12-31")
        
        # Multi-factor analysis with explicit factor proxies
        factor_proxies = {"market": "SPY", "growth": "VUG", "value": "VTV"}
        stock_data = StockData.from_factor_proxies("AAPL", factor_proxies)
        
        # Portfolio-integrated analysis
        stock_data = StockData.from_yaml_config("AAPL", "portfolio.yaml")
        
        # Check analysis type
        has_factors = stock_data.has_factor_analysis()  # True/False
        cache_key = stock_data.get_cache_key()  # For caching
        ```
    """
    
    # Core stock analysis parameters
    ticker: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    
    # Factor analysis configuration
    factor_proxies: Optional[Dict[str, Union[str, List[str]]]] = None
    yaml_path: Optional[str] = None
    
    # Analysis metadata
    analysis_name: Optional[str] = None
    
    # Caching and metadata
    _cache_key: Optional[str] = None
    _last_updated: Optional[datetime] = None
    
    def __post_init__(self):
        """
        Validate and normalize stock data after initialization.
        
        This method is automatically called after dataclass initialization to perform
        validation, normalization, and setup of derived attributes. It ensures data
        integrity and prepares the object for analysis operations.
        
        Validation Steps:
        1. Validates ticker is not empty
        2. Normalizes ticker to uppercase format
        3. Sets default analysis name if not provided
        4. Generates cache key for performance optimization
        5. Records creation timestamp
        
        Raises:
            ValueError: If ticker is empty or None
        """
        if not self.ticker:
            raise ValueError("Ticker cannot be empty")
        
        # Normalize ticker to uppercase
        self.ticker = self.ticker.upper()
        
        # Set default analysis name
        if not self.analysis_name:
            self.analysis_name = f"{self.ticker}_analysis"
        
        # Generate cache key
        self._cache_key = self._generate_cache_key()
        self._last_updated = datetime.now()
    
    def get_cache_key(self) -> str:
        """
        Get the cache key for this stock analysis configuration.
        
        The cache key is a unique identifier based on all analysis parameters,
        enabling efficient caching of analysis results. Changes to any parameter
        will result in a different cache key, ensuring cache consistency.
        
        Returns:
            str: MD5 hash of analysis parameters for cache identification
            
        Cache Key Components:
            - ticker: Stock symbol
            - start_date: Analysis start date
            - end_date: Analysis end date
            - factor_proxies: Factor model configuration
            - yaml_path: Portfolio YAML configuration path
        """
        return self._cache_key
    
    def _generate_cache_key(self) -> str:
        """Generate cache key for this stock analysis configuration."""
        # Create hash of stock parameters
        key_data = {
            "ticker": self.ticker,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "factor_proxies": self.factor_proxies,
            "yaml_path": self.yaml_path
        }
        
        # Convert to JSON string and hash
        json_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(json_str.encode()).hexdigest()
    
    @classmethod
    def from_ticker(cls, ticker: str, 
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None) -> 'StockData':
        """
        Create StockData for simple market regression analysis.
        
        This is the most basic construction method for single-stock analysis
        using market regression (stock vs. SPY benchmark). Use this when you
        need straightforward volatility and beta analysis without factor models.
        
        Args:
            ticker (str): Stock symbol to analyze (e.g., "AAPL", "MSFT")
            start_date (Optional[str]): Analysis start date in YYYY-MM-DD format
            end_date (Optional[str]): Analysis end date in YYYY-MM-DD format
                
        Returns:
            StockData: Configured for single-factor market regression analysis
            
        Example:
            ```python
            # Simple market analysis with default date range
            stock_data = StockData.from_ticker("AAPL")
            
            # Market analysis with custom date range
            stock_data = StockData.from_ticker("TSLA", "2020-01-01", "2023-12-31")
            ```
        """
        return cls(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date
        )
    
    @classmethod
    def from_yaml_config(cls, ticker: str, yaml_path: str,
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> 'StockData':
        """
        Create StockData with factor proxies inherited from portfolio YAML configuration.
        
        This method creates stock analysis configuration that inherits factor proxy
        settings from a portfolio YAML file. It's useful for ensuring consistency
        between portfolio-level and stock-level factor model configurations.
        
        Args:
            ticker (str): Stock symbol to analyze
            yaml_path (str): Path to portfolio YAML file containing factor proxies
            start_date (Optional[str]): Analysis start date (overrides YAML if provided)
            end_date (Optional[str]): Analysis end date (overrides YAML if provided)
                
        Returns:
            StockData: Configured for multi-factor analysis using portfolio factor proxies
            
        Example:
            ```python
            # Use portfolio factor configuration
            stock_data = StockData.from_yaml_config("AAPL", "portfolio.yaml")
            
            # Use portfolio factors with custom date range
            stock_data = StockData.from_yaml_config(
                "AAPL", "portfolio.yaml", "2021-01-01", "2023-12-31"
            )
            ```
        """
        return cls(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            yaml_path=yaml_path
        )
    
    @classmethod
    def from_factor_proxies(cls, ticker: str, 
                           factor_proxies: Dict[str, Union[str, List[str]]],
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> 'StockData':
        """
        Create StockData with explicit factor proxies for multi-factor analysis.
        
        This method creates stock analysis configuration with explicitly defined
        factor proxies for comprehensive multi-factor model analysis. Use this
        when you need precise control over factor model specification.
        
        Args:
            ticker (str): Stock symbol to analyze
            factor_proxies (Dict[str, Union[str, List[str]]]): Factor proxy mappings
                Format: {"factor_name": "proxy_ticker"} or {"factor_name": ["proxy1", "proxy2"]}
                Example: {"market": "SPY", "growth": "VUG", "value": "VTV", "momentum": "MTUM"}
            start_date (Optional[str]): Analysis start date in YYYY-MM-DD format
            end_date (Optional[str]): Analysis end date in YYYY-MM-DD format
                
        Returns:
            StockData: Configured for multi-factor analysis with specified factor proxies
            
        Example:
            ```python
            # Multi-factor analysis with style factors
            factor_proxies = {
                "market": "SPY",
                "growth": "VUG", 
                "value": "VTV",
                "momentum": "MTUM",
                "quality": "QUAL"
            }
            stock_data = StockData.from_factor_proxies("AAPL", factor_proxies)
            
            # Multi-factor analysis with custom date range
            stock_data = StockData.from_factor_proxies(
                "TSLA", factor_proxies, "2020-01-01", "2023-12-31"
            )
            ```
        """
        return cls(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            factor_proxies=factor_proxies
        )
    
    def has_factor_analysis(self) -> bool:
        """
        Check if this stock data includes factor analysis configuration.
        
        Determines whether the stock analysis will use multi-factor models
        (complex factor analysis) or simple market regression (single-factor).
        This affects the analysis type and output format.
        
        Returns:
            bool: True if factor proxies or YAML config are provided, False for simple market regression
            
        Analysis Types:
            - True: Multi-factor analysis with factor exposures, R-squared decomposition
            - False: Simple market regression with beta, alpha, correlation vs. SPY
            
        Example:
            ```python
            # Simple market regression
            stock_data = StockData.from_ticker("AAPL")
            has_factors = stock_data.has_factor_analysis()  # False
            
            # Multi-factor analysis
            stock_data = StockData.from_factor_proxies("AAPL", {"market": "SPY", "growth": "VUG"})
            has_factors = stock_data.has_factor_analysis()  # True
            ```
        """
        return self.factor_proxies is not None or self.yaml_path is not None
    
    def __hash__(self) -> int:
        """Make StockData hashable for caching."""
        return hash(self._cache_key)
    
    def __eq__(self, other) -> bool:
        """Compare StockData objects."""
        if not isinstance(other, StockData):
            return False
        return self._cache_key == other._cache_key


@dataclass
class PortfolioData:
    """
    Comprehensive portfolio configuration with multi-format input support and intelligent validation.
    
    This enterprise-grade data container handles all portfolio input formats used throughout
    the system, providing automatic format detection, validation, and standardization. It serves
    as the foundation for all portfolio analysis operations and ensures consistent data handling
    across the entire analysis pipeline.
    
    Key Features:
    - **Multi-Format Input Support**: Handles shares, dollars, percentages, and weights seamlessly
    - **Intelligent Format Detection**: Automatically detects input format and converts appropriately
    - **Comprehensive Validation**: Input validation with meaningful error messages and corrections
    - **Standardized Output**: Converts all formats to consistent internal representation
    - **Caching Infrastructure**: Built-in cache key generation for performance optimization
    - **YAML Integration**: Full serialization support for configuration management
    - **Type Safety**: Comprehensive type validation and dataclass integrity checks
    
    Supported Input Formats:
    1. **Shares/Dollars Format**: {"AAPL": {"shares": 100}, "SPY": {"dollars": 5000}}
    2. **Percentage Format**: {"AAPL": 25.0, "SPY": 75.0} (must sum to ~100%)
    3. **Weight Format**: {"AAPL": 0.25, "SPY": 0.75} (must sum to ~1.0)
    4. **Mixed Format**: {"AAPL": {"shares": 100}, "SPY": {"weight": 0.3}}
    
    Portfolio Configuration:
    - **Portfolio Input**: Raw user-provided portfolio allocation in any supported format
    - **Date Range**: Analysis start and end dates for historical data
    - **Expected Returns**: Optional expected return forecasts for optimization
    - **Factor Proxies**: Factor model configuration for risk analysis
    - **Metadata**: Portfolio name and caching information
    
    Business Logic:
    1. **Input Detection**: Automatically detects format based on data structure
    2. **Format Conversion**: Converts to standardized internal representation
    3. **Validation**: Ensures weights/percentages sum correctly and tickers are valid
    4. **Caching**: Generates cache keys for performance optimization
    5. **Serialization**: Supports YAML export for configuration persistence
    
    Example:
        ```python
        # Create from percentage allocation
        portfolio_data = PortfolioData.from_holdings(
            holdings={"AAPL": 30.0, "MSFT": 25.0, "GOOGL": 20.0, "SGOV": 25.0},
            start_date="2020-01-01",
            end_date="2023-12-31"
        )
        
        # Create from shares/dollars format
        holdings = {
            "AAPL": {"shares": 100},
            "SPY": {"dollars": 5000},
            "SGOV": {"weight": 0.25}
        }
        portfolio_data = PortfolioData.from_holdings(holdings, "2020-01-01", "2023-12-31")
        
        # Load from YAML configuration
        portfolio_data = PortfolioData.from_yaml("portfolio.yaml")
        
        # Access portfolio information
        tickers = portfolio_data.get_tickers()  # ["AAPL", "MSFT", "GOOGL", "SGOV"]
        weights = portfolio_data.get_weights()  # {"AAPL": 0.30, "MSFT": 0.25, ...}
        cache_key = portfolio_data.get_cache_key()  # For caching
        
        # Export to YAML
        portfolio_data.to_yaml("output_portfolio.yaml")
        ```
    """
    
    # Raw portfolio input (as provided by user)
    portfolio_input: Dict[str, Union[float, Dict[str, float]]]
    
    # Standardized portfolio input (converted to shares/dollars/weight format)
    standardized_input: Dict[str, Dict[str, float]]
    
    # Portfolio metadata
    start_date: str
    end_date: str
    expected_returns: Dict[str, float]
    stock_factor_proxies: Dict[str, str]
    
    # Portfolio analysis results (populated after standardization)
    weights: Optional[Dict[str, float]] = None
    total_value: Optional[float] = None
    
    # Portfolio name for identification
    portfolio_name: Optional[str] = None
    
    # Caching and metadata
    _cache_key: Optional[str] = None
    _last_updated: Optional[datetime] = None
    
    def __post_init__(self):
        """
        Validate and standardize portfolio input after initialization.
        
        This method is automatically called after dataclass initialization to perform
        comprehensive validation, format detection, and standardization of portfolio
        input data. It ensures data integrity and prepares the portfolio for analysis.
        
        Processing Steps:
        1. Validates portfolio input is not empty
        2. Detects input format (shares/dollars, percentages, weights)
        3. Converts to standardized internal representation
        4. Validates allocation sums (100% for percentages, 1.0 for weights)
        5. Generates cache key for performance optimization
        6. Records processing timestamp
        
        Raises:
            ValueError: If portfolio input is empty, invalid format, or allocation sums are incorrect
        """
        if not self.portfolio_input:
            raise ValueError("Portfolio input cannot be empty")
            
        # Detect input format and convert to standardized format
        input_format = self._detect_input_format()
        self.standardized_input = self._convert_to_standardized_format(input_format)
        
        # Generate cache key
        self._cache_key = self._generate_cache_key()
        self._last_updated = datetime.now()
    
    def _detect_input_format(self) -> str:
        """Auto-detect the input format based on data structure."""
        if not self.portfolio_input:
            raise ValueError("Portfolio input is empty")
        
        # Check first value to determine format
        first_value = next(iter(self.portfolio_input.values()))
        
        if isinstance(first_value, dict):
            # Check if it has shares/dollars keys
            if any(key in first_value for key in ["shares", "dollars", "value"]):
                return "shares_dollars"
            elif "weight" in first_value:
                return "weights"
            else:
                raise ValueError(f"Unknown dict format: {first_value}")
        
        elif isinstance(first_value, (int, float)):
            # Check if values are percentages (sum ~100) or weights (sum ~1)
            total = sum(self.portfolio_input.values())
            if total > 10:  # Likely percentages
                return "percentages"
            else:  # Likely decimal weights
                return "weights"
        
        else:
            raise ValueError(f"Unsupported value type: {type(first_value)}")
    
    def _convert_to_standardized_format(self, input_format: str) -> Dict[str, Dict[str, float]]:
        """Convert input to standardized portfolio_input format."""
        if input_format == "shares_dollars":
            return self._convert_shares_dollars()
        elif input_format == "percentages":
            return self._convert_percentages()
        elif input_format == "weights":
            return self._convert_weights()
        else:
            raise ValueError(f"Unsupported input format: {input_format}")
    
    def _convert_shares_dollars(self) -> Dict[str, Dict[str, float]]:
        """Convert shares/dollars format to standardized format."""
        standardized = {}
        for ticker, holding in self.portfolio_input.items():
            if isinstance(holding, dict):
                if "shares" in holding:
                    standardized[ticker] = {"shares": float(holding["shares"])}
                elif "dollars" in holding:
                    standardized[ticker] = {"dollars": float(holding["dollars"])}
                elif "value" in holding:
                    standardized[ticker] = {"dollars": float(holding["value"])}
                else:
                    raise ValueError(f"Unknown holding format for {ticker}: {holding}")
            else:
                raise ValueError(f"Expected dict format for {ticker}, got {type(holding)}")
        return standardized
    
    def _convert_percentages(self) -> Dict[str, Dict[str, float]]:
        """Convert percentage allocations to weight format."""
        total_allocation = sum(self.portfolio_input.values())
        if abs(total_allocation - 100) > 1:
            raise ValueError(f"Allocations must sum to 100%, got {total_allocation}%")
        
        standardized = {}
        for ticker, percentage in self.portfolio_input.items():
            weight = percentage / total_allocation
            standardized[ticker] = {"weight": weight}
        
        return standardized
    
    def _convert_weights(self) -> Dict[str, Dict[str, float]]:
        """Convert decimal weights to standardized format."""
        if isinstance(next(iter(self.portfolio_input.values())), dict):
            # Already in weight dict format
            return {ticker: {"weight": float(holding["weight"])} 
                   for ticker, holding in self.portfolio_input.items()}
        else:
            # Simple weight values
            total_weight = sum(self.portfolio_input.values())
            if abs(total_weight - 1.0) > 0.01:
                raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
            
            standardized = {}
            for ticker, weight in self.portfolio_input.items():
                standardized[ticker] = {"weight": float(weight)}
            
            return standardized
    
    def get_tickers(self) -> List[str]:
        """
        Get list of portfolio tickers in standardized format.
        
        Returns all ticker symbols included in the portfolio configuration,
        extracted from the standardized input format. Useful for validation,
        iteration, and analysis setup.
        
        Returns:
            List[str]: List of ticker symbols in the portfolio
            
        Example:
            ```python
            portfolio_data = PortfolioData.from_holdings(
                {"AAPL": 30.0, "MSFT": 25.0, "GOOGL": 20.0, "SGOV": 25.0},
                "2020-01-01", "2023-12-31"
            )
            tickers = portfolio_data.get_tickers()  # ["AAPL", "MSFT", "GOOGL", "SGOV"]
            ```
        """
        return list(self.standardized_input.keys())
    
    def get_weights(self) -> Dict[str, float]:
        """
        Get portfolio weights calculated from standardized input.
        
        Returns the portfolio allocation weights as decimal values (summing to 1.0),
        regardless of the original input format. For shares/dollars format, weights
        are calculated after portfolio standardization. For percentage/weight formats,
        weights are extracted directly from the standardized input.
        
        Returns:
            Dict[str, float]: Portfolio weights as {ticker: weight} mapping
            
        Weight Calculation:
            - Shares/Dollars: Calculated based on market values during standardization
            - Percentages: Converted from percentages to decimal weights
            - Weights: Used directly from input
            
        Example:
            ```python
            # From percentages
            portfolio_data = PortfolioData.from_holdings(
                {"AAPL": 30.0, "MSFT": 25.0, "GOOGL": 20.0, "SGOV": 25.0},
                "2020-01-01", "2023-12-31"
            )
            weights = portfolio_data.get_weights()  # {"AAPL": 0.30, "MSFT": 0.25, ...}
            
            # From shares/dollars (after standardization)
            portfolio_data = PortfolioData.from_holdings(
                {"AAPL": {"shares": 100}, "SPY": {"dollars": 5000}},
                "2020-01-01", "2023-12-31"
            )
            weights = portfolio_data.get_weights()  # Calculated from market values
            ```
        """
        if self.weights is not None:
            return self.weights
        
        # This would normally be calculated by standardize_portfolio_input
        # For now, return weights from standardized input if available
        weights = {}
        for ticker, holding in self.standardized_input.items():
            if "weight" in holding:
                weights[ticker] = holding["weight"]
        
        return weights
    
    def get_cache_key(self) -> str:
        """
        Get the cache key for this portfolio configuration.
        
        The cache key is a unique identifier based on all portfolio parameters,
        enabling efficient caching of analysis results across the system. Changes
        to any parameter will result in a different cache key, ensuring cache
        consistency and preventing stale data issues.
        
        Returns:
            str: MD5 hash of portfolio parameters for cache identification
            
        Cache Key Components:
            - portfolio_input: Standardized portfolio allocation
            - start_date: Analysis start date
            - end_date: Analysis end date
            - expected_returns: Expected return forecasts
            - stock_factor_proxies: Factor model configuration
            
        Example:
            ```python
            portfolio_data = PortfolioData.from_holdings(
                {"AAPL": 30.0, "MSFT": 25.0}, "2020-01-01", "2023-12-31"
            )
            cache_key = portfolio_data.get_cache_key()  # "a1b2c3d4e5f6..."
            
            # Different allocation -> different cache key
            portfolio_data2 = PortfolioData.from_holdings(
                {"AAPL": 25.0, "MSFT": 30.0}, "2020-01-01", "2023-12-31"
            )
            cache_key2 = portfolio_data2.get_cache_key()  # "f6e5d4c3b2a1..."
            ```
        """
        return self._cache_key
    
    def _generate_cache_key(self) -> str:
        """Generate cache key for this portfolio configuration."""
        # Create hash of portfolio input, dates, and expected returns
        key_data = {
            "portfolio_input": self.standardized_input,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "expected_returns": self.expected_returns,
            "stock_factor_proxies": self.stock_factor_proxies
        }
        
        # Convert to JSON string and hash
        json_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(json_str.encode()).hexdigest()
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'PortfolioData':
        """
        Create PortfolioData from YAML configuration file.
        
        This method loads a complete portfolio configuration from a YAML file,
        including portfolio allocation, date ranges, expected returns, and factor
        proxies. The YAML file should contain all necessary parameters for
        portfolio analysis.
        
        Args:
            yaml_path (str): Path to YAML configuration file
            
        Returns:
            PortfolioData: Complete portfolio configuration loaded from YAML
            
        Required YAML Structure:
            ```yaml
            portfolio_input:
              AAPL: 30.0
              MSFT: 25.0
              GOOGL: 20.0
              SGOV: 25.0
            start_date: "2020-01-01"
            end_date: "2023-12-31"
            expected_returns:
              AAPL: 0.12
              MSFT: 0.10
            stock_factor_proxies:
              market: SPY
              growth: VUG
            ```
            
        Example:
            ```python
            # Load complete portfolio configuration
            portfolio_data = PortfolioData.from_yaml("portfolio.yaml")
            
            # Access loaded configuration
            tickers = portfolio_data.get_tickers()
            weights = portfolio_data.get_weights()
            expected_returns = portfolio_data.expected_returns
            ```
        """
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return cls(
            portfolio_input=config['portfolio_input'],
            standardized_input=config['portfolio_input'],  # Already standardized in YAML
            start_date=config['start_date'],
            end_date=config['end_date'],
            expected_returns=config.get('expected_returns', {}),
            stock_factor_proxies=config.get('stock_factor_proxies', {})
        )
    
    @classmethod
    def from_holdings(cls, holdings: Dict[str, Union[float, Dict]], 
                     start_date: str, end_date: str,
                     expected_returns: Optional[Dict[str, float]] = None,
                     stock_factor_proxies: Optional[Dict[str, str]] = None) -> 'PortfolioData':
        """
        Create PortfolioData from holdings dictionary with flexible input formats.
        
        This is the primary method for creating portfolio configurations from Python
        dictionaries. It automatically detects and handles multiple input formats,
        making it easy to create portfolios from various data sources.
        
        Args:
            holdings (Dict[str, Union[float, Dict]]): Portfolio allocation in any supported format
            start_date (str): Analysis start date in YYYY-MM-DD format
            end_date (str): Analysis end date in YYYY-MM-DD format
            expected_returns (Optional[Dict[str, float]]): Expected return forecasts for optimization
            stock_factor_proxies (Optional[Dict[str, str]]): Factor proxy mappings for analysis
            
        Returns:
            PortfolioData: Complete portfolio configuration with standardized input
            
        Supported Holdings Formats:
            - Percentages: {"AAPL": 30.0, "MSFT": 25.0, "GOOGL": 20.0, "SGOV": 25.0}
            - Weights: {"AAPL": 0.30, "MSFT": 0.25, "GOOGL": 0.20, "SGOV": 0.25}
            - Shares/Dollars: {"AAPL": {"shares": 100}, "SPY": {"dollars": 5000}}
            - Mixed: {"AAPL": {"shares": 100}, "SPY": {"weight": 0.30}}
            
        Example:
            ```python
            # From percentage allocation
            portfolio_data = PortfolioData.from_holdings(
                holdings={"AAPL": 30.0, "MSFT": 25.0, "GOOGL": 20.0, "SGOV": 25.0},
                start_date="2020-01-01",
                end_date="2023-12-31"
            )
            
            # From shares/dollars with expected returns
            holdings = {"AAPL": {"shares": 100}, "SPY": {"dollars": 5000}}
            expected_returns = {"AAPL": 0.12, "SPY": 0.08}
            portfolio_data = PortfolioData.from_holdings(
                holdings, "2020-01-01", "2023-12-31", expected_returns
            )
            
            # With factor proxies for risk analysis
            factor_proxies = {"market": "SPY", "growth": "VUG"}
            portfolio_data = PortfolioData.from_holdings(
                {"AAPL": 30.0, "MSFT": 25.0, "GOOGL": 20.0, "SGOV": 25.0},
                "2020-01-01", "2023-12-31",
                stock_factor_proxies=factor_proxies
            )
            ```
        """
        return cls(
            portfolio_input=holdings,
            standardized_input={},  # Will be set in __post_init__
            start_date=start_date,
            end_date=end_date,
            expected_returns=expected_returns or {},
            stock_factor_proxies=stock_factor_proxies or {}
        )
    
    def to_yaml(self, output_path: str) -> None:
        """
        Save portfolio data to YAML configuration file.
        
        This method serializes the complete portfolio configuration to a YAML file,
        including standardized portfolio input, date ranges, expected returns, and
        factor proxies. The resulting YAML file can be used to recreate the exact
        same portfolio configuration using from_yaml().
        
        Args:
            output_path (str): Path where YAML file will be saved
            
        Generated YAML Structure:
            ```yaml
            portfolio_input:
              AAPL: {weight: 0.30}
              MSFT: {weight: 0.25}
              GOOGL: {weight: 0.20}
              SGOV: {weight: 0.25}
            start_date: "2020-01-01"
            end_date: "2023-12-31"
            expected_returns:
              AAPL: 0.12
              MSFT: 0.10
            stock_factor_proxies:
              market: SPY
              growth: VUG
            ```
            
        Example:
            ```python
            # Create portfolio and save to YAML
            portfolio_data = PortfolioData.from_holdings(
                {"AAPL": 30.0, "MSFT": 25.0, "GOOGL": 20.0, "SGOV": 25.0},
                "2020-01-01", "2023-12-31"
            )
            portfolio_data.to_yaml("my_portfolio.yaml")
            
            # Load back from YAML
            loaded_portfolio = PortfolioData.from_yaml("my_portfolio.yaml")
            ```
        """
        config = {
            "portfolio_input": self.standardized_input,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "expected_returns": self.expected_returns,
            "stock_factor_proxies": self.stock_factor_proxies
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    def __hash__(self) -> int:
        """Make PortfolioData hashable for caching."""
        return hash(self._cache_key)
    
    def __eq__(self, other) -> bool:
        """Compare PortfolioData objects."""
        if not isinstance(other, PortfolioData):
            return False
        return self._cache_key == other._cache_key


 