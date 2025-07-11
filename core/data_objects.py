"""Core data objects for the risk module."""

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
    Stock analysis data container handling stock analysis parameters.
    
    Provides structured input for individual stock analysis operations.
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
        """Validate and normalize stock data."""
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
        """Get the cache key for this stock analysis configuration."""
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
        """Create StockData from ticker and optional date range."""
        return cls(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date
        )
    
    @classmethod
    def from_yaml_config(cls, ticker: str, yaml_path: str,
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> 'StockData':
        """Create StockData with factor proxies from YAML config."""
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
        """Create StockData with explicit factor proxies."""
        return cls(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            factor_proxies=factor_proxies
        )
    
    def has_factor_analysis(self) -> bool:
        """Check if this stock data includes factor analysis configuration."""
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
    Portfolio data container handling real portfolio input formats.
    
    Supports all formats from create_portfolio_yaml:
    - Shares/dollars: {"AAPL": {"shares": 100}, "SPY": {"dollars": 5000}}
    - Percentages: {"AAPL": 25.0, "SPY": 75.0}
    - Weights: {"AAPL": 0.25, "SPY": 0.75}
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
        """Validate and standardize portfolio input."""
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
        """Get list of portfolio tickers."""
        return list(self.standardized_input.keys())
    
    def get_weights(self) -> Dict[str, float]:
        """Get portfolio weights (calculated from standardized input)."""
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
        """Get the cache key for this portfolio configuration."""
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
        """Create PortfolioData from YAML file."""
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
        """Create PortfolioData from holdings dictionary."""
        return cls(
            portfolio_input=holdings,
            standardized_input={},  # Will be set in __post_init__
            start_date=start_date,
            end_date=end_date,
            expected_returns=expected_returns or {},
            stock_factor_proxies=stock_factor_proxies or {}
        )
    
    def to_yaml(self, output_path: str) -> None:
        """Save portfolio data to YAML file."""
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


 