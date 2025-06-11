"""
Core backtesting engine for the trading platform.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
from pathlib import Path

from ..common.base_classes import (
    BaseStrategy, Order, Position, OrderType, OrderSide, OrderStatus,
    calculate_returns, calculate_volatility, calculate_sharpe_ratio, calculate_max_drawdown
)
from ..config.config import get_config


class BacktestEngine:
    """
    Main backtesting engine that simulates trading strategies.
    """
    
    def __init__(
        self,
        strategy: BaseStrategy,
        data_source: str = "csv",
        initial_capital: float = None,
        commission_rate: float = None,
        slippage_rate: float = None
    ):
        """
        Initialize the backtest engine.
        
        Args:
            strategy: Trading strategy to test
            data_source: Type of data source ('csv', 'api', etc.)
            initial_capital: Starting capital amount
            commission_rate: Commission rate per trade
            slippage_rate: Slippage rate for market orders
        """
        self.strategy = strategy
        self.data_source = data_source
        
        # Load configuration
        config = get_config()
        self.initial_capital = initial_capital or config.backtest.initial_capital
        self.commission_rate = commission_rate or config.backtest.commission_rate
        self.slippage_rate = slippage_rate or config.backtest.slippage_rate
        
        # Initialize state
        self.current_capital = self.initial_capital
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.trades: List[Dict] = []
        self.portfolio_history: List[Dict] = []
        self.current_date = None
        
        # Performance metrics
        self.metrics = {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def load_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Load historical data for backtesting.
        
        Args:
            symbols: List of symbols to load data for
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            Dictionary mapping symbols to their data
        """
        data = {}
        config = get_config()
        data_path = config.get_data_path("processed")
        
        for symbol in symbols:
            try:
                # Try to load from processed data first
                file_path = data_path / f"{symbol}_daily.csv"
                if file_path.exists():
                    df = pd.read_csv(file_path)
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                    
                    # Filter by date range
                    df = df.loc[start_date:end_date]
                    data[symbol] = df
                    self.logger.info(f"Loaded data for {symbol}: {len(df)} records")
                else:
                    self.logger.warning(f"No data file found for {symbol}")
                    
            except Exception as e:
                self.logger.error(f"Error loading data for {symbol}: {e}")
        
        return data
    
    def run_backtest(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        rebalance_frequency: str = "daily"
    ) -> Dict[str, Any]:
        """
        Run the backtest simulation.
        
        Args:
            symbols: List of symbols to trade
            start_date: Start date for backtest
            end_date: End date for backtest
            rebalance_frequency: How often to rebalance ('daily', 'weekly', 'monthly')
            
        Returns:
            Dictionary containing backtest results
        """
        self.logger.info(f"Starting backtest from {start_date} to {end_date}")
        
        # Load data
        data = self.load_data(symbols, start_date, end_date)
        if not data:
            raise ValueError("No data loaded for backtesting")
        
        # Get all trading dates
        all_dates = set()
        for symbol_data in data.values():
            all_dates.update(symbol_data.index)
        trading_dates = sorted(list(all_dates))
        
        # Run simulation
        for i, current_date in enumerate(trading_dates):
            self.current_date = current_date
            
            # Get current market data
            current_data = {}
            for symbol, symbol_data in data.items():
                if current_date in symbol_data.index:
                    current_data[symbol] = symbol_data.loc[current_date]
            
            if not current_data:
                continue
            
            # Process pending orders
            self._process_orders(current_data)
            
            # Check if it's time to rebalance
            if self._should_rebalance(current_date, rebalance_frequency, i):
                # Generate signals
                market_data = self._prepare_market_data(data, current_date)
                signals = self.strategy.generate_signals(market_data)
                
                # Generate orders based on signals
                new_orders = self._generate_orders(signals, current_data)
                self.orders.extend(new_orders)
            
            # Update portfolio value
            portfolio_value = self._calculate_portfolio_value(current_data)
            self.portfolio_history.append({
                'date': current_date,
                'portfolio_value': portfolio_value,
                'cash': self.current_capital,
                'positions': {k: v.to_dict() for k, v in self.positions.items()}
            })
        
        # Calculate performance metrics
        self.metrics = self._calculate_metrics()
        
        self.logger.info("Backtest completed successfully")
        return self._get_results()
    
    # ...existing helper methods...
    
    def _calculate_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics."""
        if not self.portfolio_history:
            return {}
        
        # Extract portfolio values
        df = pd.DataFrame(self.portfolio_history)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        portfolio_values = df['portfolio_value']
        returns = calculate_returns(portfolio_values)
        
        metrics = {
            'total_return': (portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1) * 100,
            'annualized_return': ((portfolio_values.iloc[-1] / portfolio_values.iloc[0]) ** (252 / len(portfolio_values)) - 1) * 100,
            'volatility': calculate_volatility(returns) * 100,
            'sharpe_ratio': calculate_sharpe_ratio(returns),
            'max_drawdown': calculate_max_drawdown(portfolio_values) * 100,
            'total_trades': len(self.trades),
            'final_portfolio_value': portfolio_values.iloc[-1],
            'peak_portfolio_value': portfolio_values.max(),
        }
        
        return metrics


# Legacy compatibility
class Backtester:
    def __init__(self, strategy, data):
        self.strategy = strategy
        self.data = data
        self.results = None

    def run_backtest(self):
        self.strategy.initialize()
        for index, row in self.data.iterrows():
            self.strategy.execute(row)
        self.results = self.strategy.get_results()

    def analyze_results(self):
        if self.results is None:
            raise ValueError("No results to analyze. Please run the backtest first.")
        # Implement analysis logic here
        return {
            "total_return": self.results["total_return"],
            "sharpe_ratio": self.results["sharpe_ratio"],
            "max_drawdown": self.results["max_drawdown"],
        }