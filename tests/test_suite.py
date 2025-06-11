"""
Comprehensive Testing Framework for Quantitative Trading Platform

This module provides unit tests and integration tests for all major components
of the trading platform to ensure reliability and correctness.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
from unittest.mock import Mock, patch, MagicMock
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import modules to test
from src.indicators.technical_indicators import (
    MovingAverage, ExponentialMovingAverage, RSI, MACD, BollingerBands
)
from src.risk_management.risk_manager import PositionSizeCalculator, RiskMetrics
from src.portfolio.portfolio_manager import Position, Portfolio, PortfolioManager
from src.data.loaders.data_loader import DataLoader
from src.data.processors.data_processor import DataProcessor
from src.strategies.momentum.moving_average_crossover import MovingAverageCrossoverStrategy
from src.core.backtester import BacktestEngine
from src.config.config import Config


class TestTechnicalIndicators(unittest.TestCase):
    """Test technical indicators functionality"""
    
    def setUp(self):
        """Set up test data"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)  # For reproducible tests
        
        # Create realistic price data
        base_price = 100
        returns = np.random.normal(0.001, 0.02, 100)  # Daily returns ~0.1% mean, 2% std
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        self.price_data = pd.Series(prices, index=dates)
        self.ohlcv_data = pd.DataFrame({
            'open': self.price_data * (1 + np.random.normal(0, 0.001, 100)),
            'high': self.price_data * (1 + abs(np.random.normal(0, 0.005, 100))),
            'low': self.price_data * (1 - abs(np.random.normal(0, 0.005, 100))),
            'close': self.price_data,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
    
    def test_moving_average(self):
        """Test Moving Average indicator"""
        ma = MovingAverage(period=20)
        result = ma.calculate(self.price_data)
        
        # Test basic properties
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(self.price_data))
        
        # Test that first 19 values are NaN (due to 20-period window)
        self.assertTrue(pd.isna(result.iloc[:19]).all())
        
        # Test that MA is calculated correctly for a known case
        expected_ma_20 = self.price_data.iloc[:20].mean()
        self.assertAlmostEqual(result.iloc[19], expected_ma_20, places=5)
    
    def test_exponential_moving_average(self):
        """Test Exponential Moving Average indicator"""
        ema = ExponentialMovingAverage(period=12)
        result = ema.calculate(self.price_data)
        
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(self.price_data))
        
        # EMA should have fewer NaN values than SMA
        self.assertFalse(pd.isna(result.iloc[11:]).any())
    
    def test_rsi(self):
        """Test RSI indicator"""
        rsi = RSI(period=14)
        result = rsi.calculate(self.price_data)
        
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(self.price_data))
        
        # RSI should be between 0 and 100
        valid_rsi = result.dropna()
        self.assertTrue((valid_rsi >= 0).all())
        self.assertTrue((valid_rsi <= 100).all())
    
    def test_macd(self):
        """Test MACD indicator"""
        macd = MACD(fast_period=12, slow_period=26, signal_period=9)
        result = macd.calculate(self.price_data)
        
        self.assertIsInstance(result, dict)
        self.assertIn('macd', result)
        self.assertIn('signal', result)
        self.assertIn('histogram', result)
        
        # Test that histogram = macd - signal
        valid_idx = ~(pd.isna(result['macd']) | pd.isna(result['signal']))
        macd_diff = result['macd'][valid_idx] - result['signal'][valid_idx]
        histogram = result['histogram'][valid_idx]
        
        pd.testing.assert_series_equal(macd_diff, histogram, check_names=False)
    
    def test_bollinger_bands(self):
        """Test Bollinger Bands indicator"""
        bb = BollingerBands(period=20, std_dev=2.0)
        result = bb.calculate(self.price_data)
        
        self.assertIsInstance(result, dict)
        required_keys = ['upper', 'middle', 'lower', 'width']
        for key in required_keys:
            self.assertIn(key, result)
        
        # Test that upper > middle > lower
        valid_idx = ~pd.isna(result['upper'])
        self.assertTrue((result['upper'][valid_idx] >= result['middle'][valid_idx]).all())
        self.assertTrue((result['middle'][valid_idx] >= result['lower'][valid_idx]).all())


class TestRiskManagement(unittest.TestCase):
    """Test risk management functionality"""
    
    def setUp(self):
        """Set up test data"""
        self.account_balance = 100000
        self.risk_per_trade = 0.02
        self.entry_price = 100
        self.stop_loss = 95
        
        # Create sample returns data
        np.random.seed(42)
        self.returns = pd.Series(np.random.normal(0.001, 0.02, 252))  # 1 year of daily returns
    
    def test_position_size_calculator(self):
        """Test position sizing calculations"""
        calc = PositionSizeCalculator()
        
        # Test fixed fractional sizing
        position_size = calc.calculate_position_size(
            account_balance=self.account_balance,
            risk_per_trade=self.risk_per_trade,
            entry_price=self.entry_price,
            stop_loss=self.stop_loss,
            method='fixed_fractional'
        )
        
        # Expected: (100000 * 0.02) / (100 - 95) = 2000 / 5 = 400 shares
        expected_size = (self.account_balance * self.risk_per_trade) / (self.entry_price - self.stop_loss)
        self.assertAlmostEqual(position_size, expected_size, places=2)
    
    def test_risk_metrics(self):
        """Test risk metrics calculations"""
        metrics = RiskMetrics()
        
        # Test Sharpe ratio
        sharpe = metrics.calculate_sharpe_ratio(self.returns)
        self.assertIsInstance(sharpe, float)
        self.assertFalse(np.isnan(sharpe))
        
        # Test VaR
        var_95 = metrics.calculate_var(self.returns, confidence_level=0.95)
        self.assertIsInstance(var_95, float)
        self.assertTrue(var_95 < 0)  # VaR should be negative
        
        # Test maximum drawdown
        prices = (1 + self.returns).cumprod() * 100
        max_dd = metrics.calculate_max_drawdown(prices)
        self.assertIsInstance(max_dd, float)
        self.assertTrue(max_dd <= 0)  # Max drawdown should be negative or zero


class TestPortfolioManagement(unittest.TestCase):
    """Test portfolio management functionality"""
    
    def setUp(self):
        """Set up test data"""
        self.initial_cash = 100000
        self.symbol = 'TEST'
        self.price = 100
        self.quantity = 500
    
    def test_position_creation(self):
        """Test position creation and management"""
        position = Position(
            symbol=self.symbol,
            quantity=self.quantity,
            entry_price=self.price,
            entry_time=datetime.now()
        )
        
        self.assertEqual(position.symbol, self.symbol)
        self.assertEqual(position.quantity, self.quantity)
        self.assertEqual(position.entry_price, self.price)
        self.assertEqual(position.market_value, self.quantity * self.price)
    
    def test_position_pnl(self):
        """Test position P&L calculation"""
        position = Position(
            symbol=self.symbol,
            quantity=self.quantity,
            entry_price=self.price,
            entry_time=datetime.now()
        )
        
        # Test unrealized P&L
        current_price = 105
        unrealized_pnl = position.calculate_unrealized_pnl(current_price)
        expected_pnl = self.quantity * (current_price - self.price)
        
        self.assertEqual(unrealized_pnl, expected_pnl)
    
    def test_portfolio_creation(self):
        """Test portfolio creation"""
        portfolio = Portfolio(initial_cash=self.initial_cash)
        
        self.assertEqual(portfolio.cash, self.initial_cash)
        self.assertEqual(len(portfolio.positions), 0)
        self.assertEqual(portfolio.total_value, self.initial_cash)
    
    def test_portfolio_add_position(self):
        """Test adding position to portfolio"""
        portfolio = Portfolio(initial_cash=self.initial_cash)
        
        # Add position
        portfolio.add_position(
            symbol=self.symbol,
            quantity=self.quantity,
            price=self.price
        )
        
        self.assertEqual(len(portfolio.positions), 1)
        self.assertIn(self.symbol, portfolio.positions)
        
        # Check cash reduction
        expected_cash = self.initial_cash - (self.quantity * self.price)
        self.assertEqual(portfolio.cash, expected_cash)


class TestDataLoader(unittest.TestCase):
    """Test data loading functionality"""
    
    def setUp(self):
        """Set up test data"""
        self.loader = DataLoader()
        
        # Create sample CSV data
        self.sample_data = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=100, freq='D'),
            'Open': np.random.uniform(90, 110, 100),
            'High': np.random.uniform(100, 120, 100),
            'Low': np.random.uniform(80, 100, 100),
            'Close': np.random.uniform(90, 110, 100),
            'Volume': np.random.randint(1000, 10000, 100)
        })
    
    @patch('pandas.read_csv')
    def test_load_csv_data(self, mock_read_csv):
        """Test CSV data loading"""
        mock_read_csv.return_value = self.sample_data
        
        result = self.loader.load_csv_data('dummy_path.csv')
        
        self.assertIsInstance(result, pd.DataFrame)
        mock_read_csv.assert_called_once()
    
    def test_data_validation(self):
        """Test data validation"""
        # Test valid data
        is_valid = self.loader.validate_ohlcv_data(self.sample_data)
        self.assertTrue(is_valid)
        
        # Test invalid data (missing column)
        invalid_data = self.sample_data.drop('Close', axis=1)
        is_valid = self.loader.validate_ohlcv_data(invalid_data)
        self.assertFalse(is_valid)


class TestDataProcessor(unittest.TestCase):
    """Test data processing functionality"""
    
    def setUp(self):
        """Set up test data"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        self.sample_data = pd.DataFrame({
            'open': np.random.uniform(90, 110, 100),
            'high': np.random.uniform(100, 120, 100),
            'low': np.random.uniform(80, 100, 100),
            'close': np.random.uniform(90, 110, 100),
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        # Ensure proper OHLC relationships
        for i in range(len(self.sample_data)):
            o, c = self.sample_data.iloc[i]['open'], self.sample_data.iloc[i]['close']
            self.sample_data.iloc[i, self.sample_data.columns.get_loc('high')] = max(o, c) + np.random.uniform(0, 2)
            self.sample_data.iloc[i, self.sample_data.columns.get_loc('low')] = min(o, c) - np.random.uniform(0, 2)
        
        self.processor = DataProcessor()
    
    def test_clean_ohlcv_data(self):
        """Test OHLCV data cleaning"""
        # Add some invalid data
        dirty_data = self.sample_data.copy()
        dirty_data.iloc[10, dirty_data.columns.get_loc('close')] = -5  # Negative price
        dirty_data.iloc[20] = np.nan  # Missing data
        
        cleaned_data = self.processor.clean_ohlcv_data(dirty_data)
        
        # Should have fewer rows due to cleaning
        self.assertLess(len(cleaned_data), len(dirty_data))
        
        # Should not have negative prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in cleaned_data.columns:
                self.assertTrue((cleaned_data[col] > 0).all())


class TestBacktester(unittest.TestCase):
    """Test backtesting functionality"""
    
    def setUp(self):
        """Set up test data"""
        # Create sample price data
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        np.random.seed(42)
        
        base_price = 100
        returns = np.random.normal(0.0005, 0.01, 252)  # Slight positive drift
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        self.price_data = pd.DataFrame({
            'open': pd.Series(prices) * (1 + np.random.normal(0, 0.001, 252)),
            'high': pd.Series(prices) * (1 + abs(np.random.normal(0, 0.005, 252))),
            'low': pd.Series(prices) * (1 - abs(np.random.normal(0, 0.005, 252))),
            'close': pd.Series(prices),
            'volume': np.random.randint(1000, 10000, 252)
        }, index=dates)
        
        # Create a simple test strategy
        self.strategy = MovingAverageCrossoverStrategy(
            short_window=10,
            long_window=20
        )
    
    def test_backtest_engine_initialization(self):
        """Test backtest engine initialization"""
        config = Config()
        engine = BacktestEngine(config.backtest)
        
        self.assertIsInstance(engine, BacktestEngine)
        self.assertEqual(engine.initial_capital, config.backtest.initial_capital)
    
    @patch('src.core.backtester.BacktestEngine.run_backtest')
    def test_backtest_execution(self, mock_run_backtest):
        """Test backtest execution"""
        # Mock the backtest result
        mock_result = {
            'total_return': 0.15,
            'sharpe_ratio': 1.2,
            'max_drawdown': -0.05,
            'trades': []
        }
        mock_run_backtest.return_value = mock_result
        
        config = Config()
        engine = BacktestEngine(config.backtest)
        
        result = engine.run_backtest(self.strategy, self.price_data)
        
        self.assertIsInstance(result, dict)
        self.assertIn('total_return', result)
        mock_run_backtest.assert_called_once()


class TestStrategyIntegration(unittest.TestCase):
    """Test strategy integration and execution"""
    
    def setUp(self):
        """Set up test data"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        # Create trending price data for MA crossover
        trend = np.linspace(0, 0.2, 100)  # 20% uptrend over period
        noise = np.random.normal(0, 0.01, 100)
        returns = trend + noise
        
        prices = [100]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret/100))
        
        self.data = pd.DataFrame({
            'open': pd.Series(prices) * (1 + np.random.normal(0, 0.001, 100)),
            'high': pd.Series(prices) * (1 + abs(np.random.normal(0, 0.005, 100))),
            'low': pd.Series(prices) * (1 - abs(np.random.normal(0, 0.005, 100))),
            'close': pd.Series(prices),
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
    
    def test_moving_average_strategy(self):
        """Test moving average crossover strategy"""
        strategy = MovingAverageCrossoverStrategy(short_window=5, long_window=20)
        
        # Test signal generation
        signals = strategy.generate_signals(self.data)
        
        self.assertIsInstance(signals, pd.Series)
        self.assertEqual(len(signals), len(self.data))
        
        # Signals should be -1, 0, or 1
        unique_signals = signals.dropna().unique()
        for signal in unique_signals:
            self.assertIn(signal, [-1, 0, 1])


class TestConfigManagement(unittest.TestCase):
    """Test configuration management"""
    
    def test_config_initialization(self):
        """Test configuration initialization"""
        config = Config()
        
        # Test that all major config sections exist
        self.assertIsNotNone(config.data)
        self.assertIsNotNone(config.backtest)
        self.assertIsNotNone(config.risk)
    
    def test_config_values(self):
        """Test configuration values"""
        config = Config()
        
        # Test some default values
        self.assertGreater(config.backtest.initial_capital, 0)
        self.assertGreater(config.risk.max_position_size, 0)
        self.assertLessEqual(config.risk.max_position_size, 1.0)


def run_all_tests():
    """Run all test suites"""
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestTechnicalIndicators,
        TestRiskManagement,
        TestPortfolioManagement,
        TestDataLoader,
        TestDataProcessor,
        TestBacktester,
        TestStrategyIntegration,
        TestConfigManagement
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*50}")
    
    return result


if __name__ == "__main__":
    # Run all tests
    print("Starting Quantitative Trading Platform Test Suite...")
    print("="*50)
    
    result = run_all_tests()
    
    # Exit with appropriate code
    exit_code = 0 if result.wasSuccessful() else 1
    sys.exit(exit_code)
