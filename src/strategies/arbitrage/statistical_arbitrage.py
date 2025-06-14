"""
Statistical Arbitrage Strategy.

This strategy identifies pairs of correlated stocks and trades the mean reversion
of their price ratio when it deviates from historical norms.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from scipy.stats import zscore

from ...common.base_classes import BaseStrategy, Order, OrderType, OrderSide
from ...indicators.technical_indicators import MovingAverage


class StatisticalArbitrageStrategy(BaseStrategy):
    """
    Statistical Arbitrage (Pairs Trading) Strategy.
    
    Identifies pairs of correlated stocks and trades mean reversion of their spread.
    """
    
    def __init__(
        self,
        pair_symbols: Tuple[str, str],
        lookback_period: int = 60,
        entry_zscore: float = 2.0,
        exit_zscore: float = 0.5,
        stop_loss_zscore: float = 3.0,
        position_size: float = 1000,
        min_correlation: float = 0.7
    ):
        super().__init__()
        self.symbol_a, self.symbol_b = pair_symbols
        self.lookback_period = lookback_period
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.stop_loss_zscore = stop_loss_zscore
        self.position_size = position_size
        self.min_correlation = min_correlation
        
        # Price history for both symbols
        self.prices_a = []
        self.prices_b = []
        self.spreads = []
        self.spread_ma = MovingAverage(period=lookback_period)
        
        # Position tracking
        self.position_a = 0  # Position in symbol A
        self.position_b = 0  # Position in symbol B
        self.entry_spread = None
        self.current_spread = None
        self.hedge_ratio = 1.0
        
    def initialize(self, data: pd.DataFrame) -> None:
        """Initialize the strategy with historical data."""
        super().initialize(data)
        
        # Ensure we have data for both symbols
        required_columns = [f'{self.symbol_a}_close', f'{self.symbol_b}_close']
        if not all(col in self.data.columns for col in required_columns):
            raise ValueError(f"Missing price data for symbols {self.symbol_a} and {self.symbol_b}")
        
        # Calculate spreads and statistics
        prices_a = self.data[f'{self.symbol_a}_close']
        prices_b = self.data[f'{self.symbol_b}_close']
        
        # Calculate hedge ratio using linear regression
        correlation = prices_a.corr(prices_b)
        if correlation < self.min_correlation:
            self.logger.warning(f"Low correlation {correlation:.3f} between {self.symbol_a} and {self.symbol_b}")
        
        # Calculate hedge ratio (beta)
        covariance = np.cov(prices_a, prices_b)[0, 1]
        variance_b = np.var(prices_b)
        self.hedge_ratio = covariance / variance_b if variance_b != 0 else 1.0
        
        # Calculate spread: A - hedge_ratio * B
        self.data['spread'] = prices_a - (self.hedge_ratio * prices_b)
        
        # Calculate rolling statistics
        self.data['spread_mean'] = self.data['spread'].rolling(window=self.lookback_period).mean()
        self.data['spread_std'] = self.data['spread'].rolling(window=self.lookback_period).std()
        self.data['spread_zscore'] = (
            (self.data['spread'] - self.data['spread_mean']) / self.data['spread_std']
        )
        
        # Generate signals
        self.data['long_signal'] = self.data['spread_zscore'] <= -self.entry_zscore
        self.data['short_signal'] = self.data['spread_zscore'] >= self.entry_zscore
        self.data['exit_signal'] = np.abs(self.data['spread_zscore']) <= self.exit_zscore
        
        self.logger.info(f"Statistical Arbitrage initialized for pair {self.symbol_a}/{self.symbol_b}")
        self.logger.info(f"Correlation: {correlation:.3f}, Hedge Ratio: {self.hedge_ratio:.3f}")
    
    def update_prices(self, symbol: str, price: float) -> None:
        """Update price data for a symbol."""
        if symbol == self.symbol_a:
            self.prices_a.append(price)
            if len(self.prices_a) > self.lookback_period * 2:
                self.prices_a = self.prices_a[-self.lookback_period * 2:]
        elif symbol == self.symbol_b:
            self.prices_b.append(price)
            if len(self.prices_b) > self.lookback_period * 2:
                self.prices_b = self.prices_b[-self.lookback_period * 2:]
        
        # Calculate current spread if we have both prices
        if len(self.prices_a) > 0 and len(self.prices_b) > 0:
            current_spread = self.prices_a[-1] - (self.hedge_ratio * self.prices_b[-1])
            self.spreads.append(current_spread)
            self.current_spread = current_spread
            
            # Keep spread history manageable
            if len(self.spreads) > self.lookback_period * 2:
                self.spreads = self.spreads[-self.lookback_period * 2:]
    
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Order]:
        """Generate arbitrage signals based on spread deviation."""
        orders = []
        
        if not self.is_initialized:
            return orders
        
        symbol = market_data.get('symbol')
        price = market_data.get('close', market_data.get('price'))
        
        if not symbol or price is None:
            return orders
        
        # Update price data
        self.update_prices(symbol, price)
        
        # Need sufficient data for both symbols
        if len(self.spreads) < self.lookback_period:
            return orders
        
        # Calculate current spread statistics
        recent_spreads = self.spreads[-self.lookback_period:]
        spread_mean = np.mean(recent_spreads)
        spread_std = np.std(recent_spreads)
        
        if spread_std == 0:
            return orders
        
        current_zscore = (self.current_spread - spread_mean) / spread_std
        
        # Get current prices
        if len(self.prices_a) == 0 or len(self.prices_b) == 0:
            return orders
        
        price_a = self.prices_a[-1]
        price_b = self.prices_b[-1]
        
        # Entry signals
        if abs(self.position_a) == 0:  # No current position
            
            if current_zscore <= -self.entry_zscore:
                # Spread is unusually low: Long A, Short B
                quantity_a = self.position_size / price_a
                quantity_b = (self.position_size * self.hedge_ratio) / price_b
                
                orders.append(Order(
                    symbol=self.symbol_a,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=quantity_a
                ))
                
                orders.append(Order(
                    symbol=self.symbol_b,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=quantity_b
                ))
                
                self.position_a = quantity_a
                self.position_b = -quantity_b
                self.entry_spread = self.current_spread
                
                self.logger.info(f"Statistical Arbitrage LONG spread: Z-score {current_zscore:.2f}")
            
            elif current_zscore >= self.entry_zscore:
                # Spread is unusually high: Short A, Long B
                quantity_a = self.position_size / price_a
                quantity_b = (self.position_size * self.hedge_ratio) / price_b
                
                orders.append(Order(
                    symbol=self.symbol_a,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=quantity_a
                ))
                
                orders.append(Order(
                    symbol=self.symbol_b,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=quantity_b
                ))
                
                self.position_a = -quantity_a
                self.position_b = quantity_b
                self.entry_spread = self.current_spread
                
                self.logger.info(f"Statistical Arbitrage SHORT spread: Z-score {current_zscore:.2f}")
        
        # Exit signals
        elif abs(self.position_a) > 0:  # Have position
            
            should_exit = False
            exit_reason = ""
            
            # Normal exit when spread reverts
            if abs(current_zscore) <= self.exit_zscore:
                should_exit = True
                exit_reason = f"Mean reversion: Z-score {current_zscore:.2f}"
            
            # Stop loss when spread moves too far against us
            elif abs(current_zscore) >= self.stop_loss_zscore:
                should_exit = True
                exit_reason = f"Stop loss: Z-score {current_zscore:.2f}"
            
            if should_exit:
                # Close both positions
                if self.position_a > 0:
                    orders.append(Order(
                        symbol=self.symbol_a,
                        side=OrderSide.SELL,
                        order_type=OrderType.MARKET,
                        quantity=abs(self.position_a)
                    ))
                else:
                    orders.append(Order(
                        symbol=self.symbol_a,
                        side=OrderSide.BUY,
                        order_type=OrderType.MARKET,
                        quantity=abs(self.position_a)
                    ))
                
                if self.position_b > 0:
                    orders.append(Order(
                        symbol=self.symbol_b,
                        side=OrderSide.SELL,
                        order_type=OrderType.MARKET,
                        quantity=abs(self.position_b)
                    ))
                else:
                    orders.append(Order(
                        symbol=self.symbol_b,
                        side=OrderSide.BUY,
                        order_type=OrderType.MARKET,
                        quantity=abs(self.position_b)
                    ))
                
                self.logger.info(f"Statistical Arbitrage EXIT: {exit_reason}")
                
                # Reset positions
                self.position_a = 0
                self.position_b = 0
                self.entry_spread = None
        
        return orders
    
    def get_strategy_state(self) -> Dict[str, Any]:
        """Get current strategy state."""
        current_zscore = None
        if len(self.spreads) >= self.lookback_period:
            recent_spreads = self.spreads[-self.lookback_period:]
            spread_mean = np.mean(recent_spreads)
            spread_std = np.std(recent_spreads)
            if spread_std != 0:
                current_zscore = (self.current_spread - spread_mean) / spread_std
        
        return {
            'symbol_a': self.symbol_a,
            'symbol_b': self.symbol_b,
            'position_a': self.position_a,
            'position_b': self.position_b,
            'hedge_ratio': self.hedge_ratio,
            'current_spread': self.current_spread,
            'entry_spread': self.entry_spread,
            'current_zscore': current_zscore,
            'prices_a': self.prices_a[-5:] if len(self.prices_a) >= 5 else self.prices_a,
            'prices_b': self.prices_b[-5:] if len(self.prices_b) >= 5 else self.prices_b
        }
    
    def reset(self) -> None:
        """Reset strategy state."""
        super().reset()
        self.prices_a = []
        self.prices_b = []
        self.spreads = []
        self.position_a = 0
        self.position_b = 0
        self.entry_spread = None
        self.current_spread = None
        self.spread_ma.reset()
