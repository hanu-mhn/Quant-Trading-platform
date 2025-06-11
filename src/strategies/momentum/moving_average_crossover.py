"""
Moving Average Crossover Strategy Implementation.

This strategy generates buy/sell signals based on the crossover of fast and slow moving averages.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime

from ...common.base_classes import BaseStrategy, Order, OrderType, OrderSide
from ...indicators.technical_indicators import MovingAverage


class MovingAverageCrossoverStrategy(BaseStrategy):
    """
    Moving Average Crossover Strategy.
    
    Generates buy signal when fast MA crosses above slow MA.
    Generates sell signal when fast MA crosses below slow MA.
    """
    
    def __init__(
        self,
        fast_period: int = 10,
        slow_period: int = 20,
        position_size: float = 1000,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04
    ):
        super().__init__()
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.position_size = position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # Initialize indicators
        self.fast_ma = MovingAverage(period=fast_period)
        self.slow_ma = MovingAverage(period=slow_period)
        
        # Track strategy state
        self.position = 0  # Current position size
        self.entry_price = None
        self.last_signal = None
        
    def initialize(self, data: pd.DataFrame) -> None:
        """Initialize the strategy with historical data."""
        super().initialize(data)
        
        # Calculate initial moving averages
        self.data['fast_ma'] = self.fast_ma.calculate(self.data['close'])
        self.data['slow_ma'] = self.slow_ma.calculate(self.data['close'])
        
        # Calculate crossover signals
        self.data['signal'] = 0
        self.data.loc[self.data['fast_ma'] > self.data['slow_ma'], 'signal'] = 1
        self.data.loc[self.data['fast_ma'] < self.data['slow_ma'], 'signal'] = -1
        
        # Detect signal changes (crossovers)
        self.data['signal_change'] = self.data['signal'].diff()
        
        self.logger.info(f"MovingAverageCrossover strategy initialized with periods {self.fast_period}/{self.slow_period}")
    
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Order]:
        """Generate trading signals based on MA crossover."""
        orders = []
        
        if not self.is_initialized:
            return orders
            
        current_price = market_data.get('close', market_data.get('price'))
        if current_price is None:
            return orders
        
        # Update indicators with new data
        fast_ma_value = self.fast_ma.update(current_price)
        slow_ma_value = self.slow_ma.update(current_price)
        
        if fast_ma_value is None or slow_ma_value is None:
            return orders
        
        # Determine current signal
        current_signal = 1 if fast_ma_value > slow_ma_value else -1
        
        # Check for crossover
        if self.last_signal is not None and current_signal != self.last_signal:
            if current_signal == 1 and self.position <= 0:
                # Fast MA crossed above slow MA - Buy signal
                quantity = self.position_size / current_price
                order = Order(
                    symbol=market_data.get('symbol', 'UNKNOWN'),
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=quantity
                )
                orders.append(order)
                self.entry_price = current_price
                self.position = quantity
                
                self.logger.info(f"MA Crossover BUY signal: Fast MA {fast_ma_value:.2f} > Slow MA {slow_ma_value:.2f}")
                
            elif current_signal == -1 and self.position >= 0:
                # Fast MA crossed below slow MA - Sell signal
                if self.position > 0:
                    order = Order(
                        symbol=market_data.get('symbol', 'UNKNOWN'),
                        side=OrderSide.SELL,
                        order_type=OrderType.MARKET,
                        quantity=self.position
                    )
                    orders.append(order)
                    self.position = 0
                    self.entry_price = None
                    
                    self.logger.info(f"MA Crossover SELL signal: Fast MA {fast_ma_value:.2f} < Slow MA {slow_ma_value:.2f}")
        
        # Check stop loss and take profit
        if self.position > 0 and self.entry_price is not None:
            stop_loss_price = self.entry_price * (1 - self.stop_loss_pct)
            take_profit_price = self.entry_price * (1 + self.take_profit_pct)
            
            if current_price <= stop_loss_price:
                # Stop loss triggered
                order = Order(
                    symbol=market_data.get('symbol', 'UNKNOWN'),
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=self.position
                )
                orders.append(order)
                self.position = 0
                self.entry_price = None
                self.logger.info(f"Stop loss triggered at {current_price:.2f}")
                
            elif current_price >= take_profit_price:
                # Take profit triggered
                order = Order(
                    symbol=market_data.get('symbol', 'UNKNOWN'),
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=self.position
                )
                orders.append(order)
                self.position = 0
                self.entry_price = None
                self.logger.info(f"Take profit triggered at {current_price:.2f}")
        
        self.last_signal = current_signal
        return orders
    
    def get_strategy_state(self) -> Dict[str, Any]:
        """Get current strategy state."""
        return {
            'position': self.position,
            'entry_price': self.entry_price,
            'last_signal': self.last_signal,
            'fast_ma_value': self.fast_ma.get_current_value(),
            'slow_ma_value': self.slow_ma.get_current_value()
        }
    
    def reset(self) -> None:
        """Reset strategy state."""
        super().reset()
        self.position = 0
        self.entry_price = None
        self.last_signal = None
        self.fast_ma.reset()
        self.slow_ma.reset()
