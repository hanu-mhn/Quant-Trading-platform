"""
Bollinger Band Mean Reversion Strategy.

This strategy uses Bollinger Bands to identify overbought and oversold conditions
for mean reversion trading opportunities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime

from ...common.base_classes import BaseStrategy, Order, OrderType, OrderSide
from ...indicators.technical_indicators import BollingerBands, RSI


class BollingerBandReversionStrategy(BaseStrategy):
    """
    Bollinger Band Mean Reversion Strategy.
    
    Buy when price touches lower band and RSI is oversold.
    Sell when price touches upper band and RSI is overbought.
    """
    
    def __init__(
        self,
        bb_period: int = 20,
        bb_std_dev: float = 2.0,
        rsi_period: int = 14,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        position_size: float = 1000,
        stop_loss_pct: float = 0.03,
        take_profit_pct: float = 0.06
    ):
        super().__init__()
        self.bb_period = bb_period
        self.bb_std_dev = bb_std_dev
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.position_size = position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # Initialize indicators
        self.bollinger_bands = BollingerBands(period=bb_period, std_dev=bb_std_dev)
        self.rsi = RSI(period=rsi_period)
        
        # Track strategy state
        self.position = 0  # Current position size
        self.entry_price = None
        self.last_bb_signal = None
        self.last_rsi_value = None
        
    def initialize(self, data: pd.DataFrame) -> None:
        """Initialize the strategy with historical data."""
        super().initialize(data)
        
        # Calculate Bollinger Bands
        bb_data = self.bollinger_bands.calculate(self.data['close'])
        self.data['bb_upper'] = bb_data['upper_band']
        self.data['bb_middle'] = bb_data['middle_band']
        self.data['bb_lower'] = bb_data['lower_band']
        
        # Calculate RSI
        self.data['rsi'] = self.rsi.calculate(self.data['close'])
        
        # Calculate band position (where price is relative to bands)
        bb_width = self.data['bb_upper'] - self.data['bb_lower']
        self.data['bb_position'] = (self.data['close'] - self.data['bb_lower']) / bb_width
        
        # Generate signals
        self.data['bb_lower_touch'] = self.data['close'] <= self.data['bb_lower']
        self.data['bb_upper_touch'] = self.data['close'] >= self.data['bb_upper']
        self.data['rsi_oversold'] = self.data['rsi'] <= self.rsi_oversold
        self.data['rsi_overbought'] = self.data['rsi'] >= self.rsi_overbought
        
        # Combined signals
        self.data['buy_signal'] = (
            self.data['bb_lower_touch'] & 
            self.data['rsi_oversold']
        )
        self.data['sell_signal'] = (
            self.data['bb_upper_touch'] & 
            self.data['rsi_overbought']
        )
        
        self.logger.info(f"BollingerBandReversion strategy initialized with BB({self.bb_period}, {self.bb_std_dev}) and RSI({self.rsi_period})")
    
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Order]:
        """Generate trading signals based on Bollinger Bands and RSI."""
        orders = []
        
        if not self.is_initialized:
            return orders
            
        current_price = market_data.get('close', market_data.get('price'))
        if current_price is None:
            return orders
        
        # Update indicators with new data
        bb_values = self.bollinger_bands.update(current_price)
        rsi_value = self.rsi.update(current_price)
        
        if bb_values is None or rsi_value is None:
            return orders
        
        upper_band = bb_values['upper_band']
        middle_band = bb_values['middle_band']
        lower_band = bb_values['lower_band']
        
        # Determine current conditions
        touches_lower_band = current_price <= lower_band
        touches_upper_band = current_price >= upper_band
        rsi_oversold = rsi_value <= self.rsi_oversold
        rsi_overbought = rsi_value >= self.rsi_overbought
        
        symbol = market_data.get('symbol', 'UNKNOWN')
        
        # Generate buy signal
        if touches_lower_band and rsi_oversold and self.position <= 0:
            quantity = self.position_size / current_price
            order = Order(
                symbol=symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=quantity
            )
            orders.append(order)
            self.entry_price = current_price
            self.position = quantity
            
            self.logger.info(f"BB Reversion BUY signal: Price {current_price:.2f} at lower band {lower_band:.2f}, RSI {rsi_value:.1f}")
        
        # Generate sell signal
        elif touches_upper_band and rsi_overbought and self.position >= 0:
            if self.position > 0:
                order = Order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=self.position
                )
                orders.append(order)
                self.position = 0
                self.entry_price = None
                
                self.logger.info(f"BB Reversion SELL signal: Price {current_price:.2f} at upper band {upper_band:.2f}, RSI {rsi_value:.1f}")
        
        # Mean reversion exit when price returns to middle band
        elif self.position > 0 and current_price >= middle_band:
            order = Order(
                symbol=symbol,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=self.position
            )
            orders.append(order)
            self.position = 0
            self.entry_price = None
            
            self.logger.info(f"BB Reversion EXIT: Price {current_price:.2f} returned to middle band {middle_band:.2f}")
        
        # Check stop loss and take profit
        if self.position > 0 and self.entry_price is not None:
            stop_loss_price = self.entry_price * (1 - self.stop_loss_pct)
            take_profit_price = self.entry_price * (1 + self.take_profit_pct)
            
            if current_price <= stop_loss_price:
                # Stop loss triggered
                order = Order(
                    symbol=symbol,
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
                    symbol=symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=self.position
                )
                orders.append(order)
                self.position = 0
                self.entry_price = None
                self.logger.info(f"Take profit triggered at {current_price:.2f}")
        
        # Store current values for next iteration
        self.last_bb_signal = {
            'upper_band': upper_band,
            'middle_band': middle_band,
            'lower_band': lower_band,
            'touches_upper': touches_upper_band,
            'touches_lower': touches_lower_band
        }
        self.last_rsi_value = rsi_value
        
        return orders
    
    def get_strategy_state(self) -> Dict[str, Any]:
        """Get current strategy state."""
        return {
            'position': self.position,
            'entry_price': self.entry_price,
            'last_rsi_value': self.last_rsi_value,
            'bollinger_bands': self.last_bb_signal,
            'bb_values': self.bollinger_bands.get_current_value()
        }
    
    def reset(self) -> None:
        """Reset strategy state."""
        super().reset()
        self.position = 0
        self.entry_price = None
        self.last_bb_signal = None
        self.last_rsi_value = None
        self.bollinger_bands.reset()
        self.rsi.reset()
