# Simple Moving Average Crossover Strategy
"""
Simple Moving Average Crossover Strategy - Perfect for getting started with algorithmic trading!

Copyright © 2025 Malavath Hanmanth Nayak. All rights reserved.
Developer: Malavath Hanmanth Nayak
Contact: hanmanthnayak.95@gmail.com
GitHub: https://github.com/hanu-mhn
LinkedIn: https://www.linkedin.com/in/hanmanth-nayak-m-6bbab1263/

This software is provided under the MIT License.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class SimpleMAStrategy:
    """Simple Moving Average Crossover Strategy - BUY when 20-day MA crosses above 50-day MA"""
    
    def __init__(self):
        self.name = "Simple MA Crossover"
        self.short_window = 20
        self.long_window = 50
        self.stop_loss_pct = 0.02
        self.position_size_pct = 0.05
        self.positions = {}
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        data = data.copy()
        data[f'MA_{self.short_window}'] = data['close'].rolling(window=self.short_window).mean()
        data[f'MA_{self.long_window}'] = data['close'].rolling(window=self.long_window).mean()        # Generate signals
        data['signal'] = 0
        signal_mask = np.arange(len(data)) >= self.short_window
        data.loc[signal_mask, 'signal'] = np.where(
            data[f'MA_{self.short_window}'].iloc[self.short_window:] > data[f'MA_{self.long_window}'].iloc[self.short_window:], 1, 0
        )
        data['position'] = data['signal'].diff()
        
        return data
    
    def generate_signals(self, symbol: str, data: pd.DataFrame) -> List[Dict]:
        """Generate trading signals"""
        signals = []
        
        if len(data) < self.long_window:
            return signals
            
        # Calculate indicators
        data = self.calculate_indicators(data)
        
        # Get latest data point
        latest = data.iloc[-1]
        prev = data.iloc[-2] if len(data) > 1 else None
        
        if prev is None:
            return signals
            
        # Check for crossover signals
        if latest['position'] == 1:  # Buy signal
            signals.append({
                'symbol': symbol,
                'action': 'BUY',
                'quantity': self._calculate_quantity(symbol, latest['close']),
                'price': latest['close'],
                'timestamp': latest.name,
                'reason': f"MA{self.short_window} crossed above MA{self.long_window}",
                'stop_loss': latest['close'] * (1 - self.stop_loss_pct)
            })
            
        elif latest['position'] == -1:  # Sell signal
            if symbol in self.positions:
                signals.append({
                    'symbol': symbol,
                    'action': 'SELL',
                    'quantity': self.positions[symbol]['quantity'],
                    'price': latest['close'],
                    'timestamp': latest.name,
                    'reason': f"MA{self.short_window} crossed below MA{self.long_window}"
                })
        
        return signals
    
    def _calculate_quantity(self, symbol: str, price: float) -> int:
        """Calculate position size based on portfolio percentage"""
        portfolio_value = 1000000  # 10 lakh default
        position_value = portfolio_value * self.position_size_pct
        quantity = int(position_value / price)
        return max(1, quantity)  # At least 1 share
    
    def update_position(self, symbol: str, action: str, quantity: int, price: float):
        """Update position tracking"""
        if action == 'BUY':
            self.positions[symbol] = {
                'quantity': quantity,
                'entry_price': price,
                'entry_time': datetime.now()
            }
        elif action == 'SELL' and symbol in self.positions:
            del self.positions[symbol]
    
    def check_stop_loss(self, symbol: str, current_price: float) -> bool:
        """Check if stop loss should be triggered"""
        if symbol not in self.positions:
            return False
            
        entry_price = self.positions[symbol]['entry_price']
        stop_loss_price = entry_price * (1 - self.stop_loss_pct)
        
        return current_price <= stop_loss_price
    
    def get_strategy_info(self) -> Dict:
        """Get strategy information"""
        return {
            'name': self.name,
            'type': 'Trend Following',
            'timeframe': 'Daily',
            'parameters': {
                'short_ma': self.short_window,
                'long_ma': self.long_window,
                'stop_loss': f"{self.stop_loss_pct*100}%",
                'position_size': f"{self.position_size_pct*100}%"
            },
            'description': "Buys when short MA crosses above long MA, sells when it crosses below"
        }

# Example usage and testing
if __name__ == "__main__":
    # Create strategy instance
    strategy = SimpleMAStrategy()
    
    # Generate sample data for testing
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    # Simulate price data with trend
    prices = []
    price = 100
    for i in range(len(dates)):
        change = np.random.normal(0.001, 0.02)  # Small daily changes
        if i > 100:  # Add trend after 100 days
            change += 0.002
        price *= (1 + change)
        prices.append(price)
    
    # Create DataFrame
    test_data = pd.DataFrame({
        'close': prices,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    print("Testing Simple MA Strategy...")
    print(f"Strategy Info: {strategy.get_strategy_info()}")
    
    # Test signal generation
    signals = strategy.generate_signals('TEST', test_data)
    
    print(f"\nGenerated {len(signals)} signals")
    if signals:
        print("Latest signals:")
        for signal in signals[-3:]:  # Show last 3 signals
            print(f"  {signal['timestamp']}: {signal['action']} {signal['quantity']} @ Rs{signal['price']:.2f} - {signal['reason']}")
    
    print("\nStrategy is ready for live trading!")
