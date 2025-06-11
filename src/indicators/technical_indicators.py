# filepath: d:\QUANT\QT_python\quant-trading-platform\src\indicators\technical_indicators.py
"""
Technical Indicators Module.

This module provides a comprehensive collection of technical analysis indicators
commonly used in quantitative trading strategies.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, List
from abc import ABC, abstractmethod
import warnings

from ..common.base_classes import BaseIndicator


class MovingAverage(BaseIndicator):
    """Simple Moving Average (SMA) indicator."""
    
    def __init__(self, period: int = 20):
        super().__init__(period)
        self.period = period
        self.values = []
        
    def calculate(self, data: Union[pd.Series, np.ndarray, List[float]]) -> pd.Series:
        """Calculate SMA for entire data series."""
        if isinstance(data, list):
            data = pd.Series(data)
        elif isinstance(data, np.ndarray):
            data = pd.Series(data)
            
        return data.rolling(window=self.period, min_periods=self.period).mean()
    
    def update(self, value: float) -> Optional[float]:
        """Update indicator with new value and return current SMA."""
        self.values.append(value)
        
        # Keep only required values for calculation
        if len(self.values) > self.period:
            self.values = self.values[-self.period:]
            
        if len(self.values) >= self.period:
            return sum(self.values) / len(self.values)
        return None
    
    def get_current_value(self) -> Optional[float]:
        """Get current indicator value."""
        if len(self.values) >= self.period:
            return sum(self.values) / len(self.values)
        return None
    
    def reset(self):
        """Reset indicator state."""
        self.values = []


class ExponentialMovingAverage(BaseIndicator):
    """Exponential Moving Average (EMA) indicator."""
    
    def __init__(self, period: int = 20):
        super().__init__(period)
        self.period = period
        self.multiplier = 2 / (period + 1)
        self.ema_value = None
        self.initialized = False
        
    def calculate(self, data: Union[pd.Series, np.ndarray, List[float]]) -> pd.Series:
        """Calculate EMA for entire data series."""
        if isinstance(data, list):
            data = pd.Series(data)
        elif isinstance(data, np.ndarray):
            data = pd.Series(data)
            
        return data.ewm(span=self.period, adjust=False).mean()
    
    def update(self, value: float) -> Optional[float]:
        """Update indicator with new value and return current EMA."""
        if not self.initialized:
            self.ema_value = value
            self.initialized = True
        else:
            self.ema_value = (value * self.multiplier) + (self.ema_value * (1 - self.multiplier))
            
        return self.ema_value
    
    def get_current_value(self) -> Optional[float]:
        """Get current indicator value."""
        return self.ema_value
    
    def reset(self):
        """Reset indicator state."""
        self.ema_value = None
        self.initialized = False


class RSI(BaseIndicator):
    """Relative Strength Index (RSI) indicator."""
    
    def __init__(self, period: int = 14):
        super().__init__(period)
        self.period = period
        self.gains = []
        self.losses = []
        self.last_price = None
        
    def calculate(self, data: Union[pd.Series, np.ndarray, List[float]]) -> pd.Series:
        """Calculate RSI for entire data series."""
        if isinstance(data, list):
            data = pd.Series(data)
        elif isinstance(data, np.ndarray):
            data = pd.Series(data)
            
        delta = data.diff()
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        avg_gains = gains.rolling(window=self.period).mean()
        avg_losses = losses.rolling(window=self.period).mean()
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def update(self, value: float) -> Optional[float]:
        """Update indicator with new value and return current RSI."""
        if self.last_price is not None:
            change = value - self.last_price
            gain = max(0, change)
            loss = max(0, -change)
            
            self.gains.append(gain)
            self.losses.append(loss)
            
            # Keep only required values
            if len(self.gains) > self.period:
                self.gains = self.gains[-self.period:]
                self.losses = self.losses[-self.period:]
            
            if len(self.gains) >= self.period:
                avg_gain = sum(self.gains) / len(self.gains)
                avg_loss = sum(self.losses) / len(self.losses)
                
                if avg_loss == 0:
                    return 100
                
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                self.last_price = value
                return rsi
        
        self.last_price = value
        return None
    
    def get_current_value(self) -> Optional[float]:
        """Get current indicator value."""
        if len(self.gains) >= self.period:
            avg_gain = sum(self.gains) / len(self.gains)
            avg_loss = sum(self.losses) / len(self.losses)
            
            if avg_loss == 0:
                return 100
            
            rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs))
        return None
    
    def reset(self):
        """Reset indicator state."""
        self.gains = []
        self.losses = []
        self.last_price = None


class MACD(BaseIndicator):
    """Moving Average Convergence Divergence (MACD) indicator."""
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        super().__init__(slow_period)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        
        self.fast_ema = ExponentialMovingAverage(fast_period)
        self.slow_ema = ExponentialMovingAverage(slow_period)
        self.signal_ema = ExponentialMovingAverage(signal_period)
        
        self.macd_values = []
        
    def calculate(self, data: Union[pd.Series, np.ndarray, List[float]]) -> pd.DataFrame:
        """Calculate MACD for entire data series."""
        if isinstance(data, list):
            data = pd.Series(data)
        elif isinstance(data, np.ndarray):
            data = pd.Series(data)
            
        fast_ema = data.ewm(span=self.fast_period).mean()
        slow_ema = data.ewm(span=self.slow_period).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=self.signal_period).mean()
        histogram = macd_line - signal_line
        
        return pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        })
    
    def update(self, value: float) -> Optional[dict]:
        """Update indicator with new value and return current MACD values."""
        fast_ema_value = self.fast_ema.update(value)
        slow_ema_value = self.slow_ema.update(value)
        
        if fast_ema_value is not None and slow_ema_value is not None:
            macd_value = fast_ema_value - slow_ema_value
            self.macd_values.append(macd_value)
            
            # Keep only required values for signal calculation
            if len(self.macd_values) > self.signal_period * 2:
                self.macd_values = self.macd_values[-self.signal_period * 2:]
            
            signal_value = self.signal_ema.update(macd_value)
            
            if signal_value is not None:
                histogram = macd_value - signal_value
                return {
                    'macd': macd_value,
                    'signal': signal_value,
                    'histogram': histogram
                }
        
        return None
    
    def get_current_value(self) -> Optional[dict]:
        """Get current indicator value."""
        fast_ema_value = self.fast_ema.get_current_value()
        slow_ema_value = self.slow_ema.get_current_value()
        
        if fast_ema_value is not None and slow_ema_value is not None:
            macd_value = fast_ema_value - slow_ema_value
            signal_value = self.signal_ema.get_current_value()
            
            if signal_value is not None:
                histogram = macd_value - signal_value
                return {
                    'macd': macd_value,
                    'signal': signal_value,
                    'histogram': histogram
                }
        
        return None
    
    def reset(self):
        """Reset indicator state."""
        self.fast_ema.reset()
        self.slow_ema.reset()
        self.signal_ema.reset()
        self.macd_values = []


class BollingerBands(BaseIndicator):
    """Bollinger Bands indicator."""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        super().__init__(period)
        self.period = period
        self.std_dev = std_dev
        self.values = []
        
    def calculate(self, data: Union[pd.Series, np.ndarray, List[float]]) -> pd.DataFrame:
        """Calculate Bollinger Bands for entire data series."""
        if isinstance(data, list):
            data = pd.Series(data)
        elif isinstance(data, np.ndarray):
            data = pd.Series(data)
            
        middle_band = data.rolling(window=self.period).mean()
        std = data.rolling(window=self.period).std()
        upper_band = middle_band + (std * self.std_dev)
        lower_band = middle_band - (std * self.std_dev)
        
        return pd.DataFrame({
            'upper_band': upper_band,
            'middle_band': middle_band,
            'lower_band': lower_band
        })
    
    def update(self, value: float) -> Optional[dict]:
        """Update indicator with new value and return current Bollinger Bands."""
        self.values.append(value)
        
        # Keep only required values
        if len(self.values) > self.period:
            self.values = self.values[-self.period:]
        
        if len(self.values) >= self.period:
            middle_band = sum(self.values) / len(self.values)
            variance = sum((x - middle_band) ** 2 for x in self.values) / len(self.values)
            std = variance ** 0.5
            upper_band = middle_band + (std * self.std_dev)
            lower_band = middle_band - (std * self.std_dev)
            
            return {
                'upper_band': upper_band,
                'middle_band': middle_band,
                'lower_band': lower_band
            }
        
        return None
    
    def get_current_value(self) -> Optional[dict]:
        """Get current indicator value."""
        if len(self.values) >= self.period:
            middle_band = sum(self.values) / len(self.values)
            variance = sum((x - middle_band) ** 2 for x in self.values) / len(self.values)
            std = variance ** 0.5
            upper_band = middle_band + (std * self.std_dev)
            lower_band = middle_band - (std * self.std_dev)
            
            return {
                'upper_band': upper_band,
                'middle_band': middle_band,
                'lower_band': lower_band
            }
        
        return None
    
    def reset(self):
        """Reset indicator state."""
        self.values = []


class StochasticOscillator(BaseIndicator):
    """Stochastic Oscillator indicator."""
    
    def __init__(self, k_period: int = 14, d_period: int = 3):
        super().__init__(k_period)
        self.k_period = k_period
        self.d_period = d_period
        self.highs = []
        self.lows = []
        self.closes = []
        self.k_values = []
        
    def calculate(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.DataFrame:
        """Calculate Stochastic Oscillator for entire data series."""
        lowest_low = low.rolling(window=self.k_period).min()
        highest_high = high.rolling(window=self.k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=self.d_period).mean()
        
        return pd.DataFrame({
            'k_percent': k_percent,
            'd_percent': d_percent
        })
    
    def update(self, high: float, low: float, close: float) -> Optional[dict]:
        """Update indicator with new OHLC values."""
        self.highs.append(high)
        self.lows.append(low)
        self.closes.append(close)
        
        # Keep only required values
        if len(self.highs) > self.k_period:
            self.highs = self.highs[-self.k_period:]
            self.lows = self.lows[-self.k_period:]
            self.closes = self.closes[-self.k_period:]
        
        if len(self.closes) >= self.k_period:
            lowest_low = min(self.lows)
            highest_high = max(self.highs)
            
            if highest_high != lowest_low:
                k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
                self.k_values.append(k_percent)
                
                # Keep only required values for D calculation
                if len(self.k_values) > self.d_period:
                    self.k_values = self.k_values[-self.d_period:]
                
                if len(self.k_values) >= self.d_period:
                    d_percent = sum(self.k_values) / len(self.k_values)
                    return {
                        'k_percent': k_percent,
                        'd_percent': d_percent
                    }
        
        return None
    
    def reset(self):
        """Reset indicator state."""
        self.highs = []
        self.lows = []
        self.closes = []
        self.k_values = []


class AverageTrueRange(BaseIndicator):
    """Average True Range (ATR) indicator."""
    
    def __init__(self, period: int = 14):
        super().__init__(period)
        self.period = period
        self.true_ranges = []
        self.last_close = None
        
    def calculate(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate ATR for entire data series."""
        high_low = high - low
        high_close_prev = np.abs(high - close.shift(1))
        low_close_prev = np.abs(low - close.shift(1))
        
        true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        atr = true_range.rolling(window=self.period).mean()
        
        return atr
    
    def update(self, high: float, low: float, close: float) -> Optional[float]:
        """Update indicator with new OHLC values."""
        if self.last_close is not None:
            true_range = max(
                high - low,
                abs(high - self.last_close),
                abs(low - self.last_close)
            )
            self.true_ranges.append(true_range)
            
            # Keep only required values
            if len(self.true_ranges) > self.period:
                self.true_ranges = self.true_ranges[-self.period:]
            
            if len(self.true_ranges) >= self.period:
                atr = sum(self.true_ranges) / len(self.true_ranges)
                self.last_close = close
                return atr
        
        self.last_close = close
        return None
    
    def reset(self):
        """Reset indicator state."""
        self.true_ranges = []
        self.last_close = None


# Utility functions for common technical analysis calculations
def calculate_support_resistance(data: pd.Series, window: int = 20, min_touches: int = 2) -> dict:
    """
    Calculate support and resistance levels using local minima and maxima.
    
    Args:
        data: Price data series
        window: Window size for finding local extrema
        min_touches: Minimum number of touches required for a level
        
    Returns:
        Dictionary with support and resistance levels
    """
    try:
        from scipy.signal import argrelextrema
        
        # Find local minima (support) and maxima (resistance)
        local_min_idx = argrelextrema(data.values, np.less, order=window)[0]
        local_max_idx = argrelextrema(data.values, np.greater, order=window)[0]
        
        support_levels = data.iloc[local_min_idx].values
        resistance_levels = data.iloc[local_max_idx].values
        
        return {
            'support_levels': support_levels,
            'resistance_levels': resistance_levels,
            'support_indices': local_min_idx,
            'resistance_indices': local_max_idx
        }
    except ImportError:
        # Fallback implementation without scipy
        return _calculate_support_resistance_fallback(data, window, min_touches)


def _calculate_support_resistance_fallback(data: pd.Series, window: int = 20, min_touches: int = 2) -> dict:
    """
    Fallback implementation for support/resistance calculation without scipy.
    """
    # Simple implementation using rolling windows
    rolling_min = data.rolling(window=window, center=True).min()
    rolling_max = data.rolling(window=window, center=True).max()
    
    # Find points where price equals the rolling min/max (approximate local extrema)
    support_mask = (data == rolling_min) & (data == data.rolling(window=3, center=True).min())
    resistance_mask = (data == rolling_max) & (data == data.rolling(window=3, center=True).max())
    
    support_indices = data[support_mask].index.tolist()
    resistance_indices = data[resistance_mask].index.tolist()
    
    support_levels = data[support_mask].values
    resistance_levels = data[resistance_mask].values
    
    return {
        'support_levels': support_levels,
        'resistance_levels': resistance_levels,
        'support_indices': support_indices,
        'resistance_indices': resistance_indices
    }


def calculate_pivot_points(high: float, low: float, close: float) -> dict:
    """
    Calculate pivot points for the next trading session.
    
    Args:
        high: Previous session high
        low: Previous session low
        close: Previous session close
        
    Returns:
        Dictionary with pivot points and support/resistance levels
    """
    pivot = (high + low + close) / 3
    
    # Support levels
    s1 = (2 * pivot) - high
    s2 = pivot - (high - low)
    s3 = low - 2 * (high - pivot)
    
    # Resistance levels
    r1 = (2 * pivot) - low
    r2 = pivot + (high - low)
    r3 = high + 2 * (pivot - low)
    
    return {
        'pivot': pivot,
        'support_1': s1,
        'support_2': s2,
        'support_3': s3,
        'resistance_1': r1,
        'resistance_2': r2,
        'resistance_3': r3
    }