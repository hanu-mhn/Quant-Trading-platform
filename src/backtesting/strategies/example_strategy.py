"""
Example trading strategy implementation for backtesting
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List

class ExampleStrategy:
    """
    Example strategy for demonstration purposes
    """
    
    def __init__(self, params=None):
        """
        Initialize the strategy with parameters
        
        Parameters:
        -----------
        params : Dict[str, Any], optional
            Strategy parameters
        """
        self.params = params or {
            'ma_short': 50,
            'ma_long': 200
        }
        
    def generate_signals(self, data):
        """
        Generate trading signals based on the strategy logic
        
        Parameters:
        -----------
        data : pd.DataFrame
            Market data with OHLCV columns
            
        Returns:
        --------
        pd.Series
            Series of trading signals (1 for buy, -1 for sell, 0 for hold)
        """
        # This would normally implement real strategy logic
        # For testing purposes, we'll return a random series of signals
        
        np.random.seed(42)  # For reproducible testing
        signals = np.random.choice([0, 1, -1], size=len(data), p=[0.8, 0.1, 0.1])
        return pd.Series(signals, index=data.index)
