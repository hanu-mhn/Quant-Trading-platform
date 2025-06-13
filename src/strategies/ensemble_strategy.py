"""
Ensemble Strategy Example

This strategy combines signals from multiple sub-strategies (e.g., ML, deep learning, rule-based) and applies risk management.
"""

import pandas as pd
from typing import List
from ..common.base_classes import BaseStrategy, Order

class EnsembleStrategy(BaseStrategy):
    def __init__(self, strategies: List[BaseStrategy], voting: str = 'majority'):
        super().__init__()
        self.strategies = strategies
        self.voting = voting

    def generate_signals(self, data: pd.DataFrame) -> list:
        # Collect signals from all strategies
        all_signals = [s.generate_signals(data) for s in self.strategies]
        # Simple majority voting for buy/sell/hold
        # (You can implement more advanced weighting/aggregation)
        # This is a placeholder for demonstration
        return all_signals[0] if all_signals else []
