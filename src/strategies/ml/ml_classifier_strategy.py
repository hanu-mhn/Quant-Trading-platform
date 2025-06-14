"""
ML Classifier Strategy Example

This strategy uses a trained scikit-learn classifier to generate trading signals based on input features.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict
from ...common.base_classes import BaseStrategy, Order, OrderType, OrderSide
from joblib import load

class MLClassifierStrategy(BaseStrategy):
    def __init__(self, model_path: str, feature_columns: list, position_size: float = 1000):
        super().__init__()
        self.model = load(model_path)
        self.feature_columns = feature_columns
        self.position_size = position_size

    def generate_signals(self, data: pd.DataFrame) -> list:
        # Extract features
        X = data[self.feature_columns].values
        preds = self.model.predict(X)
        signals = []
        for i, pred in enumerate(preds):
            if pred == 1:
                signals.append(Order(
                    symbol=data.iloc[i]['symbol'],
                    order_type=OrderType.MARKET,
                    side=OrderSide.BUY,
                    quantity=self.position_size
                ))
            elif pred == -1:
                signals.append(Order(
                    symbol=data.iloc[i]['symbol'],
                    order_type=OrderType.MARKET,
                    side=OrderSide.SELL,
                    quantity=self.position_size
                ))
        return signals
