"""
LSTM Deep Learning Strategy Example

This strategy uses a trained LSTM model (Keras) to predict price movement and generate trading signals.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict
from ...common.base_classes import BaseStrategy, Order, OrderType, OrderSide
from tensorflow.keras.models import load_model

class LSTMDeepStrategy(BaseStrategy):
    def __init__(self, model_path: str, sequence_length: int = 30, position_size: float = 1000):
        super().__init__()
        self.model = load_model(model_path)
        self.sequence_length = sequence_length
        self.position_size = position_size

    def generate_signals(self, data: pd.DataFrame) -> list:
        # Assume data is sorted by date ascending
        signals = []
        for i in range(self.sequence_length, len(data)):
            X_seq = data.iloc[i-self.sequence_length:i].values.reshape(1, self.sequence_length, -1)
            pred = self.model.predict(X_seq)[0, 0]
            if pred > 0.5:
                signals.append(Order(
                    symbol=data.iloc[i]['symbol'],
                    order_type=OrderType.MARKET,
                    side=OrderSide.BUY,
                    quantity=self.position_size
                ))
            elif pred < -0.5:
                signals.append(Order(
                    symbol=data.iloc[i]['symbol'],
                    order_type=OrderType.MARKET,
                    side=OrderSide.SELL,
                    quantity=self.position_size
                ))
        return signals
