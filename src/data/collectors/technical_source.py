import pandas as pd
import talib


class TechnicalSource:
    def __init__(self, data):
        """
        Initialize with historical price data (e.g., a DataFrame with OHLC data).
        """
        self.data = data

    def calculate_moving_average(self, period):
        """
        Calculate the moving average.
        """
        self.data[f"MA_{period}"] = talib.SMA(self.data["close"], timeperiod=period)
        return self.data

    def calculate_rsi(self, period):
        """
        Calculate the Relative Strength Index (RSI).
        """
        self.data[f"RSI_{period}"] = talib.RSI(self.data["close"], timeperiod=period)
        return self.data