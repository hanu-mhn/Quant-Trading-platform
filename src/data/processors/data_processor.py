"""
Data Processing Module.

This module provides data cleaning, transformation, and feature engineering
capabilities for market data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging

from ...indicators.technical_indicators import (
    MovingAverage, ExponentialMovingAverage, RSI, MACD, 
    BollingerBands, StochasticOscillator, AverageTrueRange
)


class DataProcessor:
    """Comprehensive data processing and feature engineering."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def clean_ohlcv_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean OHLCV data by removing invalid records and fixing inconsistencies.
        
        Args:
            df: Raw OHLCV DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        initial_rows = len(df_clean)
        
        # Remove rows with missing essential data
        essential_columns = ['open', 'high', 'low', 'close']
        df_clean = df_clean.dropna(subset=[col for col in essential_columns if col in df_clean.columns])
        
        # Remove rows with negative or zero prices
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in df_clean.columns:
                df_clean = df_clean[df_clean[col] > 0]
        
        # Fix OHLC inconsistencies
        if all(col in df_clean.columns for col in ['open', 'high', 'low', 'close']):
            # Ensure high is the maximum
            df_clean['high'] = df_clean[['open', 'high', 'low', 'close']].max(axis=1)
            
            # Ensure low is the minimum
            df_clean['low'] = df_clean[['open', 'high', 'low', 'close']].min(axis=1)
        
        # Handle volume data
        if 'volume' in df_clean.columns:
            # Replace negative volumes with 0
            df_clean['volume'] = df_clean['volume'].clip(lower=0)
            
            # Fill missing volumes with median
            df_clean['volume'] = df_clean['volume'].fillna(df_clean['volume'].median())
        
        # Remove duplicate dates for same symbol
        if 'date' in df_clean.columns and 'symbol' in df_clean.columns:
            df_clean = df_clean.drop_duplicates(subset=['date', 'symbol'], keep='last')
        
        # Sort by date
        if 'date' in df_clean.columns:
            df_clean = df_clean.sort_values('date').reset_index(drop=True)
        
        rows_removed = initial_rows - len(df_clean)
        if rows_removed > 0:
            self.logger.info(f"Cleaned data: removed {rows_removed} invalid rows ({rows_removed/initial_rows:.1%})")
        
        return df_clean
    
    def add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic derived features to OHLCV data.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            DataFrame with additional features
        """
        df_features = df.copy()
        
        # Price-based features
        if all(col in df_features.columns for col in ['open', 'high', 'low', 'close']):
            # Daily returns
            df_features['returns'] = df_features['close'].pct_change()
            
            # Log returns
            df_features['log_returns'] = np.log(df_features['close'] / df_features['close'].shift(1))
            
            # Price ranges
            df_features['range'] = df_features['high'] - df_features['low']
            df_features['range_pct'] = df_features['range'] / df_features['close']
            
            # Body and shadows (candlestick analysis)
            df_features['body'] = abs(df_features['close'] - df_features['open'])
            df_features['body_pct'] = df_features['body'] / df_features['close']
            df_features['upper_shadow'] = df_features['high'] - np.maximum(df_features['open'], df_features['close'])
            df_features['lower_shadow'] = np.minimum(df_features['open'], df_features['close']) - df_features['low']
            
            # Gap analysis
            df_features['gap'] = df_features['open'] - df_features['close'].shift(1)
            df_features['gap_pct'] = df_features['gap'] / df_features['close'].shift(1)
            
            # Intraday movement
            df_features['intraday_return'] = (df_features['close'] - df_features['open']) / df_features['open']
        
        # Volume-based features
        if 'volume' in df_features.columns:
            # Volume moving averages
            df_features['volume_ma_5'] = df_features['volume'].rolling(window=5).mean()
            df_features['volume_ma_20'] = df_features['volume'].rolling(window=20).mean()
            
            # Volume ratio
            df_features['volume_ratio'] = df_features['volume'] / df_features['volume_ma_20']
            
            # Price-volume features
            if 'close' in df_features.columns:
                df_features['vwap'] = (df_features['close'] * df_features['volume']).rolling(window=20).sum() / \
                                     df_features['volume'].rolling(window=20).sum()
        
        # Volatility features
        if 'returns' in df_features.columns:
            df_features['volatility_5'] = df_features['returns'].rolling(window=5).std()
            df_features['volatility_20'] = df_features['returns'].rolling(window=20).std()
            df_features['volatility_ratio'] = df_features['volatility_5'] / df_features['volatility_20']
        
        return df_features
    
    def add_technical_indicators(
        self, 
        df: pd.DataFrame,
        indicators: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Add technical indicators to DataFrame.
        
        Args:
            df: OHLCV DataFrame
            indicators: List of indicators to calculate (None for all)
            
        Returns:
            DataFrame with technical indicators
        """
        df_tech = df.copy()
        
        if indicators is None:
            indicators = ['sma', 'ema', 'rsi', 'macd', 'bollinger', 'stochastic', 'atr']
        
        # Moving Averages
        if 'sma' in indicators and 'close' in df_tech.columns:
            sma_periods = [5, 10, 20, 50, 200]
            for period in sma_periods:
                sma = MovingAverage(period=period)
                df_tech[f'sma_{period}'] = sma.calculate(df_tech['close'])
        
        if 'ema' in indicators and 'close' in df_tech.columns:
            ema_periods = [12, 26, 50]
            for period in ema_periods:
                ema = ExponentialMovingAverage(period=period)
                df_tech[f'ema_{period}'] = ema.calculate(df_tech['close'])
        
        # RSI
        if 'rsi' in indicators and 'close' in df_tech.columns:
            rsi = RSI(period=14)
            df_tech['rsi'] = rsi.calculate(df_tech['close'])
            df_tech['rsi_oversold'] = df_tech['rsi'] < 30
            df_tech['rsi_overbought'] = df_tech['rsi'] > 70
        
        # MACD
        if 'macd' in indicators and 'close' in df_tech.columns:
            macd = MACD(fast_period=12, slow_period=26, signal_period=9)
            macd_data = macd.calculate(df_tech['close'])
            df_tech['macd'] = macd_data['macd']
            df_tech['macd_signal'] = macd_data['signal']
            df_tech['macd_histogram'] = macd_data['histogram']
            df_tech['macd_bullish'] = df_tech['macd'] > df_tech['macd_signal']
        
        # Bollinger Bands
        if 'bollinger' in indicators and 'close' in df_tech.columns:
            bb = BollingerBands(period=20, std_dev=2.0)
            bb_data = bb.calculate(df_tech['close'])
            df_tech['bb_upper'] = bb_data['upper_band']
            df_tech['bb_middle'] = bb_data['middle_band']
            df_tech['bb_lower'] = bb_data['lower_band']
            df_tech['bb_width'] = (bb_data['upper_band'] - bb_data['lower_band']) / bb_data['middle_band']
            df_tech['bb_position'] = (df_tech['close'] - bb_data['lower_band']) / (bb_data['upper_band'] - bb_data['lower_band'])
        
        # Stochastic Oscillator
        if 'stochastic' in indicators and all(col in df_tech.columns for col in ['high', 'low', 'close']):
            stoch = StochasticOscillator(k_period=14, d_period=3)
            stoch_data = stoch.calculate(df_tech['high'], df_tech['low'], df_tech['close'])
            df_tech['stoch_k'] = stoch_data['k_percent']
            df_tech['stoch_d'] = stoch_data['d_percent']
            df_tech['stoch_oversold'] = df_tech['stoch_k'] < 20
            df_tech['stoch_overbought'] = df_tech['stoch_k'] > 80
        
        # Average True Range
        if 'atr' in indicators and all(col in df_tech.columns for col in ['high', 'low', 'close']):
            atr = AverageTrueRange(period=14)
            df_tech['atr'] = atr.calculate(df_tech['high'], df_tech['low'], df_tech['close'])
            df_tech['atr_percent'] = df_tech['atr'] / df_tech['close']
        
        return df_tech
    
    def add_time_features(self, df: pd.DataFrame, date_column: str = 'date') -> pd.DataFrame:
        """
        Add time-based features to DataFrame.
        
        Args:
            df: DataFrame with date column
            date_column: Name of date column
            
        Returns:
            DataFrame with time features
        """
        df_time = df.copy()
        
        if date_column not in df_time.columns:
            self.logger.warning(f"Date column '{date_column}' not found")
            return df_time
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df_time[date_column]):
            df_time[date_column] = pd.to_datetime(df_time[date_column])
        
        # Basic time features
        df_time['year'] = df_time[date_column].dt.year
        df_time['month'] = df_time[date_column].dt.month
        df_time['day'] = df_time[date_column].dt.day
        df_time['dayofweek'] = df_time[date_column].dt.dayofweek
        df_time['dayofyear'] = df_time[date_column].dt.dayofyear
        df_time['quarter'] = df_time[date_column].dt.quarter
        df_time['week'] = df_time[date_column].dt.isocalendar().week
        
        # Trading day features
        df_time['is_monday'] = df_time['dayofweek'] == 0
        df_time['is_friday'] = df_time['dayofweek'] == 4
        df_time['is_month_end'] = df_time[date_column].dt.is_month_end
        df_time['is_month_start'] = df_time[date_column].dt.is_month_start
        df_time['is_quarter_end'] = df_time[date_column].dt.is_quarter_end
        
        # Cyclical encoding for time features
        df_time['month_sin'] = np.sin(2 * np.pi * df_time['month'] / 12)
        df_time['month_cos'] = np.cos(2 * np.pi * df_time['month'] / 12)
        df_time['dayofweek_sin'] = np.sin(2 * np.pi * df_time['dayofweek'] / 7)
        df_time['dayofweek_cos'] = np.cos(2 * np.pi * df_time['dayofweek'] / 7)
        
        return df_time
    
    def add_lag_features(
        self, 
        df: pd.DataFrame, 
        columns: List[str], 
        lags: List[int],
        group_by: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Add lagged features to DataFrame.
        
        Args:
            df: Input DataFrame
            columns: Columns to create lags for
            lags: List of lag periods
            group_by: Column to group by (e.g., 'symbol')
            
        Returns:
            DataFrame with lag features
        """
        df_lag = df.copy()
        
        for col in columns:
            if col not in df_lag.columns:
                continue
            
            for lag in lags:
                lag_col_name = f'{col}_lag_{lag}'
                
                if group_by and group_by in df_lag.columns:
                    df_lag[lag_col_name] = df_lag.groupby(group_by)[col].shift(lag)
                else:
                    df_lag[lag_col_name] = df_lag[col].shift(lag)
        
        return df_lag
    
    def add_rolling_features(
        self, 
        df: pd.DataFrame, 
        columns: List[str], 
        windows: List[int],
        operations: List[str] = ['mean', 'std', 'min', 'max'],
        group_by: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Add rolling window features to DataFrame.
        
        Args:
            df: Input DataFrame
            columns: Columns to create rolling features for
            windows: List of window sizes
            operations: List of operations ('mean', 'std', 'min', 'max', 'median')
            group_by: Column to group by (e.g., 'symbol')
            
        Returns:
            DataFrame with rolling features
        """
        df_rolling = df.copy()
        
        for col in columns:
            if col not in df_rolling.columns:
                continue
            
            for window in windows:
                for operation in operations:
                    feature_name = f'{col}_rolling_{window}_{operation}'
                    
                    if group_by and group_by in df_rolling.columns:
                        grouped = df_rolling.groupby(group_by)[col].rolling(window=window, min_periods=1)
                    else:
                        grouped = df_rolling[col].rolling(window=window, min_periods=1)
                    
                    if operation == 'mean':
                        df_rolling[feature_name] = grouped.mean().values
                    elif operation == 'std':
                        df_rolling[feature_name] = grouped.std().values
                    elif operation == 'min':
                        df_rolling[feature_name] = grouped.min().values
                    elif operation == 'max':
                        df_rolling[feature_name] = grouped.max().values
                    elif operation == 'median':
                        df_rolling[feature_name] = grouped.median().values
        
        return df_rolling
    
    def normalize_features(
        self, 
        df: pd.DataFrame, 
        columns: List[str], 
        method: str = 'zscore',
        group_by: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Normalize specified columns in DataFrame.
        
        Args:
            df: Input DataFrame
            columns: Columns to normalize
            method: Normalization method ('zscore', 'minmax', 'robust')
            group_by: Column to group by for normalization
            
        Returns:
            DataFrame with normalized features
        """
        df_norm = df.copy()
        
        for col in columns:
            if col not in df_norm.columns:
                continue
            
            if group_by and group_by in df_norm.columns:
                if method == 'zscore':
                    df_norm[f'{col}_norm'] = df_norm.groupby(group_by)[col].transform(
                        lambda x: (x - x.mean()) / x.std()
                    )
                elif method == 'minmax':
                    df_norm[f'{col}_norm'] = df_norm.groupby(group_by)[col].transform(
                        lambda x: (x - x.min()) / (x.max() - x.min())
                    )
                elif method == 'robust':
                    df_norm[f'{col}_norm'] = df_norm.groupby(group_by)[col].transform(
                        lambda x: (x - x.median()) / x.mad()
                    )
            else:
                if method == 'zscore':
                    df_norm[f'{col}_norm'] = (df_norm[col] - df_norm[col].mean()) / df_norm[col].std()
                elif method == 'minmax':
                    df_norm[f'{col}_norm'] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())
                elif method == 'robust':
                    df_norm[f'{col}_norm'] = (df_norm[col] - df_norm[col].median()) / df_norm[col].mad()
        
        return df_norm
    
    def create_features_pipeline(
        self, 
        df: pd.DataFrame,
        include_basic: bool = True,
        include_technical: bool = True,
        include_time: bool = True,
        technical_indicators: Optional[List[str]] = None,
        lag_periods: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Complete feature engineering pipeline.
        
        Args:
            df: Input OHLCV DataFrame
            include_basic: Include basic derived features
            include_technical: Include technical indicators
            include_time: Include time-based features
            technical_indicators: Specific technical indicators to include
            lag_periods: Lag periods for lag features
            rolling_windows: Window sizes for rolling features
            
        Returns:
            DataFrame with all features
        """
        # Start with cleaned data
        df_processed = self.clean_ohlcv_data(df)
        
        # Add basic features
        if include_basic:
            df_processed = self.add_basic_features(df_processed)
        
        # Add technical indicators
        if include_technical:
            df_processed = self.add_technical_indicators(df_processed, technical_indicators)
        
        # Add time features
        if include_time and 'date' in df_processed.columns:
            df_processed = self.add_time_features(df_processed)
        
        # Add lag features
        if lag_periods:
            price_columns = ['close', 'volume', 'returns']
            available_columns = [col for col in price_columns if col in df_processed.columns]
            df_processed = self.add_lag_features(
                df_processed, 
                available_columns, 
                lag_periods,
                group_by='symbol' if 'symbol' in df_processed.columns else None
            )
        
        # Add rolling features
        if rolling_windows:
            price_columns = ['close', 'volume', 'returns']
            available_columns = [col for col in price_columns if col in df_processed.columns]
            df_processed = self.add_rolling_features(
                df_processed, 
                available_columns, 
                rolling_windows,
                group_by='symbol' if 'symbol' in df_processed.columns else None
            )
        
        self.logger.info(f"Feature engineering completed. Added {len(df_processed.columns) - len(df.columns)} new features")
        
        return df_processed
