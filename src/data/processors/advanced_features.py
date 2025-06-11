"""
Advanced Feature Engineering Module

This module provides sophisticated feature engineering capabilities for quantitative trading,
including market microstructure features, regime detection, and cross-asset features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import warnings
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from scipy import stats
import talib

warnings.filterwarnings('ignore')


class AdvancedFeatureEngineer:
    """
    Advanced feature engineering for quantitative trading strategies
    """
    
    def __init__(self):
        self.scalers = {}
        self.pca_models = {}
        self.regime_models = {}
    
    def create_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create market microstructure features
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            DataFrame with microstructure features
        """
        result = df.copy()
        
        # Price impact and liquidity proxies
        result['amihud_illiquidity'] = abs(result['close'].pct_change()) / result['volume']
        result['roll_spread'] = 2 * np.sqrt(abs(result['close'].diff().rolling(20).cov(result['close'].shift(1))))
        
        # Order flow imbalance proxy
        result['price_volume_imbalance'] = result['close'].pct_change() * result['volume']
        result['buying_pressure'] = np.where(result['close'] > result['open'], result['volume'], -result['volume'])
        result['cumulative_buying_pressure'] = result['buying_pressure'].rolling(20).sum()
        
        # Tick direction (simplified)
        result['tick_direction'] = np.sign(result['close'].diff())
        result['tick_momentum'] = result['tick_direction'].rolling(10).sum()
        
        # Volatility clustering
        returns = result['close'].pct_change()
        result['vol_clustering'] = returns.rolling(20).std() / returns.rolling(60).std()
        
        # Jump detection
        result['jump_intensity'] = abs(returns) / returns.rolling(20).std()
        result['is_jump'] = (result['jump_intensity'] > 3).astype(int)
        
        return result
    
    def create_regime_detection_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create market regime detection features
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            DataFrame with regime features
        """
        result = df.copy()
        returns = result['close'].pct_change()
        
        # Volatility regimes
        vol_short = returns.rolling(20).std()
        vol_long = returns.rolling(60).std()
        result['vol_regime_ratio'] = vol_short / vol_long
        
        # Trend regimes
        sma_short = result['close'].rolling(20).mean()
        sma_long = result['close'].rolling(60).mean()
        result['trend_regime'] = (sma_short / sma_long - 1) * 100
        
        # Momentum regimes
        result['momentum_regime'] = returns.rolling(20).mean() / returns.rolling(20).std()
        
        # Crisis detection (extreme volatility)
        vol_threshold = vol_long.quantile(0.95)
        result['crisis_regime'] = (vol_short > vol_threshold).astype(int)
        
        # Market stress indicator
        result['stress_indicator'] = (
            result['vol_regime_ratio'] * 0.4 +
            abs(result['momentum_regime']) * 0.3 +
            result['crisis_regime'] * 0.3
        )
        
        # Regime persistence
        result['regime_persistence'] = result['stress_indicator'].rolling(10).std()
        
        return result
    
    def create_cross_asset_features(self, main_df: pd.DataFrame, 
                                  market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create cross-asset and market-wide features
        
        Args:
            main_df: Primary asset DataFrame
            market_data: Dictionary of market data (e.g., {'VIX': vix_df, 'SPY': spy_df})
            
        Returns:
            DataFrame with cross-asset features
        """
        result = main_df.copy()
        
        for asset_name, asset_df in market_data.items():
            if asset_df.empty:
                continue
                
            # Align indices
            common_dates = result.index.intersection(asset_df.index)
            if len(common_dates) == 0:
                continue
            
            asset_aligned = asset_df.loc[common_dates, 'close']
            main_aligned = result.loc[common_dates, 'close']
            
            # Correlation features
            corr_30d = main_aligned.pct_change().rolling(30).corr(asset_aligned.pct_change())
            result.loc[common_dates, f'{asset_name}_correlation_30d'] = corr_30d
            
            # Beta calculation
            returns_main = main_aligned.pct_change()
            returns_asset = asset_aligned.pct_change()
            
            covariance = returns_main.rolling(60).cov(returns_asset)
            variance_asset = returns_asset.rolling(60).var()
            beta = covariance / variance_asset
            result.loc[common_dates, f'{asset_name}_beta_60d'] = beta
            
            # Relative performance
            rel_perf = (main_aligned / asset_aligned).pct_change()
            result.loc[common_dates, f'{asset_name}_relative_performance'] = rel_perf
            
            # Lead-lag relationships
            for lag in [1, 2, 3, 5]:
                lagged_corr = returns_main.rolling(30).corr(returns_asset.shift(lag))
                result.loc[common_dates, f'{asset_name}_lag_{lag}_correlation'] = lagged_corr
        
        return result
    
    def create_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create market sentiment proxy features
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            DataFrame with sentiment features
        """
        result = df.copy()
        
        # Price momentum sentiment
        for period in [5, 10, 20, 30]:
            momentum = result['close'].pct_change(period)
            result[f'momentum_sentiment_{period}d'] = momentum.rolling(20).rank(pct=True)
        
        # Volume sentiment
        vol_ma = result['volume'].rolling(20).mean()
        result['volume_sentiment'] = (result['volume'] / vol_ma).rolling(10).rank(pct=True)
        
        # Volatility sentiment (fear/greed)
        vol = result['close'].pct_change().rolling(20).std()
        result['volatility_sentiment'] = 1 - vol.rolling(60).rank(pct=True)  # Inverse: low vol = positive sentiment
        
        # Price position sentiment
        high_52w = result['close'].rolling(252).max()
        low_52w = result['close'].rolling(252).min()
        result['price_position_sentiment'] = (result['close'] - low_52w) / (high_52w - low_52w)
        
        # Composite sentiment score
        sentiment_components = [
            'momentum_sentiment_20d',
            'volume_sentiment', 
            'volatility_sentiment',
            'price_position_sentiment'
        ]
        
        available_components = [col for col in sentiment_components if col in result.columns]
        if available_components:
            result['composite_sentiment'] = result[available_components].mean(axis=1)
        
        return result
    
    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create statistical and distributional features
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            DataFrame with statistical features
        """
        result = df.copy()
        returns = result['close'].pct_change()
        
        # Rolling statistical moments
        for window in [20, 60]:
            result[f'skewness_{window}d'] = returns.rolling(window).skew()
            result[f'kurtosis_{window}d'] = returns.rolling(window).kurt()
            
        # Quantile features
        for window in [20, 60]:
            for quantile in [0.05, 0.25, 0.75, 0.95]:
                col_name = f'return_q{int(quantile*100)}_{window}d'
                result[col_name] = returns.rolling(window).quantile(quantile)
        
        # Tail risk measures
        result['var_5pct_20d'] = returns.rolling(20).quantile(0.05)
        result['cvar_5pct_20d'] = returns[returns <= result['var_5pct_20d']].rolling(20).mean()
        
        # Normality tests (rolling)
        def rolling_jarque_bera(x):
            if len(x) < 8:
                return np.nan
            try:
                stat, p_value = stats.jarque_bera(x.dropna())
                return p_value
            except:
                return np.nan
        
        result['normality_test_60d'] = returns.rolling(60).apply(rolling_jarque_bera)
        
        # Distribution stability
        result['distribution_stability'] = (
            result['skewness_20d'].rolling(10).std() + 
            result['kurtosis_20d'].rolling(10).std()
        )
        
        return result
    
    def create_fractal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create fractal and complexity features
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            DataFrame with fractal features
        """
        result = df.copy()
        
        # Hurst exponent approximation
        def hurst_exponent(ts, max_lag=20):
            """Calculate Hurst exponent for time series"""
            if len(ts) < max_lag * 2:
                return np.nan
            
            lags = range(2, max_lag)
            tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
            
            if len(tau) < 2 or any(t <= 0 for t in tau):
                return np.nan
            
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        
        # Rolling Hurst exponent
        price_series = result['close']
        result['hurst_exponent_60d'] = price_series.rolling(60).apply(
            lambda x: hurst_exponent(x.values), raw=False
        )
        
        # Fractal dimension approximation
        result['fractal_dimension'] = 2 - result['hurst_exponent_60d']
        
        # Complexity measures
        def approximate_entropy(data, m=2, r=None):
            """Calculate approximate entropy"""
            if len(data) < 10:
                return np.nan
            
            if r is None:
                r = 0.2 * np.std(data)
            
            def _maxdist(xi, xj, m):
                return max([abs(ua - va) for ua, va in zip(xi, xj)])
            
            def _phi(m):
                patterns = np.array([data[i:i+m] for i in range(len(data) - m + 1)])
                C = np.zeros(len(patterns))
                for i in range(len(patterns)):
                    template_i = patterns[i]
                    for j in range(len(patterns)):
                        if _maxdist(template_i, patterns[j], m) <= r:
                            C[i] += 1.0
                C = C / float(len(patterns))
                phi = np.mean(np.log(C))
                return phi
            
            return _phi(m) - _phi(m + 1)
        
        returns = result['close'].pct_change().fillna(0)
        result['approximate_entropy_60d'] = returns.rolling(60).apply(
            lambda x: approximate_entropy(x.values), raw=False
        )
        
        return result
    
    def create_technical_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical pattern recognition features
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            DataFrame with pattern features
        """
        result = df.copy()
        
        # Ensure we have the required columns
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in result.columns for col in required_cols):
            return result
        
        try:
            # Convert to numpy arrays for TA-Lib
            open_prices = result['open'].values.astype(float)
            high_prices = result['high'].values.astype(float)
            low_prices = result['low'].values.astype(float)
            close_prices = result['close'].values.astype(float)
            
            # Candlestick patterns
            patterns = {
                'doji': talib.CDLDOJI(open_prices, high_prices, low_prices, close_prices),
                'hammer': talib.CDLHAMMER(open_prices, high_prices, low_prices, close_prices),
                'shooting_star': talib.CDLSHOOTINGSTAR(open_prices, high_prices, low_prices, close_prices),
                'engulfing': talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices),
                'harami': talib.CDLHARAMI(open_prices, high_prices, low_prices, close_prices),
                'morning_star': talib.CDLMORNINGSTAR(open_prices, high_prices, low_prices, close_prices),
                'evening_star': talib.CDLEVENINGSTAR(open_prices, high_prices, low_prices, close_prices)
            }
            
            for pattern_name, pattern_values in patterns.items():
                result[f'pattern_{pattern_name}'] = pattern_values
            
            # Pattern strength (sum of recent patterns)
            pattern_cols = [col for col in result.columns if col.startswith('pattern_')]
            result['pattern_strength'] = result[pattern_cols].rolling(5).sum().sum(axis=1)
            
        except Exception as e:
            print(f"Warning: Could not calculate technical patterns: {e}")
        
        return result
    
    def create_alternative_data_features(self, df: pd.DataFrame, 
                                       alt_data: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
        """
        Create features from alternative data sources
        
        Args:
            df: OHLCV DataFrame
            alt_data: Dictionary of alternative data (e.g., news sentiment, social media)
            
        Returns:
            DataFrame with alternative data features
        """
        result = df.copy()
        
        if alt_data is None:
            alt_data = {}
        
        # Process each alternative data source
        for data_name, data_df in alt_data.items():
            if data_df.empty:
                continue
            
            # Align data
            common_dates = result.index.intersection(data_df.index)
            if len(common_dates) == 0:
                continue
            
            # Add features based on data type
            if 'sentiment' in data_name.lower():
                # Sentiment scores
                if 'score' in data_df.columns:
                    sentiment = data_df.loc[common_dates, 'score']
                    result.loc[common_dates, f'{data_name}_score'] = sentiment
                    result.loc[common_dates, f'{data_name}_ma_5d'] = sentiment.rolling(5).mean()
                    result.loc[common_dates, f'{data_name}_change'] = sentiment.diff()
            
            elif 'volume' in data_name.lower() or 'activity' in data_name.lower():
                # Activity/volume metrics
                if 'count' in data_df.columns:
                    activity = data_df.loc[common_dates, 'count']
                    result.loc[common_dates, f'{data_name}_activity'] = activity
                    result.loc[common_dates, f'{data_name}_activity_ma'] = activity.rolling(7).mean()
            
            elif 'economic' in data_name.lower():
                # Economic indicators
                for col in data_df.columns:
                    if col in ['value', 'index', 'rate']:
                        indicator = data_df.loc[common_dates, col]
                        result.loc[common_dates, f'{data_name}_{col}'] = indicator
                        result.loc[common_dates, f'{data_name}_{col}_change'] = indicator.pct_change()
        
        return result
    
    def apply_feature_selection(self, df: pd.DataFrame, 
                              target_col: str,
                              method: str = 'correlation',
                              top_k: int = 50) -> Tuple[pd.DataFrame, List[str]]:
        """
        Apply feature selection to reduce dimensionality
        
        Args:
            df: DataFrame with features
            target_col: Target column name
            method: Selection method ('correlation', 'mutual_info', 'pca')
            top_k: Number of top features to select
            
        Returns:
            Tuple of (selected features DataFrame, selected feature names)
        """
        feature_cols = [col for col in df.columns if col != target_col and not col.startswith('target_')]
        
        if method == 'correlation':
            # Select features based on correlation with target
            correlations = df[feature_cols].corrwith(df[target_col]).abs()
            top_features = correlations.nlargest(top_k).index.tolist()
            
        elif method == 'mutual_info':
            from sklearn.feature_selection import mutual_info_regression
            
            # Fill NaN values for mutual info calculation
            X = df[feature_cols].fillna(0)
            y = df[target_col].fillna(0)
            
            mi_scores = mutual_info_regression(X, y)
            mi_df = pd.DataFrame({'feature': feature_cols, 'score': mi_scores})
            top_features = mi_df.nlargest(top_k, 'score')['feature'].tolist()
            
        elif method == 'pca':
            # PCA-based feature selection
            X = df[feature_cols].fillna(0)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            pca = PCA(n_components=min(top_k, X.shape[1]))
            X_pca = pca.fit_transform(X_scaled)
            
            # Create PCA feature names
            top_features = [f'pca_component_{i}' for i in range(X_pca.shape[1])]
            
            # Create new DataFrame with PCA components
            result_df = df[[target_col]].copy()
            for i, feature_name in enumerate(top_features):
                result_df[feature_name] = X_pca[:, i]
            
            self.scalers['pca_scaler'] = scaler
            self.pca_models['feature_pca'] = pca
            
            return result_df, top_features
        
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        # Return selected features
        selected_df = df[[target_col] + top_features].copy()
        return selected_df, top_features
    
    def create_interaction_features(self, df: pd.DataFrame, 
                                  feature_groups: Optional[List[List[str]]] = None) -> pd.DataFrame:
        """
        Create interaction features between existing features
        
        Args:
            df: DataFrame with features
            feature_groups: List of feature groups to create interactions within
            
        Returns:
            DataFrame with interaction features
        """
        result = df.copy()
        
        if feature_groups is None:
            # Default feature groups
            price_features = [col for col in df.columns if any(x in col.lower() for x in ['price', 'close', 'sma', 'ema'])]
            volume_features = [col for col in df.columns if 'volume' in col.lower()]
            volatility_features = [col for col in df.columns if 'vol' in col.lower()]
            
            feature_groups = [price_features[:5], volume_features[:3], volatility_features[:3]]
        
        # Create interactions within each group
        for group_idx, group in enumerate(feature_groups):
            if len(group) < 2:
                continue
            
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    feat1, feat2 = group[i], group[j]
                    if feat1 in result.columns and feat2 in result.columns:
                        # Multiplication interaction
                        result[f'{feat1}_x_{feat2}'] = result[feat1] * result[feat2]
                        
                        # Ratio interaction (avoid division by zero)
                        mask = result[feat2] != 0
                        result[f'{feat1}_div_{feat2}'] = np.where(mask, result[feat1] / result[feat2], 0)
        
        return result
    
    def get_feature_importance(self, df: pd.DataFrame, 
                             target_col: str,
                             method: str = 'random_forest') -> pd.DataFrame:
        """
        Calculate feature importance using various methods
        
        Args:
            df: DataFrame with features
            target_col: Target column name
            method: Importance calculation method
            
        Returns:
            DataFrame with feature importance scores
        """
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.inspection import permutation_importance
        
        feature_cols = [col for col in df.columns if col != target_col and not col.startswith('target_')]
        
        # Prepare data
        X = df[feature_cols].fillna(0)
        y = df[target_col].fillna(0)
        
        if method == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
        elif method == 'permutation':
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X, y)
            
            perm_importance = permutation_importance(model, X, y, n_repeats=5, random_state=42)
            
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': perm_importance.importances_mean,
                'std': perm_importance.importances_std
            }).sort_values('importance', ascending=False)
        
        else:
            raise ValueError(f"Unknown importance method: {method}")
        
        return importance_df


# Example usage
if __name__ == "__main__":
    # Initialize feature engineer
    feature_engineer = AdvancedFeatureEngineer()
    
    # Create sample data
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
    sample_data = pd.DataFrame({
        'open': np.random.randn(len(dates)).cumsum() + 100,
        'high': np.random.randn(len(dates)).cumsum() + 105,
        'low': np.random.randn(len(dates)).cumsum() + 95,
        'close': np.random.randn(len(dates)).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    # Ensure proper OHLC relationships
    sample_data['high'] = sample_data[['open', 'close']].max(axis=1) + abs(np.random.randn(len(dates)))
    sample_data['low'] = sample_data[['open', 'close']].min(axis=1) - abs(np.random.randn(len(dates)))
    
    # Apply feature engineering
    try:
        print("Creating microstructure features...")
        result = feature_engineer.create_microstructure_features(sample_data)
        
        print("Creating regime detection features...")
        result = feature_engineer.create_regime_detection_features(result)
        
        print("Creating sentiment features...")
        result = feature_engineer.create_sentiment_features(result)
        
        print("Creating statistical features...")
        result = feature_engineer.create_statistical_features(result)
        
        print(f"Final feature count: {len(result.columns)}")
        print(f"Sample features: {list(result.columns[:10])}")
        
    except Exception as e:
        print(f"Error in feature engineering: {e}")
