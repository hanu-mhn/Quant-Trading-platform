"""
Machine Learning Models for Trading Prediction.

This module provides various ML models for price prediction, trend analysis,
and signal generation in quantitative trading.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
import logging
from abc import ABC, abstractmethod

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, classification_report
import joblib


class BasePredictionModel(ABC):
    """Abstract base class for prediction models."""
    
    def __init__(self, model_type: str = "regression"):
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.is_trained = False
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        pass
    
    def save_model(self, filepath: str) -> None:
        """Save trained model to file."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load trained model from file."""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.is_trained = True
        self.logger.info(f"Model loaded from {filepath}")


class PricePredictor(BasePredictionModel):
    """Model for predicting future prices."""
    
    def __init__(
        self,
        model_name: str = "random_forest",
        lookback_period: int = 30,
        prediction_horizon: int = 1,
        use_scaling: bool = True
    ):
        super().__init__("regression")
        self.model_name = model_name
        self.lookback_period = lookback_period
        self.prediction_horizon = prediction_horizon
        self.use_scaling = use_scaling
        
        # Initialize model based on name
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the underlying ML model."""
        if self.model_name == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_name == "linear_regression":
            self.model = LinearRegression()
        elif self.model_name == "svr":
            self.model = SVR(kernel='rbf', C=1.0, gamma='scale')
        elif self.model_name == "neural_network":
            self.model = MLPRegressor(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")
        
        if self.use_scaling:
            self.scaler = StandardScaler()
    
    def prepare_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for training.
        
        Args:
            data: OHLCV DataFrame with technical indicators
            
        Returns:
            Features DataFrame and target Series
        """
        # Select relevant features
        feature_columns = []
        
        # Price features
        price_features = ['open', 'high', 'low', 'close', 'volume']
        feature_columns.extend([col for col in price_features if col in data.columns])
        
        # Technical indicator features
        tech_features = [col for col in data.columns if any(indicator in col.lower() 
                        for indicator in ['sma', 'ema', 'rsi', 'macd', 'bb_', 'stoch', 'atr'])]
        feature_columns.extend(tech_features)
        
        # Basic derived features
        derived_features = [col for col in data.columns if any(feature in col.lower()
                           for feature in ['returns', 'volatility', 'range', 'volume_ratio'])]
        feature_columns.extend(derived_features)
        
        # Remove duplicates and ensure columns exist
        feature_columns = list(set(feature_columns))
        feature_columns = [col for col in feature_columns if col in data.columns]
        
        # Create lagged features
        features_df = pd.DataFrame()
        for i in range(self.lookback_period):
            lagged_data = data[feature_columns].shift(i).add_suffix(f'_lag_{i}')
            features_df = pd.concat([features_df, lagged_data], axis=1)
        
        # Create target (future price)
        target = data['close'].shift(-self.prediction_horizon)
        
        # Remove rows with NaN values
        valid_indices = features_df.notna().all(axis=1) & target.notna()
        features_df = features_df[valid_indices]
        target = target[valid_indices]
        
        self.feature_names = features_df.columns.tolist()
        
        return features_df, target
    
    def train(
        self,
        data: pd.DataFrame,
        test_size: float = 0.2,
        validation_split: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the price prediction model.
        
        Args:
            data: Training data with OHLCV and indicators
            test_size: Proportion of data for testing
            validation_split: Use time series validation
            **kwargs: Additional training parameters
            
        Returns:
            Training results and metrics
        """
        # Prepare features and target
        X, y = self.prepare_features(data)
        
        if len(X) == 0:
            raise ValueError("No valid training data after feature preparation")
        
        # Split data
        if validation_split:
            # Use time series split to maintain temporal order
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        
        # Scale features if enabled
        if self.use_scaling:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_train_scaled = X_train.values
            X_test_scaled = X_test.values
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Make predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        # Calculate percentage errors
        train_mape = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
        test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
        
        results = {
            'model_name': self.model_name,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_mape': train_mape,
            'test_mape': test_mape,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'feature_count': len(self.feature_names)
        }
        
        # Feature importance for tree-based models
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            results['feature_importance'] = feature_importance
        
        self.logger.info(f"Model training completed. Test MAPE: {test_mape:.2f}%")
        
        return results
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make price predictions.
        
        Args:
            data: Input data for prediction
            
        Returns:
            Predicted prices
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare features
        X, _ = self.prepare_features(data)
        
        if len(X) == 0:
            return np.array([])
        
        # Scale features if enabled
        if self.use_scaling:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        return predictions


class TrendClassifier(BasePredictionModel):
    """Model for classifying price trends (up/down/sideways)."""
    
    def __init__(
        self,
        model_name: str = "random_forest",
        trend_threshold: float = 0.02,
        lookback_period: int = 20,
        use_scaling: bool = True
    ):
        super().__init__("classification")
        self.model_name = model_name
        self.trend_threshold = trend_threshold
        self.lookback_period = lookback_period
        self.use_scaling = use_scaling
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the underlying ML model."""
        if self.model_name == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_name == "logistic_regression":
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        elif self.model_name == "svc":
            self.model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        elif self.model_name == "neural_network":
            self.model = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")
        
        if self.use_scaling:
            self.scaler = StandardScaler()
    
    def create_trend_labels(self, data: pd.DataFrame, periods: int = 5) -> pd.Series:
        """
        Create trend labels based on future price movement.
        
        Args:
            data: OHLCV DataFrame
            periods: Number of periods to look ahead
            
        Returns:
            Trend labels (0: down, 1: sideways, 2: up)
        """
        future_returns = data['close'].shift(-periods) / data['close'] - 1
        
        labels = pd.Series(index=data.index, dtype=int)
        labels[future_returns <= -self.trend_threshold] = 0  # Down
        labels[future_returns >= self.trend_threshold] = 2   # Up
        labels[(future_returns > -self.trend_threshold) & 
               (future_returns < self.trend_threshold)] = 1   # Sideways
        
        return labels
    
    def prepare_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for training.
        
        Args:
            data: OHLCV DataFrame with technical indicators
            
        Returns:
            Features DataFrame and target Series
        """
        # Select relevant features (similar to PricePredictor)
        feature_columns = []
        
        # Technical indicator features
        tech_features = [col for col in data.columns if any(indicator in col.lower() 
                        for indicator in ['sma', 'ema', 'rsi', 'macd', 'bb_', 'stoch', 'atr'])]
        feature_columns.extend(tech_features)
        
        # Returns and momentum features
        momentum_features = [col for col in data.columns if any(feature in col.lower()
                            for feature in ['returns', 'volatility', 'volume_ratio'])]
        feature_columns.extend(momentum_features)
        
        # Remove duplicates and ensure columns exist
        feature_columns = list(set(feature_columns))
        feature_columns = [col for col in feature_columns if col in data.columns]
        
        # Use current values (no lagging for classification)
        features_df = data[feature_columns].copy()
        
        # Create trend labels
        target = self.create_trend_labels(data)
        
        # Remove rows with NaN values
        valid_indices = features_df.notna().all(axis=1) & target.notna()
        features_df = features_df[valid_indices]
        target = target[valid_indices]
        
        self.feature_names = features_df.columns.tolist()
        
        return features_df, target
    
    def train(
        self,
        data: pd.DataFrame,
        test_size: float = 0.2,
        validation_split: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the trend classification model.
        
        Args:
            data: Training data with OHLCV and indicators
            test_size: Proportion of data for testing
            validation_split: Use time series validation
            **kwargs: Additional training parameters
            
        Returns:
            Training results and metrics
        """
        # Prepare features and target
        X, y = self.prepare_features(data)
        
        if len(X) == 0:
            raise ValueError("No valid training data after feature preparation")
        
        # Split data
        if validation_split:
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        
        # Scale features if enabled
        if self.use_scaling:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_train_scaled = X_train.values
            X_test_scaled = X_test.values
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Make predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        # Classification report
        class_report = classification_report(
            y_test, y_test_pred,
            target_names=['Down', 'Sideways', 'Up'],
            output_dict=True
        )
        
        results = {
            'model_name': self.model_name,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'classification_report': class_report,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'feature_count': len(self.feature_names)
        }
        
        # Feature importance for tree-based models
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            results['feature_importance'] = feature_importance
        
        self.logger.info(f"Model training completed. Test accuracy: {test_accuracy:.2%}")
        
        return results
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make trend predictions.
        
        Args:
            data: Input data for prediction
            
        Returns:
            Predicted trend classes
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare features
        X, _ = self.prepare_features(data)
        
        if len(X) == 0:
            return np.array([])
        
        # Scale features if enabled
        if self.use_scaling:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            data: Input data for prediction
            
        Returns:
            Prediction probabilities for each class
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("Model does not support probability predictions")
        
        # Prepare features
        X, _ = self.prepare_features(data)
        
        if len(X) == 0:
            return np.array([])
        
        # Scale features if enabled
        if self.use_scaling:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
        
        # Get probabilities
        probabilities = self.model.predict_proba(X_scaled)
        
        return probabilities


class ModelEnsemble:
    """Ensemble of multiple prediction models."""
    
    def __init__(self, models: List[BasePredictionModel]):
        self.models = models
        self.weights = None
        self.logger = logging.getLogger(__name__)
    
    def train_ensemble(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Train all models in the ensemble."""
        results = {}
        
        for i, model in enumerate(self.models):
            self.logger.info(f"Training model {i+1}/{len(self.models)}: {model.model_name}")
            model_results = model.train(data, **kwargs)
            results[f'model_{i}'] = model_results
        
        # Set equal weights by default
        self.weights = np.ones(len(self.models)) / len(self.models)
        
        return results
    
    def predict(self, data: pd.DataFrame, method: str = 'average') -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            data: Input data
            method: Ensemble method ('average', 'weighted', 'majority')
            
        Returns:
            Ensemble predictions
        """
        predictions = []
        
        for model in self.models:
            pred = model.predict(data)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        if method == 'average':
            return np.mean(predictions, axis=0)
        elif method == 'weighted':
            return np.average(predictions, axis=0, weights=self.weights)
        elif method == 'majority':
            # For classification
            from scipy.stats import mode
            return mode(predictions, axis=0)[0].flatten()
        else:
            raise ValueError(f"Unknown ensemble method: {method}")


def create_model_pipeline(
    model_type: str = "price_prediction",
    data: pd.DataFrame = None,
    **kwargs
) -> BasePredictionModel:
    """
    Create and optionally train a complete ML model pipeline.
    
    Args:
        model_type: Type of model ("price_prediction" or "trend_classification")
        data: Training data (optional)
        **kwargs: Model and training parameters
        
    Returns:
        Trained or initialized model
    """
    if model_type == "price_prediction":
        model = PricePredictor(**kwargs)
    elif model_type == "trend_classification":
        model = TrendClassifier(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train if data is provided
    if data is not None:
        model.train(data)
    
    return model
