"""
Risk Management Module.

This module provides comprehensive risk management tools for portfolio management
and position sizing in quantitative trading strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import logging

from ..common.base_classes import BaseRiskManager, Order, OrderSide
from ..config.config import Config


class PositionSizeCalculator:
    """Calculate position sizes based on various risk models."""
    
    @staticmethod
    def fixed_fractional(
        account_value: float,
        risk_percent: float,
        entry_price: float,
        stop_loss_price: float
    ) -> float:
        """
        Calculate position size using fixed fractional method.
        
        Args:
            account_value: Total account value
            risk_percent: Risk percentage (0.01 = 1%)
            entry_price: Entry price for the position
            stop_loss_price: Stop loss price
            
        Returns:
            Position size in shares/units
        """
        if entry_price <= 0 or stop_loss_price <= 0:
            return 0
        
        risk_amount = account_value * risk_percent
        price_risk = abs(entry_price - stop_loss_price)
        
        if price_risk <= 0:
            return 0
        
        position_size = risk_amount / price_risk
        return max(0, position_size)
    
    @staticmethod
    def kelly_criterion(
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        account_value: float
    ) -> float:
        """
        Calculate position size using Kelly Criterion.
        
        Args:
            win_rate: Historical win rate (0.6 = 60%)
            avg_win: Average winning trade amount
            avg_loss: Average losing trade amount
            account_value: Total account value
            
        Returns:
            Optimal fraction of account to risk
        """
        if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
            return 0
        
        win_loss_ratio = avg_win / abs(avg_loss)
        kelly_fraction = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        
        # Cap Kelly fraction to prevent over-leveraging
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Max 25% of account
        
        return kelly_fraction * account_value
    
    @staticmethod
    def volatility_based(
        price: float,
        volatility: float,
        target_volatility: float,
        account_value: float
    ) -> float:
        """
        Calculate position size based on volatility targeting.
        
        Args:
            price: Current price
            volatility: Historical volatility (annualized)
            target_volatility: Target portfolio volatility
            account_value: Total account value
            
        Returns:
            Position size in shares/units
        """
        if volatility <= 0 or price <= 0:
            return 0
        
        volatility_ratio = target_volatility / volatility
        position_value = account_value * volatility_ratio
        position_size = position_value / price
        
        return max(0, position_size)


class RiskMetrics:
    """Calculate various risk metrics for portfolio analysis."""
    
    @staticmethod
    def value_at_risk(
        returns: pd.Series,
        confidence_level: float = 0.95,
        time_horizon: int = 1
    ) -> float:
        """
        Calculate Value at Risk (VaR) using historical simulation.
        
        Args:
            returns: Historical returns series
            confidence_level: Confidence level (0.95 = 95%)
            time_horizon: Time horizon in days
            
        Returns:
            VaR value
        """
        if len(returns) == 0:
            return 0
        
        alpha = 1 - confidence_level
        var = np.percentile(returns, alpha * 100)
        
        # Scale for time horizon
        if time_horizon > 1:
            var = var * np.sqrt(time_horizon)
        
        return abs(var)
    
    @staticmethod
    def conditional_var(
        returns: pd.Series,
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Conditional Value at Risk (CVaR/Expected Shortfall).
        
        Args:
            returns: Historical returns series
            confidence_level: Confidence level (0.95 = 95%)
            
        Returns:
            CVaR value
        """
        if len(returns) == 0:
            return 0
        
        var = RiskMetrics.value_at_risk(returns, confidence_level)
        cvar = returns[returns <= -var].mean()
        
        return abs(cvar) if not np.isnan(cvar) else var
    
    @staticmethod
    def maximum_drawdown(prices: pd.Series) -> Dict[str, float]:
        """
        Calculate maximum drawdown and related metrics.
        
        Args:
            prices: Price series
            
        Returns:
            Dictionary with drawdown metrics
        """
        if len(prices) == 0:
            return {'max_drawdown': 0, 'drawdown_duration': 0, 'recovery_time': 0}
        
        cumulative = prices / prices.iloc[0]
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        
        max_drawdown = drawdown.min()
        max_dd_date = drawdown.idxmin()
        
        # Find drawdown duration
        drawdown_start = running_max[:max_dd_date].idxmax()
        drawdown_duration = (max_dd_date - drawdown_start).days if hasattr(max_dd_date, 'date') else 0
        
        # Find recovery time
        recovery_time = 0
        if max_dd_date < prices.index[-1]:
            post_dd = running_max[max_dd_date:]
            recovery_point = post_dd[post_dd > running_max.loc[max_dd_date]].index
            if len(recovery_point) > 0:
                recovery_time = (recovery_point[0] - max_dd_date).days if hasattr(recovery_point[0], 'date') else 0
        
        return {
            'max_drawdown': abs(max_drawdown),
            'drawdown_duration': drawdown_duration,
            'recovery_time': recovery_time,
            'drawdown_start': drawdown_start,
            'drawdown_end': max_dd_date
        }
    
    @staticmethod
    def sharpe_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: Returns series
            risk_free_rate: Risk-free rate (annualized)
            periods_per_year: Number of periods per year
            
        Returns:
            Sharpe ratio
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0
        
        excess_returns = returns - (risk_free_rate / periods_per_year)
        sharpe = excess_returns.mean() / returns.std()
        
        return sharpe * np.sqrt(periods_per_year)
    
    @staticmethod
    def sortino_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Sortino ratio (downside deviation).
        
        Args:
            returns: Returns series
            risk_free_rate: Risk-free rate (annualized)
            periods_per_year: Number of periods per year
            
        Returns:
            Sortino ratio
        """
        if len(returns) == 0:
            return 0
        
        excess_returns = returns - (risk_free_rate / periods_per_year)
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0
        
        sortino = excess_returns.mean() / downside_returns.std()
        
        return sortino * np.sqrt(periods_per_year)


class PortfolioRiskManager(BaseRiskManager):
    """Comprehensive portfolio risk management system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.config_manager = Config()
        self.config = config or self.config_manager.get_risk_config()
        
        # Risk limits
        self.max_portfolio_risk = self.config.max_portfolio_risk_percent / 100
        self.max_position_size = self.config.max_single_position_percent / 100
        self.max_sector_exposure = self.config.max_sector_exposure_percent / 100
        self.stop_loss_pct = self.config.stop_loss_percent / 100
        self.max_drawdown = self.config.max_drawdown_percent / 100
        self.var_confidence = self.config.var_confidence_level
        
        # Portfolio state
        self.positions = {}
        self.sector_exposures = {}
        self.portfolio_value = 0
        self.daily_returns = []
        self.risk_metrics_cache = {}
        
        self.logger = logging.getLogger(__name__)
    
    def check_risk_limits(self, order: Order, current_price: float, portfolio_value: float) -> bool:
        """
        Check if order violates risk limits.
        
        Args:
            order: Order to check
            current_price: Current market price
            portfolio_value: Current portfolio value
            
        Returns:
            True if order passes risk checks, False otherwise
        """
        try:
            # Calculate position value
            position_value = order.quantity * current_price
            
            # Check maximum position size
            position_pct = position_value / portfolio_value
            if position_pct > self.max_position_size:
                self.logger.warning(f"Order rejected: Position size {position_pct:.2%} exceeds limit {self.max_position_size:.2%}")
                return False
            
            # Check portfolio risk
            current_portfolio_risk = self.calculate_portfolio_risk()
            if current_portfolio_risk > self.max_portfolio_risk:
                self.logger.warning(f"Order rejected: Portfolio risk {current_portfolio_risk:.2%} exceeds limit {self.max_portfolio_risk:.2%}")
                return False
            
            # Check sector exposure (if sector info available)
            if hasattr(order, 'sector') and order.sector:
                current_sector_exposure = self.sector_exposures.get(order.sector, 0)
                new_sector_exposure = (current_sector_exposure + position_value) / portfolio_value
                
                if new_sector_exposure > self.max_sector_exposure:
                    self.logger.warning(f"Order rejected: Sector exposure {new_sector_exposure:.2%} exceeds limit {self.max_sector_exposure:.2%}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in risk check: {e}")
            return False
    
    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss_price: Optional[float] = None,
        method: str = 'fixed_fractional'
    ) -> float:
        """
        Calculate optimal position size based on risk management rules.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            stop_loss_price: Stop loss price
            method: Position sizing method
            
        Returns:
            Recommended position size
        """
        if self.portfolio_value <= 0:
            return 0
        
        if method == 'fixed_fractional':
            if stop_loss_price is None:
                # Use default stop loss percentage
                stop_loss_price = entry_price * (1 - self.stop_loss_pct)
            
            return PositionSizeCalculator.fixed_fractional(
                self.portfolio_value,
                self.max_portfolio_risk,
                entry_price,
                stop_loss_price
            )
        
        elif method == 'volatility_based':
            # Get historical volatility for the symbol
            volatility = self.get_symbol_volatility(symbol)
            target_vol = 0.15  # 15% target volatility
            
            return PositionSizeCalculator.volatility_based(
                entry_price,
                volatility,
                target_vol,
                self.portfolio_value
            )
        
        else:
            # Default to 1% risk
            return self.portfolio_value * 0.01 / entry_price
    
    def calculate_portfolio_risk(self) -> float:
        """Calculate current portfolio risk metrics."""
        if len(self.daily_returns) < 30:  # Need at least 30 days
            return 0
        
        returns_series = pd.Series(self.daily_returns)
        var = RiskMetrics.value_at_risk(returns_series, self.var_confidence)
        
        return var
    
    def update_portfolio_state(
        self,
        positions: Dict[str, float],
        portfolio_value: float,
        daily_return: Optional[float] = None
    ):
        """Update current portfolio state for risk calculations."""
        self.positions = positions.copy()
        self.portfolio_value = portfolio_value
        
        if daily_return is not None:
            self.daily_returns.append(daily_return)
            
            # Keep only last 252 days (1 year)
            if len(self.daily_returns) > 252:
                self.daily_returns = self.daily_returns[-252:]
    
    def get_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report."""
        if len(self.daily_returns) < 2:
            return {'status': 'Insufficient data for risk analysis'}
        
        returns_series = pd.Series(self.daily_returns)
        
        # Calculate risk metrics
        var_95 = RiskMetrics.value_at_risk(returns_series, 0.95)
        var_99 = RiskMetrics.value_at_risk(returns_series, 0.99)
        cvar_95 = RiskMetrics.conditional_var(returns_series, 0.95)
        
        # Create price series for drawdown calculation
        cumulative_returns = (1 + returns_series).cumprod()
        dd_metrics = RiskMetrics.maximum_drawdown(cumulative_returns)
        
        sharpe = RiskMetrics.sharpe_ratio(returns_series)
        sortino = RiskMetrics.sortino_ratio(returns_series)
        
        # Portfolio composition
        total_positions = sum(abs(pos) for pos in self.positions.values())
        position_weights = {
            symbol: abs(position) / total_positions 
            for symbol, position in self.positions.items()
        } if total_positions > 0 else {}
        
        return {
            'portfolio_value': self.portfolio_value,
            'num_positions': len(self.positions),
            'total_exposure': total_positions,
            'position_weights': position_weights,
            'risk_metrics': {
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'max_drawdown': dd_metrics['max_drawdown'],
                'sharpe_ratio': sharpe,
                'sortino_ratio': sortino,
                'volatility': returns_series.std() * np.sqrt(252)
            },
            'risk_limits': {
                'max_portfolio_risk': self.max_portfolio_risk,
                'max_position_size': self.max_position_size,
                'max_sector_exposure': self.max_sector_exposure,
                'current_risk_usage': self.calculate_portfolio_risk() / self.max_portfolio_risk
            }
        }
    
    def get_symbol_volatility(self, symbol: str, lookback_days: int = 30) -> float:
        """Get historical volatility for a symbol."""
        # This would normally fetch historical data
        # For now, return a default volatility
        return 0.25  # 25% annualized volatility
    
    def reset(self):
        """Reset risk manager state."""
        self.positions = {}
        self.sector_exposures = {}
        self.portfolio_value = 0
        self.daily_returns = []
        self.risk_metrics_cache = {}
