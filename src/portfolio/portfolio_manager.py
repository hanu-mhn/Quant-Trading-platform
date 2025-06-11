"""
Portfolio Management Module.

This module provides comprehensive portfolio management capabilities including
position tracking, performance calculation, and portfolio optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import logging

from ..common.base_classes import BasePortfolioManager, Order, OrderStatus
from ..config.config import Config


class Position:
    """Represents a single position in the portfolio."""
    
    def __init__(
        self,
        symbol: str,
        quantity: float,
        avg_price: float,
        timestamp: datetime = None
    ):
        self.symbol = symbol
        self.quantity = quantity
        self.avg_price = avg_price
        self.timestamp = timestamp or datetime.now()
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
    
    def update_price(self, current_price: float) -> None:
        """Update unrealized P&L based on current price."""
        self.unrealized_pnl = (current_price - self.avg_price) * self.quantity
    
    def add_quantity(self, quantity: float, price: float) -> None:
        """Add quantity to position (average down/up)."""
        total_cost = (self.quantity * self.avg_price) + (quantity * price)
        self.quantity += quantity
        if self.quantity != 0:
            self.avg_price = total_cost / self.quantity
    
    def reduce_quantity(self, quantity: float, price: float) -> float:
        """Reduce position quantity and return realized P&L."""
        if quantity > abs(self.quantity):
            quantity = abs(self.quantity)
        
        realized_pnl = (price - self.avg_price) * quantity * (1 if self.quantity > 0 else -1)
        self.realized_pnl += realized_pnl
        
        if self.quantity > 0:
            self.quantity = max(0, self.quantity - quantity)
        else:
            self.quantity = min(0, self.quantity + quantity)
        
        return realized_pnl
    
    def get_market_value(self, current_price: float) -> float:
        """Get current market value of position."""
        return self.quantity * current_price
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary."""
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'avg_price': self.avg_price,
            'timestamp': self.timestamp,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl
        }


class Portfolio:
    """Portfolio data structure and calculations."""
    
    def __init__(self, initial_cash: float = 100000.0):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Dict[str, Any]] = []
        self.daily_values: List[Dict[str, Any]] = []
        self.created_at = datetime.now()
        
    def add_trade(
        self,
        symbol: str,
        quantity: float,
        price: float,
        commission: float = 0.0,
        timestamp: datetime = None
    ) -> None:
        """Add a trade to the portfolio."""
        timestamp = timestamp or datetime.now()
        trade_value = quantity * price
        
        # Update cash
        self.cash -= (trade_value + commission)
        
        # Update or create position
        if symbol in self.positions:
            if (self.positions[symbol].quantity > 0 and quantity > 0) or \
               (self.positions[symbol].quantity < 0 and quantity < 0):
                # Adding to existing position
                self.positions[symbol].add_quantity(quantity, price)
            else:
                # Closing or reversing position
                realized_pnl = self.positions[symbol].reduce_quantity(abs(quantity), price)
                if self.positions[symbol].quantity == 0:
                    del self.positions[symbol]
                
                # If there's remaining quantity, create new position
                remaining_qty = quantity - (abs(quantity) - abs(self.positions.get(symbol, Position('', 0, 0)).quantity))
                if remaining_qty != 0:
                    self.positions[symbol] = Position(symbol, remaining_qty, price, timestamp)
        else:
            # New position
            self.positions[symbol] = Position(symbol, quantity, price, timestamp)
        
        # Record trade
        trade_record = {
            'timestamp': timestamp,
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'commission': commission,
            'trade_value': trade_value
        }
        self.trade_history.append(trade_record)
    
    def get_total_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value."""
        market_value = sum(
            pos.get_market_value(current_prices.get(symbol, pos.avg_price))
            for symbol, pos in self.positions.items()
        )
        return self.cash + market_value
    
    def get_positions_summary(self, current_prices: Dict[str, float]) -> pd.DataFrame:
        """Get summary of all positions."""
        if not self.positions:
            return pd.DataFrame()
        
        positions_data = []
        for symbol, position in self.positions.items():
            current_price = current_prices.get(symbol, position.avg_price)
            position.update_price(current_price)
            
            pos_data = position.to_dict()
            pos_data.update({
                'current_price': current_price,
                'market_value': position.get_market_value(current_price),
                'total_pnl': position.realized_pnl + position.unrealized_pnl
            })
            positions_data.append(pos_data)
        
        return pd.DataFrame(positions_data)
    
    def get_performance_metrics(self, current_prices: Dict[str, float]) -> Dict[str, Any]:
        """Calculate portfolio performance metrics."""
        if not self.daily_values:
            return {}
        
        values_df = pd.DataFrame(self.daily_values)
        if 'total_value' not in values_df.columns:
            return {}
        
        returns = values_df['total_value'].pct_change().dropna()
        
        if len(returns) < 2:
            return {}
        
        current_value = self.get_total_value(current_prices)
        total_return = (current_value - self.initial_cash) / self.initial_cash
        
        metrics = {
            'total_return': total_return,
            'total_value': current_value,
            'initial_cash': self.initial_cash,
            'current_cash': self.cash,
            'num_positions': len(self.positions),
            'num_trades': len(self.trade_history)
        }
        
        if len(returns) > 0:
            metrics.update({
                'avg_daily_return': returns.mean(),
                'daily_volatility': returns.std(),
                'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
                'max_drawdown': self._calculate_max_drawdown(values_df['total_value']),
                'win_rate': len(returns[returns > 0]) / len(returns)
            })
        
        return metrics
    
    def _calculate_max_drawdown(self, values: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if len(values) < 2:
            return 0
        
        cumulative = values / values.iloc[0]
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        
        return abs(drawdown.min())
    
    def record_daily_value(self, current_prices: Dict[str, float], date: datetime = None) -> None:
        """Record daily portfolio value for performance tracking."""
        date = date or datetime.now().date()
        total_value = self.get_total_value(current_prices)
        
        self.daily_values.append({
            'date': date,
            'total_value': total_value,
            'cash': self.cash,
            'market_value': total_value - self.cash
        })


class PortfolioManager(BasePortfolioManager):
    """Advanced portfolio management with optimization and rebalancing."""
    
    def __init__(self, initial_cash: float = 100000.0, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.config_manager = Config()
        self.config = config or {}
        
        self.portfolio = Portfolio(initial_cash)
        self.current_prices: Dict[str, float] = {}
        self.target_weights: Dict[str, float] = {}
        self.rebalance_threshold = 0.05  # 5% deviation threshold
        
        self.logger = logging.getLogger(__name__)
    
    def execute_order(self, order: Order, fill_price: float, commission: float = 0.0) -> bool:
        """Execute an order and update portfolio."""
        try:
            # Determine quantity based on order side
            quantity = order.quantity if order.side.value == 'BUY' else -order.quantity
            
            # Check if we have enough cash for buy orders
            if order.side.value == 'BUY':
                required_cash = order.quantity * fill_price + commission
                if required_cash > self.portfolio.cash:
                    self.logger.warning(f"Insufficient cash for order. Required: {required_cash}, Available: {self.portfolio.cash}")
                    return False
            
            # Execute trade
            self.portfolio.add_trade(order.symbol, quantity, fill_price, commission)
            
            self.logger.info(f"Executed {order.side.value} order: {order.quantity} shares of {order.symbol} at {fill_price}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error executing order: {e}")
            return False
    
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get position information for a symbol."""
        if symbol not in self.portfolio.positions:
            return None
        
        position = self.portfolio.positions[symbol]
        current_price = self.current_prices.get(symbol, position.avg_price)
        position.update_price(current_price)
        
        return position.to_dict()
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary."""
        positions_df = self.portfolio.get_positions_summary(self.current_prices)
        performance_metrics = self.portfolio.get_performance_metrics(self.current_prices)
        
        return {
            'portfolio_value': self.portfolio.get_total_value(self.current_prices),
            'cash': self.portfolio.cash,
            'positions': positions_df.to_dict('records') if not positions_df.empty else [],
            'performance_metrics': performance_metrics,
            'num_positions': len(self.portfolio.positions),
            'num_trades': len(self.portfolio.trade_history)
        }
    
    def update_prices(self, prices: Dict[str, float]) -> None:
        """Update current market prices."""
        self.current_prices.update(prices)
        
        # Record daily value if it's a new day
        today = datetime.now().date()
        if not self.portfolio.daily_values or self.portfolio.daily_values[-1]['date'] != today:
            self.portfolio.record_daily_value(self.current_prices, today)
    
    def set_target_weights(self, weights: Dict[str, float]) -> None:
        """Set target portfolio weights for rebalancing."""
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            self.target_weights = {symbol: weight / total_weight for symbol, weight in weights.items()}
        else:
            self.target_weights = {}
    
    def check_rebalancing_needed(self) -> bool:
        """Check if portfolio needs rebalancing based on target weights."""
        if not self.target_weights:
            return False
        
        total_value = self.portfolio.get_total_value(self.current_prices)
        if total_value <= 0:
            return False
        
        # Calculate current weights
        current_weights = {}
        for symbol, position in self.portfolio.positions.items():
            current_price = self.current_prices.get(symbol, position.avg_price)
            market_value = position.get_market_value(current_price)
            current_weights[symbol] = market_value / total_value
        
        # Check deviations
        for symbol, target_weight in self.target_weights.items():
            current_weight = current_weights.get(symbol, 0)
            deviation = abs(current_weight - target_weight)
            
            if deviation > self.rebalance_threshold:
                return True
        
        return False
    
    def generate_rebalancing_orders(self) -> List[Order]:
        """Generate orders to rebalance portfolio to target weights."""
        if not self.target_weights:
            return []
        
        orders = []
        total_value = self.portfolio.get_total_value(self.current_prices)
        
        if total_value <= 0:
            return orders
        
        # Calculate target positions
        for symbol, target_weight in self.target_weights.items():
            target_value = total_value * target_weight
            current_price = self.current_prices.get(symbol)
            
            if not current_price:
                continue
            
            target_quantity = target_value / current_price
            current_quantity = self.portfolio.positions.get(symbol, Position('', 0, 0)).quantity
            
            quantity_diff = target_quantity - current_quantity
            
            # Only create order if difference is significant
            if abs(quantity_diff) > 0.01:  # Minimum 0.01 shares
                side = OrderSide.BUY if quantity_diff > 0 else OrderSide.SELL
                
                order = Order(
                    symbol=symbol,
                    side=side,
                    order_type='MARKET',
                    quantity=abs(quantity_diff)
                )
                orders.append(order)
        
        return orders
    
    def calculate_portfolio_metrics(self) -> Dict[str, Any]:
        """Calculate advanced portfolio metrics."""
        return self.portfolio.get_performance_metrics(self.current_prices)
    
    def reset(self) -> None:
        """Reset portfolio manager state."""
        self.portfolio = Portfolio(self.portfolio.initial_cash)
        self.current_prices = {}
        self.target_weights = {}
    
    def calculate_portfolio_value(self, positions: List[Position], current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value."""
        total_value = self.portfolio.cash
        
        for position in positions:
            current_price = current_prices.get(position.symbol, position.avg_price)
            total_value += position.get_market_value(current_price)
        
        return total_value
    
    def calculate_allocation(self, positions: List[Position]) -> Dict[str, float]:
        """Calculate portfolio allocation by symbol."""
        total_value = self.portfolio.get_total_value(self.current_prices)
        if total_value <= 0:
            return {}
        
        allocation = {}
        for position in positions:
            current_price = self.current_prices.get(position.symbol, position.avg_price)
            market_value = position.get_market_value(current_price)
            allocation[position.symbol] = market_value / total_value
        
        return allocation
    
    def rebalance_portfolio(self, target_allocation: Dict[str, float]) -> List[Order]:
        """Generate orders to rebalance portfolio."""
        self.set_target_weights(target_allocation)
        return self.generate_rebalancing_orders()
