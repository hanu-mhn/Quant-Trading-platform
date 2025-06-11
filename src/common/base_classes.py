"""
Base classes and interfaces for the trading platform.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import pandas as pd
from datetime import datetime
from enum import Enum


class OrderType(Enum):
    """Order types for trading."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderSide(Enum):
    """Order sides for trading."""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """Order status for tracking."""
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class Order:
    """Represents a trading order."""
    
    def __init__(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        order_id: Optional[str] = None
    ):
        self.symbol = symbol
        self.side = side
        self.order_type = order_type
        self.quantity = quantity
        self.price = price
        self.stop_price = stop_price
        self.order_id = order_id or self._generate_order_id()
        self.status = OrderStatus.PENDING
        self.filled_quantity = 0.0
        self.filled_price = 0.0
        self.timestamp = datetime.now()
    
    def _generate_order_id(self) -> str:
        """Generate a unique order ID."""
        return f"ORD_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    
    def to_dict(self) -> Dict:
        """Convert order to dictionary."""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'quantity': self.quantity,
            'price': self.price,
            'stop_price': self.stop_price,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'filled_price': self.filled_price,
            'timestamp': self.timestamp
        }


class Position:
    """Represents a trading position."""
    
    def __init__(self, symbol: str, quantity: float = 0.0, avg_price: float = 0.0):
        self.symbol = symbol
        self.quantity = quantity
        self.avg_price = avg_price
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
    
    def update_position(self, trade_quantity: float, trade_price: float):
        """Update position with a new trade."""
        if self.quantity == 0:
            # New position
            self.quantity = trade_quantity
            self.avg_price = trade_price
        elif (self.quantity > 0 and trade_quantity > 0) or (self.quantity < 0 and trade_quantity < 0):
            # Adding to existing position
            total_cost = (self.quantity * self.avg_price) + (trade_quantity * trade_price)
            self.quantity += trade_quantity
            self.avg_price = total_cost / self.quantity if self.quantity != 0 else 0
        else:
            # Reducing or closing position
            if abs(trade_quantity) >= abs(self.quantity):
                # Closing position completely or reversing
                self.realized_pnl += (trade_price - self.avg_price) * self.quantity
                remaining_quantity = trade_quantity + self.quantity
                self.quantity = remaining_quantity
                self.avg_price = trade_price if remaining_quantity != 0 else 0
            else:
                # Partial close
                self.realized_pnl += (trade_price - self.avg_price) * abs(trade_quantity)
                self.quantity += trade_quantity
    
    def calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L."""
        if self.quantity == 0:
            return 0.0
        self.unrealized_pnl = (current_price - self.avg_price) * self.quantity
        return self.unrealized_pnl
    
    def to_dict(self) -> Dict:
        """Convert position to dictionary."""
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'avg_price': self.avg_price,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl
        }


class BaseDataSource(ABC):
    """Abstract base class for data sources."""
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to the data source."""
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """Disconnect from the data source."""
        pass
    
    @abstractmethod
    def get_historical_data(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime,
        timeframe: str = '1D'
    ) -> pd.DataFrame:
        """Get historical market data."""
        pass
    
    @abstractmethod
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        pass


class BaseBroker(ABC):
    """Abstract base class for broker implementations."""
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to the broker."""
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """Disconnect from the broker."""
        pass
    
    @abstractmethod
    def place_order(self, order: Order) -> str:
        """Place an order and return order ID."""
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status."""
        pass
    
    @abstractmethod
    def get_positions(self) -> List[Position]:
        """Get current positions."""
        pass
    
    @abstractmethod
    def get_account_balance(self) -> float:
        """Get account balance."""
        pass


class BaseStrategy(ABC):
    """Abstract base class for trading strategies."""
    
    def __init__(self, name: str):
        self.name = name
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.performance_metrics: Dict[str, Any] = {}
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate trading signals based on market data."""
        pass
    
    @abstractmethod
    def calculate_position_size(self, signal: Dict[str, Any], portfolio_value: float) -> float:
        """Calculate position size for a signal."""
        pass
    
    def add_order(self, order: Order):
        """Add an order to the strategy."""
        self.orders.append(order)
    
    def update_position(self, symbol: str, quantity: float, price: float):
        """Update position for a symbol."""
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol)
        self.positions[symbol].update_position(quantity, price)
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        return self.positions.get(symbol)


class BaseIndicator(ABC):
    """Abstract base class for technical indicators."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate the indicator values."""
        pass
    
    def validate_data(self, data: pd.DataFrame, required_columns: List[str] = None) -> bool:
        """Validate input data."""
        if required_columns is None:
            required_columns = ['close']
        
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        if data.empty:
            raise ValueError("Data is empty")
        
        return True


class BaseRiskManager(ABC):
    """Abstract base class for risk management."""
    
    @abstractmethod
    def check_risk_limits(self, order: Order, portfolio: Dict) -> bool:
        """Check if order violates risk limits."""
        pass
    
    @abstractmethod
    def calculate_position_risk(self, position: Position, current_price: float) -> Dict[str, float]:
        """Calculate risk metrics for a position."""
        pass
    
    @abstractmethod
    def suggest_stop_loss(self, entry_price: float, side: OrderSide) -> float:
        """Suggest stop loss price."""
        pass


class BasePortfolioManager(ABC):
    """Abstract base class for portfolio management."""
    
    @abstractmethod
    def calculate_portfolio_value(self, positions: List[Position], current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value."""
        pass
    
    @abstractmethod
    def calculate_allocation(self, positions: List[Position]) -> Dict[str, float]:
        """Calculate portfolio allocation by symbol."""
        pass
    
    @abstractmethod
    def rebalance_portfolio(self, target_allocation: Dict[str, float]) -> List[Order]:
        """Generate orders to rebalance portfolio."""
        pass


# Common utility functions
def validate_symbol(symbol: str) -> bool:
    """Validate trading symbol format."""
    if not symbol or not isinstance(symbol, str):
        return False
    return len(symbol.strip()) > 0


def calculate_returns(prices: pd.Series) -> pd.Series:
    """Calculate returns from price series."""
    return prices.pct_change().dropna()


def calculate_volatility(returns: pd.Series, annualize: bool = True) -> float:
    """Calculate volatility from returns."""
    vol = returns.std()
    if annualize:
        vol *= (252 ** 0.5)  # Assuming 252 trading days per year
    return vol


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculate Sharpe ratio."""
    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
    if excess_returns.std() == 0:
        return 0.0
    return excess_returns.mean() / excess_returns.std() * (252 ** 0.5)


def calculate_max_drawdown(prices: pd.Series) -> float:
    """Calculate maximum drawdown."""
    cumulative = (1 + calculate_returns(prices)).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()
