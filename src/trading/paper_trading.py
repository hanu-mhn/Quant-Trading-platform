"""
Paper Trading System

This module provides a comprehensive paper trading environment for testing
strategies without risking real capital. It simulates realistic market conditions
including slippage, latency, and commission costs.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
import asyncio
import json
import sqlite3
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from concurrent.futures import ThreadPoolExecutor
import time
import threading
from pathlib import Path

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from src.common.base_classes import BaseBroker, BaseStrategy
    from src.config.config import Config, BrokerConfig
    from src.portfolio.portfolio_manager import Portfolio, Position
    from src.risk_management.risk_manager import PortfolioRiskManager
    from src.data.loaders.data_loader import DataLoader
    from src.utils.logging_system import get_logging_manager
except ImportError:
    # Fallback for relative imports when run as module
    from ..common.base_classes import BaseBroker, BaseStrategy
    from ..config.config import Config, BrokerConfig
    from ..portfolio.portfolio_manager import Portfolio, Position
    from ..risk_management.risk_manager import PortfolioRiskManager
    from ..data.loaders.data_loader import DataLoader
    from ..utils.logging_system import get_logging_manager


class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


@dataclass
class PaperOrder:
    """Paper trading order structure"""
    order_id: str
    symbol: str
    quantity: int
    order_type: OrderType
    side: str  # 'BUY' or 'SELL'
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = 'DAY'
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    notes: str = ""


@dataclass
class MarketSimulation:
    """Market simulation parameters"""
    latency_ms: Tuple[int, int] = (50, 200)  # Min, max latency
    slippage_bps: Tuple[int, int] = (1, 5)   # Min, max slippage in basis points
    commission_per_share: float = 0.01       # Commission per share
    market_impact_threshold: int = 1000      # Shares threshold for market impact
    market_impact_bps: float = 2.0           # Additional slippage for large orders
    fill_probability: float = 0.95           # Probability of order fill
    partial_fill_probability: float = 0.1    # Probability of partial fill


class PaperBroker(BaseBroker):
    """
    Paper trading broker that simulates real trading conditions
    """
    
    def __init__(self, 
                 config: BrokerConfig,
                 initial_cash: float = 100000.0,
                 simulation_params: Optional[MarketSimulation] = None):
        """
        Initialize paper broker
        
        Args:
            config: Broker configuration
            initial_cash: Initial cash balance
            simulation_params: Market simulation parameters
        """
        super().__init__(config)
        
        self.initial_cash = initial_cash
        self.simulation = simulation_params or MarketSimulation()
        
        # Account state
        self.cash_balance = initial_cash
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, PaperOrder] = {}
        self.trade_history: List[Dict] = []
        
        # Market data
        self.data_loader = DataLoader()
        self.current_prices: Dict[str, float] = {}
        self.price_history: Dict[str, pd.DataFrame] = {}
        
        # Order processing
        self.order_queue = asyncio.Queue()
        self.is_processing = False
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Performance tracking
        self.start_time = datetime.now()
        self.total_trades = 0
        self.winning_trades = 0
        self.total_commission = 0.0
        self.total_slippage = 0.0
        
        # Database for persistence
        self.db_path = Path("data/paper_trading.db")
        self._init_database()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Paper broker initialized with ${initial_cash:,.2f}")
    
    def _init_database(self):
        """Initialize SQLite database for paper trading data"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Orders table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                symbol TEXT,
                quantity INTEGER,
                order_type TEXT,
                side TEXT,
                price REAL,
                stop_price REAL,
                time_in_force TEXT,
                submitted_at TEXT,
                filled_at TEXT,
                status TEXT,
                filled_quantity INTEGER,
                avg_fill_price REAL,
                commission REAL,
                slippage REAL,
                notes TEXT
            )
        ''')
        
        # Trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                order_id TEXT,
                symbol TEXT,
                quantity INTEGER,
                price REAL,
                side TEXT,
                commission REAL,
                slippage REAL,
                timestamp TEXT,
                pnl REAL
            )
        ''')
        
        # Portfolio snapshots
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                timestamp TEXT,
                cash_balance REAL,
                total_value REAL,
                pnl REAL,
                positions_json TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def connect(self) -> bool:
        """Connect to paper trading environment"""
        self.is_connected = True
        self.logger.info("Connected to paper trading environment")
        
        # Start order processing
        asyncio.create_task(self._process_orders())
        
        return True
    
    def disconnect(self) -> bool:
        """Disconnect from paper trading environment"""
        self.is_connected = False
        self.is_processing = False
        self.logger.info("Disconnected from paper trading environment")
        return True
    
    async def place_order(self, 
                         symbol: str,
                         quantity: int,
                         order_type: OrderType,
                         side: str,
                         price: Optional[float] = None,
                         stop_price: Optional[float] = None) -> str:
        """
        Place a paper trading order
        
        Args:
            symbol: Trading symbol
            quantity: Number of shares
            order_type: Order type
            side: 'BUY' or 'SELL'
            price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            
        Returns:
            Order ID
        """
        # Generate unique order ID
        order_id = str(uuid.uuid4())
        
        # Create order
        order = PaperOrder(
            order_id=order_id,
            symbol=symbol,
            quantity=quantity,
            order_type=order_type,
            side=side,
            price=price,
            stop_price=stop_price,
            submitted_at=datetime.now(),
            status=OrderStatus.SUBMITTED
        )
        
        # Validate order
        if not self._validate_order(order):
            order.status = OrderStatus.REJECTED
            order.notes = "Order validation failed"
            self.orders[order_id] = order
            return order_id
        
        # Add to orders and queue for processing
        self.orders[order_id] = order
        await self.order_queue.put(order_id)
        
        self.logger.info(f"Order submitted: {order_id} - {side} {quantity} {symbol}")
        
        # Save to database
        self._save_order_to_db(order)
        
        return order_id
    
    def _validate_order(self, order: PaperOrder) -> bool:
        """Validate order parameters"""
        # Check for sufficient buying power
        if order.side == 'BUY':
            current_price = self._get_current_price(order.symbol)
            if current_price is None:
                return False
            
            estimated_cost = order.quantity * current_price
            if estimated_cost > self.cash_balance:
                order.notes = "Insufficient buying power"
                return False
        
        # Check for sufficient shares to sell
        elif order.side == 'SELL':
            if order.symbol not in self.positions:
                order.notes = "No position to sell"
                return False
            
            position = self.positions[order.symbol]
            if position.quantity < order.quantity:
                order.notes = "Insufficient shares to sell"
                return False
        
        return True
    
    async def _process_orders(self):
        """Process orders asynchronously"""
        self.is_processing = True
        
        while self.is_processing:
            try:
                # Get order from queue with timeout
                order_id = await asyncio.wait_for(
                    self.order_queue.get(), 
                    timeout=1.0
                )
                
                # Process the order
                await self._execute_order(order_id)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error processing orders: {e}")
    
    async def _execute_order(self, order_id: str):
        """Execute a single order with realistic simulation"""
        order = self.orders[order_id]
        
        # Simulate network latency
        latency = np.random.randint(*self.simulation.latency_ms)
        await asyncio.sleep(latency / 1000.0)
        
        # Get current market price
        current_price = self._get_current_price(order.symbol)
        if current_price is None:
            order.status = OrderStatus.REJECTED
            order.notes = "Unable to get market price"
            return
        
        # Determine execution price based on order type
        execution_price = self._calculate_execution_price(order, current_price)
        if execution_price is None:
            # Order not fillable at current price
            return
        
        # Simulate fill probability
        if np.random.random() > self.simulation.fill_probability:
            order.status = OrderStatus.REJECTED
            order.notes = "Order not filled - market conditions"
            return
        
        # Calculate slippage and commission
        slippage = self._calculate_slippage(order, execution_price)
        commission = self._calculate_commission(order)
        
        # Apply slippage to execution price
        if order.side == 'BUY':
            execution_price += slippage
        else:
            execution_price -= slippage
        
        # Determine fill quantity (partial fills possible)
        fill_quantity = order.quantity
        if np.random.random() < self.simulation.partial_fill_probability:
            fill_quantity = int(order.quantity * np.random.uniform(0.5, 0.9))
        
        # Execute the trade
        self._execute_trade(order, fill_quantity, execution_price, commission, slippage)
        
        self.logger.info(
            f"Order executed: {order_id} - {fill_quantity}/{order.quantity} "
            f"@ ${execution_price:.2f} (slippage: ${slippage:.3f})"
        )
    
    def _calculate_execution_price(self, order: PaperOrder, current_price: float) -> Optional[float]:
        """Calculate execution price based on order type"""
        if order.order_type == OrderType.MARKET:
            return current_price
        
        elif order.order_type == OrderType.LIMIT:
            if order.side == 'BUY' and order.price >= current_price:
                return min(order.price, current_price)
            elif order.side == 'SELL' and order.price <= current_price:
                return max(order.price, current_price)
            else:
                return None  # Order not fillable
        
        elif order.order_type == OrderType.STOP:
            if order.side == 'BUY' and current_price >= order.stop_price:
                return current_price
            elif order.side == 'SELL' and current_price <= order.stop_price:
                return current_price
            else:
                return None
        
        return current_price
    
    def _calculate_slippage(self, order: PaperOrder, price: float) -> float:
        """Calculate realistic slippage"""
        base_slippage_bps = np.random.uniform(*self.simulation.slippage_bps)
        
        # Add market impact for large orders
        if order.quantity > self.simulation.market_impact_threshold:
            impact_factor = order.quantity / self.simulation.market_impact_threshold
            base_slippage_bps += self.simulation.market_impact_bps * impact_factor
        
        return price * (base_slippage_bps / 10000.0)
    
    def _calculate_commission(self, order: PaperOrder) -> float:
        """Calculate commission costs"""
        return order.quantity * self.simulation.commission_per_share
    
    def _execute_trade(self, order: PaperOrder, quantity: int, price: float, 
                      commission: float, slippage: float):
        """Execute the actual trade and update portfolio"""
        
        # Update order status
        order.filled_quantity += quantity
        order.avg_fill_price = ((order.avg_fill_price * (order.filled_quantity - quantity)) + 
                               (price * quantity)) / order.filled_quantity
        order.commission += commission
        order.slippage += slippage
        order.filled_at = datetime.now()
        
        if order.filled_quantity == order.quantity:
            order.status = OrderStatus.FILLED
        else:
            order.status = OrderStatus.PARTIALLY_FILLED
        
        # Update portfolio
        if order.side == 'BUY':
            self._add_position(order.symbol, quantity, price)
            self.cash_balance -= (quantity * price + commission)
        else:
            self._reduce_position(order.symbol, quantity, price)
            self.cash_balance += (quantity * price - commission)
        
        # Record trade
        trade_record = {
            'trade_id': str(uuid.uuid4()),
            'order_id': order.order_id,
            'symbol': order.symbol,
            'quantity': quantity,
            'price': price,
            'side': order.side,
            'commission': commission,
            'slippage': slippage,
            'timestamp': datetime.now(),
            'pnl': self._calculate_trade_pnl(order.symbol, quantity, price, order.side)
        }
        
        self.trade_history.append(trade_record)
        self._save_trade_to_db(trade_record)
        
        # Update statistics
        self.total_trades += 1
        self.total_commission += commission
        self.total_slippage += slippage
        
        if trade_record['pnl'] > 0:
            self.winning_trades += 1
    
    def _add_position(self, symbol: str, quantity: int, price: float):
        """Add to existing position or create new one"""
        if symbol in self.positions:
            position = self.positions[symbol]
            total_cost = (position.quantity * position.entry_price) + (quantity * price)
            total_quantity = position.quantity + quantity
            position.entry_price = total_cost / total_quantity
            position.quantity = total_quantity
        else:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                entry_price=price,
                entry_time=datetime.now()
            )
    
    def _reduce_position(self, symbol: str, quantity: int, price: float):
        """Reduce existing position"""
        if symbol in self.positions:
            position = self.positions[symbol]
            position.quantity -= quantity
            
            if position.quantity <= 0:
                del self.positions[symbol]
    
    def _calculate_trade_pnl(self, symbol: str, quantity: int, price: float, side: str) -> float:
        """Calculate P&L for a trade"""
        if side == 'SELL' and symbol in self.positions:
            position = self.positions[symbol]
            return quantity * (price - position.entry_price)
        return 0.0
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for symbol"""
        # In a real implementation, this would fetch from live data
        # For paper trading, we can use cached prices or simulate
        if symbol in self.current_prices:
            # Add some random price movement
            base_price = self.current_prices[symbol]
            change_pct = np.random.normal(0, 0.01)  # 1% daily volatility
            return base_price * (1 + change_pct)
        
        # If no cached price, try to get from data loader
        try:
            data = self.data_loader.load_symbol_data(symbol, period='1d')
            if data is not None and not data.empty:
                latest_price = data['close'].iloc[-1]
                self.current_prices[symbol] = latest_price
                return latest_price
        except:
            pass
        
        return None
    
    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        total_value = self.cash_balance
        
        for symbol, position in self.positions.items():
            current_price = self._get_current_price(symbol)
            if current_price:
                total_value += position.quantity * current_price
        
        return total_value
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary"""
        portfolio_value = self.get_portfolio_value()
        total_pnl = portfolio_value - self.initial_cash
        
        return {
            'initial_cash': self.initial_cash,
            'current_cash': self.cash_balance,
            'portfolio_value': portfolio_value,
            'total_pnl': total_pnl,
            'total_pnl_pct': (total_pnl / self.initial_cash) * 100,
            'positions_count': len(self.positions),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': (self.winning_trades / self.total_trades) if self.total_trades > 0 else 0,
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage,
            'trading_days': (datetime.now() - self.start_time).days,
            'positions': {symbol: asdict(pos) for symbol, pos in self.positions.items()}
        }
    
    def _save_order_to_db(self, order: PaperOrder):
        """Save order to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            order.order_id, order.symbol, order.quantity, order.order_type.value,
            order.side, order.price, order.stop_price, order.time_in_force,
            order.submitted_at.isoformat() if order.submitted_at else None,
            order.filled_at.isoformat() if order.filled_at else None,
            order.status.value, order.filled_quantity, order.avg_fill_price,
            order.commission, order.slippage, order.notes
        ))
        
        conn.commit()
        conn.close()
    
    def _save_trade_to_db(self, trade: Dict):
        """Save trade to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO trades VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade['trade_id'], trade['order_id'], trade['symbol'],
            trade['quantity'], trade['price'], trade['side'],
            trade['commission'], trade['slippage'],
            trade['timestamp'].isoformat(), trade['pnl']
        ))
        
        conn.commit()
        conn.close()


class PaperTradingManager:
    """Manager for paper trading sessions"""
    
    def __init__(self, config: Config):
        self.config = config
        self.active_sessions: Dict[str, PaperBroker] = {}
        self.logger = logging.getLogger(__name__)
    
    def create_session(self, 
                      session_id: str,
                      initial_cash: float = 100000.0,
                      simulation_params: Optional[MarketSimulation] = None) -> PaperBroker:
        """Create a new paper trading session"""
        
        broker_config = BrokerConfig(
            name="paper_broker",
            commission=0.01,
            slippage=0.001
        )
        
        broker = PaperBroker(
            config=broker_config,
            initial_cash=initial_cash,
            simulation_params=simulation_params
        )
        
        broker.connect()
        self.active_sessions[session_id] = broker
        
        self.logger.info(f"Created paper trading session: {session_id}")
        return broker
    
    def get_session(self, session_id: str) -> Optional[PaperBroker]:
        """Get an existing session"""
        return self.active_sessions.get(session_id)
    
    def close_session(self, session_id: str):
        """Close a paper trading session"""
        if session_id in self.active_sessions:
            broker = self.active_sessions[session_id]
            broker.disconnect()
            del self.active_sessions[session_id]
            
            self.logger.info(f"Closed paper trading session: {session_id}")


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Create paper trading manager
        config = Config()
        manager = PaperTradingManager(config)
        
        # Create a session
        session = manager.create_session("test_session", initial_cash=50000.0)
        
        # Place some test orders
        order1 = await session.place_order("AAPL", 100, OrderType.MARKET, "BUY")
        print(f"Order 1 placed: {order1}")
        
        # Wait a bit for processing
        await asyncio.sleep(2)
        
        order2 = await session.place_order("AAPL", 50, OrderType.LIMIT, "SELL", price=150.0)
        print(f"Order 2 placed: {order2}")
        
        # Wait and check status
        await asyncio.sleep(2)
        
        # Get portfolio summary
        summary = session.get_portfolio_summary()
        print(f"Portfolio Summary: {summary}")
        
        # Close session
        manager.close_session("test_session")
    
    # Run the example
    asyncio.run(main())