"""
Live trading engine for real-time strategy execution.
"""

import threading
import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from queue import Queue, Empty
import json
from pathlib import Path

from ..common.base_classes import (
    BaseStrategy, BaseBroker, Order, Position, OrderType, OrderSide, OrderStatus
)
from ..config.config import get_config


class LiveTradingEngine:
    """
    Main live trading engine that executes strategies in real-time.
    """
    
    def __init__(
        self,
        strategy: BaseStrategy,
        broker: BaseBroker,
        risk_manager: Optional[Any] = None,
        paper_trading: bool = True
    ):
        """Initialize the live trading engine."""
        self.strategy = strategy
        self.broker = broker
        self.risk_manager = risk_manager
        self.paper_trading = paper_trading
        
        # Load configuration
        self.config = get_config()
        
        # Initialize state
        self.is_running = False
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.order_queue = Queue()
        self.market_data_queue = Queue()
        
        # Performance tracking
        self.trades_today = 0
        self.daily_pnl = 0.0
        self.session_start_time = None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Session tracking
        self.session_id = self._generate_session_id()
        self.session_log = []
    
    def start_trading(self, symbols: List[str]):
        """Start the live trading session."""
        if self.is_running:
            self.logger.warning("Trading session is already running")
            return
        
        self.logger.info(f"Starting live trading session {self.session_id}")
        
        # Connect to broker
        if not self.broker.connect():
            raise RuntimeError("Failed to connect to broker")
        
        # Initialize session
        self.is_running = True
        self.session_start_time = datetime.now()
        self.symbols = symbols
        
        self.logger.info("Live trading session started successfully")
    
    def stop_trading(self):
        """Stop the live trading session."""
        if not self.is_running:
            return
        
        self.logger.info("Stopping live trading session")
        self.is_running = False
        
        # Disconnect from broker
        self.broker.disconnect()
        
        self.logger.info("Live trading session stopped")
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        return f"SESSION_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get current portfolio summary."""
        return {
            'positions': {k: v.to_dict() for k, v in self.positions.items()},
            'trades_today': self.trades_today,
            'session_id': self.session_id,
            'is_running': self.is_running
        }


# Legacy compatibility
class LiveTrader:
    def __init__(self, broker):
        self.broker = broker
        self.is_trading = False

    def start_trading(self):
        if not self.is_trading:
            self.broker.connect()
            self.is_trading = True
            print("Live trading session started.")

    def stop_trading(self):
        if self.is_trading:
            self.is_trading = False
            print("Live trading session stopped.")