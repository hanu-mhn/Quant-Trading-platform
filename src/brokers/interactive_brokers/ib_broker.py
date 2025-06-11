"""
Interactive Brokers (IB) Integration

This module provides integration with Interactive Brokers API for live trading,
market data, and portfolio management.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
import logging
import time
import threading
from dataclasses import dataclass
from enum import Enum
import queue

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

try:
    from src.common.base_classes import BaseBroker
    from src.config.config import BrokerConfig
except ImportError:
    # Fallback for relative imports when run as module
    from ...common.base_classes import BaseBroker
    from ...config.config import BrokerConfig

# IB API imports (requires ibapi package)
try:
    from ibapi.client import EClient
    from ibapi.wrapper import EWrapper
    from ibapi.contract import Contract
    from ibapi.order import Order
    from ibapi.common import OrderId, TickerId
    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False
    print("Warning: Interactive Brokers API not available. Install with: pip install ibapi")


class IBOrderType(Enum):
    """IB Order types"""
    MARKET = "MKT"
    LIMIT = "LMT"
    STOP = "STP"
    STOP_LIMIT = "STP LMT"
    TRAIL = "TRAIL"
    TRAIL_LIMIT = "TRAIL LIMIT"


class IBTimeInForce(Enum):
    """IB Time in force"""
    DAY = "DAY"
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill


@dataclass
class IBOrderDetails:
    """IB Order details structure"""
    symbol: str
    exchange: str
    quantity: int
    order_type: IBOrderType
    action: str  # 'BUY' or 'SELL'
    price: Optional[float] = None
    aux_price: Optional[float] = None  # For stop orders
    time_in_force: IBTimeInForce = IBTimeInForce.DAY
    outside_rth: bool = False  # Outside regular trading hours
    hidden: bool = False
    block_order: bool = False
    sweep_to_fill: bool = False
    display_size: int = 0
    trigger_method: int = 0
    good_after_time: Optional[str] = None
    good_till_date: Optional[str] = None
    oca_group: Optional[str] = None  # One-Cancels-All group
    oca_type: int = 0


class IBWrapper(EWrapper):
    """IB API Wrapper to handle callbacks"""
    
    def __init__(self, broker_instance):
        EWrapper.__init__(self)
        self.broker = broker_instance
        self.logger = logging.getLogger(__name__)
    
    def error(self, reqId: TickerId, errorCode: int, errorString: str):
        """Handle error messages"""
        self.logger.error(f"IB Error {errorCode}: {errorString} (ReqId: {reqId})")
        self.broker.last_error = {'code': errorCode, 'message': errorString, 'reqId': reqId}
    
    def connectAck(self):
        """Connection acknowledgment"""
        self.logger.info("IB Connection acknowledged")
        self.broker.is_connected = True
    
    def nextValidId(self, orderId: OrderId):
        """Receive next valid order ID"""
        self.broker.next_order_id = orderId
        self.logger.info(f"Next valid order ID: {orderId}")
    
    def managedAccounts(self, accountsList: str):
        """Receive managed accounts"""
        accounts = accountsList.split(',')
        self.broker.managed_accounts = accounts
        self.logger.info(f"Managed accounts: {accounts}")
    
    def accountSummary(self, reqId: int, account: str, tag: str, value: str, currency: str):
        """Account summary data"""
        if reqId not in self.broker.account_data:
            self.broker.account_data[reqId] = {}
        
        self.broker.account_data[reqId][tag] = {
            'value': value,
            'currency': currency
        }
    
    def position(self, account: str, contract: Contract, position: float, avgCost: float):
        """Position data"""
        position_data = {
            'account': account,
            'symbol': contract.symbol,
            'exchange': contract.exchange,
            'currency': contract.currency,
            'position': position,
            'avgCost': avgCost,
            'marketValue': position * avgCost
        }
        
        self.broker.positions[f"{contract.symbol}_{contract.exchange}"] = position_data
    
    def orderStatus(self, orderId: OrderId, status: str, filled: float,
                   remaining: float, avgFillPrice: float, permId: int,
                   parentId: int, lastFillPrice: float, clientId: int, whyHeld: str):
        """Order status updates"""
        order_data = {
            'orderId': orderId,
            'status': status,
            'filled': filled,
            'remaining': remaining,
            'avgFillPrice': avgFillPrice,
            'lastFillPrice': lastFillPrice,
            'whyHeld': whyHeld
        }
        
        self.broker.order_status[orderId] = order_data
        self.logger.info(f"Order {orderId} status: {status}")
    
    def openOrder(self, orderId: OrderId, contract: Contract, order: Order, orderState):
        """Open order details"""
        order_data = {
            'orderId': orderId,
            'contract': contract,
            'order': order,
            'orderState': orderState
        }
        
        self.broker.open_orders[orderId] = order_data
    
    def execDetails(self, reqId: int, contract: Contract, execution):
        """Execution details"""
        exec_data = {
            'execId': execution.execId,
            'time': execution.time,
            'account': execution.acctNumber,
            'exchange': execution.exchange,
            'side': execution.side,
            'shares': execution.shares,
            'price': execution.price,
            'permId': execution.permId,
            'clientId': execution.clientId,
            'orderId': execution.orderId,
            'liquidation': execution.liquidation
        }
        
        self.broker.executions[execution.execId] = exec_data
    
    def tickPrice(self, reqId: TickerId, tickType, price: float, attrib):
        """Market data price ticks"""
        if reqId not in self.broker.market_data:
            self.broker.market_data[reqId] = {}
        
        tick_types = {
            1: 'bid',
            2: 'ask', 
            4: 'last',
            6: 'high',
            7: 'low',
            9: 'close'
        }
        
        if tickType in tick_types:
            self.broker.market_data[reqId][tick_types[tickType]] = price
    
    def tickSize(self, reqId: TickerId, tickType, size: int):
        """Market data size ticks"""
        if reqId not in self.broker.market_data:
            self.broker.market_data[reqId] = {}
        
        tick_types = {
            0: 'bid_size',
            3: 'ask_size',
            5: 'last_size',
            8: 'volume'
        }
        
        if tickType in tick_types:
            self.broker.market_data[reqId][tick_types[tickType]] = size
    
    def historicalData(self, reqId: int, bar):
        """Historical data bars"""
        if reqId not in self.broker.historical_data:
            self.broker.historical_data[reqId] = []
        
        bar_data = {
            'date': bar.date,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume,
            'count': bar.count,
            'wap': bar.wap
        }
        
        self.broker.historical_data[reqId].append(bar_data)


class IBBroker(BaseBroker):
    """
    Interactive Brokers API integration
    """
    
    def __init__(self, config: BrokerConfig, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1):
        """
        Initialize IB broker
        
        Args:
            config: Broker configuration
            host: TWS/Gateway host
            port: TWS/Gateway port (7497 for TWS, 4001 for Gateway)
            client_id: Unique client ID
        """
        if not IB_AVAILABLE:
            raise ImportError("Interactive Brokers API not available. Install with: pip install ibapi")
        
        super().__init__(config)
        
        self.host = host
        self.port = port
        self.client_id = client_id
        
        # Create wrapper and client
        self.wrapper = IBWrapper(self)
        self.client = EClient(self.wrapper)
        
        # Connection status
        self.is_connected = False
        self.connection_thread = None
        
        # Data storage
        self.next_order_id = None
        self.managed_accounts = []
        self.account_data = {}
        self.positions = {}
        self.order_status = {}
        self.open_orders = {}
        self.executions = {}
        self.market_data = {}
        self.historical_data = {}
        self.last_error = None
        
        # Request ID management
        self.next_req_id = 1000
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def connect(self) -> bool:
        """
        Connect to Interactive Brokers TWS/Gateway
        
        Returns:
            True if connection successful
        """
        try:
            self.client.connect(self.host, self.port, self.client_id)
            
            # Start connection thread
            self.connection_thread = threading.Thread(target=self.client.run, daemon=True)
            self.connection_thread.start()
            
            # Wait for connection acknowledgment
            timeout = 10
            start_time = time.time()
            
            while not self.is_connected and (time.time() - start_time) < timeout:
                time.sleep(0.1)
            
            if self.is_connected:
                # Request managed accounts
                self.client.reqManagedAccts()
                
                # Wait for account data
                time.sleep(1)
                
                self.logger.info(f"Connected to IB - Accounts: {self.managed_accounts}")
                return True
            else:
                self.logger.error("Failed to connect to IB within timeout")
                
        except Exception as e:
            self.logger.error(f"Error connecting to IB: {e}")
            
        return False
    
    def disconnect(self) -> bool:
        """
        Disconnect from Interactive Brokers
        
        Returns:
            True if disconnection successful
        """
        try:
            if self.client.isConnected():
                self.client.disconnect()
                
            if self.connection_thread and self.connection_thread.is_alive():
                self.connection_thread.join(timeout=5)
            
            self.is_connected = False
            self.logger.info("Disconnected from IB")
            return True
            
        except Exception as e:
            self.logger.error(f"Error disconnecting from IB: {e}")
            return False
    
    def _get_next_req_id(self) -> int:
        """Get next request ID"""
        self.next_req_id += 1
        return self.next_req_id
    
    def create_stock_contract(self, symbol: str, exchange: str = "SMART", currency: str = "USD") -> Contract:
        """
        Create stock contract
        
        Args:
            symbol: Stock symbol
            exchange: Exchange (SMART for best execution)
            currency: Currency
            
        Returns:
            Contract object
        """
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = exchange
        contract.currency = currency
        return contract
    
    def create_order(self, order_details: IBOrderDetails) -> Order:
        """
        Create order object
        
        Args:
            order_details: Order details
            
        Returns:
            Order object
        """
        order = Order()
        order.action = order_details.action
        order.totalQuantity = order_details.quantity
        order.orderType = order_details.order_type.value
        order.tif = order_details.time_in_force.value
        
        # Set price based on order type
        if order_details.order_type in [IBOrderType.LIMIT, IBOrderType.STOP_LIMIT]:
            order.lmtPrice = order_details.price or 0.0
            
        if order_details.order_type in [IBOrderType.STOP, IBOrderType.STOP_LIMIT]:
            order.auxPrice = order_details.aux_price or 0.0
        
        # Additional order parameters
        order.outsideRth = order_details.outside_rth
        order.hidden = order_details.hidden
        order.blockOrder = order_details.block_order
        order.sweepToFill = order_details.sweep_to_fill
        order.displaySize = order_details.display_size
        order.triggerMethod = order_details.trigger_method
        
        if order_details.good_after_time:
            order.goodAfterTime = order_details.good_after_time
            
        if order_details.good_till_date:
            order.goodTillDate = order_details.good_till_date
        
        if order_details.oca_group:
            order.ocaGroup = order_details.oca_group
            order.ocaType = order_details.oca_type
        
        return order
    
    def place_order(self, order_details: IBOrderDetails) -> Optional[int]:
        """
        Place a new order
        
        Args:
            order_details: Order details
            
        Returns:
            Order ID if successful
        """
        try:
            if not self.next_order_id:
                self.logger.error("No valid order ID available")
                return None
            
            contract = self.create_stock_contract(order_details.symbol, order_details.exchange)
            order = self.create_order(order_details)
            
            order_id = self.next_order_id
            self.client.placeOrder(order_id, contract, order)
            self.next_order_id += 1
            
            self.logger.info(f"Order placed: {order_id} - {order_details.action} {order_details.quantity} {order_details.symbol}")
            return order_id
            
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return None
    
    def cancel_order(self, order_id: int) -> bool:
        """
        Cancel existing order
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if successful
        """
        try:
            self.client.cancelOrder(order_id)
            self.logger.info(f"Order cancellation requested: {order_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    def get_account_summary(self, account: str = None) -> Dict[str, Any]:
        """
        Get account summary
        
        Args:
            account: Account ID (uses first managed account if None)
            
        Returns:
            Account summary data
        """
        if not account and self.managed_accounts:
            account = self.managed_accounts[0]
        
        if not account:
            return {}
        
        req_id = self._get_next_req_id()
        
        # Request account summary
        tags = "NetLiquidation,TotalCashValue,SettledCash,AccruedCash,BuyingPower,EquityWithLoanValue,GrossPositionValue"
        self.client.reqAccountSummary(req_id, "All", tags)
        
        # Wait for data
        timeout = 5
        start_time = time.time()
        
        while req_id not in self.account_data and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        
        self.client.cancelAccountSummary(req_id)
        
        return self.account_data.get(req_id, {})
    
    def get_positions(self, account: str = None) -> List[Dict]:
        """
        Get current positions
        
        Args:
            account: Account ID
            
        Returns:
            List of positions
        """
        # Clear existing positions
        self.positions.clear()
        
        # Request positions
        self.client.reqPositions()
        
        # Wait for data
        time.sleep(2)
        
        self.client.cancelPositions()
        
        # Filter by account if specified
        if account:
            return [pos for pos in self.positions.values() if pos['account'] == account]
        
        return list(self.positions.values())
    
    def get_open_orders(self) -> List[Dict]:
        """
        Get open orders
        
        Returns:
            List of open orders
        """
        # Clear existing orders
        self.open_orders.clear()
        
        # Request open orders
        self.client.reqOpenOrders()
        
        # Wait for data
        time.sleep(2)
        
        return list(self.open_orders.values())
    
    def get_executions(self, account: str = None) -> List[Dict]:
        """
        Get executions
        
        Args:
            account: Account ID filter
            
        Returns:
            List of executions
        """
        from ibapi.execution import ExecutionFilter
        
        # Clear existing executions
        self.executions.clear()
        
        # Create execution filter
        exec_filter = ExecutionFilter()
        if account:
            exec_filter.acctCode = account
        
        req_id = self._get_next_req_id()
        self.client.reqExecutions(req_id, exec_filter)
        
        # Wait for data
        time.sleep(2)
        
        return list(self.executions.values())
    
    def request_market_data(self, symbol: str, exchange: str = "SMART") -> int:
        """
        Request real-time market data
        
        Args:
            symbol: Stock symbol
            exchange: Exchange
            
        Returns:
            Request ID for the subscription
        """
        req_id = self._get_next_req_id()
        contract = self.create_stock_contract(symbol, exchange)
        
        # Request market data
        self.client.reqMktData(req_id, contract, "", False, False, [])
        
        self.logger.info(f"Market data requested for {symbol} (ReqId: {req_id})")
        return req_id
    
    def cancel_market_data(self, req_id: int):
        """
        Cancel market data subscription
        
        Args:
            req_id: Request ID to cancel
        """
        self.client.cancelMktData(req_id)
        if req_id in self.market_data:
            del self.market_data[req_id]
    
    def get_market_data(self, req_id: int) -> Optional[Dict]:
        """
        Get latest market data for a subscription
        
        Args:
            req_id: Request ID
            
        Returns:
            Market data dictionary
        """
        return self.market_data.get(req_id)
    
    def get_historical_data(self, symbol: str, duration: str = "1 Y", 
                          bar_size: str = "1 day", exchange: str = "SMART") -> pd.DataFrame:
        """
        Get historical data
        
        Args:
            symbol: Stock symbol
            duration: Duration (e.g., "1 Y", "6 M", "30 D")
            bar_size: Bar size (e.g., "1 day", "1 hour", "5 mins")
            exchange: Exchange
            
        Returns:
            Historical data DataFrame
        """
        req_id = self._get_next_req_id()
        contract = self.create_stock_contract(symbol, exchange)
        
        # Clear existing data
        if req_id in self.historical_data:
            del self.historical_data[req_id]
        
        # Request historical data
        end_datetime = datetime.now().strftime("%Y%m%d %H:%M:%S")
        self.client.reqHistoricalData(
            req_id, contract, end_datetime, duration, bar_size, 
            "TRADES", 1, 1, False, []
        )
        
        # Wait for data
        timeout = 30
        start_time = time.time()
        
        while req_id not in self.historical_data and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        
        if req_id in self.historical_data:
            # Convert to DataFrame
            data = self.historical_data[req_id]
            if data:
                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                return df
        
        return pd.DataFrame()
    
    def validate_connection(self) -> bool:
        """
        Validate broker connection
        
        Returns:
            True if connection is valid
        """
        return self.client.isConnected() and self.is_connected
    
    def get_account_balance(self, account: str = None) -> float:
        """
        Get available account balance
        
        Args:
            account: Account ID
            
        Returns:
            Available balance
        """
        summary = self.get_account_summary(account)
        if 'SettledCash' in summary:
            return float(summary['SettledCash']['value'])
        return 0.0
    
    def get_buying_power(self, account: str = None) -> float:
        """
        Get total buying power
        
        Args:
            account: Account ID
            
        Returns:
            Total buying power
        """
        summary = self.get_account_summary(account)
        if 'BuyingPower' in summary:
            return float(summary['BuyingPower']['value'])
        return 0.0
    
    def get_portfolio_value(self, account: str = None) -> float:
        """
        Get total portfolio value
        
        Args:
            account: Account ID
            
        Returns:
            Total portfolio value
        """
        summary = self.get_account_summary(account)
        if 'NetLiquidation' in summary:
            return float(summary['NetLiquidation']['value'])
        return 0.0


# Factory function for easy broker creation
def create_ib_broker(host: str = "127.0.0.1", port: int = 7497, client_id: int = 1) -> IBBroker:
    """
    Create Interactive Brokers broker instance
    
    Args:
        host: TWS/Gateway host
        port: TWS/Gateway port
        client_id: Client ID
        
    Returns:
        IBBroker instance
    """
    # Default broker config
    config = BrokerConfig(
        name='interactive_brokers',
        commission=0.005,  # $0.005 per share
        slippage=0.001     # 0.1%
    )
    
    return IBBroker(config, host, port, client_id)


# Example usage
if __name__ == "__main__":
    if IB_AVAILABLE:
        # Create broker instance
        broker = create_ib_broker()
        
        # Connect to TWS/Gateway
        if broker.connect():
            print("Connected to Interactive Brokers successfully!")
            
            # Get account summary
            summary = broker.get_account_summary()
            print(f"Account Summary: {summary}")
            
            # Get positions
            positions = broker.get_positions()
            print(f"Positions: {len(positions)}")
            
            # Request market data for AAPL
            req_id = broker.request_market_data("AAPL")
            time.sleep(2)  # Wait for data
            
            market_data = broker.get_market_data(req_id)
            if market_data:
                print(f"AAPL Market Data: {market_data}")
            
            # Cancel market data
            broker.cancel_market_data(req_id)
            
            # Get historical data
            hist_data = broker.get_historical_data("AAPL", "30 D", "1 day")
            print(f"Historical data shape: {hist_data.shape}")
            
            # Disconnect
            broker.disconnect()
        else:
            print("Failed to connect to Interactive Brokers")
    else:
        print("Interactive Brokers API not available")
