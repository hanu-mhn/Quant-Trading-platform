"""
REST API Server for Quantitative Trading Platform

This module provides a comprehensive REST API for interacting with the trading platform,
including portfolio management, strategy control, market data access, and system monitoring.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import logging
import asyncio
import json
from pathlib import Path
import jwt
from passlib.context import CryptContext
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from src.config.config import Config
    from src.portfolio.portfolio_manager import PortfolioManager
    from src.data.loaders.data_loader import DataLoader
    from src.trading.paper_trading import PaperTradingManager, OrderType
    from src.utils.logging_system import get_logging_manager
    from src.core.backtester import BacktestEngine
    from src.strategies.momentum.moving_average_crossover import MovingAverageCrossoverStrategy
except ImportError:
    # Fallback for relative imports when run as module
    from ..config.config import Config
    from ..portfolio.portfolio_manager import PortfolioManager
    from ..data.loaders.data_loader import DataLoader
    from ..trading.paper_trading import PaperTradingManager, OrderType
    from ..utils.logging_system import get_logging_manager
    from ..core.backtester import BacktestEngine
    from ..strategies.momentum.moving_average_crossover import MovingAverageCrossoverStrategy


# Pydantic models for API requests/responses
class OrderRequest(BaseModel):
    symbol: str = Field(..., description="Trading symbol")
    quantity: int = Field(..., gt=0, description="Number of shares")
    order_type: str = Field(..., description="Order type: MARKET, LIMIT, STOP")
    side: str = Field(..., description="Order side: BUY or SELL")
    price: Optional[float] = Field(None, description="Limit price for limit orders")
    stop_price: Optional[float] = Field(None, description="Stop price for stop orders")


class PortfolioResponse(BaseModel):
    total_value: float
    cash_balance: float
    total_pnl: float
    total_pnl_pct: float
    positions_count: int
    positions: Dict[str, Any]


class StrategyRequest(BaseModel):
    strategy_type: str = Field(..., description="Strategy type")
    parameters: Dict[str, Any] = Field(..., description="Strategy parameters")
    symbols: List[str] = Field(..., description="Symbols to trade")


class BacktestRequest(BaseModel):
    strategy: StrategyRequest
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    initial_capital: float = Field(100000, description="Initial capital")


class SystemStatusResponse(BaseModel):
    timestamp: datetime
    system_health: str
    active_strategies: int
    total_trades: int
    api_version: str
    uptime_seconds: float


class UserCredentials(BaseModel):
    username: str
    password: str


# Initialize FastAPI app
app = FastAPI(
    title="Quantitative Trading Platform API",
    description="REST API for quantitative trading platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
config = None
portfolio_manager = None
data_loader = None
paper_trading_manager = None
logging_manager = None
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
SECRET_KEY = "your-secret-key-change-this-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Server start time for uptime calculation
server_start_time = datetime.now()


async def startup_event():
    """Initialize services on startup"""
    global config, portfolio_manager, data_loader, paper_trading_manager, logging_manager
    
    # Initialize configuration
    config = Config()
      # Initialize services
    portfolio_manager = PortfolioManager(initial_cash=100000.0)
    data_loader = DataLoader()
    paper_trading_manager = PaperTradingManager(config)
    logging_manager = get_logging_manager()
    
    logging.info("API server initialized successfully")


app.add_event_handler("startup", startup_event)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token"""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        return username
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")


# Authentication endpoints
@app.post("/auth/login")
async def login(credentials: UserCredentials):
    """Login and get access token"""
    # Simple authentication - replace with proper user authentication
    if credentials.username == "admin" and credentials.password == "password":
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": credentials.username}, expires_delta=access_token_expires
        )
        return {"access_token": access_token, "token_type": "bearer"}
    raise HTTPException(status_code=401, detail="Incorrect username or password")


# System endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Quantitative Trading Platform API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now()}


@app.get("/status", response_model=SystemStatusResponse)
async def get_system_status(username: str = Depends(verify_token)):
    """Get comprehensive system status"""
    uptime = (datetime.now() - server_start_time).total_seconds()
    
    return SystemStatusResponse(
        timestamp=datetime.now(),
        system_health="Good",
        active_strategies=len(paper_trading_manager.active_sessions) if paper_trading_manager else 0,
        total_trades=0,  # Would fetch from database
        api_version="1.0.0",
        uptime_seconds=uptime
    )


# Portfolio endpoints
@app.get("/portfolio", response_model=PortfolioResponse)
async def get_portfolio(session_id: str = "default", username: str = Depends(verify_token)):
    """Get portfolio information"""
    session = paper_trading_manager.get_session(session_id)
    if not session:
        # Create a new session if it doesn't exist
        session = paper_trading_manager.create_session(session_id)
    
    summary = session.get_portfolio_summary()
    
    return PortfolioResponse(
        total_value=summary['portfolio_value'],
        cash_balance=summary['current_cash'],
        total_pnl=summary['total_pnl'],
        total_pnl_pct=summary['total_pnl_pct'],
        positions_count=summary['positions_count'],
        positions=summary['positions']
    )


@app.get("/portfolio/positions")
async def get_positions(session_id: str = "default", username: str = Depends(verify_token)):
    """Get all positions"""
    session = paper_trading_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Trading session not found")
    
    return {"positions": session.positions}


@app.get("/portfolio/performance")
async def get_portfolio_performance(session_id: str = "default", username: str = Depends(verify_token)):
    """Get portfolio performance metrics"""
    session = paper_trading_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Trading session not found")
    
    metrics = session.get_performance_metrics()
    return {"performance": metrics}


# Trading endpoints
@app.post("/orders")
async def place_order(order: OrderRequest, session_id: str = "default", username: str = Depends(verify_token)):
    """Place a new trading order"""
    session = paper_trading_manager.get_session(session_id)
    if not session:
        session = paper_trading_manager.create_session(session_id)
    
    try:
        order_type = OrderType(order.order_type.upper())
        
        order_id = await session.place_order(
            symbol=order.symbol,
            quantity=order.quantity,
            order_type=order_type,
            side=order.side.upper(),
            price=order.price,
            stop_price=order.stop_price
        )
        
        return {"order_id": order_id, "status": "submitted"}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/orders")
async def get_orders(session_id: str = "default", username: str = Depends(verify_token)):
    """Get all orders"""
    session = paper_trading_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Trading session not found")
    
    orders = {}
    for order_id, order in session.orders.items():
        orders[order_id] = {
            "order_id": order.order_id,
            "symbol": order.symbol,
            "quantity": order.quantity,
            "order_type": order.order_type.value,
            "side": order.side,
            "status": order.status.value,
            "submitted_at": order.submitted_at.isoformat() if order.submitted_at else None,
            "filled_at": order.filled_at.isoformat() if order.filled_at else None,
            "avg_fill_price": order.avg_fill_price
        }
    
    return {"orders": orders}


@app.delete("/orders/{order_id}")
async def cancel_order(order_id: str, session_id: str = "default", username: str = Depends(verify_token)):
    """Cancel an existing order"""
    session = paper_trading_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Trading session not found")
    
    success = session.cancel_order(order_id)
    if success:
        return {"message": "Order cancelled successfully"}
    else:
        raise HTTPException(status_code=400, detail="Unable to cancel order")


@app.get("/trades")
async def get_trades(session_id: str = "default", username: str = Depends(verify_token)):
    """Get trade history"""
    session = paper_trading_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Trading session not found")
    
    return {"trades": session.trade_history}


# Market data endpoints
@app.get("/market/quote/{symbol}")
async def get_quote(symbol: str, username: str = Depends(verify_token)):
    """Get current quote for a symbol"""
    try:
        # This would fetch real-time data in production
        data = data_loader.load_symbol_data(symbol, period='1d')
        if data is not None and not data.empty:
            latest = data.iloc[-1]
            return {
                "symbol": symbol,
                "price": float(latest['close']),
                "open": float(latest['open']),
                "high": float(latest['high']),
                "low": float(latest['low']),
                "volume": int(latest['volume']),
                "timestamp": latest.name.isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail="Symbol not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/market/historical/{symbol}")
async def get_historical_data(
    symbol: str,
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
    username: str = Depends(verify_token)
):
    """Get historical data for a symbol"""
    try:
        data = data_loader.load_symbol_data(symbol, start_date, end_date)
        if data is not None and not data.empty:
            # Convert DataFrame to JSON-serializable format
            result = []
            for idx, row in data.iterrows():
                result.append({
                    "date": idx.isoformat(),
                    "open": float(row['open']),
                    "high": float(row['high']),
                    "low": float(row['low']),
                    "close": float(row['close']),
                    "volume": int(row['volume'])
                })
            return {"symbol": symbol, "data": result}
        else:
            raise HTTPException(status_code=404, detail="No data found for symbol")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Strategy endpoints
@app.post("/strategies/backtest")
async def run_backtest(request: BacktestRequest, background_tasks: BackgroundTasks, username: str = Depends(verify_token)):
    """Run a strategy backtest"""
    try:
        # Create strategy based on request
        if request.strategy.strategy_type.lower() == "moving_average_crossover":
            params = request.strategy.parameters
            strategy = MovingAverageCrossoverStrategy(
                short_window=params.get('short_window', 10),
                long_window=params.get('long_window', 20)
            )
        else:
            raise HTTPException(status_code=400, detail="Unsupported strategy type")
        
        # Load data for backtest
        symbol = request.strategy.symbols[0] if request.strategy.symbols else "AAPL"
        data = data_loader.load_symbol_data(symbol, request.start_date, request.end_date)
        
        if data is None or data.empty:
            raise HTTPException(status_code=404, detail="No data available for backtest")
        
        # Run backtest
        backtest_engine = BacktestEngine(config.backtest)
        results = backtest_engine.run_backtest(strategy, data)
        
        return {
            "backtest_id": f"bt_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "status": "completed",
            "results": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/strategies/list")
async def list_strategies(username: str = Depends(verify_token)):
    """List available strategies"""
    return {
        "strategies": [
            {
                "name": "moving_average_crossover",
                "description": "Moving Average Crossover Strategy",
                "parameters": ["short_window", "long_window"]
            },
            {
                "name": "bollinger_band_reversion",
                "description": "Bollinger Band Mean Reversion Strategy",
                "parameters": ["period", "std_dev"]
            }
        ]
    }


# Paper trading session management
@app.post("/paper-trading/sessions")
async def create_paper_trading_session(
    session_id: str,
    initial_cash: float = 100000.0,
    username: str = Depends(verify_token)
):
    """Create a new paper trading session"""
    try:
        session = paper_trading_manager.create_session(session_id, initial_cash)
        return {
            "session_id": session_id,
            "initial_cash": initial_cash,
            "status": "created"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/paper-trading/sessions/{session_id}")
async def close_paper_trading_session(session_id: str, username: str = Depends(verify_token)):
    """Close a paper trading session"""
    try:
        paper_trading_manager.close_session(session_id)
        return {"message": "Session closed successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/paper-trading/sessions")
async def list_paper_trading_sessions(username: str = Depends(verify_token)):
    """List all paper trading sessions"""
    sessions = paper_trading_manager.get_all_sessions_summary()
    return {"sessions": sessions}


# Monitoring endpoints
@app.get("/monitoring/logs")
async def get_logs(
    hours: int = Query(24, description="Hours to look back"),
    level: Optional[str] = Query(None, description="Log level filter"),
    limit: int = Query(100, description="Maximum number of logs"),
    username: str = Depends(verify_token)
):
    """Get system logs"""
    try:
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        logs = logging_manager.query_logs(
            start_time=start_time,
            end_time=end_time,
            level=level,
            limit=limit
        )
        
        # Convert DataFrame to records
        if not logs.empty:
            logs_list = logs.to_dict('records')
        else:
            logs_list = []
        
        return {"logs": logs_list}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/monitoring/performance")
async def get_performance_metrics(username: str = Depends(verify_token)):
    """Get system performance metrics"""
    try:
        metrics = logging_manager.performance_monitor.get_performance_summary()
        return {"performance": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket endpoints for real-time updates
@app.websocket("/ws/portfolio/{session_id}")
async def websocket_portfolio_updates(websocket, session_id: str):
    """WebSocket endpoint for real-time portfolio updates"""
    await websocket.accept()
    
    try:
        while True:
            session = paper_trading_manager.get_session(session_id)
            if session:
                summary = session.get_portfolio_summary()
                await websocket.send_json(summary)
            
            await asyncio.sleep(5)  # Update every 5 seconds
    
    except Exception as e:
        logging.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()


if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )