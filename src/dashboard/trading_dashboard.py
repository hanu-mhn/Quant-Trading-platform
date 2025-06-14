"""
Web Dashboard for Quantitative Trading Platform

This module provides a web-based dashboard for monitoring trading strategies,
portfolio performance, system health, and real-time metrics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sqlite3
from datetime import datetime, timedelta
import json
import sys
import os
from pathlib import Path
import time
import threading
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from src.utils.logging_system import get_logging_manager, TradingMetrics
    from src.portfolio.portfolio_manager import PortfolioManager
    from src.data.loaders.data_loader import DataLoader
    from src.config.config import Config
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    st.stop()


# Page configuration
st.set_page_config(
    page_title="Quantitative Trading Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    
    .profit {
        color: #00ff00;
        font-weight: bold;
    }
    
    .loss {
        color: #ff0000;
        font-weight: bold;
    }
    
    .status-indicator {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 5px;
    }
    
    .status-online {
        background-color: #00ff00;
    }
    
    .status-offline {
        background-color: #ff0000;
    }
    
    .status-warning {
        background-color: #ffaa00;
    }
</style>
""", unsafe_allow_html=True)


class DashboardData:
    """Data provider for the dashboard"""
    
    def __init__(self):
        self.config = Config()
        self.data_loader = DataLoader()
        self.logging_manager = get_logging_manager()
        
        # Cache for expensive operations
        self._cache = {}
        self._cache_timeout = 60  # seconds
    
    def get_cached_data(self, key: str, fetch_func, timeout: int = None):
        """Get data from cache or fetch if expired"""
        timeout = timeout or self._cache_timeout
        current_time = time.time()
        
        if key in self._cache:
            data, timestamp = self._cache[key]
            if current_time - timestamp < timeout:
                return data
        
        # Fetch fresh data
        try:
            data = fetch_func()
            self._cache[key] = (data, current_time)
            return data
        except Exception as e:
            st.error(f"Failed to fetch {key}: {e}")
            return None
    
    def get_portfolio_data(self) -> Dict[str, Any]:
        """Get portfolio performance data"""
        def fetch_portfolio():
            # Simulated portfolio data
            # In real implementation, this would fetch from portfolio manager
            return {
                'total_value': 125000.00,
                'cash_balance': 25000.00,
                'invested_value': 100000.00,
                'total_pnl': 25000.00,
                'day_pnl': 1250.50,
                'positions': 15,
                'strategies_active': 3
            }
        
        return self.get_cached_data('portfolio', fetch_portfolio)
    
    def get_performance_metrics(self) -> pd.DataFrame:
        """Get system performance metrics"""
        def fetch_performance():
            metrics = self.logging_manager.performance_monitor.get_recent_metrics(60)
            if metrics:
                return pd.DataFrame([{
                    'timestamp': m.timestamp,
                    'cpu_percent': m.cpu_percent,
                    'memory_percent': m.memory_percent,
                    'memory_mb': m.memory_mb,
                    'active_threads': m.active_threads
                } for m in metrics])
            return pd.DataFrame()
        
        return self.get_cached_data('performance', fetch_performance)
    
    def get_trading_history(self, days: int = 30) -> pd.DataFrame:
        """Get trading history"""
        def fetch_trading():
            # Simulated trading data
            dates = pd.date_range(datetime.now() - timedelta(days=days), datetime.now(), freq='D')
            
            # Generate realistic P&L data
            np.random.seed(42)
            daily_returns = np.random.normal(0.001, 0.02, len(dates))
            cumulative_pnl = np.cumsum(daily_returns) * 10000  # Scale to dollar amounts
            
            return pd.DataFrame({
                'date': dates,
                'daily_pnl': daily_returns * 10000,
                'cumulative_pnl': cumulative_pnl,
                'trades': np.random.poisson(5, len(dates)),
                'win_rate': np.random.uniform(0.4, 0.7, len(dates))
            })
        
        return self.get_cached_data(f'trading_{days}d', fetch_trading)
    
    def get_strategy_performance(self) -> pd.DataFrame:
        """Get strategy performance data"""
        def fetch_strategies():
            strategies = [
                {'name': 'MA_Crossover', 'status': 'Active', 'pnl': 5250.75, 'trades': 45, 'win_rate': 0.62},
                {'name': 'Mean_Reversion', 'status': 'Active', 'pnl': 3125.25, 'trades': 28, 'win_rate': 0.57},
                {'name': 'Momentum', 'status': 'Paused', 'pnl': -850.50, 'trades': 12, 'win_rate': 0.42},
                {'name': 'Arbitrage', 'status': 'Active', 'pnl': 1875.00, 'trades': 62, 'win_rate': 0.71}
            ]
            return pd.DataFrame(strategies)
        
        return self.get_cached_data('strategies', fetch_strategies)
    
    def get_recent_logs(self, hours: int = 24, level: str = None) -> pd.DataFrame:
        """Get recent log entries"""
        def fetch_logs():
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            return self.logging_manager.query_logs(
                start_time=start_time,
                end_time=end_time,
                level=level,
                limit=100
            )
        
        cache_key = f'logs_{hours}h_{level or "all"}'
        return self.get_cached_data(cache_key, fetch_logs, timeout=30)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        def fetch_status():
            return {
                'timestamp': datetime.now(),
                'market_hours': self._is_market_hours(),
                'data_feed': 'Connected',
                'broker_connection': 'Connected',
                'strategies_running': 3,
                'last_trade': datetime.now() - timedelta(minutes=5),
                'system_health': 'Good'
            }
        
        return self.get_cached_data('system_status', fetch_status, timeout=10)
    
    def _is_market_hours(self) -> bool:
        """Check if current time is within market hours"""
        now = datetime.now()
        if now.weekday() >= 5:  # Weekend
            return False
        
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return market_open <= now <= market_close


@st.cache_resource
def load_dashboard_data():
    """Load and cache dashboard data"""
    return DashboardData()


def create_portfolio_overview(data_provider: DashboardData):
    """Create portfolio overview section"""
    st.markdown('<div class="main-header">📊 Portfolio Overview</div>', unsafe_allow_html=True)
    
    portfolio_data = data_provider.get_portfolio_data()
    
    if portfolio_data:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Portfolio Value",
                value=f"${portfolio_data['total_value']:,.2f}",
                delta=f"${portfolio_data['day_pnl']:,.2f}"
            )
        
        with col2:
            st.metric(
                label="Total P&L",
                value=f"${portfolio_data['total_pnl']:,.2f}",
                delta=f"{(portfolio_data['total_pnl']/100000)*100:.2f}%"
            )
        
        with col3:
            st.metric(
                label="Cash Balance",
                value=f"${portfolio_data['cash_balance']:,.2f}"
            )
        
        with col4:
            st.metric(
                label="Active Positions",
                value=portfolio_data['positions']
            )


def create_performance_chart(data_provider: DashboardData):
    """Create performance charts"""
    st.subheader("📈 Performance Analytics")
    
    # Portfolio performance over time
    trading_data = data_provider.get_trading_history(30)
    
    if not trading_data.empty:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Cumulative P&L', 'Daily P&L', 'Trade Count', 'Win Rate'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Cumulative P&L
        fig.add_trace(
            go.Scatter(
                x=trading_data['date'],
                y=trading_data['cumulative_pnl'],
                name='Cumulative P&L',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Daily P&L
        colors = ['green' if x > 0 else 'red' for x in trading_data['daily_pnl']]
        fig.add_trace(
            go.Bar(
                x=trading_data['date'],
                y=trading_data['daily_pnl'],
                name='Daily P&L',
                marker_color=colors
            ),
            row=1, col=2
        )
        
        # Trade count
        fig.add_trace(
            go.Scatter(
                x=trading_data['date'],
                y=trading_data['trades'],
                name='Trades',
                line=dict(color='orange')
            ),
            row=2, col=1
        )
        
        # Win rate
        fig.add_trace(
            go.Scatter(
                x=trading_data['date'],
                y=trading_data['win_rate'],
                name='Win Rate',
                line=dict(color='purple')
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No trading data available")


def create_strategy_dashboard(data_provider: DashboardData):
    """Create strategy performance dashboard"""
    st.subheader("🎯 Strategy Performance")
    
    strategy_data = data_provider.get_strategy_performance()
    
    if not strategy_data.empty:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Strategy P&L chart
            fig = go.Figure()
            
            colors = ['green' if x > 0 else 'red' for x in strategy_data['pnl']]
            fig.add_trace(go.Bar(
                x=strategy_data['name'],
                y=strategy_data['pnl'],
                marker_color=colors,
                text=[f"${x:,.0f}" for x in strategy_data['pnl']],
                textposition='outside'
            ))
            
            fig.update_layout(
                title="Strategy P&L",
                xaxis_title="Strategy",
                yaxis_title="P&L ($)",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Strategy Status")
            for _, row in strategy_data.iterrows():
                status_color = "status-online" if row['status'] == 'Active' else "status-offline"
                pnl_class = "profit" if row['pnl'] > 0 else "loss"
                
                st.markdown(f"""
                <div class="metric-card">
                    <strong>{row['name']}</strong><br>
                    <span class="status-indicator {status_color}"></span>{row['status']}<br>
                    P&L: <span class="{pnl_class}">${row['pnl']:,.2f}</span><br>
                    Trades: {row['trades']}<br>
                    Win Rate: {row['win_rate']:.1%}
                </div>
                """, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)


def create_system_monitoring(data_provider: DashboardData):
    """Create system monitoring dashboard"""
    st.subheader("🖥️ System Monitoring")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Performance metrics
        perf_data = data_provider.get_performance_metrics()
        
        if not perf_data.empty:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('CPU Usage (%)', 'Memory Usage (%)', 'Memory (MB)', 'Active Threads')
            )
            
            fig.add_trace(
                go.Scatter(x=perf_data['timestamp'], y=perf_data['cpu_percent'], name='CPU'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=perf_data['timestamp'], y=perf_data['memory_percent'], name='Memory %'),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Scatter(x=perf_data['timestamp'], y=perf_data['memory_mb'], name='Memory MB'),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=perf_data['timestamp'], y=perf_data['active_threads'], name='Threads'),
                row=2, col=2
            )
            
            fig.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No performance data available")
    
    with col2:
        # System status
        status = data_provider.get_system_status()
        
        st.markdown("### System Status")
        
        market_status = "🟢 Open" if status['market_hours'] else "🔴 Closed"
        st.write(f"**Market Hours:** {market_status}")
        
        data_status = "🟢" if status['data_feed'] == 'Connected' else "🔴"
        st.write(f"**Data Feed:** {data_status} {status['data_feed']}")
        
        broker_status = "🟢" if status['broker_connection'] == 'Connected' else "🔴"
        st.write(f"**Broker:** {broker_status} {status['broker_connection']}")
        
        st.write(f"**Active Strategies:** {status['strategies_running']}")
        st.write(f"**System Health:** {status['system_health']}")
        st.write(f"**Last Update:** {status['timestamp'].strftime('%H:%M:%S')}")


def create_logs_viewer(data_provider: DashboardData):
    """Create logs viewer"""
    st.subheader("📝 System Logs")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        hours = st.selectbox("Time Period", [1, 6, 24, 72], index=2)
    
    with col2:
        level = st.selectbox("Log Level", ["All", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        level = None if level == "All" else level
    
    logs_data = data_provider.get_recent_logs(hours, level)
    
    if not logs_data.empty:
        # Log level distribution
        if 'level' in logs_data.columns:
            level_counts = logs_data['level'].value_counts()
            
            fig = go.Figure(data=[
                go.Bar(x=level_counts.index, y=level_counts.values)
            ])
            fig.update_layout(
                title=f"Log Distribution (Last {hours}h)",
                xaxis_title="Log Level",
                yaxis_title="Count"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent logs table
        st.markdown("### Recent Logs")
        
        # Format logs for display
        display_logs = logs_data.copy()
        if 'timestamp' in display_logs.columns:
            display_logs['timestamp'] = pd.to_datetime(display_logs['timestamp']).dt.strftime('%H:%M:%S')
        
        # Show only relevant columns
        columns_to_show = ['timestamp', 'level', 'logger_name', 'message']
        columns_to_show = [col for col in columns_to_show if col in display_logs.columns]
        
        st.dataframe(
            display_logs[columns_to_show].head(20),
            use_container_width=True,
            height=400
        )
    else:
        st.info(f"No logs found for the last {hours} hours")


def create_trading_controls():
    """Create trading controls section"""
    st.subheader("🎮 Trading Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Refresh Data", type="primary"):
            st.cache_data.clear()
            st.rerun()
    
    with col2:
        if st.button("⏸️ Pause All Strategies", type="secondary"):
            st.warning("Strategy pause functionality not implemented in demo")
    
    with col3:
        if st.button("🛑 Emergency Stop", type="secondary"):
            st.error("Emergency stop functionality not implemented in demo")
    
    # Settings
    with st.expander("⚙️ Settings"):
        st.slider("Refresh Interval (seconds)", 10, 300, 60)
        st.checkbox("Auto-refresh enabled", value=True)
        st.checkbox("Sound alerts", value=False)
        st.selectbox("Theme", ["Light", "Dark"], index=0)


def main():
    """Main dashboard function"""
    st.markdown('<div class="main-header">Quantitative Trading Platform Dashboard</div>', 
                unsafe_allow_html=True)
    
    # Initialize data provider
    try:
        data_provider = load_dashboard_data()
    except Exception as e:
        st.error(f"Failed to initialize dashboard: {e}")
        st.stop()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    page = st.sidebar.selectbox(
        "Select Dashboard",
        ["Overview", "Performance", "Strategies", "System", "Logs", "Controls"]
    )
    
    # Auto-refresh option
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
    
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    # Main content based on selected page
    if page == "Overview":
        create_portfolio_overview(data_provider)
        st.markdown("---")
        create_performance_chart(data_provider)
    
    elif page == "Performance":
        create_performance_chart(data_provider)
    
    elif page == "Strategies":
        create_strategy_dashboard(data_provider)
    
    elif page == "System":
        create_system_monitoring(data_provider)
    
    elif page == "Logs":
        create_logs_viewer(data_provider)
    
    elif page == "Controls":
        create_trading_controls()
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"Dashboard last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        f"Data refresh: {data_provider._cache_timeout}s"
    )


if __name__ == "__main__":
    main()
