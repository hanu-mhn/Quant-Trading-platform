"""
Simplified Streamlit Dashboard for Cloud Deployment
This is a demo version with reduced dependencies for Streamlit Community Cloud

Copyright © 2025 Malavath Hanmanth Nayak. All rights reserved.
Developer: Malavath Hanmanth Nayak
Contact: hanmanthnayak.95@gmail.com
GitHub: https://github.com/hanu-mhn
LinkedIn: https://www.linkedin.com/in/hanmanth-nayak-m-6bbab1263/

This software is provided under the MIT License.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time

# Page configuration
st.set_page_config(
    page_title="Quantitative Trading Platform - Demo",
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
    .success-metric {
        border-left-color: #28a745;
    }
    .warning-metric {
        border-left-color: #ffc107;
    }
    .danger-metric {
        border-left-color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Check if cache_data is available, else use cache
try:
    cache_decorator = st.cache_data
except AttributeError:
    # Fallback for older Streamlit versions
    cache_decorator = st.cache

@cache_decorator
def generate_sample_data():
    """Generate sample trading data for demo"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
    
    # Generate sample price data
    initial_price = 100
    returns = np.random.normal(0.0005, 0.02, len(dates))
    prices = initial_price * np.exp(np.cumsum(returns))
    
    # Create sample portfolio data
    portfolio_data = pd.DataFrame({
        'date': dates,
        'portfolio_value': 100000 * np.exp(np.cumsum(returns * 0.8)),
        'cash': 100000 - np.cumsum(np.random.randint(1000, 5000, len(dates))),
        'spy_price': prices
    })
    
    return portfolio_data

@cache_decorator  
def get_sample_positions():
    """Generate sample positions data"""
    positions = [
        {'symbol': 'AAPL', 'quantity': 150, 'avg_price': 175.50, 'current_price': 182.30, 'pnl': 1020},
        {'symbol': 'GOOGL', 'quantity': 50, 'avg_price': 2750.00, 'current_price': 2820.50, 'pnl': 3525},
        {'symbol': 'MSFT', 'quantity': 200, 'avg_price': 335.75, 'current_price': 342.10, 'pnl': 1270},
        {'symbol': 'TSLA', 'quantity': 75, 'avg_price': 245.20, 'current_price': 238.90, 'pnl': -472.5},
        {'symbol': 'NVDA', 'quantity': 100, 'avg_price': 450.30, 'current_price': 478.20, 'pnl': 2790}
    ]
    return pd.DataFrame(positions)

def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<div class="main-header">📊 Quantitative Trading Platform</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎛️ Control Panel")
    
    # Demo notice
    st.sidebar.info("🚀 **Live Demo Version**\n\nThis is a demonstration of the quantitative trading platform running on Streamlit Community Cloud.")
    
    # Navigation
    page = st.sidebar.selectbox("Navigate", [
        "📊 Portfolio Overview", 
        "📈 Performance Analytics", 
        "🤖 Strategy Monitor", 
        "📱 Trade Simulator",
        "ℹ️ About Platform"
    ])
    
    # Load sample data
    portfolio_data = generate_sample_data()
    positions_df = get_sample_positions()
    
    if page == "📊 Portfolio Overview":
        show_portfolio_overview(portfolio_data, positions_df)
    elif page == "📈 Performance Analytics":
        show_performance_analytics(portfolio_data)
    elif page == "🤖 Strategy Monitor":
        show_strategy_monitor()
    elif page == "📱 Trade Simulator":
        show_trade_simulator()
    elif page == "ℹ️ About Platform":
        show_about_platform()

def show_portfolio_overview(portfolio_data, positions_df):
    """Show portfolio overview dashboard"""
    st.header("📊 Portfolio Overview")
    
    # Current metrics
    current_value = portfolio_data['portfolio_value'].iloc[-1]
    initial_value = portfolio_data['portfolio_value'].iloc[0]
    total_return = (current_value - initial_value) / initial_value * 100
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Portfolio Value",
            f"${current_value:,.0f}",
            f"{total_return:+.2f}%"
        )
    
    with col2:
        total_pnl = positions_df['pnl'].sum()
        st.metric(
            "Total P&L",
            f"${total_pnl:,.0f}",
            f"{total_pnl/current_value*100:+.2f}%"
        )
    
    with col3:
        st.metric(
            "Positions",
            len(positions_df),
            "5 Active"
        )
    
    with col4:
        volatility = portfolio_data['portfolio_value'].pct_change().std() * np.sqrt(252) * 100
        st.metric(
            "Volatility",
            f"{volatility:.1f}%",
            "Annualized"
        )
    
    # Portfolio chart
    st.subheader("📈 Portfolio Performance")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=portfolio_data['date'],
        y=portfolio_data['portfolio_value'],
        name='Portfolio Value',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.update_layout(
        title="Portfolio Value Over Time",
        xaxis_title="Date",
        yaxis_title="Value ($)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Current positions
    st.subheader("📋 Current Positions")
    
    # Style the dataframe
    def color_pnl(val):
        color = 'green' if val > 0 else 'red'
        return f'color: {color}'
    
    styled_positions = positions_df.style.applymap(color_pnl, subset=['pnl'])
    st.dataframe(styled_positions, use_container_width=True)

def show_performance_analytics(portfolio_data):
    """Show performance analytics"""
    st.header("📈 Performance Analytics")
    
    # Calculate metrics
    returns = portfolio_data['portfolio_value'].pct_change().dropna()
    cumulative_returns = (1 + returns).cumprod() - 1
    
    # Performance metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Key Metrics")
        
        metrics = {
            "Total Return": f"{cumulative_returns.iloc[-1]*100:.2f}%",
            "Sharpe Ratio": f"{returns.mean()/returns.std()*np.sqrt(252):.2f}",
            "Max Drawdown": f"{((portfolio_data['portfolio_value']/portfolio_data['portfolio_value'].expanding().max()-1).min()*100):.2f}%",
            "Volatility": f"{returns.std()*np.sqrt(252)*100:.2f}%",
            "Best Day": f"{returns.max()*100:.2f}%",
            "Worst Day": f"{returns.min()*100:.2f}%"
        }
        
        for metric, value in metrics.items():
            st.metric(metric, value)
    
    with col2:
        st.subheader("📈 Returns Distribution")
        
        fig = px.histogram(
            returns*100,
            nbins=50,
            title="Daily Returns Distribution",
            labels={'value': 'Daily Return (%)', 'count': 'Frequency'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Monthly returns heatmap
    st.subheader("🔥 Monthly Returns Heatmap")
    
    monthly_returns = portfolio_data.set_index('date')['portfolio_value'].resample('M').last().pct_change()
    monthly_returns.index = monthly_returns.index.strftime('%Y-%m')
    
    # Create a simple monthly returns table
    if len(monthly_returns) > 0:
        months_per_row = 6
        monthly_data = monthly_returns.values.reshape(-1, months_per_row) if len(monthly_returns) >= months_per_row else monthly_returns.values.reshape(1, -1)
        
        fig = px.imshow(
            monthly_data,
            color_continuous_scale='RdYlGn',
            title="Monthly Returns (%)"
        )
        st.plotly_chart(fig, use_container_width=True)

def show_strategy_monitor():
    """Show strategy monitoring"""
    st.header("🤖 Strategy Monitor")
    
    st.info("💡 **Strategy Simulation**\n\nThis demo shows how strategies would be monitored in the full platform.")
    
    # Strategy performance
    strategies = [
        {"name": "MA Crossover", "status": "Active", "return": "12.5%", "sharpe": "1.8", "trades": 45},
        {"name": "Mean Reversion", "status": "Active", "return": "8.3%", "sharpe": "1.2", "trades": 32},
        {"name": "Momentum", "status": "Paused", "return": "15.7%", "sharpe": "2.1", "trades": 28},
    ]
    
    st.subheader("📋 Active Strategies")
    
    for strategy in strategies:
        with st.expander(f"🎯 {strategy['name']} - {strategy['status']}"):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Return", strategy['return'])
            with col2:
                st.metric("Sharpe Ratio", strategy['sharpe'])
            with col3:
                st.metric("Total Trades", strategy['trades'])
            with col4:
                status_color = "🟢" if strategy['status'] == "Active" else "🟡"
                st.write(f"Status: {status_color} {strategy['status']}")
    
    # Strategy settings
    st.subheader("⚙️ Strategy Configuration")
    
    with st.form("strategy_config"):
        strategy_type = st.selectbox("Strategy Type", ["Moving Average Crossover", "RSI Mean Reversion", "Bollinger Bands"])
        
        if strategy_type == "Moving Average Crossover":
            short_ma = st.slider("Short MA Period", 5, 50, 20)
            long_ma = st.slider("Long MA Period", 20, 200, 50)
            st.write(f"Configuration: {short_ma}-day MA crossing {long_ma}-day MA")
        
        submitted = st.form_submit_button("Update Strategy")
        if submitted:
            st.success("Strategy configuration updated!")

def show_trade_simulator():
    """Show trade simulator"""
    st.header("📱 Trade Simulator")
    
    st.info("🎮 **Paper Trading Simulator**\n\nTest your trading ideas risk-free!")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Place Order")
        
        with st.form("place_order"):
            symbol = st.text_input("Symbol", value="AAPL")
            side = st.selectbox("Side", ["BUY", "SELL"])
            quantity = st.number_input("Quantity", min_value=1, value=100)
            order_type = st.selectbox("Order Type", ["MARKET", "LIMIT"])
            
            if order_type == "LIMIT":
                price = st.number_input("Limit Price", min_value=0.01, value=180.00)
            
            submitted = st.form_submit_button("🚀 Place Order")
            
            if submitted:
                st.success(f"✅ {side} order for {quantity} shares of {symbol} submitted!")
    
    with col2:
        st.subheader("📊 Order History")
        
        # Sample order history
        orders = [
            {"Time": "09:30:15", "Symbol": "AAPL", "Side": "BUY", "Qty": 100, "Price": "$182.30", "Status": "FILLED"},
            {"Time": "10:15:22", "Symbol": "GOOGL", "Side": "SELL", "Qty": 25, "Price": "$2820.50", "Status": "FILLED"},
            {"Time": "11:45:33", "Symbol": "MSFT", "Side": "BUY", "Qty": 50, "Price": "$342.10", "Status": "PENDING"},
        ]
        
        st.dataframe(pd.DataFrame(orders), use_container_width=True)

def show_about_platform():
    """Show about information"""
    st.header("ℹ️ About the Platform")
    
    st.markdown("""
    ## 🚀 Quantitative Trading Platform
    
    This is a **comprehensive algorithmic trading platform** built with Python, designed for both 
    professional traders and quantitative researchers.
    
    ### ✨ Key Features
    
    - **📊 Real-time Portfolio Management**: Track positions, P&L, and risk metrics
    - **🤖 Strategy Engine**: Deploy and monitor multiple trading strategies
    - **📈 Advanced Backtesting**: Historical performance analysis with detailed metrics
    - **📱 Paper Trading**: Risk-free strategy testing environment
    - **🔧 REST API**: Complete programmatic access to all features
    - **📊 Interactive Dashboard**: Web-based monitoring and control interface
    
    ### 🛠️ Technology Stack
    
    - **Backend**: Python, FastAPI, SQLAlchemy
    - **Frontend**: Streamlit, Plotly, Bootstrap
    - **Database**: PostgreSQL, Redis
    - **Deployment**: Docker, Nginx, Prometheus
    - **Brokers**: Interactive Brokers, Zerodha (planned)
    
    ### 📈 Supported Strategies
    
    - Moving Average Crossover
    - RSI Mean Reversion  
    - Bollinger Band Breakout
    - Statistical Arbitrage
    - Custom Strategy Framework
    
    ### 🔗 Links
    
    - **GitHub Repository**: [View Source Code](https://github.com/yourusername/quant-trading-platform)
    - **Documentation**: [Read the Docs](https://your-docs-url.com)
    - **API Reference**: [API Documentation](https://your-api-url.com/docs)
    
    ### ⚠️ Disclaimer
    
    This platform is for **educational and research purposes**. Trading involves risk, 
    and past performance does not guarantee future results. Always understand the risks 
    before trading with real money.
    """)
    
    # System status
    st.subheader("🔧 Demo System Status")
    
    status_items = [
        ("📊 Dashboard", "🟢 Online"),
        ("🤖 Strategy Engine", "🟢 Active"),
        ("📱 Paper Trading", "🟢 Available"),
        ("📈 Data Feed", "🟢 Connected"),
        ("🔧 API Services", "🟢 Running")    ]
    
    for service, status in status_items:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(service)
        with col2:
            st.write(status)
    
    # Developer information footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 20px; color: #666; font-size: 14px;'>
        <p><strong>🚀 Quantitative Trading Platform</strong></p>
        <p>Developed by <strong>Malavath Hanmanth Nayak</strong></p>
        <p>📧 <a href="mailto:hanmanthnayak.95@gmail.com">hanmanthnayak.95@gmail.com</a> | 
           💻 <a href="https://github.com/hanu-mhn" target="_blank">GitHub</a> | 
           🔗 <a href="https://www.linkedin.com/in/hanmanth-nayak-m-6bbab1263/" target="_blank">LinkedIn</a></p>
        <p><small>Copyright © 2025 Malavath Hanmanth Nayak. All rights reserved.</small></p>
        <p><small>This platform is for educational and demonstration purposes.</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
