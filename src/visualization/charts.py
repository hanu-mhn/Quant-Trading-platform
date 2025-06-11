"""
Visualization Module.

This module provides comprehensive charting and visualization tools for
market data analysis, strategy performance, and risk metrics.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
import logging


class TradingCharts:
    """Comprehensive trading chart visualization tools."""
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (15, 10)):
        self.style = style
        self.figsize = figsize
        self.logger = logging.getLogger(__name__)
        
        # Set matplotlib style
        plt.style.use(style)
        sns.set_palette("husl")
    
    def plot_candlestick_chart(
        self,
        data: pd.DataFrame,
        title: str = "Candlestick Chart",
        volume: bool = True,
        indicators: Optional[List[str]] = None,
        interactive: bool = False
    ) -> Union[plt.Figure, go.Figure]:
        """
        Create candlestick chart with optional volume and indicators.
        
        Args:
            data: OHLCV DataFrame
            title: Chart title
            volume: Include volume subplot
            indicators: List of indicator columns to plot
            interactive: Create interactive Plotly chart
            
        Returns:
            Matplotlib or Plotly figure
        """
        if interactive:
            return self._plot_interactive_candlestick(data, title, volume, indicators)
        else:
            return self._plot_matplotlib_candlestick(data, title, volume, indicators)
    
    def _plot_matplotlib_candlestick(
        self,
        data: pd.DataFrame,
        title: str,
        volume: bool,
        indicators: Optional[List[str]]
    ) -> plt.Figure:
        """Create matplotlib candlestick chart."""
        from matplotlib.patches import Rectangle
        
        # Determine subplot layout
        n_subplots = 1 + (1 if volume else 0) + (1 if indicators else 0)
        height_ratios = [3] + ([1] if volume else []) + ([1] if indicators else [])
        
        fig, axes = plt.subplots(
            n_subplots, 1, 
            figsize=self.figsize,
            height_ratios=height_ratios,
            sharex=True
        )
        
        if n_subplots == 1:
            axes = [axes]
        
        ax_main = axes[0]
        
        # Plot candlesticks
        for i, (idx, row) in enumerate(data.iterrows()):
            open_price = row['open']
            high_price = row['high']
            low_price = row['low']
            close_price = row['close']
            
            # Determine color
            color = 'green' if close_price >= open_price else 'red'
            
            # Draw high-low line
            ax_main.plot([i, i], [low_price, high_price], color='black', linewidth=1)
            
            # Draw body rectangle
            body_height = abs(close_price - open_price)
            body_bottom = min(open_price, close_price)
            
            rect = Rectangle(
                (i - 0.3, body_bottom), 0.6, body_height,
                facecolor=color, alpha=0.7, edgecolor='black'
            )
            ax_main.add_patch(rect)
        
        ax_main.set_title(title, fontsize=16, fontweight='bold')
        ax_main.set_ylabel('Price (₹)', fontsize=12)
        ax_main.grid(True, alpha=0.3)
        
        # Set x-axis ticks
        if 'date' in data.columns:
            tick_positions = range(0, len(data), max(1, len(data) // 10))
            tick_labels = [data.iloc[i]['date'].strftime('%Y-%m-%d') if pd.notna(data.iloc[i]['date']) else '' 
                          for i in tick_positions]
            ax_main.set_xticks(tick_positions)
            ax_main.set_xticklabels(tick_labels, rotation=45)
        
        subplot_idx = 1
        
        # Plot volume
        if volume and 'volume' in data.columns:
            ax_vol = axes[subplot_idx]
            colors = ['green' if data.iloc[i]['close'] >= data.iloc[i]['open'] else 'red' 
                     for i in range(len(data))]
            ax_vol.bar(range(len(data)), data['volume'], color=colors, alpha=0.7)
            ax_vol.set_ylabel('Volume', fontsize=12)
            ax_vol.grid(True, alpha=0.3)
            subplot_idx += 1
        
        # Plot indicators
        if indicators and subplot_idx < len(axes):
            ax_ind = axes[subplot_idx]
            for indicator in indicators:
                if indicator in data.columns:
                    ax_ind.plot(range(len(data)), data[indicator], label=indicator, linewidth=2)
            ax_ind.set_ylabel('Indicators', fontsize=12)
            ax_ind.legend()
            ax_ind.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _plot_interactive_candlestick(
        self,
        data: pd.DataFrame,
        title: str,
        volume: bool,
        indicators: Optional[List[str]]
    ) -> go.Figure:
        """Create interactive Plotly candlestick chart."""
        # Determine subplot specifications
        subplot_titles = ['Price']
        if volume:
            subplot_titles.append('Volume')
        if indicators:
            subplot_titles.append('Indicators')
        
        rows = len(subplot_titles)
        row_heights = [0.6] + ([0.2] if volume else []) + ([0.2] if indicators else [])
        
        fig = make_subplots(
            rows=rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=subplot_titles,
            row_heights=row_heights
        )
        
        # Main candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data['date'] if 'date' in data.columns else data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        row_idx = 2
        
        # Volume chart
        if volume and 'volume' in data.columns:
            colors = ['green' if close >= open else 'red' 
                     for close, open in zip(data['close'], data['open'])]
            
            fig.add_trace(
                go.Bar(
                    x=data['date'] if 'date' in data.columns else data.index,
                    y=data['volume'],
                    marker_color=colors,
                    name='Volume',
                    opacity=0.7
                ),
                row=row_idx, col=1
            )
            row_idx += 1
        
        # Indicators
        if indicators and row_idx <= rows:
            for indicator in indicators:
                if indicator in data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=data['date'] if 'date' in data.columns else data.index,
                            y=data[indicator],
                            mode='lines',
                            name=indicator,
                            line=dict(width=2)
                        ),
                        row=row_idx, col=1
                    )
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_rangeslider_visible=False,
            height=600 * rows / 2,
            showlegend=True
        )
        
        return fig
    
    def plot_strategy_performance(
        self,
        portfolio_history: List[Dict[str, Any]],
        benchmark_data: Optional[pd.DataFrame] = None,
        title: str = "Strategy Performance"
    ) -> plt.Figure:
        """
        Plot strategy performance with portfolio value, drawdown, and benchmarks.
        
        Args:
            portfolio_history: List of portfolio value records
            benchmark_data: Benchmark price data for comparison
            title: Chart title
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(3, 1, figsize=self.figsize, height_ratios=[2, 1, 1])
        
        # Extract portfolio data
        dates = [record['date'] for record in portfolio_history]
        values = [record['total_value'] for record in portfolio_history]
        
        # Portfolio value
        axes[0].plot(dates, values, label='Portfolio Value', linewidth=2, color='blue')
        
        # Benchmark comparison
        if benchmark_data is not None and 'close' in benchmark_data.columns:
            # Normalize benchmark to start at same value
            benchmark_normalized = benchmark_data['close'] / benchmark_data['close'].iloc[0] * values[0]
            axes[0].plot(benchmark_data['date'], benchmark_normalized, 
                        label='Benchmark', linewidth=2, color='gray', alpha=0.7)
        
        axes[0].set_title(title, fontsize=16, fontweight='bold')
        axes[0].set_ylabel('Portfolio Value (₹)', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Calculate and plot returns
        returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]
        return_dates = dates[1:]
        
        axes[1].plot(return_dates, np.array(returns) * 100, color='green', alpha=0.7)
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1].set_ylabel('Daily Returns (%)', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        # Calculate and plot drawdown
        running_max = np.maximum.accumulate(values)
        drawdown = [(val - running_max[i]) / running_max[i] * 100 for i, val in enumerate(values)]
        
        axes[2].fill_between(dates, drawdown, 0, alpha=0.3, color='red', label='Drawdown')
        axes[2].plot(dates, drawdown, color='red', linewidth=1)
        axes[2].set_ylabel('Drawdown (%)', fontsize=12)
        axes[2].set_xlabel('Date', fontsize=12)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_risk_metrics(
        self,
        returns: pd.Series,
        risk_metrics: Dict[str, float],
        title: str = "Risk Analysis"
    ) -> plt.Figure:
        """
        Plot comprehensive risk analysis charts.
        
        Args:
            returns: Daily returns series
            risk_metrics: Dictionary of calculated risk metrics
            title: Chart title
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # Returns distribution
        axes[0, 0].hist(returns, bins=50, alpha=0.7, density=True, color='blue')
        axes[0, 0].axvline(risk_metrics.get('VaR_95', 0), color='red', linestyle='--', 
                          label=f'VaR 95%: {risk_metrics.get("VaR_95", 0):.2%}')
        axes[0, 0].axvline(risk_metrics.get('VaR_99', 0), color='orange', linestyle='--',
                          label=f'VaR 99%: {risk_metrics.get("VaR_99", 0):.2%}')
        axes[0, 0].set_title('Returns Distribution')
        axes[0, 0].set_xlabel('Daily Return')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Rolling volatility
        rolling_vol = returns.rolling(window=20).std() * np.sqrt(252) * 100
        axes[0, 1].plot(rolling_vol.index, rolling_vol.values, color='purple')
        axes[0, 1].set_title('Rolling 20-Day Volatility (Annualized)')
        axes[0, 1].set_ylabel('Volatility (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Risk metrics comparison
        metrics_names = ['Sharpe Ratio', 'Sortino Ratio']
        metrics_values = [risk_metrics.get('Sharpe_Ratio', 0), risk_metrics.get('Sortino_Ratio', 0)]
        axes[1, 0].bar(metrics_names, metrics_values, color=['blue', 'green'], alpha=0.7)
        axes[1, 0].set_title('Risk-Adjusted Return Metrics')
        axes[1, 0].set_ylabel('Ratio')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Cumulative returns
        cumulative_returns = (1 + returns).cumprod() - 1
        axes[1, 1].plot(cumulative_returns.index, cumulative_returns.values * 100, color='green')
        axes[1, 1].set_title('Cumulative Returns')
        axes[1, 1].set_ylabel('Cumulative Return (%)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_correlation_matrix(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        title: str = "Correlation Matrix"
    ) -> plt.Figure:
        """
        Plot correlation matrix heatmap.
        
        Args:
            data: DataFrame with numerical columns
            columns: Specific columns to include
            title: Chart title
            
        Returns:
            Matplotlib figure
        """
        if columns:
            corr_data = data[columns]
        else:
            # Select numerical columns
            corr_data = data.select_dtypes(include=[np.number])
        
        correlation_matrix = corr_data.corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap='RdYlBu_r',
            center=0,
            square=True,
            fmt='.2f',
            cbar_kws={'shrink': 0.8},
            ax=ax
        )
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def plot_portfolio_composition(
        self,
        positions: List[Dict[str, Any]],
        title: str = "Portfolio Composition"
    ) -> plt.Figure:
        """
        Plot portfolio composition pie chart and bar chart.
        
        Args:
            positions: List of position dictionaries
            title: Chart title
            
        Returns:
            Matplotlib figure
        """
        if not positions:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No positions to display', ha='center', va='center', fontsize=16)
            ax.set_title(title)
            return fig
        
        # Extract position data
        symbols = [pos['symbol'] for pos in positions]
        values = [abs(pos['market_value']) for pos in positions]
        pnl = [pos.get('unrealized_pnl', 0) for pos in positions]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Pie chart
        colors = plt.cm.Set3(np.linspace(0, 1, len(symbols)))
        wedges, texts, autotexts = ax1.pie(
            values, 
            labels=symbols, 
            autopct='%1.1f%%',
            colors=colors,
            startangle=90
        )
        ax1.set_title('Portfolio Allocation', fontsize=14, fontweight='bold')
        
        # Bar chart with P&L colors
        bar_colors = ['green' if p >= 0 else 'red' for p in pnl]
        bars = ax2.bar(symbols, values, color=bar_colors, alpha=0.7)
        ax2.set_title('Position Values', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Market Value (₹)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add P&L labels on bars
        for bar, p in zip(bars, pnl):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'₹{p:,.0f}', ha='center', va='bottom', fontsize=10)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def save_chart(self, fig: plt.Figure, filename: str, dpi: int = 300) -> None:
        """
        Save chart to file.
        
        Args:
            fig: Matplotlib figure
            filename: Output filename
            dpi: Resolution
        """
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        self.logger.info(f"Chart saved to {filename}")


class InteractiveCharts:
    """Interactive Plotly-based charts for web dashboards."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_dashboard_chart(
        self,
        data: pd.DataFrame,
        chart_type: str = 'candlestick'
    ) -> go.Figure:
        """
        Create interactive dashboard chart.
        
        Args:
            data: Market data DataFrame
            chart_type: Type of chart ('candlestick', 'line', 'area')
            
        Returns:
            Plotly figure
        """
        if chart_type == 'candlestick':
            fig = go.Figure(data=go.Candlestick(
                x=data['date'],
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close']
            ))
        elif chart_type == 'line':
            fig = go.Figure(data=go.Scatter(
                x=data['date'],
                y=data['close'],
                mode='lines',
                name='Close Price'
            ))
        elif chart_type == 'area':
            fig = go.Figure(data=go.Scatter(
                x=data['date'],
                y=data['close'],
                mode='lines',
                fill='tozeroy',
                name='Close Price'
            ))
        
        fig.update_layout(
            title='Interactive Market Chart',
            xaxis_title='Date',
            yaxis_title='Price (₹)',
            xaxis_rangeslider_visible=False
        )
        
        return fig
