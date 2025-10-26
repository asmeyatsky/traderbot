"""
Advanced Charting and Visualization

This module implements advanced charting capabilities for the trading platform
including interactive charts, performance visualization, and risk analytics.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
from datetime import datetime, timedelta
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from io import BytesIO
import base64

from src.domain.entities.trading import Position, Portfolio
from src.domain.entities.user import User
from src.domain.value_objects import Symbol
from src.infrastructure.data_processing.backtesting import BacktestResult, BacktestTrade
from src.domain.ports import MarketDataPort


class ChartGenerator:
    """
    Class to generate various types of trading charts and visualizations.
    """
    
    def __init__(self):
        pass
    
    def generate_candlestick_chart(
        self, 
        symbol: Symbol, 
        data: List[Dict[str, float]], 
        title: str = None
    ) -> str:
        """
        Generate a candlestick chart for a given symbol.
        """
        if not data:
            return ""
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp']) if 'timestamp' in df.columns else pd.date_range(
            start=datetime.now() - timedelta(days=len(data)), 
            periods=len(data),
            freq='D'
        )
        
        # Create candlestick chart
        fig = go.Figure(data=go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close']
        ))
        
        fig.update_layout(
            title=f"{symbol} - Candlestick Chart" if title is None else title,
            xaxis_title="Date",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False
        )
        
        return fig.to_html(include_plotlyjs='cdn')
    
    def generate_portfolio_performance_chart(
        self, 
        portfolio: Portfolio, 
        historical_values: List[Tuple[datetime, float]]
    ) -> str:
        """
        Generate a portfolio performance chart.
        """
        if not historical_values:
            return ""
        
        dates, values = zip(*historical_values)
        
        fig = go.Figure()
        
        # Portfolio value
        fig.add_trace(go.Scatter(
            x=list(dates), 
            y=list(values), 
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Add a baseline at initial value
        initial_value = values[0] if values else 0
        fig.add_trace(go.Scatter(
            x=[dates[0], dates[-1]], 
            y=[initial_value, initial_value], 
            mode='lines',
            name='Initial Value',
            line=dict(color='red', width=1, dash='dash')
        ))
        
        fig.update_layout(
            title="Portfolio Performance Over Time",
            xaxis_title="Date",
            yaxis_title="Value ($)",
            hovermode='x unified'
        )
        
        return fig.to_html(include_plotlyjs='cdn')
    
    def generate_position_allocation_chart(self, portfolio: Portfolio) -> str:
        """
        Generate a pie chart showing position allocation.
        """
        if not portfolio.positions:
            return ""
        
        symbols = []
        values = []
        colors = px.colors.qualitative.Set3
        
        for i, position in enumerate(portfolio.positions):
            symbols.append(str(position.symbol))
            values.append(float(position.market_value.amount))
        
        # Add cash as a position
        if float(portfolio.cash_balance.amount) > 0:
            symbols.append("Cash")
            values.append(float(portfolio.cash_balance.amount))
        
        fig = go.Figure(data=[go.Pie(
            labels=symbols, 
            values=values,
            textinfo='label+percent',
            insidetextorientation='radial',
            marker=dict(colors=colors[:len(symbols)])
        )])
        
        fig.update_layout(
            title="Portfolio Allocation",
            showlegend=False
        )
        
        return fig.to_html(include_plotlyjs='cdn')
    
    def generate_risk_metrics_visualization(
        self, 
        portfolio: Portfolio, 
        risk_metrics: Dict[str, float]
    ) -> str:
        """
        Generate visualization of risk metrics.
        """
        metrics = list(risk_metrics.keys())
        values = list(risk_metrics.values())
        
        # Create gauge chart for key risk metrics
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Volatility', 'Value at Risk', 'Beta', 'Sharpe Ratio'),
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}]]
        )
        
        # Add indicators for each metric
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=risk_metrics.get('volatility', 0) * 100,  # Convert to percentage
            domain={'row': 0, 'column': 0},
            title={'text': "Volatility (%)"},
            gauge={'axis': {'range': [None, 50]},
                   'bar': {'color': "darkblue"},
                   'steps': [
                       {'range': [0, 10], 'color': "lightgreen"},
                       {'range': [10, 25], 'color': "orange"},
                       {'range': [25, 50], 'color': "red"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': risk_metrics.get('volatility', 0) * 100}}
        ), row=1, col=1)
        
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=risk_metrics.get('value_at_risk', 0),
            domain={'row': 0, 'column': 1},
            title={'text': "VaR"},
            gauge={'axis': {'range': [None, portfolio.total_value.amount * 0.5]},
                   'bar': {'color': "darkblue"},
                   'steps': [
                       {'range': [0, portfolio.total_value.amount * 0.1], 'color': "lightgreen"},
                       {'range': [portfolio.total_value.amount * 0.1, portfolio.total_value.amount * 0.2], 'color': "orange"},
                       {'range': [portfolio.total_value.amount * 0.2, portfolio.total_value.amount * 0.5], 'color': "red"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': risk_metrics.get('value_at_risk', 0)}}
        ), row=1, col=2)
        
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=risk_metrics.get('beta', 1.0),
            domain={'row': 1, 'column': 0},
            title={'text': "Beta"},
            gauge={'axis': {'range': [0, 3]},
                   'bar': {'color': "darkblue"},
                   'steps': [
                       {'range': [0, 1], 'color': "lightgreen"},
                       {'range': [1, 2], 'color': "orange"},
                       {'range': [2, 3], 'color': "red"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': risk_metrics.get('beta', 1.0)}}
        ), row=2, col=1)
        
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=risk_metrics.get('sharpe_ratio', 0.0),
            domain={'row': 1, 'column': 1},
            title={'text': "Sharpe Ratio"},
            gauge={'axis': {'range': [-2, 3]},
                   'bar': {'color': "darkblue"},
                   'steps': [
                       {'range': [-2, 0], 'color': "red"},
                       {'range': [0, 1.5], 'color': "orange"},
                       {'range': [1.5, 3], 'color': "lightgreen"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': risk_metrics.get('sharpe_ratio', 0.0)}}
        ), row=2, col=2)
        
        fig.update_layout(
            title="Risk Metrics Dashboard"
        )
        
        return fig.to_html(include_plotlyjs='cdn')
    
    def generate_backtest_results_chart(self, backtest_result: BacktestResult) -> str:
        """
        Generate charts for backtest results.
        """
        if not backtest_result.trades:
            return ""
        
        # Create subplots for equity curve and drawdown
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Equity Curve', 'Drawdown'),
            row_width=[0.7, 0.3]
        )
        
        # Calculate equity curve
        equity_values = [backtest_result.initial_capital]
        for trade in backtest_result.trades:
            equity_values.append(equity_values[-1] + trade.pnl)
        
        # Calculate drawdown
        running_max = [equity_values[0]]
        drawdowns = [0]
        for value in equity_values[1:]:
            running_max.append(max(running_max[-1], value))
            drawdown = (running_max[-1] - value) / running_max[-1] if running_max[-1] > 0 else 0
            drawdowns.append(drawdown)
        
        # Add equity curve
        fig.add_trace(
            go.Scatter(
                x=list(range(len(equity_values))),
                y=equity_values,
                mode='lines',
                name='Equity Curve',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Add drawdown
        fig.add_trace(
            go.Scatter(
                x=list(range(len(drawdowns))),
                y=[d * 100 for d in drawdowns],  # Convert to percentage
                mode='lines',
                name='Drawdown (%)',
                line=dict(color='red', width=1)
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f"Backtest Results: {backtest_result.strategy_name}",
            height=600
        )
        
        fig.update_xaxes(title_text="Trade Number", row=2, col=1)
        fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        
        return fig.to_html(include_plotlyjs='cdn')
    
    def generate_trading_performance_summary(self, backtest_result: BacktestResult) -> str:
        """
        Generate a summary of trading performance metrics.
        """
        fig = go.Figure()
        
        # Prepare data
        metrics = [
            'Total Return %',
            'Sharpe Ratio',
            'Max Drawdown %',
            'Win Rate %',
            'Profit Factor'
        ]
        
        values = [
            backtest_result.total_return_pct,
            backtest_result.sharpe_ratio,
            -backtest_result.max_drawdown * 100,  # Convert to positive percentage
            backtest_result.win_rate,
            (backtest_result.avg_win * backtest_result.winning_trades) / 
            (backtest_result.avg_loss * backtest_result.losing_trades) if backtest_result.avg_loss > 0 else 0
        ]
        
        colors = ['positive' if v >= 0 else 'negative' for v in values[:4]] + ['positive']
        color_map = {'positive': 'lightgreen', 'negative': 'lightcoral'}
        bar_colors = [color_map[c] if isinstance(c, str) else 'lightblue' for c in colors]
        
        fig.add_trace(go.Bar(
            x=metrics,
            y=values,
            marker_color=bar_colors
        ))
        
        fig.update_layout(
            title="Trading Performance Metrics",
            yaxis_title="Value",
            xaxis_tickangle=-45
        )
        
        return fig.to_html(include_plotlyjs='cdn')
    
    def generate_technical_indicators_chart(
        self, 
        symbol: Symbol, 
        price_data: List[Dict[str, Any]], 
        indicators: Dict[str, List[float]]
    ) -> str:
        """
        Generate a chart with technical indicators.
        """
        if not price_data:
            return ""
        
        # Convert to DataFrame
        df = pd.DataFrame(price_data)
        df['timestamp'] = pd.to_datetime(df['timestamp']) if 'timestamp' in df.columns else pd.date_range(
            start=datetime.now() - timedelta(days=len(price_data)), 
            periods=len(price_data),
            freq='D'
        )
        
        # Create subplots: price on main chart, indicators on subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_width=[0.7, 0.15, 0.15],
            subplot_titles=(f'{symbol} Price', 'RSI', 'Volume')
        )
        
        # Add price line
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['close'],
                mode='lines',
                name='Close Price',
                line=dict(color='black')
            ),
            row=1, col=1
        )
        
        # Add Moving Averages if provided
        if 'sma_20' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['sma_20'],
                    mode='lines',
                    name='SMA 20',
                    line=dict(color='blue', dash='dash')
                ),
                row=1, col=1
            )
        
        if 'sma_50' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['sma_50'],
                    mode='lines',
                    name='SMA 50',
                    line=dict(color='orange', dash='dash')
                ),
                row=1, col=1
            )
        
        # Add RSI if provided
        if 'rsi' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['rsi'],
                    mode='lines',
                    name='RSI',
                    line=dict(color='purple')
                ),
                row=2, col=1
            )
            
            # Add RSI overbought/oversold lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # Add volume if provided
        if 'volume' in df.columns:
            fig.add_trace(
                go.Bar(
                    x=df['timestamp'],
                    y=df['volume'],
                    name='Volume',
                    marker_color='lightblue'
                ),
                row=3, col=1
            )
        
        fig.update_layout(
            title=f"Technical Analysis for {symbol}",
            height=700
        )
        
        return fig.to_html(include_plotlyjs='cdn')


class DashboardVisualizer:
    """
    Integrates charts into a comprehensive dashboard.
    """
    
    def __init__(self):
        self.chart_generator = ChartGenerator()
    
    def generate_portfolio_dashboard(self, user: User, portfolio: Portfolio) -> str:
        """
        Generate a complete portfolio dashboard with multiple charts.
        """
        html_parts = []
        html_parts.append(f"<h2>Portfolio Dashboard for {user.first_name} {user.last_name}</h2>")
        
        # Add performance chart
        # This would use historical portfolio value data in a real implementation
        historical_values = [(datetime.now() - timedelta(days=i), float(portfolio.total_value.amount)) 
                            for i in range(30, 0, -1)]
        performance_chart = self.chart_generator.generate_portfolio_performance_chart(
            portfolio, historical_values
        )
        html_parts.append(performance_chart)
        
        # Add allocation chart
        allocation_chart = self.chart_generator.generate_position_allocation_chart(portfolio)
        html_parts.append(allocation_chart)
        
        # Calculate and add risk metrics visualization
        # In a real implementation, these would come from a risk service
        risk_metrics = {
            'volatility': 0.15,
            'value_at_risk': float(portfolio.total_value.amount) * 0.05,
            'beta': 1.0,
            'sharpe_ratio': 0.8
        }
        risk_chart = self.chart_generator.generate_risk_metrics_visualization(portfolio, risk_metrics)
        html_parts.append(risk_chart)
        
        # Combine all HTML parts
        full_html = "<div>" + "".join(html_parts) + "</div>"
        return full_html
    
    def generate_backtest_dashboard(self, backtest_result: BacktestResult) -> str:
        """
        Generate a backtest results dashboard.
        """
        html_parts = []
        html_parts.append(f"<h2>Backtest Results: {backtest_result.strategy_name}</h2>")
        
        # Add performance chart
        equity_chart = self.chart_generator.generate_backtest_results_chart(backtest_result)
        html_parts.append(equity_chart)
        
        # Add performance metrics
        metrics_chart = self.chart_generator.generate_trading_performance_summary(backtest_result)
        html_parts.append(metrics_chart)
        
        # Combine all HTML parts
        full_html = "<div>" + "".join(html_parts) + "</div>"
        return full_html


class ChartAPI:
    """
    API interface for generating charts programmatically.
    """
    
    def __init__(self):
        self.visualizer = DashboardVisualizer()
        self.chart_generator = ChartGenerator()
    
    def get_portfolio_charts(self, portfolio_id: str) -> Dict[str, str]:
        """
        Get all portfolio charts as a dictionary of HTML strings.
        """
        # This would normally fetch portfolio data from a repository
        # For now, returning empty placeholders
        return {
            "performance_chart": "",
            "allocation_chart": "",
            "risk_chart": ""
        }
    
    def get_symbol_chart(self, symbol: Symbol, days: int = 30) -> str:
        """
        Get a chart for a specific symbol.
        """
        # This would fetch real market data
        # Creating mock data for demonstration
        dates = pd.date_range(end=datetime.now(), periods=days)
        prices = 100 + np.cumsum(np.random.randn(days) * 0.5)
        
        data = []
        for i, date in enumerate(dates):
            price = prices[i]
            data.append({
                'timestamp': date,
                'open': price * (1 + np.random.uniform(-0.02, 0.02)),
                'high': price * (1 + np.random.uniform(0, 0.03)),
                'low': price * (1 - np.random.uniform(0, 0.03)),
                'close': price
            })
        
        return self.chart_generator.generate_candlestick_chart(symbol, data)