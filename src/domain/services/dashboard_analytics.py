"""
Enhanced Dashboard Analytics Service

Implements advanced dashboard features including technical indicators,
portfolio analytics, and customizable alerts
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from decimal import Decimal
import numpy as np
import pandas as pd

from src.domain.entities.trading import Portfolio, Position, Order
from src.domain.entities.user import User
from src.domain.value_objects import Money, Price, Symbol


@dataclass
class TechnicalIndicator:
    """Data class for technical indicators"""
    symbol: Symbol
    sma_20: Optional[Decimal] = None
    sma_50: Optional[Decimal] = None
    ema_12: Optional[Decimal] = None
    ema_26: Optional[Decimal] = None
    rsi: Optional[Decimal] = None
    macd: Optional[Decimal] = None
    macd_signal: Optional[Decimal] = None
    bollinger_upper: Optional[Decimal] = None
    bollinger_lower: Optional[Decimal] = None
    atr: Optional[Decimal] = None
    calculated_at: Optional[datetime] = None


@dataclass
class DashboardMetrics:
    """Data class for dashboard metrics"""
    total_value: Money
    daily_pnl: Money
    daily_pnl_percentage: Decimal
    positions_count: int
    active_orders_count: int
    unrealized_pnl: Money
    realized_pnl: Money
    top_gainers: List[Tuple[Symbol, Decimal]]  # (symbol, percentage_gain)
    top_losers: List[Tuple[Symbol, Decimal]]   # (symbol, percentage_loss)
    allocation_by_sector: Dict[str, Decimal]  # sector -> percentage
    allocation_by_asset: Dict[Symbol, Decimal]  # symbol -> percentage
    technical_indicators: List[TechnicalIndicator]
    risk_metrics: Dict[str, Decimal]  # metric_name -> value
    performance_chart_data: List[Dict[str, Decimal]]  # Historical portfolio values


class DashboardAnalyticsService(ABC):
    """
    Abstract base class for dashboard analytics services.
    """
    
    @abstractmethod
    def get_dashboard_metrics(self, portfolio: Portfolio, user: User) -> DashboardMetrics:
        """Calculate comprehensive dashboard metrics"""
        pass
    
    @abstractmethod
    def calculate_technical_indicators(self, symbol: Symbol, days: int = 90) -> TechnicalIndicator:
        """Calculate technical indicators for a symbol"""
        pass
    
    @abstractmethod
    def get_performance_chart_data(self, portfolio: Portfolio, days: int = 30) -> List[Dict[str, Decimal]]:
        """Get historical portfolio performance data for charting"""
        pass
    
    @abstractmethod
    def generate_sector_allocation(self, portfolio: Portfolio) -> Dict[str, Decimal]:
        """Generate sector allocation breakdown (mock implementation)"""
        pass


class DefaultDashboardAnalyticsService(DashboardAnalyticsService):
    """
    Default implementation of dashboard analytics services.
    Note: This is a simplified implementation using mock data - in production,
    this would connect to market data and use actual historical prices
    """
    
    def __init__(self):
        self._mock_price_history = self._generate_mock_price_history()
    
    def _generate_mock_price_history(self) -> Dict[str, List[Decimal]]:
        """Generate mock historical price data for common symbols"""
        np.random.seed(42)  # For reproducible mock results
        
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'SPY', 'QQQ', 'DIS', 'MCD']
        history = {}
        
        for symbol in symbols:
            # Generate mock historical prices (simulating realistic price movements)
            base_price = np.random.uniform(50, 300)  # Random base price
            prices = [Decimal(str(base_price))]
            
            for _ in range(90):  # 90 days of data
                # Generate price change (typically 0-3% daily movement)
                change_pct = np.random.normal(0.001, 0.02)  # 0.1% average daily return, 2% std dev
                new_price = prices[-1] * Decimal(str(1 + change_pct))
                
                # Ensure price doesn't go below $1
                if new_price < Decimal('1.0'):
                    new_price = Decimal('1.0')
                
                prices.append(new_price)
            
            history[symbol] = prices
        
        return history
    
    def get_dashboard_metrics(self, portfolio: Portfolio, user: User) -> DashboardMetrics:
        """
        Calculate comprehensive dashboard metrics
        """
        # Calculate portfolio total value
        total_value = portfolio.total_value
        
        # Calculate daily P&L (mock - would use previous day's value in production)
        daily_pnl = Money(Decimal(str(round(np.random.uniform(-1000, 2000), 2))), 'USD')
        daily_pnl_percentage = (daily_pnl.amount / total_value.amount) * 100 if total_value.amount > 0 else Decimal('0')
        
        # Count positions and active orders (mock values)
        positions_count = len(portfolio.positions)
        active_orders_count = np.random.randint(0, 5)  # Mock count of active orders
        
        # Calculate P&L metrics
        unrealized_pnl = Money(Decimal('0'), 'USD')
        realized_pnl = Money(Decimal('0'), 'USD')
        
        for position in portfolio.positions:
            pnl = position.unrealized_pnl_amount
            if pnl.is_positive():
                unrealized_pnl = Money(unrealized_pnl.amount + pnl.amount, pnl.currency)
            else:
                unrealized_pnl = Money(unrealized_pnl.amount + pnl.amount, pnl.currency)
        
        # Calculate top gainers and losers
        top_gainers = []
        top_losers = []
        
        for position in portfolio.positions:
            if position.average_buy_price.amount > 0:
                pct_change = ((position.current_price.amount - position.average_buy_price.amount) / 
                             position.average_buy_price.amount) * 100
                symbol = position.symbol
                
                if pct_change > 0:
                    top_gainers.append((symbol, pct_change))
                else:
                    top_losers.append((symbol, abs(pct_change)))
        
        # Sort and limit to top 3
        top_gainers = sorted(top_gainers, key=lambda x: x[1], reverse=True)[:3]
        top_losers = sorted(top_losers, key=lambda x: x[1], reverse=True)[:3]
        
        # Generate allocation data
        allocation_by_asset = {}
        if total_value.amount > 0:
            for position in portfolio.positions:
                allocation = (position.market_value.amount / total_value.amount) * 100
                allocation_by_asset[position.symbol] = allocation
        
        # Mock sector allocation (in reality, this would map symbols to sectors)
        allocation_by_sector = {
            "Technology": Decimal('45.0'),
            "Healthcare": Decimal('20.0'),
            "Financials": Decimal('15.0'),
            "Consumer": Decimal('12.0'),
            "Industrials": Decimal('8.0')
        }
        
        # Calculate technical indicators for portfolio holdings
        technical_indicators = []
        for position in portfolio.positions:
            tech_ind = self.calculate_technical_indicators(position.symbol)
            technical_indicators.append(tech_ind)
        
        # Mock risk metrics
        risk_metrics = {
            "volatility_30d": Decimal('1.2'),
            "max_drawdown": Decimal('8.5'),
            "sharpe_ratio": Decimal('1.4'),
            "beta": Decimal('1.1')
        }
        
        # Performance chart data
        performance_chart_data = self.get_performance_chart_data(portfolio)
        
        return DashboardMetrics(
            total_value=total_value,
            daily_pnl=daily_pnl,
            daily_pnl_percentage=daily_pnl_percentage,
            positions_count=positions_count,
            active_orders_count=active_orders_count,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            top_gainers=top_gainers,
            top_losers=top_losers,
            allocation_by_sector=allocation_by_sector,
            allocation_by_asset=allocation_by_asset,
            technical_indicators=technical_indicators,
            risk_metrics=risk_metrics,
            performance_chart_data=performance_chart_data
        )
    
    def calculate_technical_indicators(self, symbol: Symbol, days: int = 90) -> TechnicalIndicator:
        """
        Calculate technical indicators for a symbol
        """
        # Generate mock price data for this symbol
        if str(symbol) not in self._mock_price_history:
            # If symbol not in mock data, create mock data
            np.random.seed(hash(str(symbol)) % 2**32)  # Deterministic seed based on symbol
            base_price = np.random.uniform(50, 300)
            prices = [Decimal(str(base_price))]
            for _ in range(90):
                change_pct = np.random.normal(0.001, 0.02)
                new_price = prices[-1] * Decimal(str(1 + change_pct))
                if new_price < Decimal('1.0'):
                    new_price = Decimal('1.0')
                prices.append(new_price)
            self._mock_price_history[str(symbol)] = prices
        else:
            prices = self._mock_price_history[str(symbol)]
        
        # Convert to numpy array for calculations
        prices_array = np.array([float(p) for p in prices[-days:]])
        
        # Calculate Simple Moving Averages
        sma_20 = None
        sma_50 = None
        
        if len(prices_array) >= 20:
            sma_20 = Decimal(str(round(prices_array[-20:].mean(), 2)))
        
        if len(prices_array) >= 50:
            sma_50 = Decimal(str(round(prices_array[-50:].mean(), 2)))
        
        # Calculate Exponential Moving Averages
        ema_12 = None
        ema_26 = None
        
        if len(prices_array) >= 12:
            ema_12 = Decimal(str(round(pd.Series(prices_array).ewm(span=12).mean().iloc[-1], 2)))
        
        if len(prices_array) >= 26:
            ema_26 = Decimal(str(round(pd.Series(prices_array).ewm(span=26).mean().iloc[-1], 2)))
        
        # Calculate RSI
        rsi = None
        if len(prices_array) > 14:
            delta = np.diff(prices_array)
            gains = np.where(delta > 0, delta, 0)
            losses = np.where(delta < 0, -delta, 0)
            
            avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else Decimal('0')
            avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else Decimal('0')
            
            if avg_loss != 0:
                rs = avg_gain / avg_loss
                rsi = Decimal(str(round(100 - (100 / (1 + rs)), 2)))
            else:
                rsi = Decimal('100')  # If no losses, RSI is 100
        
        # Calculate MACD
        macd = None
        macd_signal = None
        if ema_12 and ema_26:
            macd = ema_12 - ema_26
            # Simplified MACD signal line (9-day EMA of MACD)
            macd_signal = macd * Decimal('0.7')  # Mock signal line calculation
        
        # Calculate Bollinger Bands
        bollinger_upper = None
        bollinger_lower = None
        if sma_20:
            std_20 = Decimal(str(round(np.std(prices_array[-20:]), 2)))
            bollinger_upper = sma_20 + (std_20 * 2)
            bollinger_lower = sma_20 - (std_20 * 2)
        
        # Calculate Average True Range (ATR)
        atr = None
        if len(prices_array) > 14:
            # Simplified ATR calculation (using daily range)
            daily_ranges = np.abs(np.diff(prices_array))
            if len(daily_ranges) >= 14:
                atr = Decimal(str(round(np.mean(daily_ranges[-14:]), 2)))
        
        return TechnicalIndicator(
            symbol=symbol,
            sma_20=sma_20,
            sma_50=sma_50,
            ema_12=ema_12,
            ema_26=ema_26,
            rsi=rsi,
            macd=macd,
            macd_signal=macd_signal,
            bollinger_upper=bollinger_upper,
            bollinger_lower=bollinger_lower,
            atr=atr,
            calculated_at=datetime.now()
        )
    
    def get_performance_chart_data(self, portfolio: Portfolio, days: int = 30) -> List[Dict[str, Decimal]]:
        """
        Get historical portfolio performance data for charting
        """
        # Generate mock historical portfolio values
        current_value = float(portfolio.total_value.amount)
        chart_data = []
        
        # Generate data points going back 'days' days
        np.random.seed(42)  # For consistent mock data
        
        for i in range(days, 0, -1):
            # Simulate daily fluctuations around current value
            day_change = np.random.uniform(-0.02, 0.03)  # -2% to +3% daily change
            value = current_value * (1 + day_change)
            
            # Add some trend to make it more realistic
            trend_factor = 1 + (i / days) * 0.001  # Slight upward bias
            value = value * trend_factor
            
            chart_data.append({
                "date": (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d"),
                "value": Decimal(str(round(value, 2)))
            })
            
            current_value = value
        
        return chart_data
    
    def generate_sector_allocation(self, portfolio: Portfolio) -> Dict[str, Decimal]:
        """
        Generate sector allocation breakdown
        Note: This is a mock implementation that assigns sectors randomly
        In production, this would use actual sector mappings
        """
        sectors = ["Technology", "Healthcare", "Financials", "Consumer", "Industrials", "Utilities", "Energy"]
        allocation = {}
        
        total_value = portfolio.total_value.amount
        if total_value <= 0:
            return allocation
        
        remaining_percentage = Decimal('100.0')
        
        # Assign percentages to each sector except the last one
        for sector in sectors[:-1]:
            # Assign a random percentage between 5% and 25%
            percentage = min(remaining_percentage - Decimal(str(len(sectors) - len(allocation) - 1)) * Decimal('5'), 
                           Decimal(str(round(np.random.uniform(5, 25), 2))))
            allocation[sector] = percentage
            remaining_percentage -= percentage
        
        # Assign remaining percentage to the last sector
        allocation[sectors[-1]] = remaining_percentage if remaining_percentage > 0 else Decimal('0')
        
        return allocation