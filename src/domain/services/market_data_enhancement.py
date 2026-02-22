"""
Market Data Enhancement Service

Implements enhanced market data capabilities including 
additional data providers, news sentiment scoring, and economic calendar integration
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from decimal import Decimal
import numpy as np
import requests
from enum import Enum

from src.domain.value_objects import Symbol, Price, NewsSentiment
from src.domain.entities.trading import Order
from src.domain.ports import MarketDataPort


class MarketDataSource(Enum):
    POLYGON = "polygon"
    ALPHA_VANTAGE = "alpha_vantage"
    YAHOO_FINANCE = "yahoo_finance"
    FINNHUB = "finnhub"
    MOCK_DATA = "mock_data"


@dataclass
class MarketDataPoint:
    """Data class for a single market data point"""
    symbol: Symbol
    price: Price
    volume: int
    timestamp: datetime
    source: MarketDataSource
    high: Optional[Price] = None
    low: Optional[Price] = None
    open: Optional[Price] = None
    close: Optional[Price] = None
    change: Optional[Decimal] = None
    change_percent: Optional[Decimal] = None


@dataclass
class NewsArticle:
    """Data class for news articles"""
    id: str
    title: str
    summary: str
    content: str
    source: str
    published_at: datetime
    symbols: List[Symbol]
    sentiment: NewsSentiment
    relevance_score: Decimal  # 0-100, how relevant to the symbols


@dataclass
class EconomicEvent:
    """Data class for economic calendar events"""
    event_id: str
    event_name: str
    country: str
    date: datetime
    impact_level: str  # 'low', 'medium', 'high'
    forecast_value: Optional[str] = None
    previous_value: Optional[str] = None
    actual_value: Optional[str] = None
    currency: Optional[str] = None


@dataclass
class EnhancedMarketData:
    """Data class for enhanced market data"""
    symbol: Symbol
    current_price: Price
    market_data_points: List[MarketDataPoint]
    news_sentiment: List[NewsArticle]
    volatility_forecast: Optional[Decimal] = None
    economic_events: List[EconomicEvent] = None
    technical_signals: Dict[str, str] = None  # e.g., {"RSI": "OVERSOLD", "MACD": "BULLISH"}


class MarketDataEnhancementService(ABC):
    """
    Abstract base class for market data enhancement services.
    """
    
    @abstractmethod
    def get_enhanced_market_data(self, symbol: Symbol, 
                                sources: List[MarketDataSource] = None) -> EnhancedMarketData:
        """Get enhanced market data from multiple sources"""
        pass
    
    @abstractmethod
    def get_news_sentiment(self, symbol: Symbol, days: int = 7) -> List[NewsArticle]:
        """Get news sentiment for a symbol"""
        pass
    
    @abstractmethod
    def get_economic_calendar(self, days_ahead: int = 7) -> List[EconomicEvent]:
        """Get economic calendar events"""
        pass
    
    @abstractmethod
    def calculate_volatility_forecast(self, symbol: Symbol, lookback_days: int = 30) -> Decimal:
        """Calculate volatility forecast for a symbol"""
        pass
    
    @abstractmethod
    def get_technical_signals(self, symbol: Symbol) -> Dict[str, str]:
        """Get technical analysis signals for a symbol"""
        pass


class DefaultMarketDataEnhancementService(MarketDataEnhancementService, MarketDataPort):
    """
    Default implementation of market data enhancement services.
    Note: This is a simplified implementation using mock data - in production,
    this would connect to real market data APIs
    """
    
    def __init__(self):
        self._mock_prices = self._generate_mock_prices()
        self._mock_news = self._generate_mock_news()
        self._mock_economic_events = self._generate_mock_economic_events()
    
    # ------------------------------------------------------------------
    # MarketDataPort implementation
    # ------------------------------------------------------------------

    def get_current_price(self, symbol: Symbol) -> Optional[Price]:
        """Get mock current price for a symbol."""
        points = self._mock_prices.get(str(symbol), [])
        if points:
            return points[-1].price
        # Return a deterministic fallback for unknown symbols
        return Price(Decimal('100.00'), 'USD')

    def get_historical_prices(self, symbol: Symbol, start_date=None, end_date=None) -> List[Price]:
        """Get mock historical prices for a symbol."""
        points = self._mock_prices.get(str(symbol), [])
        return [p.price for p in points]

    def get_market_news(self, symbol: Symbol) -> List[str]:
        """Get mock news headlines for a symbol."""
        articles = self._mock_news.get(str(symbol), [])
        return [a.title for a in articles]

    # ------------------------------------------------------------------
    # Internal data generation
    # ------------------------------------------------------------------

    def _generate_mock_prices(self) -> Dict[str, List[MarketDataPoint]]:
        """Generate mock historical price data"""
        np.random.seed(42)  # For reproducible mock results
        
        symbols = ['AAPL', 'GOOG', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NVDA', 'TSLA', 'SPY', 'QQQ', 'DIS', 'MCD']
        data = {}
        
        for symbol_str in symbols:
            symbol = Symbol(symbol_str)
            base_price = np.random.uniform(50, 300)
            price_points = []
            
            for i in range(30):  # 30 days of data
                # Generate realistic price movement
                change_pct = np.random.normal(0.001, 0.02)  # 0.1% average daily return, 2% std dev
                current_price = base_price * (1 + change_pct)
                
                # Update base price for next iteration
                base_price = current_price
                
                price_points.append(
                    MarketDataPoint(
                        symbol=symbol,
                        price=Price(Decimal(str(round(current_price, 2))), 'USD'),
                        volume=np.random.randint(1000000, 10000000),
                        timestamp=datetime.now() - timedelta(days=30-i),
                        source=MarketDataSource.MOCK_DATA,
                        high=Price(Decimal(str(round(current_price * 1.02, 2))), 'USD'),
                        low=Price(Decimal(str(round(current_price * 0.98, 2))), 'USD'),
                        open=Price(Decimal(str(round(current_price * 0.995, 2))), 'USD'),
                        close=Price(Decimal(str(round(current_price, 2))), 'USD'),
                        change=Decimal(str(round(change_pct * 100, 4))),
                        change_percent=Decimal(str(round(change_pct * 100, 4)))
                    )
                )
            
            data[symbol_str] = price_points
        
        return data
    
    def _generate_mock_news(self) -> Dict[str, List[NewsArticle]]:
        """Generate mock news data"""
        symbols = ['AAPL', 'GOOG', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NVDA', 'TSLA']
        news_data = {}
        
        for symbol in symbols:
            news_articles = []
            for i in range(5):  # 5 mock articles per symbol
                # Generate mock sentiment score
                sentiment_score = Decimal(str(round(np.random.uniform(-50, 50), 2)))
                
                news_articles.append(
                    NewsArticle(
                        id=f"news_{symbol}_{i}",
                        title=f"Mock news title for {symbol} - {i}",
                        summary=f"Summary of news article for {symbol} showing market movements",
                        content=f"Full content of the news article about {symbol} and how it might affect the market...",
                        source="Mock News Source",
                        published_at=datetime.now() - timedelta(hours=np.random.randint(1, 24)),
                        symbols=[Symbol(symbol)],
                        sentiment=NewsSentiment(
                            score=sentiment_score,
                            confidence=Decimal('85.0'),
                            source="Mock Analyzer"
                        ),
                        relevance_score=Decimal(str(round(np.random.uniform(70, 100), 2)))
                    )
                )
            
            news_data[symbol] = news_articles
        
        return news_data
    
    def _generate_mock_economic_events(self) -> List[EconomicEvent]:
        """Generate mock economic calendar events"""
        events = []
        
        # Generate some mock economic events
        event_names = [
            "Non-Farm Employment Change", "Consumer Price Index", "GDP Growth Rate",
            "Federal Reserve Interest Rate Decision", "Retail Sales", "Manufacturing PMI"
        ]
        
        for i, name in enumerate(event_names):
            events.append(
                EconomicEvent(
                    event_id=f"event_{i}",
                    event_name=name,
                    country="US",
                    date=datetime.now() + timedelta(days=np.random.randint(0, 14)),
                    impact_level=np.random.choice(["low", "medium", "high"]),
                    forecast_value=str(round(np.random.uniform(-2, 4), 2)),
                    previous_value=str(round(np.random.uniform(-2, 4), 2)),
                    currency="USD"
                )
            )
        
        return events
    
    def get_enhanced_market_data(self, symbol: Symbol, 
                                sources: List[MarketDataSource] = None) -> EnhancedMarketData:
        """
        Get enhanced market data from multiple sources
        """
        if sources is None:
            sources = [MarketDataSource.MOCK_DATA]  # Default to mock data
        
        # Get current price (last in our mock data)
        current_price = self._mock_prices.get(str(symbol), [])[-1].price if self._mock_prices.get(str(symbol)) else Price(Decimal('100.00'), 'USD')
        
        # Get historical data points
        market_data_points = self._mock_prices.get(str(symbol), [])
        
        # Get news sentiment
        news_sentiment = self.get_news_sentiment(symbol)
        
        # Calculate volatility forecast
        volatility = self.calculate_volatility_forecast(symbol)
        
        # Get technical signals
        technical_signals = self.get_technical_signals(symbol)
        
        # Get upcoming economic events
        economic_events = self.get_economic_calendar(7)
        
        return EnhancedMarketData(
            symbol=symbol,
            current_price=current_price,
            market_data_points=market_data_points,
            news_sentiment=news_sentiment,
            volatility_forecast=volatility,
            economic_events=economic_events,
            technical_signals=technical_signals
        )
    
    def get_news_sentiment(self, symbol: Symbol, days: int = 7) -> List[NewsArticle]:
        """
        Get news sentiment for a symbol
        """
        # Return mock news for this symbol
        symbol_str = str(symbol)
        return self._mock_news.get(symbol_str, [])
    
    def get_economic_calendar(self, days_ahead: int = 7) -> List[EconomicEvent]:
        """
        Get economic calendar events
        """
        # Filter events within the specified range
        cutoff_date = datetime.now() + timedelta(days=days_ahead)
        return [event for event in self._mock_economic_events if event.date <= cutoff_date]
    
    def calculate_volatility_forecast(self, symbol: Symbol, lookback_days: int = 30) -> Decimal:
        """
        Calculate volatility forecast for a symbol
        """
        # Mock calculation based on historical data
        price_data = self._mock_prices.get(str(symbol), [])
        if not price_data:
            return Decimal('0.20')  # Default 20% volatility
        
        # Calculate daily returns volatility
        prices = [point.price.amount for point in price_data[-lookback_days:]]
        if len(prices) < 2:
            return Decimal('0.20')
        
        returns = []
        for i in range(1, len(prices)):
            ret = (prices[i] - prices[i-1]) / prices[i-1]
            returns.append(float(ret))
        
        # Calculate annualized volatility
        if returns:
            daily_vol = np.std(returns)
            annualized_vol = daily_vol * Decimal(str(np.sqrt(252)))  # 252 trading days
            return min(annualized_vol, Decimal('1.0'))  # Cap at 100%
        
        return Decimal('0.20')
    
    def get_technical_signals(self, symbol: Symbol) -> Dict[str, str]:
        """
        Get technical analysis signals for a symbol
        """
        # Mock technical signals
        signals = {}
        
        # Generate random signals for demo
        if np.random.random() > 0.5:
            signals["RSI"] = "OVERSOLD" if np.random.random() > 0.5 else "OVERBOUGHT"
        else:
            signals["RSI"] = "NEUTRAL"
        
        if np.random.random() > 0.5:
            signals["MACD"] = "BULLISH" if np.random.random() > 0.5 else "BEARISH"
        else:
            signals["MACD"] = "NEUTRAL"
        
        if np.random.random() > 0.5:
            signals["Moving_Average"] = "BULLISH" if np.random.random() > 0.5 else "BEARISH"
        else:
            signals["Moving_Average"] = "NEUTRAL"
        
        signals["Bollinger_Bands"] = "NEUTRAL"
        
        return signals