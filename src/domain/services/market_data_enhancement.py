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
    url: str = ""


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


class DefaultMarketDataEnhancementService(MarketDataEnhancementService):
    """
    Enhanced market data service that delegates to a real MarketDataPort
    for live prices and enriches with news, technical signals, and economic data.

    When no real market data provider is available, falls back to mock data.
    """

    def __init__(self, market_data_provider: Optional[MarketDataPort] = None,
                 sentiment_service=None):
        self._provider = market_data_provider
        self._sentiment_service = sentiment_service
        self._mock_economic_events = self._generate_mock_economic_events()

    def _get_live_price(self, symbol: Symbol) -> Optional[Price]:
        """Fetch live price from the real provider, return None on failure."""
        if self._provider is None:
            return None
        try:
            return self._provider.get_current_price(symbol)
        except Exception:
            return None

    def _fetch_historical_data(self, symbol: Symbol, days: int = 30) -> List[MarketDataPoint]:
        """Fetch historical price data from the real provider with full OHLCV."""
        if self._provider is None:
            return []
        try:
            from datetime import date as date_type
            from src.infrastructure.api_clients.market_data import YahooFinanceAdapter
            end = date_type.today()
            start = end - timedelta(days=days)

            # Use OHLCV endpoint when Yahoo adapter is available
            yahoo_adapter = None
            if isinstance(self._provider, YahooFinanceAdapter):
                yahoo_adapter = self._provider
            elif hasattr(self._provider, '_adapters'):
                for adapter in self._provider._adapters:
                    if isinstance(adapter, YahooFinanceAdapter):
                        yahoo_adapter = adapter
                        break

            if yahoo_adapter is not None:
                ohlcv = yahoo_adapter.get_historical_ohlcv(symbol, start, end)
                if ohlcv:
                    points = []
                    for bar in ohlcv:
                        close_price = Price(amount=Decimal(str(round(bar['close'], 4))), currency='USD')
                        points.append(
                            MarketDataPoint(
                                symbol=symbol,
                                price=close_price,
                                volume=bar['volume'],
                                timestamp=bar['date'] if isinstance(bar['date'], datetime) else datetime.combine(bar['date'], datetime.min.time()),
                                source=MarketDataSource.YAHOO_FINANCE,
                                open=Price(amount=Decimal(str(round(bar['open'], 4))), currency='USD'),
                                high=Price(amount=Decimal(str(round(bar['high'], 4))), currency='USD'),
                                low=Price(amount=Decimal(str(round(bar['low'], 4))), currency='USD'),
                                close=close_price,
                            )
                        )
                    return points

            # Fallback: close-only data from generic provider
            prices = self._provider.get_historical_prices(symbol, start, end)
            points = []
            for i, price in enumerate(prices):
                points.append(
                    MarketDataPoint(
                        symbol=symbol,
                        price=price,
                        volume=0,
                        timestamp=datetime.now() - timedelta(days=len(prices) - i),
                        source=MarketDataSource.YAHOO_FINANCE,
                        close=price,
                    )
                )
            return points
        except Exception:
            return []
    
    def _fetch_yahoo_news(self, symbol: Symbol) -> List[NewsArticle]:
        """Fetch real news articles from Yahoo Finance via yfinance."""
        try:
            import yfinance as yf
        except ImportError:
            return []

        try:
            ticker = yf.Ticker(str(symbol))
            raw_news = ticker.news
            if not raw_news:
                return []

            articles = []
            for i, item in enumerate(raw_news[:10]):
                title = item.get('title', '')
                if not title:
                    continue

                published_ts = item.get('providerPublishTime', 0)
                published_at = (
                    datetime.fromtimestamp(published_ts)
                    if published_ts
                    else datetime.now() - timedelta(hours=i)
                )

                sentiment = self._analyze_article_sentiment(title)

                articles.append(
                    NewsArticle(
                        id=item.get('uuid', f"yahoo_{symbol}_{i}"),
                        title=title,
                        summary=title,
                        content='',
                        source=item.get('publisher', 'Yahoo Finance'),
                        published_at=published_at,
                        symbols=[symbol],
                        sentiment=sentiment,
                        relevance_score=Decimal('80'),
                        url=item.get('link', ''),
                    )
                )
            return articles
        except Exception:
            return []

    def _analyze_article_sentiment(self, text: str) -> NewsSentiment:
        """Score article text using the injected sentiment service, with a neutral fallback."""
        if self._sentiment_service is not None:
            try:
                return self._sentiment_service.analyze_sentiment(text)
            except Exception:
                pass
        return NewsSentiment(
            score=Decimal('0'),
            confidence=Decimal('30'),
            source='none',
        )

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
        Get enhanced market data from real API providers with enrichment.
        """
        # Fetch live current price
        live_price = self._get_live_price(symbol)
        current_price = live_price if live_price else Price(Decimal('0.00'), 'USD')

        # Fetch historical data points from real provider
        market_data_points = self._fetch_historical_data(symbol)

        # Get news sentiment
        news_sentiment = self.get_news_sentiment(symbol)

        # Calculate volatility forecast from historical data
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
        Get news for a symbol. Tries Yahoo Finance as the primary free source.
        """
        articles = self._fetch_yahoo_news(symbol)
        if articles:
            return articles
        return []
    
    def get_economic_calendar(self, days_ahead: int = 7) -> List[EconomicEvent]:
        """
        Get economic calendar events
        """
        # Filter events within the specified range
        cutoff_date = datetime.now() + timedelta(days=days_ahead)
        return [event for event in self._mock_economic_events if event.date <= cutoff_date]
    
    def calculate_volatility_forecast(self, symbol: Symbol, lookback_days: int = 30) -> Decimal:
        """
        Calculate annualised volatility forecast from historical prices.
        """
        price_data = self._fetch_historical_data(symbol, days=lookback_days)
        if len(price_data) < 2:
            return Decimal('0.20')  # Default 20% when no data

        prices = [float(point.price.amount) for point in price_data]
        returns = [(prices[i] - prices[i - 1]) / prices[i - 1]
                    for i in range(1, len(prices)) if prices[i - 1] != 0]

        if not returns:
            return Decimal('0.20')

        daily_vol = float(np.std(returns))
        annualized_vol = daily_vol * float(np.sqrt(252))
        return min(Decimal(str(round(annualized_vol, 4))), Decimal('1.0'))
    
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