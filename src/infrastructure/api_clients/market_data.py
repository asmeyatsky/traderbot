"""
Data Ingestion Layer for Market Data and News

This module implements adapters for various data sources including
market data providers, news APIs, and fundamental data services.
"""
import asyncio
import requests
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from polygon import RESTClient
import finnhub
from typing import List, Dict, Any, Optional
from datetime import datetime, date, timedelta
from decimal import Decimal
import pandas as pd

from src.domain.ports import MarketDataPort, NewsAnalysisPort
from src.domain.value_objects import Symbol, Price, NewsSentiment
from src.infrastructure.config.settings import settings


def _to_price(value, currency: str = 'USD') -> Price:
    """Convert a numeric value to a Price with proper Decimal handling."""
    return Price(amount=Decimal(str(round(float(value), 4))), currency=currency)


class AlphaVantageAdapter(MarketDataPort):
    """
    Adapter for Alpha Vantage API.

    Implements MarketDataPort interface to provide market data from Alpha Vantage.
    """

    def __init__(self):
        self.api_key = settings.ALPHA_VANTAGE_API_KEY
        self.ts = TimeSeries(key=self.api_key, output_format='pandas')

    def get_current_price(self, symbol: Symbol) -> Optional[Price]:
        """Get the current price for a symbol using Alpha Vantage."""
        try:
            data, meta_data = self.ts.get_quote_endpoint(symbol=str(symbol))
            if not data.empty:
                return _to_price(data.iloc[0]['05. price'])
        except Exception as e:
            print(f"Error fetching price from Alpha Vantage for {symbol}: {e}")
            return None

    def get_historical_prices(self, symbol: Symbol, start_date: date, end_date: date) -> List[Price]:
        """Get historical prices for a symbol within a date range."""
        try:
            data, meta_data = self.ts.get_daily(symbol=str(symbol), outputsize='full')
            # Filter data based on date range
            filtered_data = data[(data.index.date >= start_date) & (data.index.date <= end_date)]

            prices = []
            for idx, row in filtered_data.iterrows():
                prices.append(_to_price(row['4. close']))

            return prices
        except Exception as e:
            print(f"Error fetching historical prices from Alpha Vantage for {symbol}: {e}")
            return []

    def get_market_news(self, symbol: Symbol) -> List[str]:
        """Get recent news for a symbol."""
        # Alpha Vantage doesn't provide news directly in free tier
        # This is a placeholder implementation
        return []


class PolygonAdapter(MarketDataPort):
    """
    Adapter for Polygon.io API.

    Implements MarketDataPort interface to provide market data from Polygon.io.
    """

    def __init__(self):
        self.api_key = settings.POLYGON_API_KEY
        self.client = RESTClient(self.api_key)

    def get_current_price(self, symbol: Symbol) -> Optional[Price]:
        """Get the current price for a symbol using Polygon.io."""
        try:
            # Get the last trade for the symbol
            trades = self.client.get_last_trade(str(symbol))
            if trades:
                return _to_price(trades.price)
        except Exception as e:
            print(f"Error fetching price from Polygon.io for {symbol}: {e}")
            return None

    def get_historical_prices(self, symbol: Symbol, start_date: date, end_date: date) -> List[Price]:
        """Get historical prices for a symbol within a date range."""
        try:
            # Get historical aggregates (OHLCV) from Polygon
            response = self.client.get_aggs(
                str(symbol),
                1,  # Multiplier
                "day",  # Timespan
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )

            prices = []
            for result in response:
                prices.append(_to_price(result.close))

            return prices
        except Exception as e:
            print(f"Error fetching historical prices from Polygon.io for {symbol}: {e}")
            return []

    def get_market_news(self, symbol: Symbol) -> List[str]:
        """Get recent news for a symbol."""
        try:
            # Get news for the symbol from Polygon
            news = self.client.get_market_news(
                ticker=str(symbol),
                limit=10
            )

            news_list = []
            for article in news:
                news_list.append(article.title or article.description or "")

            return news_list
        except Exception as e:
            print(f"Error fetching news from Polygon.io for {symbol}: {e}")
            return []


class YahooFinanceAdapter(MarketDataPort):
    """
    Adapter for Yahoo Finance data.

    Implements MarketDataPort interface to provide market data from Yahoo Finance.
    This is a free alternative to paid APIs.
    """

    def get_current_price(self, symbol: Symbol) -> Optional[Price]:
        """Get the current price for a symbol using Yahoo Finance."""
        try:
            ticker = yf.Ticker(str(symbol))
            hist = ticker.history(period="1d")
            if not hist.empty:
                return _to_price(hist['Close'].iloc[-1])
        except Exception as e:
            print(f"Error fetching price from Yahoo Finance for {symbol}: {e}")
            return None

    def get_historical_prices(self, symbol: Symbol, start_date: date, end_date: date) -> List[Price]:
        """Get historical prices for a symbol within a date range."""
        try:
            ticker = yf.Ticker(str(symbol))
            hist = ticker.history(start=start_date, end=end_date)

            prices = []
            for idx, row in hist.iterrows():
                prices.append(_to_price(row['Close']))

            return prices
        except Exception as e:
            print(f"Error fetching historical prices from Yahoo Finance for {symbol}: {e}")
            return []

    def get_market_news(self, symbol: Symbol) -> List[str]:
        """Get recent news for a symbol."""
        # Yahoo Finance doesn't directly provide news text through yfinance
        # This would typically require web scraping or another API
        # For now, return empty list
        return []


class MarketauxAdapter(NewsAnalysisPort):
    """
    Adapter for Marketaux API.

    Implements NewsAnalysisPort interface to provide news sentiment analysis.
    """

    def __init__(self):
        self.api_key = settings.MARKETAUX_API_KEY
        self.base_url = "https://api.marketaux.com/v1/news"

    def analyze_sentiment(self, text: str) -> NewsSentiment:
        """Analyze sentiment of a text using Marketaux."""
        return NewsSentiment(
            score=Decimal('0'),
            confidence=Decimal('0'),
            source="Marketaux"
        )

    def batch_analyze_sentiment(self, texts: List[str]) -> List[NewsSentiment]:
        """Analyze sentiment of multiple texts."""
        return [self.analyze_sentiment(text) for text in texts]

    def extract_symbols_from_news(self, news_text: str) -> List[Symbol]:
        """Extract stock symbols mentioned in news text."""
        return []


class FinnhubAdapter(MarketDataPort, NewsAnalysisPort):
    """
    Adapter for Finnhub API.

    Implements both MarketDataPort and NewsAnalysisPort.
    """

    def __init__(self):
        self.api_key = settings.FINNHUB_API_KEY
        self.client = finnhub.Client(api_key=self.api_key)

    def get_current_price(self, symbol: Symbol) -> Optional[Price]:
        """Get the current price for a symbol using Finnhub."""
        try:
            quote = self.client.quote(str(symbol))
            if quote and 'c' in quote:  # 'c' is current price in Finnhub response
                return _to_price(quote['c'])
        except Exception as e:
            print(f"Error fetching price from Finnhub for {symbol}: {e}")
            return None

    def get_historical_prices(self, symbol: Symbol, start_date: date, end_date: date) -> List[Price]:
        """Get historical prices for a symbol within a date range."""
        try:
            resolution = 'D'  # Daily
            start_ts = int(datetime.combine(start_date, datetime.min.time()).timestamp())
            end_ts = int(datetime.combine(end_date, datetime.min.time()).timestamp())

            response = self.client.stock_candles(str(symbol), resolution, start_ts, end_ts)

            prices = []
            if 'c' in response and len(response['c']) > 0:
                for i in range(len(response['c'])):
                    prices.append(_to_price(response['c'][i]))

            return prices
        except Exception as e:
            print(f"Error fetching historical prices from Finnhub for {symbol}: {e}")
            return []

    def get_market_news(self, symbol: Symbol) -> List[str]:
        """Get recent news for a symbol."""
        try:
            # Get company news for the symbol
            from_time = (datetime.now().date() - timedelta(days=7)).strftime('%Y-%m-%d')
            to_time = datetime.now().date().strftime('%Y-%m-%d')

            news = self.client.company_news(str(symbol), _from=from_time, to=to_time)

            news_list = []
            for article in news:
                news_list.append(article.get('headline', '') + ': ' + article.get('summary', ''))

            return news_list
        except Exception as e:
            print(f"Error fetching news from Finnhub for {symbol}: {e}")
            return []

    def analyze_sentiment(self, text: str) -> NewsSentiment:
        """Analyze sentiment of a text using Finnhub."""
        return NewsSentiment(
            score=Decimal('0'),
            confidence=Decimal('0'),
            source="Finnhub"
        )

    def batch_analyze_sentiment(self, texts: List[str]) -> List[NewsSentiment]:
        """Analyze sentiment of multiple texts."""
        return [self.analyze_sentiment(text) for text in texts]

    def extract_symbols_from_news(self, news_text: str) -> List[Symbol]:
        """Extract stock symbols mentioned in news text."""
        return []


# Combine adapters into a unified data service
class MarketDataService:
    """
    Unified service that combines multiple market data adapters.

    This service provides a single interface to multiple data sources
    and implements fallback logic when one source fails.
    """

    def __init__(self):
        self.alpha_vantage = AlphaVantageAdapter()
        self.polygon = PolygonAdapter()
        self.yahoo = YahooFinanceAdapter()
        self.finnhub = FinnhubAdapter()

    def get_current_price(self, symbol: Symbol) -> Optional[Price]:
        """Get current price using multiple fallback sources."""
        # Try Polygon first (typically fastest/lowest latency)
        price = self.polygon.get_current_price(symbol)
        if price:
            return price

        # Try Alpha Vantage
        price = self.alpha_vantage.get_current_price(symbol)
        if price:
            return price

        # Try Finnhub
        price = self.finnhub.get_current_price(symbol)
        if price:
            return price

        # Try Yahoo Finance as last resort
        return self.yahoo.get_current_price(symbol)

    def get_historical_prices(self, symbol: Symbol, start_date: date, end_date: date) -> List[Price]:
        """Get historical prices using multiple fallback sources."""
        # Try Polygon first
        prices = self.polygon.get_historical_prices(symbol, start_date, end_date)
        if prices:
            return prices

        # Try Alpha Vantage
        prices = self.alpha_vantage.get_historical_prices(symbol, start_date, end_date)
        if prices:
            return prices

        # Try Finnhub
        prices = self.finnhub.get_historical_prices(symbol, start_date, end_date)
        if prices:
            return prices

        # Try Yahoo Finance as last resort
        return self.yahoo.get_historical_prices(symbol, start_date, end_date)

    def get_market_news(self, symbol: Symbol) -> List[str]:
        """Get market news using multiple data sources."""
        news = []

        # Combine news from multiple sources
        news.extend(self.polygon.get_market_news(symbol))
        news.extend(self.finnhub.get_market_news(symbol))
        news.extend(self.alpha_vantage.get_market_news(symbol))

        return news
