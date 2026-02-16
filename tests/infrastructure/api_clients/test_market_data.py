"""
Tests for Market Data Adapters and Fallback Chain

Tests AlphaVantageAdapter, PolygonAdapter, FinnhubAdapter, MarketauxAdapter,
YahooFinanceAdapter, and the composite MarketDataService with its fallback chain.
All external API calls are mocked.
"""
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from decimal import Decimal
from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np

from src.domain.value_objects import Symbol, Price, NewsSentiment
from src.infrastructure.api_clients.market_data import (
    AlphaVantageAdapter,
    PolygonAdapter,
    YahooFinanceAdapter,
    FinnhubAdapter,
    MarketauxAdapter,
    MarketDataService,
)


AAPL = Symbol("AAPL")
TODAY = date.today()
WEEK_AGO = TODAY - timedelta(days=7)


# ===========================================================================
# AlphaVantageAdapter Tests
# ===========================================================================

class TestAlphaVantageAdapter:

    @patch("src.infrastructure.api_clients.market_data.TimeSeries")
    @patch("src.infrastructure.api_clients.market_data.settings")
    def test_get_current_price_success(self, mock_settings, mock_ts_cls):
        """Should return Price when API returns data."""
        mock_settings.ALPHA_VANTAGE_API_KEY = "test_key"
        mock_ts = MagicMock()
        mock_ts_cls.return_value = mock_ts

        df = pd.DataFrame({"05. price": [150.25]})
        mock_ts.get_quote_endpoint.return_value = (df, {})

        adapter = AlphaVantageAdapter()
        price = adapter.get_current_price(AAPL)

        assert price is not None
        assert isinstance(price, Price)
        assert float(price.amount) == 150.25
        assert price.currency == "USD"

    @patch("src.infrastructure.api_clients.market_data.TimeSeries")
    @patch("src.infrastructure.api_clients.market_data.settings")
    def test_get_current_price_empty_data(self, mock_settings, mock_ts_cls):
        """Should return None when API returns empty DataFrame."""
        mock_settings.ALPHA_VANTAGE_API_KEY = "test_key"
        mock_ts = MagicMock()
        mock_ts_cls.return_value = mock_ts
        mock_ts.get_quote_endpoint.return_value = (pd.DataFrame(), {})

        adapter = AlphaVantageAdapter()
        price = adapter.get_current_price(AAPL)
        assert price is None

    @patch("src.infrastructure.api_clients.market_data.TimeSeries")
    @patch("src.infrastructure.api_clients.market_data.settings")
    def test_get_current_price_api_error(self, mock_settings, mock_ts_cls):
        """Should return None when API raises an exception."""
        mock_settings.ALPHA_VANTAGE_API_KEY = "test_key"
        mock_ts = MagicMock()
        mock_ts_cls.return_value = mock_ts
        mock_ts.get_quote_endpoint.side_effect = Exception("API rate limit")

        adapter = AlphaVantageAdapter()
        price = adapter.get_current_price(AAPL)
        assert price is None

    @patch("src.infrastructure.api_clients.market_data.TimeSeries")
    @patch("src.infrastructure.api_clients.market_data.settings")
    def test_get_historical_prices_success(self, mock_settings, mock_ts_cls):
        """Should return list of Price objects for date range."""
        mock_settings.ALPHA_VANTAGE_API_KEY = "test_key"
        mock_ts = MagicMock()
        mock_ts_cls.return_value = mock_ts

        dates = pd.date_range(start=WEEK_AGO, end=TODAY, freq="B")
        df = pd.DataFrame(
            {"4. close": [150.0 + i for i in range(len(dates))]},
            index=dates,
        )
        mock_ts.get_daily.return_value = (df, {})

        adapter = AlphaVantageAdapter()
        prices = adapter.get_historical_prices(AAPL, WEEK_AGO, TODAY)

        assert len(prices) > 0
        assert all(isinstance(p, Price) for p in prices)
        assert all(p.currency == "USD" for p in prices)

    @patch("src.infrastructure.api_clients.market_data.TimeSeries")
    @patch("src.infrastructure.api_clients.market_data.settings")
    def test_get_historical_prices_api_error(self, mock_settings, mock_ts_cls):
        """Should return empty list on API error."""
        mock_settings.ALPHA_VANTAGE_API_KEY = "test_key"
        mock_ts = MagicMock()
        mock_ts_cls.return_value = mock_ts
        mock_ts.get_daily.side_effect = Exception("Network error")

        adapter = AlphaVantageAdapter()
        prices = adapter.get_historical_prices(AAPL, WEEK_AGO, TODAY)
        assert prices == []

    @patch("src.infrastructure.api_clients.market_data.TimeSeries")
    @patch("src.infrastructure.api_clients.market_data.settings")
    def test_get_market_news_returns_empty(self, mock_settings, mock_ts_cls):
        """Alpha Vantage free tier doesn't provide news."""
        mock_settings.ALPHA_VANTAGE_API_KEY = "test_key"
        mock_ts_cls.return_value = MagicMock()

        adapter = AlphaVantageAdapter()
        news = adapter.get_market_news(AAPL)
        assert news == []


# ===========================================================================
# PolygonAdapter Tests
# ===========================================================================

class TestPolygonAdapter:

    @patch("src.infrastructure.api_clients.market_data.RESTClient")
    @patch("src.infrastructure.api_clients.market_data.settings")
    def test_get_current_price_success(self, mock_settings, mock_client_cls):
        """Should return Price from last trade."""
        mock_settings.POLYGON_API_KEY = "test_key"
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_trade = MagicMock()
        mock_trade.price = 155.50
        mock_client.get_last_trade.return_value = mock_trade

        adapter = PolygonAdapter()
        price = adapter.get_current_price(AAPL)

        assert price is not None
        assert float(price.amount) == 155.50

    @patch("src.infrastructure.api_clients.market_data.RESTClient")
    @patch("src.infrastructure.api_clients.market_data.settings")
    def test_get_current_price_no_trades(self, mock_settings, mock_client_cls):
        """Should return None when no trades available."""
        mock_settings.POLYGON_API_KEY = "test_key"
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.get_last_trade.return_value = None

        adapter = PolygonAdapter()
        price = adapter.get_current_price(AAPL)
        assert price is None

    @patch("src.infrastructure.api_clients.market_data.RESTClient")
    @patch("src.infrastructure.api_clients.market_data.settings")
    def test_get_historical_prices_success(self, mock_settings, mock_client_cls):
        """Should return prices from aggregates."""
        mock_settings.POLYGON_API_KEY = "test_key"
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_results = []
        for i in range(5):
            r = MagicMock()
            r.close = 150.0 + i
            mock_results.append(r)
        mock_client.get_aggs.return_value = mock_results

        adapter = PolygonAdapter()
        prices = adapter.get_historical_prices(AAPL, WEEK_AGO, TODAY)

        assert len(prices) == 5
        assert all(isinstance(p, Price) for p in prices)

    @patch("src.infrastructure.api_clients.market_data.RESTClient")
    @patch("src.infrastructure.api_clients.market_data.settings")
    def test_get_market_news_success(self, mock_settings, mock_client_cls):
        """Should return news titles from Polygon."""
        mock_settings.POLYGON_API_KEY = "test_key"
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        articles = []
        for title in ["Apple earnings beat", "New iPhone announced"]:
            a = MagicMock()
            a.title = title
            a.description = f"Details about {title}"
            articles.append(a)
        mock_client.get_market_news.return_value = articles

        adapter = PolygonAdapter()
        news = adapter.get_market_news(AAPL)

        assert len(news) == 2
        assert "Apple earnings beat" in news[0]

    @patch("src.infrastructure.api_clients.market_data.RESTClient")
    @patch("src.infrastructure.api_clients.market_data.settings")
    def test_get_market_news_api_error(self, mock_settings, mock_client_cls):
        """Should return empty list on API error."""
        mock_settings.POLYGON_API_KEY = "test_key"
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.get_market_news.side_effect = Exception("API error")

        adapter = PolygonAdapter()
        news = adapter.get_market_news(AAPL)
        assert news == []


# ===========================================================================
# YahooFinanceAdapter Tests
# ===========================================================================

class TestYahooFinanceAdapter:

    @patch("src.infrastructure.api_clients.market_data.yf")
    def test_get_current_price_success(self, mock_yf):
        """Should return Price from Yahoo Finance."""
        mock_ticker = MagicMock()
        hist = pd.DataFrame({"Close": [148.75]})
        mock_ticker.history.return_value = hist
        mock_yf.Ticker.return_value = mock_ticker

        adapter = YahooFinanceAdapter()
        price = adapter.get_current_price(AAPL)

        assert price is not None
        assert float(price.amount) == 148.75

    @patch("src.infrastructure.api_clients.market_data.yf")
    def test_get_current_price_empty(self, mock_yf):
        """Should return None when no data available."""
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_yf.Ticker.return_value = mock_ticker

        adapter = YahooFinanceAdapter()
        price = adapter.get_current_price(AAPL)
        assert price is None

    @patch("src.infrastructure.api_clients.market_data.yf")
    def test_get_historical_prices_success(self, mock_yf):
        """Should return list of historical prices."""
        mock_ticker = MagicMock()
        dates = pd.date_range(start=WEEK_AGO, end=TODAY, freq="B")
        hist = pd.DataFrame(
            {"Close": [150.0 + i for i in range(len(dates))]},
            index=dates,
        )
        mock_ticker.history.return_value = hist
        mock_yf.Ticker.return_value = mock_ticker

        adapter = YahooFinanceAdapter()
        prices = adapter.get_historical_prices(AAPL, WEEK_AGO, TODAY)

        assert len(prices) == len(dates)
        assert all(isinstance(p, Price) for p in prices)

    @patch("src.infrastructure.api_clients.market_data.yf")
    def test_get_market_news_returns_empty(self, mock_yf):
        """Yahoo Finance adapter doesn't provide news."""
        adapter = YahooFinanceAdapter()
        news = adapter.get_market_news(AAPL)
        assert news == []


# ===========================================================================
# FinnhubAdapter Tests
# ===========================================================================

class TestFinnhubAdapter:

    @patch("src.infrastructure.api_clients.market_data.finnhub.Client")
    @patch("src.infrastructure.api_clients.market_data.settings")
    def test_get_current_price_success(self, mock_settings, mock_client_cls):
        """Should return Price from Finnhub quote."""
        mock_settings.FINNHUB_API_KEY = "test_key"
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.quote.return_value = {"c": 152.30, "h": 155.0, "l": 150.0}

        adapter = FinnhubAdapter()
        price = adapter.get_current_price(AAPL)

        assert price is not None
        assert float(price.amount) == 152.30

    @patch("src.infrastructure.api_clients.market_data.finnhub.Client")
    @patch("src.infrastructure.api_clients.market_data.settings")
    def test_get_current_price_no_quote(self, mock_settings, mock_client_cls):
        """Should return None when quote is empty."""
        mock_settings.FINNHUB_API_KEY = "test_key"
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.quote.return_value = {}

        adapter = FinnhubAdapter()
        price = adapter.get_current_price(AAPL)
        assert price is None

    @patch("src.infrastructure.api_clients.market_data.finnhub.Client")
    @patch("src.infrastructure.api_clients.market_data.settings")
    def test_get_historical_prices_success(self, mock_settings, mock_client_cls):
        """Should return prices from stock candles."""
        mock_settings.FINNHUB_API_KEY = "test_key"
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.stock_candles.return_value = {
            "c": [150.0, 151.5, 152.0, 153.0, 154.5],
            "t": [1000000 + i * 86400 for i in range(5)],
        }

        adapter = FinnhubAdapter()
        prices = adapter.get_historical_prices(AAPL, WEEK_AGO, TODAY)

        assert len(prices) == 5
        assert all(isinstance(p, Price) for p in prices)

    @patch("src.infrastructure.api_clients.market_data.finnhub.Client")
    @patch("src.infrastructure.api_clients.market_data.settings")
    def test_get_market_news_success(self, mock_settings, mock_client_cls):
        """Should return formatted news headlines."""
        mock_settings.FINNHUB_API_KEY = "test_key"
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.company_news.return_value = [
            {"headline": "Apple beats Q4", "summary": "Revenue up 15%"},
            {"headline": "New product launch", "summary": "iPhone 16 announced"},
        ]

        adapter = FinnhubAdapter()
        news = adapter.get_market_news(AAPL)

        assert len(news) == 2
        assert "Apple beats Q4" in news[0]

    @patch("src.infrastructure.api_clients.market_data.finnhub.Client")
    @patch("src.infrastructure.api_clients.market_data.settings")
    def test_analyze_sentiment_placeholder(self, mock_settings, mock_client_cls):
        """Finnhub analyze_sentiment returns placeholder NewsSentiment."""
        mock_settings.FINNHUB_API_KEY = "test_key"
        mock_client_cls.return_value = MagicMock()

        adapter = FinnhubAdapter()
        result = adapter.analyze_sentiment("Some financial text")

        assert isinstance(result, NewsSentiment)
        assert result.source == "Finnhub"

    @patch("src.infrastructure.api_clients.market_data.finnhub.Client")
    @patch("src.infrastructure.api_clients.market_data.settings")
    def test_batch_analyze_sentiment(self, mock_settings, mock_client_cls):
        """batch_analyze_sentiment should return one result per text."""
        mock_settings.FINNHUB_API_KEY = "test_key"
        mock_client_cls.return_value = MagicMock()

        adapter = FinnhubAdapter()
        results = adapter.batch_analyze_sentiment(["text1", "text2", "text3"])

        assert len(results) == 3


# ===========================================================================
# MarketauxAdapter Tests
# ===========================================================================

class TestMarketauxAdapter:

    @patch("src.infrastructure.api_clients.market_data.settings")
    def test_analyze_sentiment(self, mock_settings):
        """Should return placeholder NewsSentiment."""
        mock_settings.MARKETAUX_API_KEY = "test_key"
        adapter = MarketauxAdapter()
        result = adapter.analyze_sentiment("Apple stock analysis")

        assert isinstance(result, NewsSentiment)
        assert result.source == "Marketaux"

    @patch("src.infrastructure.api_clients.market_data.settings")
    def test_batch_analyze_sentiment(self, mock_settings):
        """Should return one result per text."""
        mock_settings.MARKETAUX_API_KEY = "test_key"
        adapter = MarketauxAdapter()
        results = adapter.batch_analyze_sentiment(["text1", "text2"])
        assert len(results) == 2

    @patch("src.infrastructure.api_clients.market_data.settings")
    def test_extract_symbols_returns_empty(self, mock_settings):
        """extract_symbols_from_news should return empty list (placeholder)."""
        mock_settings.MARKETAUX_API_KEY = "test_key"
        adapter = MarketauxAdapter()
        symbols = adapter.extract_symbols_from_news("AAPL is up today")
        assert symbols == []


# ===========================================================================
# MarketDataService (Fallback Chain) Tests
# ===========================================================================

class TestMarketDataService:

    def _make_service_with_mocks(self):
        """Create MarketDataService with all adapters mocked."""
        with patch("src.infrastructure.api_clients.market_data.settings") as mock_settings:
            mock_settings.POLYGON_API_KEY = "test"
            mock_settings.ALPHA_VANTAGE_API_KEY = "test"
            mock_settings.FINNHUB_API_KEY = "test"

            with patch("src.infrastructure.api_clients.market_data.RESTClient"), \
                 patch("src.infrastructure.api_clients.market_data.TimeSeries"), \
                 patch("src.infrastructure.api_clients.market_data.finnhub.Client"):
                svc = MarketDataService()

        # Replace adapters with fresh mocks for fine-grained control
        svc.polygon = MagicMock()
        svc.alpha_vantage = MagicMock()
        svc.finnhub = MagicMock()
        svc.yahoo = MagicMock()
        return svc

    # -- get_current_price fallback chain --

    def test_current_price_polygon_first(self):
        """Should return Polygon price when available."""
        svc = self._make_service_with_mocks()
        expected = Price(amount=Decimal("155.50"), currency="USD")
        svc.polygon.get_current_price.return_value = expected

        result = svc.get_current_price(AAPL)

        assert result == expected
        svc.alpha_vantage.get_current_price.assert_not_called()
        svc.finnhub.get_current_price.assert_not_called()
        svc.yahoo.get_current_price.assert_not_called()

    def test_current_price_falls_to_alpha_vantage(self):
        """Should fall back to Alpha Vantage when Polygon fails."""
        svc = self._make_service_with_mocks()
        svc.polygon.get_current_price.return_value = None
        expected = Price(amount=Decimal("155.25"), currency="USD")
        svc.alpha_vantage.get_current_price.return_value = expected

        result = svc.get_current_price(AAPL)

        assert result == expected
        svc.finnhub.get_current_price.assert_not_called()

    def test_current_price_falls_to_finnhub(self):
        """Should fall back to Finnhub when Polygon and Alpha Vantage fail."""
        svc = self._make_service_with_mocks()
        svc.polygon.get_current_price.return_value = None
        svc.alpha_vantage.get_current_price.return_value = None
        expected = Price(amount=Decimal("155.00"), currency="USD")
        svc.finnhub.get_current_price.return_value = expected

        result = svc.get_current_price(AAPL)

        assert result == expected
        svc.yahoo.get_current_price.assert_not_called()

    def test_current_price_falls_to_yahoo(self):
        """Should fall back to Yahoo Finance as last resort."""
        svc = self._make_service_with_mocks()
        svc.polygon.get_current_price.return_value = None
        svc.alpha_vantage.get_current_price.return_value = None
        svc.finnhub.get_current_price.return_value = None
        expected = Price(amount=Decimal("154.75"), currency="USD")
        svc.yahoo.get_current_price.return_value = expected

        result = svc.get_current_price(AAPL)
        assert result == expected

    def test_current_price_all_fail(self):
        """Should return None when all providers fail."""
        svc = self._make_service_with_mocks()
        svc.polygon.get_current_price.return_value = None
        svc.alpha_vantage.get_current_price.return_value = None
        svc.finnhub.get_current_price.return_value = None
        svc.yahoo.get_current_price.return_value = None

        result = svc.get_current_price(AAPL)
        assert result is None

    # -- get_historical_prices fallback chain --

    def test_historical_prices_polygon_first(self):
        """Should return Polygon historical prices when available."""
        svc = self._make_service_with_mocks()
        expected = [Price(amount=Decimal("150"), currency="USD")]
        svc.polygon.get_historical_prices.return_value = expected

        result = svc.get_historical_prices(AAPL, WEEK_AGO, TODAY)

        assert result == expected
        svc.alpha_vantage.get_historical_prices.assert_not_called()

    def test_historical_prices_falls_to_alpha_vantage(self):
        """Should fall back to Alpha Vantage for historical prices."""
        svc = self._make_service_with_mocks()
        svc.polygon.get_historical_prices.return_value = []
        expected = [Price(amount=Decimal("151"), currency="USD")]
        svc.alpha_vantage.get_historical_prices.return_value = expected

        result = svc.get_historical_prices(AAPL, WEEK_AGO, TODAY)
        assert result == expected

    def test_historical_prices_all_fail(self):
        """Should return empty list when all providers fail."""
        svc = self._make_service_with_mocks()
        svc.polygon.get_historical_prices.return_value = []
        svc.alpha_vantage.get_historical_prices.return_value = []
        svc.finnhub.get_historical_prices.return_value = []
        svc.yahoo.get_historical_prices.return_value = []

        result = svc.get_historical_prices(AAPL, WEEK_AGO, TODAY)
        assert result == []

    # -- get_market_news aggregation --

    def test_market_news_aggregates_sources(self):
        """News should aggregate from multiple sources (not fallback)."""
        svc = self._make_service_with_mocks()
        svc.polygon.get_market_news.return_value = ["Polygon news"]
        svc.finnhub.get_market_news.return_value = ["Finnhub news 1", "Finnhub news 2"]
        svc.alpha_vantage.get_market_news.return_value = []

        result = svc.get_market_news(AAPL)

        assert len(result) == 3
        assert "Polygon news" in result
        assert "Finnhub news 1" in result

    def test_market_news_all_empty(self):
        """Should return empty list when no sources have news."""
        svc = self._make_service_with_mocks()
        svc.polygon.get_market_news.return_value = []
        svc.finnhub.get_market_news.return_value = []
        svc.alpha_vantage.get_market_news.return_value = []

        result = svc.get_market_news(AAPL)
        assert result == []
