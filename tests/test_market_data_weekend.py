"""
Market Data Edge Case Tests

Regression suite for Bug #2: weekend/holiday market data gap.
period="1d" returned empty DataFrame on weekends; fix uses "5d".
Also tests the MarketDataService fallback chain.
"""
import pytest
from decimal import Decimal
from unittest.mock import patch, MagicMock, PropertyMock

import pandas as pd

from src.domain.value_objects import Symbol, Price
from src.infrastructure.api_clients.market_data import (
    YahooFinanceAdapter,
    MarketDataService,
    _to_price,
)


class TestYahooFinanceAdapter:
    """Tests for YahooFinanceAdapter.get_current_price()."""

    @patch("src.infrastructure.api_clients.market_data.yf")
    def test_yahoo_uses_5d_period(self, mock_yf):
        """Verify the adapter calls ticker.history(period='5d'), not '1d'."""
        mock_ticker = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker
        df = pd.DataFrame({"Close": [150.0, 151.0, 152.0]})
        mock_ticker.history.return_value = df

        adapter = YahooFinanceAdapter()
        adapter.get_current_price(Symbol("AAPL"))

        mock_ticker.history.assert_called_once_with(period="5d")

    @patch("src.infrastructure.api_clients.market_data.yf")
    def test_yahoo_returns_last_close_from_5d(self, mock_yf):
        """5-day window returns the last available close (e.g. Friday's on Saturday)."""
        mock_ticker = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker
        df = pd.DataFrame({"Close": [148.0, 149.0, 150.5]})
        mock_ticker.history.return_value = df

        adapter = YahooFinanceAdapter()
        price = adapter.get_current_price(Symbol("AAPL"))

        assert price is not None
        assert price.amount == Decimal(str(round(150.5, 4)))

    @patch("src.infrastructure.api_clients.market_data.yf")
    def test_yahoo_empty_history_returns_none(self, mock_yf):
        """Empty DataFrame (e.g. delisted symbol) returns None, not crash."""
        mock_ticker = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker
        mock_ticker.history.return_value = pd.DataFrame()

        adapter = YahooFinanceAdapter()
        result = adapter.get_current_price(Symbol("AAPL"))

        assert result is None

    @patch("src.infrastructure.api_clients.market_data.yf")
    def test_yahoo_exception_returns_none(self, mock_yf):
        """Network errors are caught and return None."""
        mock_ticker = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker
        mock_ticker.history.side_effect = Exception("Network error")

        adapter = YahooFinanceAdapter()
        result = adapter.get_current_price(Symbol("AAPL"))

        assert result is None


class TestMarketDataServiceFallback:
    """Tests for MarketDataService adapter fallback chain."""

    def test_fallback_chain_first_fails_second_succeeds(self):
        """If the first adapter returns None, the service tries the next."""
        adapter1 = MagicMock()
        adapter1.get_current_price.return_value = None

        adapter2 = MagicMock()
        expected = Price(Decimal("155.00"), "USD")
        adapter2.get_current_price.return_value = expected

        service = MarketDataService.__new__(MarketDataService)
        service._adapters = [adapter1, adapter2]

        result = service.get_current_price(Symbol("AAPL"))
        assert result == expected
        adapter1.get_current_price.assert_called_once()
        adapter2.get_current_price.assert_called_once()

    def test_all_adapters_fail_returns_none(self):
        """When every adapter returns None, service returns None."""
        adapter1 = MagicMock()
        adapter1.get_current_price.return_value = None
        adapter2 = MagicMock()
        adapter2.get_current_price.side_effect = Exception("API down")

        service = MarketDataService.__new__(MarketDataService)
        service._adapters = [adapter1, adapter2]

        result = service.get_current_price(Symbol("AAPL"))
        assert result is None


class TestToPriceHelper:
    """Tests for the _to_price conversion helper."""

    def test_converts_float_to_decimal(self):
        price = _to_price(150.1234)
        assert isinstance(price.amount, Decimal)
        assert price.amount == Decimal("150.1234")
        assert price.currency == "USD"

    def test_rounds_to_4dp(self):
        price = _to_price(150.12349)
        assert price.amount == Decimal("150.1235")  # rounded

    def test_integer_input(self):
        price = _to_price(150)
        assert price.amount == Decimal("150.0")
