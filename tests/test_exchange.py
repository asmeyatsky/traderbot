"""
Tests for Exchange Domain Entity and Exchange Registry

Tests cover:
- Exchange.is_open() returns bool based on time
- Exchange frozen dataclass
- ExchangeRegistry lists all exchanges and returns statuses
"""
import pytest
from datetime import time

from src.domain.entities.exchange import Exchange
from src.domain.services.exchange_registry import ExchangeRegistry


class TestExchange:
    def test_frozen(self):
        ex = Exchange(
            code="NYSE", name="NYSE", timezone="America/New_York",
            open_time=time(9, 30), close_time=time(16, 0),
            country="US", currency="USD",
        )
        with pytest.raises(AttributeError):
            ex.code = "NOPE"  # type: ignore

    def test_is_open_returns_bool(self):
        ex = Exchange(
            code="NYSE", name="NYSE", timezone="America/New_York",
            open_time=time(9, 30), close_time=time(16, 0),
            country="US", currency="USD",
        )
        assert isinstance(ex.is_open(), bool)

    def test_next_open_returns_datetime(self):
        ex = Exchange(
            code="NYSE", name="NYSE", timezone="America/New_York",
            open_time=time(9, 30), close_time=time(16, 0),
            country="US", currency="USD",
        )
        dt = ex.next_open()
        assert dt.tzinfo is not None

    def test_next_close_returns_datetime(self):
        ex = Exchange(
            code="NYSE", name="NYSE", timezone="America/New_York",
            open_time=time(9, 30), close_time=time(16, 0),
            country="US", currency="USD",
        )
        dt = ex.next_close()
        assert dt.tzinfo is not None


class TestExchangeRegistry:
    def test_list_all(self):
        registry = ExchangeRegistry()
        exchanges = registry.list_all()
        assert len(exchanges) >= 8
        codes = {ex.code for ex in exchanges}
        assert "NYSE" in codes
        assert "LSE" in codes
        assert "TSE" in codes

    def test_get(self):
        registry = ExchangeRegistry()
        nyse = registry.get("NYSE")
        assert nyse is not None
        assert nyse.currency == "USD"

    def test_get_unknown(self):
        registry = ExchangeRegistry()
        assert registry.get("UNKNOWN") is None

    def test_get_all_statuses(self):
        registry = ExchangeRegistry()
        statuses = registry.get_all_statuses()
        assert "exchanges" in statuses
        assert len(statuses["exchanges"]) >= 8
        for s in statuses["exchanges"]:
            assert "code" in s
            assert "is_open" in s
