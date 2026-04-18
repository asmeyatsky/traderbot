"""
Tests for CircuitBrokenBrokerAdapter — verifies that place_order outcomes
feed the breaker correctly while all other calls pass through untouched.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import requests

from src.domain.services.broker_circuit_breaker import BrokerCircuitBreaker
from src.infrastructure.adapters.circuit_broken_broker import (
    CircuitBrokenBrokerAdapter,
)
from src.infrastructure.broker_integration import (
    BrokerAuthenticationError,
    BrokerOrderResponse,
)


def _order():
    o = MagicMock()
    o.id = "order-1"
    return o


@pytest.fixture
def breaker():
    return BrokerCircuitBreaker(max_consecutive_errors=3)


@pytest.fixture
def delegate():
    m = MagicMock()
    m.paper_trading = False
    return m


@pytest.fixture
def adapter(delegate, breaker):
    return CircuitBrokenBrokerAdapter(delegate, breaker)


class TestPlaceOrderOutcomes:
    def test_success_calls_record_success(self, adapter, delegate, breaker):
        delegate.place_order.return_value = BrokerOrderResponse(
            broker_order_id="ok", status="pending",
        )
        # Preload some errors so we can see the reset.
        breaker.record_error()
        breaker.record_error()

        adapter.place_order(_order())

        assert breaker._consecutive_errors == 0

    def test_auth_error_increments_breaker(self, adapter, delegate, breaker):
        delegate.place_order.side_effect = BrokerAuthenticationError("bad keys")
        with pytest.raises(BrokerAuthenticationError):
            adapter.place_order(_order())
        assert breaker._consecutive_errors == 1

    def test_five_xx_increments_breaker(self, adapter, delegate, breaker):
        resp = MagicMock()
        resp.status_code = 503
        exc = requests.exceptions.HTTPError("service unavailable")
        exc.response = resp
        delegate.place_order.side_effect = exc

        with pytest.raises(requests.exceptions.HTTPError):
            adapter.place_order(_order())
        assert breaker._consecutive_errors == 1

    def test_four_xx_does_not_increment_breaker(self, adapter, delegate, breaker):
        """422 validation errors are the caller's problem, not a broker outage."""
        resp = MagicMock()
        resp.status_code = 422
        exc = requests.exceptions.HTTPError("bad request")
        exc.response = resp
        delegate.place_order.side_effect = exc

        with pytest.raises(requests.exceptions.HTTPError):
            adapter.place_order(_order())
        assert breaker._consecutive_errors == 0

    def test_network_error_increments_breaker(self, adapter, delegate, breaker):
        delegate.place_order.side_effect = requests.exceptions.ConnectionError(
            "dns failed"
        )
        with pytest.raises(requests.exceptions.ConnectionError):
            adapter.place_order(_order())
        assert breaker._consecutive_errors == 1

    def test_three_consecutive_errors_trip_breaker(self, adapter, delegate, breaker):
        delegate.place_order.side_effect = BrokerAuthenticationError("rotated")
        for _ in range(3):
            with pytest.raises(BrokerAuthenticationError):
                adapter.place_order(_order())
        assert breaker.is_open()


class TestPassThrough:
    def test_get_account_info_is_pass_through(self, adapter, delegate):
        delegate.get_account_info.return_value = {"buying_power": 1000}
        assert adapter.get_account_info("u") == {"buying_power": 1000}
        delegate.get_account_info.assert_called_once_with("u")

    def test_cancel_is_pass_through(self, adapter, delegate):
        delegate.cancel_order.return_value = True
        assert adapter.cancel_order("o1") is True

    def test_account_info_errors_do_not_trip_breaker(self, adapter, delegate, breaker):
        """Balance reads are not on the breaker-guarded path."""
        delegate.get_account_info.side_effect = requests.exceptions.ConnectionError()
        with pytest.raises(requests.exceptions.ConnectionError):
            adapter.get_account_info("u")
        assert breaker._consecutive_errors == 0
