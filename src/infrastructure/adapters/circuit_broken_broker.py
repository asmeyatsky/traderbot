"""
Circuit-breaker-aware broker adapter.

Architectural Intent:
- Wraps a live `AlpacaBrokerService` so that `place_order` failures feed the
  `BrokerCircuitBreaker`. A successful live place clears the counter; a 5xx
  or auth rejection bumps it, and when the threshold hits the breaker trips.
- The factory wraps only the LIVE broker — paper broker traffic hits the
  Alpaca paper endpoint with the same credentials shape but has no financial
  blast radius, so we don't want paper hiccups to halt live.
- All non-place_order calls are passed through unchanged (the breaker is
  order-placement-specific; balance reads happen on different endpoints and
  have different failure modes).
"""
from __future__ import annotations

import logging

import requests

from src.domain.services.broker_circuit_breaker import BrokerCircuitBreaker
from src.infrastructure.broker_integration import (
    AlpacaBrokerService,
    BrokerAuthenticationError,
    BrokerOrderResponse,
)

logger = logging.getLogger(__name__)


class CircuitBrokenBrokerAdapter:
    """Proxy around AlpacaBrokerService that reports outcomes to a breaker."""

    def __init__(
        self,
        delegate: AlpacaBrokerService,
        breaker: BrokerCircuitBreaker,
    ) -> None:
        self._delegate = delegate
        self._breaker = breaker
        # Preserve attribute access used by callers that peek at paper_trading
        # (e.g., logging) — the proxy is otherwise transparent.
        self.paper_trading = getattr(delegate, "paper_trading", False)

    def place_order(self, order) -> BrokerOrderResponse:
        try:
            result = self._delegate.place_order(order)
            self._breaker.record_success()
            return result
        except BrokerAuthenticationError:
            self._breaker.record_error()
            raise
        except requests.exceptions.HTTPError as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", 0) or 0
            if 500 <= status_code < 600:
                self._breaker.record_error()
            raise
        except requests.exceptions.RequestException:
            # Network errors (DNS, connection reset, timeout) count too —
            # they block the live path just as effectively as a 5xx.
            self._breaker.record_error()
            raise

    def __getattr__(self, name: str):
        """Pass-through for every other method (cancel, status, balance)."""
        return getattr(self._delegate, name)
