"""
Broker circuit breaker.

Architectural Intent (ADR-002, rewrite181226.md Phase 6):
- When a live broker returns 5xx or auth rejections in quick succession it's
  signalling that live order-placement is unsafe right now (outage, rotated
  keys, rate-limit cascade). Retrying hammers the broker and multiplies the
  risk to user funds.
- This breaker counts consecutive failures across live `place_order` calls.
  Three failures in a row trip it; while tripped, the factory refuses to
  return the live broker — live users are forced to fail-closed rather than
  retry blindly. The cool-off is 5 minutes by default; a successful live
  place_order resets the counter instantly.
- The breaker is intentionally separate from the market-volatility breaker
  (risk_management.CircuitBreakerService). Different failure modes, different
  windows, different operators watching them.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class BrokerCircuitBreaker:
    """Tracks consecutive broker failures and trips after a threshold.

    The default 3 consecutive errors / 5-minute cool-off matches ADR-002's
    tuning. Operators may override per environment via the factory.
    """

    max_consecutive_errors: int = 3
    cool_off: timedelta = timedelta(minutes=5)
    _consecutive_errors: int = 0
    _tripped_at: Optional[datetime] = None

    def is_open(self) -> bool:
        """Return True if live trading should be refused right now.

        Uses a wall-clock check rather than a timer thread — the breaker is
        passive; callers poll it on the hot path.
        """
        if self._tripped_at is None:
            return False
        if datetime.now(timezone.utc) - self._tripped_at > self.cool_off:
            logger.info("broker_circuit_breaker_reset after %s", self.cool_off)
            self._tripped_at = None
            self._consecutive_errors = 0
            return False
        return True

    def record_success(self) -> None:
        """Reset the consecutive-error counter after a successful live call."""
        if self._consecutive_errors > 0:
            logger.debug(
                "broker_circuit_breaker_success_reset prior_errors=%d",
                self._consecutive_errors,
            )
        self._consecutive_errors = 0

    def record_error(self) -> None:
        """Record a live-path failure; trip the breaker if threshold met."""
        self._consecutive_errors += 1
        logger.warning(
            "broker_circuit_breaker_error_recorded consecutive=%d threshold=%d",
            self._consecutive_errors, self.max_consecutive_errors,
        )
        if (
            self._consecutive_errors >= self.max_consecutive_errors
            and self._tripped_at is None
        ):
            self._tripped_at = datetime.now(timezone.utc)
            logger.error(
                "broker_circuit_breaker_tripped cool_off=%s", self.cool_off,
            )
