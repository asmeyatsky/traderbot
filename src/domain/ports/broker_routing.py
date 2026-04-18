"""
Broker routing port.

Architectural Intent (ADR-002):
- Application use cases cannot construct broker adapters directly because
  paper/live selection depends on per-user state (trading_mode, daily-loss
  cap) plus platform kill-switches (EMERGENCY_HALT, ENABLE_LIVE_TRADING).
- This port is the single abstraction the application layer uses to obtain
  the broker for a given user. The concrete factory lives in infrastructure
  and composes the kill-switch + loss-cap checks.
- Keeping the decision here (and not inside CreateOrderUseCase) preserves
  application-layer purity — the use case just asks "who is the broker for
  this user?" and calls `place_order`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from src.domain.entities.user import User
from src.domain.ports import TradingExecutionPort


class BrokerRoutingPort(ABC):
    """Resolve the broker service appropriate for a specific user."""

    @abstractmethod
    def for_user(self, user: User) -> TradingExecutionPort:
        """Return the broker adapter to use for this user's next order.

        Implementations must honour platform-level kill switches
        (EMERGENCY_HALT → paper, ENABLE_LIVE_TRADING=false → paper) and
        per-user safety caps (daily_loss_cap_usd). Callers should treat the
        result as opaque and never conditionally switch behaviour based on it.

        Raises:
            BrokerAuthenticationError-family exceptions when live routing is
            requested but policy blocks it (e.g. daily-loss cap breached).
        """
