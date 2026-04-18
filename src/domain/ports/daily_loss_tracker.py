"""
Daily Loss Tracker — port.

Architectural Intent (ADR-002 §Per-order guards):
- Live-mode users set a `daily_loss_cap_usd` at enablement.
- Before every live order, we check: has this user already lost more than
  their cap today? If yes, refuse the order AND auto-revert them to paper
  until they re-attest.
- Domain defines the contract; infrastructure implements against Redis (keys
  expire at end-of-day UTC so the tracker self-cleans).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal


class DailyLossTrackerPort(ABC):
    """Tracks cumulative realised losses per user per UTC day."""

    @abstractmethod
    def record_loss(self, user_id: str, loss_usd: Decimal) -> Decimal:
        """Add `loss_usd` (positive) to the user's today total. Returns new total."""

    @abstractmethod
    def today_loss(self, user_id: str) -> Decimal:
        """Current UTC-day realised loss total for the user. 0 when no entries."""

    @abstractmethod
    def reset_today(self, user_id: str) -> None:
        """Clear today's running total (e.g., after re-enablement)."""
