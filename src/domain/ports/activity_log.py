"""
Activity log port.

Architectural Intent:
- Structured, user-scoped event log used by the autonomous trading loop
  (signal generation, order placement, fills, risk blocks, stop-loss /
  take-profit triggers) and by anywhere else the app wants to record
  a user-visible event for later display.
- Distinct from `AuditEventSink` (Phase 3): audit events are security /
  compliance records (append-only, separate IAM); activity log entries
  are user-facing history that shows up in the Trading Activity view.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ActivityLogPort(ABC):
    """Record a user-visible activity event.

    The kwargs signature reflects the existing concrete repository — fields
    vary per event type (`signal`, `confidence`, `order_id`, etc.). The
    port intentionally doesn't constrain the shape because different event
    types carry different payloads; implementations are expected to persist
    whatever is passed and gracefully drop unknown keys.
    """

    @abstractmethod
    def log_event(
        self,
        *,
        user_id: str,
        event_type: str,
        message: str = "",
        **fields: Any,
    ) -> None:
        """Persist a single activity record.

        Must not raise on persistence failure — the caller (a trading loop)
        cannot meaningfully react, and blocking the loop on log failure
        would trade a minor observability loss for a major operational
        outage. Implementations should log-and-swallow.
        """
