"""
WebSocket Event Handler

Architectural Intent:
- Subscribes to domain events via the EventBus
- Pushes real-time updates to connected WebSocket clients
- Infrastructure concern — bridges domain events to WebSocket transport
"""
from __future__ import annotations

import logging
from typing import Optional

from src.domain.events import DomainEvent
from src.infrastructure.event_bus import EventHandler
from src.infrastructure.websocket_manager import get_websocket_manager

logger = logging.getLogger(__name__)


class WebSocketEventHandler(EventHandler):
    """
    Handles domain events by pushing updates to WebSocket clients.

    Subscribes to:
    - OrderExecutedEvent -> pushes to user's "orders" channel
    - AutoTradeExecutedEvent -> pushes to user's "auto-trading" channel
    - RiskAlertEvent -> pushes to user's "risk-alerts" channel
    """

    async def handle(self, event: DomainEvent) -> None:
        """Dispatch domain event to appropriate WebSocket channel."""
        manager = get_websocket_manager()
        event_type = type(event).__name__

        user_id = getattr(event, "user_id", None)
        if not user_id:
            user_id = getattr(event, "aggregate_id", None)

        if not user_id:
            logger.warning(f"Cannot route {event_type} — no user_id found")
            return

        if event_type in ("OrderExecutedEvent", "OrderPlacedEvent", "OrderCancelledEvent"):
            await manager.send_to_user(user_id, {
                "type": "order_update",
                "event": event_type,
                "data": self._serialize_event(event),
            })

        elif event_type == "AutoTradeExecutedEvent":
            await manager.send_to_user(user_id, {
                "type": "auto_trade",
                "event": event_type,
                "data": self._serialize_event(event),
            })

        elif event_type == "RiskAlertEvent":
            await manager.send_to_user(user_id, {
                "type": "risk_alert",
                "event": event_type,
                "data": self._serialize_event(event),
            })

        else:
            logger.debug(f"Unhandled event type for WebSocket: {event_type}")

    def can_handle(self, event: DomainEvent) -> bool:
        """Check if this handler can process the event."""
        return type(event).__name__ in (
            "OrderExecutedEvent",
            "OrderPlacedEvent",
            "OrderCancelledEvent",
            "AutoTradeExecutedEvent",
            "RiskAlertEvent",
        )

    def _serialize_event(self, event: DomainEvent) -> dict:
        """Serialize a domain event to a dict for WebSocket transmission."""
        data = {}
        for key in ("aggregate_id", "aggregate_type", "event_type", "occurred_at"):
            val = getattr(event, key, None)
            if val is not None:
                data[key] = str(val)

        event_data = getattr(event, "event_data", None)
        if isinstance(event_data, dict):
            data.update(event_data)

        return data
