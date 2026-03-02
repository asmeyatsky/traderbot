"""
WebSocket Connection Manager

Architectural Intent:
- Manages WebSocket connections per user
- Supports channel-based message routing (prices, orders, alerts)
- Uses in-memory dict for connection tracking (KeyDB pub/sub for multi-process)
- Infrastructure concern — domain layer is unaware of WebSocket specifics
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Set

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketManager:
    """
    Manages WebSocket connections and message routing.

    Channels:
    - prices:{symbol} — real-time price updates
    - orders — order status changes
    - auto-trading — autonomous trading activity
    - risk-alerts — risk threshold warnings
    """

    def __init__(self):
        # user_id -> set of WebSocket connections
        self._connections: Dict[str, Set[WebSocket]] = {}
        # user_id -> set of subscribed channels
        self._subscriptions: Dict[str, Set[str]] = {}

    async def connect(self, websocket: WebSocket, user_id: str) -> None:
        """Accept a WebSocket connection and register it for a user."""
        await websocket.accept()
        if user_id not in self._connections:
            self._connections[user_id] = set()
            self._subscriptions[user_id] = set()
        self._connections[user_id].add(websocket)
        logger.info(f"WebSocket connected for user {user_id}")

    def disconnect(self, websocket: WebSocket, user_id: str) -> None:
        """Remove a WebSocket connection for a user."""
        if user_id in self._connections:
            self._connections[user_id].discard(websocket)
            if not self._connections[user_id]:
                del self._connections[user_id]
                self._subscriptions.pop(user_id, None)
        logger.info(f"WebSocket disconnected for user {user_id}")

    def subscribe(self, user_id: str, channel: str) -> None:
        """Subscribe a user to a channel."""
        if user_id in self._subscriptions:
            self._subscriptions[user_id].add(channel)

    def unsubscribe(self, user_id: str, channel: str) -> None:
        """Unsubscribe a user from a channel."""
        if user_id in self._subscriptions:
            self._subscriptions[user_id].discard(channel)

    async def send_to_user(
        self, user_id: str, message: Dict[str, Any]
    ) -> None:
        """Send a message to all connections for a user."""
        if user_id not in self._connections:
            return

        disconnected = set()
        data = json.dumps(message)

        for ws in self._connections[user_id]:
            try:
                await ws.send_text(data)
            except Exception:
                disconnected.add(ws)

        for ws in disconnected:
            self._connections[user_id].discard(ws)

    async def broadcast_to_channel(
        self, channel: str, message: Dict[str, Any]
    ) -> None:
        """Broadcast a message to all users subscribed to a channel."""
        data = json.dumps(message)

        for user_id, channels in self._subscriptions.items():
            if channel in channels and user_id in self._connections:
                disconnected = set()
                for ws in self._connections[user_id]:
                    try:
                        await ws.send_text(data)
                    except Exception:
                        disconnected.add(ws)
                for ws in disconnected:
                    self._connections[user_id].discard(ws)

    async def broadcast_price_update(
        self, symbol: str, price: float, change_pct: float
    ) -> None:
        """Broadcast a price update to users watching a symbol."""
        await self.broadcast_to_channel(
            f"prices:{symbol}",
            {
                "type": "price_update",
                "symbol": symbol,
                "price": price,
                "change_pct": change_pct,
            },
        )

    @property
    def active_connections(self) -> int:
        return sum(len(conns) for conns in self._connections.values())

    @property
    def active_users(self) -> int:
        return len(self._connections)


# Singleton instance
_ws_manager: Optional[WebSocketManager] = None


def get_websocket_manager() -> WebSocketManager:
    """Get the global WebSocket manager instance."""
    global _ws_manager
    if _ws_manager is None:
        _ws_manager = WebSocketManager()
    return _ws_manager
