"""
WebSocket Manager Tests

Tests the WebSocket connection manager, channel subscriptions,
message routing, and subscription limits.
"""
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.infrastructure.websocket_manager import WebSocketManager


@pytest.fixture
def manager():
    return WebSocketManager()


def _mock_ws():
    """Create a mock WebSocket."""
    ws = AsyncMock()
    ws.send_text = AsyncMock()
    return ws


# ---------------------------------------------------------------------------
# Connection Management
# ---------------------------------------------------------------------------

class TestConnectionManagement:

    @pytest.mark.asyncio
    async def test_connect_accepts_websocket(self, manager):
        ws = _mock_ws()
        await manager.connect(ws, "user-1")
        ws.accept.assert_awaited_once()
        assert manager.active_users == 1
        assert manager.active_connections == 1

    @pytest.mark.asyncio
    async def test_disconnect_removes_websocket(self, manager):
        ws = _mock_ws()
        await manager.connect(ws, "user-1")
        manager.disconnect(ws, "user-1")
        assert manager.active_users == 0
        assert manager.active_connections == 0

    @pytest.mark.asyncio
    async def test_multiple_connections_per_user(self, manager):
        ws1 = _mock_ws()
        ws2 = _mock_ws()
        await manager.connect(ws1, "user-1")
        await manager.connect(ws2, "user-1")
        assert manager.active_users == 1
        assert manager.active_connections == 2

    @pytest.mark.asyncio
    async def test_disconnect_partial(self, manager):
        ws1 = _mock_ws()
        ws2 = _mock_ws()
        await manager.connect(ws1, "user-1")
        await manager.connect(ws2, "user-1")
        manager.disconnect(ws1, "user-1")
        assert manager.active_connections == 1
        assert manager.active_users == 1


# ---------------------------------------------------------------------------
# Subscriptions
# ---------------------------------------------------------------------------

class TestSubscriptions:

    @pytest.mark.asyncio
    async def test_subscribe_to_channel(self, manager):
        ws = _mock_ws()
        await manager.connect(ws, "u1")
        manager.subscribe("u1", "prices:AAPL")
        assert manager.get_subscription_count("u1") == 1

    @pytest.mark.asyncio
    async def test_unsubscribe_from_channel(self, manager):
        ws = _mock_ws()
        await manager.connect(ws, "u1")
        manager.subscribe("u1", "prices:AAPL")
        manager.unsubscribe("u1", "prices:AAPL")
        assert manager.get_subscription_count("u1") == 0

    def test_subscription_count_unknown_user(self, manager):
        assert manager.get_subscription_count("nonexistent") == 0


# ---------------------------------------------------------------------------
# Message Sending
# ---------------------------------------------------------------------------

class TestMessageSending:

    @pytest.mark.asyncio
    async def test_send_to_user(self, manager):
        ws = _mock_ws()
        await manager.connect(ws, "u1")
        await manager.send_to_user("u1", {"type": "test", "data": "hello"})
        ws.send_text.assert_awaited_once()
        sent = json.loads(ws.send_text.call_args[0][0])
        assert sent["type"] == "test"

    @pytest.mark.asyncio
    async def test_send_to_nonexistent_user(self, manager):
        # Should not raise
        await manager.send_to_user("ghost", {"type": "test"})

    @pytest.mark.asyncio
    async def test_broadcast_to_channel(self, manager):
        ws1 = _mock_ws()
        ws2 = _mock_ws()
        await manager.connect(ws1, "u1")
        await manager.connect(ws2, "u2")
        manager.subscribe("u1", "orders")
        manager.subscribe("u2", "orders")

        await manager.broadcast_to_channel("orders", {"type": "order_update"})
        ws1.send_text.assert_awaited_once()
        ws2.send_text.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_broadcast_only_to_subscribers(self, manager):
        ws1 = _mock_ws()
        ws2 = _mock_ws()
        await manager.connect(ws1, "u1")
        await manager.connect(ws2, "u2")
        manager.subscribe("u1", "orders")
        # u2 is NOT subscribed

        await manager.broadcast_to_channel("orders", {"type": "order_update"})
        ws1.send_text.assert_awaited_once()
        ws2.send_text.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_broadcast_price_update(self, manager):
        ws = _mock_ws()
        await manager.connect(ws, "u1")
        manager.subscribe("u1", "prices:AAPL")

        await manager.broadcast_price_update("AAPL", 150.0, 2.5)
        sent = json.loads(ws.send_text.call_args[0][0])
        assert sent["type"] == "price_update"
        assert sent["symbol"] == "AAPL"
        assert sent["price"] == 150.0

    @pytest.mark.asyncio
    async def test_broken_connection_removed(self, manager):
        ws = _mock_ws()
        ws.send_text.side_effect = Exception("connection closed")
        await manager.connect(ws, "u1")

        await manager.send_to_user("u1", {"type": "test"})
        # Connection should be removed after failure
        assert manager.active_connections == 0
