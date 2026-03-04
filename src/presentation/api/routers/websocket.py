"""
WebSocket API Router

Provides real-time bidirectional communication for:
- Live price updates
- Order status changes
- Auto-trading activity
- Risk alerts

Architectural Intent:
- Presentation layer only — delegates to WebSocketManager
- JWT-authenticated via query parameter (not URL path, to avoid token leakage in logs)
- Supports channel subscriptions for targeted updates
"""
from __future__ import annotations

import json
import logging
import re

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
import jwt

from src.infrastructure.config.settings import settings
from src.infrastructure.websocket_manager import get_websocket_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])

# Channel name must be alphanumeric with colons, hyphens, underscores
_CHANNEL_PATTERN = re.compile(r"^[a-zA-Z0-9:_-]{1,100}$")

# Max inbound message size (bytes)
_MAX_MESSAGE_SIZE = 4096

# Max subscriptions per user
_MAX_SUBSCRIPTIONS = 50


def _authenticate_ws_token(token: str) -> str | None:
    """Validate JWT token and return user_id, or None if invalid."""
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        if payload.get("type") != "access":
            return None
        return payload.get("sub")
    except jwt.PyJWTError as e:
        logger.warning(f"WebSocket authentication failed: {e}")
        return None


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    token: str = Query(..., description="JWT access token"),
):
    """
    WebSocket endpoint for real-time updates.

    Authentication: JWT token passed as query parameter (?token=...).

    Client messages (JSON):
    - {"action": "subscribe", "channel": "prices:AAPL"}
    - {"action": "unsubscribe", "channel": "prices:AAPL"}
    - {"action": "ping"}

    Server messages (JSON):
    - {"type": "price_update", "symbol": "AAPL", "price": 150.0, ...}
    - {"type": "order_update", "order_id": "...", "status": "EXECUTED", ...}
    - {"type": "risk_alert", "message": "...", ...}
    - {"type": "pong"}
    - {"type": "error", "message": "..."}
    """
    user_id = _authenticate_ws_token(token)
    if not user_id:
        await websocket.close(code=4001, reason="Invalid token")
        return

    manager = get_websocket_manager()
    await manager.connect(websocket, user_id)

    # Auto-subscribe to user-specific channels
    manager.subscribe(user_id, "orders")
    manager.subscribe(user_id, "auto-trading")
    manager.subscribe(user_id, "risk-alerts")

    try:
        while True:
            data = await websocket.receive_text()

            # Enforce message size limit
            if len(data) > _MAX_MESSAGE_SIZE:
                await websocket.send_text(
                    json.dumps({"type": "error", "message": "Message too large"})
                )
                continue

            try:
                message = json.loads(data)
                action = message.get("action")

                if action == "subscribe":
                    channel = message.get("channel", "")
                    if not _CHANNEL_PATTERN.match(channel):
                        await websocket.send_text(
                            json.dumps({"type": "error", "message": "Invalid channel name"})
                        )
                        continue
                    current_subs = manager.get_subscription_count(user_id)
                    if current_subs >= _MAX_SUBSCRIPTIONS:
                        await websocket.send_text(
                            json.dumps({"type": "error", "message": "Subscription limit reached"})
                        )
                        continue
                    manager.subscribe(user_id, channel)
                    await websocket.send_text(
                        json.dumps({"type": "subscribed", "channel": channel})
                    )

                elif action == "unsubscribe":
                    channel = message.get("channel", "")
                    manager.unsubscribe(user_id, channel)
                    await websocket.send_text(
                        json.dumps({"type": "unsubscribed", "channel": channel})
                    )

                elif action == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))

                else:
                    await websocket.send_text(
                        json.dumps({"type": "error", "message": f"Unknown action: {action}"})
                    )

            except json.JSONDecodeError:
                await websocket.send_text(
                    json.dumps({"type": "error", "message": "Invalid JSON"})
                )

    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {e}")
        manager.disconnect(websocket, user_id)
