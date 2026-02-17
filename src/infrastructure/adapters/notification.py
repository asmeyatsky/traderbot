"""
Notification Adapter

Logging-based implementation of NotificationPort.
Logs trade notifications and risk alerts to Python logger.
"""
from __future__ import annotations

import logging

from src.domain.entities.trading import Order
from src.domain.entities.user import User
from src.domain.ports import NotificationPort
from src.domain.value_objects import Symbol

logger = logging.getLogger(__name__)


class LoggingNotificationAdapter(NotificationPort):
    """
    Notification adapter that logs all notifications.

    Replaces the placeholder object() in the DI container so that
    RiskManager and other services can send notifications without error.
    A future implementation could send emails, push notifications, or SMS.
    """

    def send_trade_notification(self, user: User, order: Order) -> bool:
        logger.info(
            f"[TRADE] User {user.id}: {order.position_type.value} "
            f"{order.quantity} {order.symbol} @ {order.price}"
        )
        return True

    def send_risk_alert(self, user: User, message: str) -> bool:
        logger.warning(f"[RISK ALERT] User {user.id}: {message}")
        return True

    def send_market_alert(self, user: User, symbol: Symbol, message: str) -> bool:
        logger.info(f"[MARKET ALERT] User {user.id} ({symbol}): {message}")
        return True
