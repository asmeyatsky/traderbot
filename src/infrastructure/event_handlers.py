"""
Domain Event Handlers

Concrete implementations of event handlers for various domain events.
These handlers react to domain events and perform side effects like notifications,
logging, and audit trail management.

Architectural Intent:
- Handlers are loosely coupled to the domain layer via the event bus
- Each handler focuses on a single responsibility
- Handlers can be added/removed without changing domain logic
- Enables building complex workflows through event chains
"""
from __future__ import annotations

import logging
from datetime import datetime

from src.domain.events import (
    OrderPlacedEvent, OrderExecutedEvent, OrderCancelledEvent,
    PositionOpenedEvent, PositionClosedEvent,
    PortfolioUpdatedEvent, UserCreatedEvent,
    RiskLimitBreachedEvent, TradingPausedEvent,
    SentimentAnalysisCompletedEvent, MarketAlertTriggeredEvent,
    DomainEvent
)
from src.infrastructure.event_bus import EventHandler

logger = logging.getLogger(__name__)


class OrderPlacedEventHandler(EventHandler):
    """
    Handles OrderPlacedEvent.

    Responsibilities:
    - Log order placement
    - Create audit trail entry
    - Send notification to user
    """

    async def handle(self, event: OrderPlacedEvent) -> None:
        """Handle an order placed event."""
        logger.info(
            f"Order placed: User={event.user_id}, Symbol={event.symbol}, "
            f"Type={event.order_type}, Quantity={event.quantity}, Price={event.price.amount}"
        )

        # In a real system, this would:
        # 1. Send email/SMS notification to user
        # 2. Update user's activity log
        # 3. Trigger risk analysis
        # 4. Update real-time dashboard

        await self._send_notification(event)

    def can_handle(self, event: DomainEvent) -> bool:
        """Check if this handler can process OrderPlacedEvent."""
        return isinstance(event, OrderPlacedEvent)

    async def _send_notification(self, event: OrderPlacedEvent) -> None:
        """Send notification about order placement."""
        logger.info(f"Sending notification to user {event.user_id} about order placement")
        # In production: integrate with email/SMS service


class OrderExecutedEventHandler(EventHandler):
    """
    Handles OrderExecutedEvent.

    Responsibilities:
    - Update position tracking
    - Update portfolio valuation
    - Calculate and track gains/losses
    - Send execution confirmation
    """

    async def handle(self, event: OrderExecutedEvent) -> None:
        """Handle an order executed event."""
        logger.info(
            f"Order executed: Order={event.order_id}, Symbol={event.symbol}, "
            f"ExecutionPrice={event.execution_price.amount}, "
            f"FilledQuantity={event.filled_quantity}"
        )

        # Calculate execution fees and P&L
        commission_pct = 0.001  # 0.1% commission
        commission = event.execution_price.amount * event.filled_quantity * commission_pct

        logger.debug(
            f"Order commission: ${commission}, Total value: "
            f"${event.execution_price.amount * event.filled_quantity}"
        )

        # In production: update position, portfolio, and send confirmation
        await self._update_portfolio(event)
        await self._send_execution_confirmation(event)

    def can_handle(self, event: DomainEvent) -> bool:
        """Check if this handler can process OrderExecutedEvent."""
        return isinstance(event, OrderExecutedEvent)

    async def _update_portfolio(self, event: OrderExecutedEvent) -> None:
        """Update portfolio after order execution."""
        logger.info(f"Updating portfolio for user {event.user_id}")

    async def _send_execution_confirmation(self, event: OrderExecutedEvent) -> None:
        """Send execution confirmation to user."""
        logger.info(f"Sending execution confirmation to user {event.user_id}")


class PositionClosedEventHandler(EventHandler):
    """
    Handles PositionClosedEvent.

    Responsibilities:
    - Calculate realized P&L
    - Update performance metrics
    - Send trade summary
    """

    async def handle(self, event: PositionClosedEvent) -> None:
        """Handle a position closed event."""
        pnl_percentage = (event.realized_pnl.amount / (event.quantity * 100)) * 100  # Simplified

        logger.info(
            f"Position closed: User={event.user_id}, Symbol={event.symbol}, "
            f"Quantity={event.quantity}, ClosingPrice={event.closing_price.amount}, "
            f"RealizedP&L=${event.realized_pnl.amount} ({pnl_percentage:.2f}%)"
        )

        # In production: update user statistics, send trade summary
        await self._record_trade_summary(event)

    def can_handle(self, event: DomainEvent) -> bool:
        """Check if this handler can process PositionClosedEvent."""
        return isinstance(event, PositionClosedEvent)

    async def _record_trade_summary(self, event: PositionClosedEvent) -> None:
        """Record trade summary for user statistics."""
        logger.info(f"Recording trade summary for user {event.user_id}")


class RiskLimitBreachedEventHandler(EventHandler):
    """
    Handles RiskLimitBreachedEvent.

    Responsibilities:
    - Send critical alerts
    - Log risk violations
    - Update risk status
    - Potentially pause trading
    """

    async def handle(self, event: RiskLimitBreachedEvent) -> None:
        """Handle a risk limit breached event."""
        logger.warning(
            f"Risk limit breached: User={event.user_id}, "
            f"LimitType={event.limit_type}, "
            f"Limit=${event.limit_value.amount}, "
            f"Current=${event.current_value.amount}, "
            f"Severity={event.severity}"
        )

        # In production: send urgent alert and consider trading pause
        await self._send_risk_alert(event)

        if event.severity == "CRITICAL":
            await self._trigger_emergency_response(event)

    def can_handle(self, event: DomainEvent) -> bool:
        """Check if this handler can process RiskLimitBreachedEvent."""
        return isinstance(event, RiskLimitBreachedEvent)

    async def _send_risk_alert(self, event: RiskLimitBreachedEvent) -> None:
        """Send risk alert to user."""
        logger.warning(f"Sending risk alert to user {event.user_id}")

    async def _trigger_emergency_response(self, event: RiskLimitBreachedEvent) -> None:
        """Trigger emergency response for critical risk breach."""
        logger.critical(f"Triggering emergency response for user {event.user_id}")


class TradingPausedEventHandler(EventHandler):
    """
    Handles TradingPausedEvent.

    Responsibilities:
    - Notify user of pause
    - Log trading halt reason
    - Schedule resume check
    """

    async def handle(self, event: TradingPausedEvent) -> None:
        """Handle a trading paused event."""
        logger.warning(
            f"Trading paused: User={event.user_id}, "
            f"Reason={event.reason}, "
            f"ResumeAfter={event.resume_after}"
        )

        # In production: notify user and schedule automated resume
        await self._notify_trading_pause(event)

    def can_handle(self, event: DomainEvent) -> bool:
        """Check if this handler can process TradingPausedEvent."""
        return isinstance(event, TradingPausedEvent)

    async def _notify_trading_pause(self, event: TradingPausedEvent) -> None:
        """Notify user about trading pause."""
        logger.info(f"Notifying user {event.user_id} about trading pause")


class SentimentAnalysisEventHandler(EventHandler):
    """
    Handles SentimentAnalysisCompletedEvent.

    Responsibilities:
    - Store sentiment scores
    - Trigger trading signals based on sentiment
    - Update market analysis dashboard
    """

    async def handle(self, event: SentimentAnalysisCompletedEvent) -> None:
        """Handle sentiment analysis completion."""
        sentiment_label = "POSITIVE" if event.sentiment_score > 0.5 else (
            "NEGATIVE" if event.sentiment_score < -0.5 else "NEUTRAL"
        )

        logger.info(
            f"Sentiment analysis completed: Symbol={event.symbol}, "
            f"Score={event.sentiment_score}, "
            f"Confidence={event.confidence}, "
            f"Label={sentiment_label}, "
            f"Source={event.source}"
        )

        # In production: update trading signals and alerts
        await self._update_trading_signals(event)

    def can_handle(self, event: DomainEvent) -> bool:
        """Check if this handler can process SentimentAnalysisCompletedEvent."""
        return isinstance(event, SentimentAnalysisCompletedEvent)

    async def _update_trading_signals(self, event: SentimentAnalysisCompletedEvent) -> None:
        """Update trading signals based on sentiment."""
        logger.info(f"Updating trading signals for {event.symbol}")


class MarketAlertEventHandler(EventHandler):
    """
    Handles MarketAlertTriggeredEvent.

    Responsibilities:
    - Notify affected users
    - Log market events
    - Trigger defensive actions if needed
    """

    async def handle(self, event: MarketAlertTriggeredEvent) -> None:
        """Handle market alert."""
        logger.info(
            f"Market alert triggered: Symbol={event.symbol}, "
            f"Type={event.alert_type}, "
            f"Message={event.message}, "
            f"Severity={event.severity}"
        )

        # In production: broadcast to users with positions in this symbol
        await self._notify_affected_users(event)

    def can_handle(self, event: DomainEvent) -> bool:
        """Check if this handler can process MarketAlertTriggeredEvent."""
        return isinstance(event, MarketAlertTriggeredEvent)

    async def _notify_affected_users(self, event: MarketAlertTriggeredEvent) -> None:
        """Notify users affected by market alert."""
        logger.info(f"Notifying users about {event.alert_type} for {event.symbol}")


# Factory function to create and register all handlers
def create_event_handlers() -> list[EventHandler]:
    """
    Create all event handlers.

    Returns:
        List of all event handler instances
    """
    return [
        OrderPlacedEventHandler(),
        OrderExecutedEventHandler(),
        PositionClosedEventHandler(),
        RiskLimitBreachedEventHandler(),
        TradingPausedEventHandler(),
        SentimentAnalysisEventHandler(),
        MarketAlertEventHandler(),
    ]
