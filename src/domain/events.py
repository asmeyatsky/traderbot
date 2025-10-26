"""
Domain Events

This module defines domain events that are published when important business
events occur in the trading domain. Events enable integration between bounded
contexts and provide an audit trail of business activities.

Following Event Sourcing and CQRS patterns.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from abc import ABC
from typing import Any

from src.domain.value_objects import Money, Symbol, Price


@dataclass(frozen=True)
class DomainEvent(ABC):
    """
    Base class for all domain events.

    Architectural Intent:
    - Immutable record of something that happened in the domain
    - Used for event sourcing and integration between bounded contexts
    - Provides audit trail and enables event replay
    """
    aggregate_id: str
    occurred_at: datetime
    event_id: str  # Unique identifier for the event

    def __post_init__(self):
        """Validate event after initialization"""
        if not self.aggregate_id:
            raise ValueError("aggregate_id cannot be empty")
        if not self.event_id:
            raise ValueError("event_id cannot be empty")


@dataclass(frozen=True)
class OrderPlacedEvent(DomainEvent):
    """Event published when a new order is placed."""
    user_id: str
    symbol: Symbol
    order_type: str  # MARKET, LIMIT, STOP_LOSS, TRAILING_STOP
    position_type: str  # LONG, SHORT
    quantity: int
    price: Price
    stop_price: Price | None = None


@dataclass(frozen=True)
class OrderExecutedEvent(DomainEvent):
    """Event published when an order is executed."""
    user_id: str
    order_id: str
    symbol: Symbol
    execution_price: Price
    filled_quantity: int
    executed_at: datetime
    commission: Money | None = None


@dataclass(frozen=True)
class OrderCancelledEvent(DomainEvent):
    """Event published when an order is cancelled."""
    user_id: str
    order_id: str
    symbol: Symbol
    cancelled_at: datetime
    reason: str | None = None


@dataclass(frozen=True)
class PositionOpenedEvent(DomainEvent):
    """Event published when a new position is opened."""
    user_id: str
    symbol: Symbol
    position_type: str
    quantity: int
    average_price: Price
    opened_at: datetime


@dataclass(frozen=True)
class PositionClosedEvent(DomainEvent):
    """Event published when a position is closed."""
    user_id: str
    symbol: Symbol
    quantity: int
    closing_price: Price
    realized_pnl: Money
    closed_at: datetime


@dataclass(frozen=True)
class PortfolioUpdatedEvent(DomainEvent):
    """Event published when portfolio is updated."""
    user_id: str
    total_value: Money
    cash_balance: Money
    positions_count: int
    updated_at: datetime


@dataclass(frozen=True)
class UserCreatedEvent(DomainEvent):
    """Event published when a new user is created."""
    email: str
    first_name: str
    last_name: str
    risk_tolerance: str
    investment_goal: str
    created_at: datetime


@dataclass(frozen=True)
class UserPreferencesUpdatedEvent(DomainEvent):
    """Event published when user preferences are updated."""
    user_id: str
    field_name: str
    old_value: Any
    new_value: Any
    updated_at: datetime


@dataclass(frozen=True)
class RiskLimitBreachedEvent(DomainEvent):
    """Event published when a risk limit is breached."""
    user_id: str
    limit_type: str  # DAILY, WEEKLY, MONTHLY, DRAWDOWN
    limit_value: Money
    current_value: Money
    breached_at: datetime
    severity: str = "WARNING"  # WARNING, CRITICAL


@dataclass(frozen=True)
class TradingPausedEvent(DomainEvent):
    """Event published when trading is paused due to risk constraints."""
    user_id: str
    reason: str
    paused_at: datetime
    resume_after: datetime | None = None


@dataclass(frozen=True)
class SentimentAnalysisCompletedEvent(DomainEvent):
    """Event published when sentiment analysis is completed."""
    symbol: Symbol
    sentiment_score: float
    confidence: float
    source: str
    analyzed_at: datetime


@dataclass(frozen=True)
class MarketAlertTriggeredEvent(DomainEvent):
    """Event published when a market alert is triggered."""
    symbol: Symbol
    alert_type: str  # PRICE_MOVEMENT, VOLUME_SPIKE, NEWS_EVENT
    message: str
    triggered_at: datetime
    severity: str = "INFO"  # INFO, WARNING, CRITICAL
