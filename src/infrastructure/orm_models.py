"""
SQLAlchemy ORM Models for the Trading Platform

This module defines the database models that map domain entities to relational tables.
Following clean architecture, these are infrastructure-layer concerns and should only
be used by the repository implementations.

Architectural Intent:
- ORM models are infrastructure concerns, separate from domain entities
- Domain entities are independent of persistence mechanisms
- Repositories translate between ORM models and domain entities
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from sqlalchemy import Column, String, DateTime, Numeric, Boolean, Integer, Enum as SQLEnum, ForeignKey, Index
from sqlalchemy.orm import relationship
from sqlalchemy.types import JSON

from src.infrastructure.database import Base
from src.domain.entities.user import RiskTolerance, InvestmentGoal
from src.domain.entities.trading import OrderType, PositionType, OrderStatus


class UserORM(Base):
    """ORM model for User entity."""
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    first_name = Column(String(255), nullable=False)
    last_name = Column(String(255), nullable=False)
    password_hash = Column(String(255), nullable=False)

    # Risk and investment preferences
    risk_tolerance = Column(SQLEnum(RiskTolerance), nullable=False, default=RiskTolerance.MODERATE)
    investment_goal = Column(SQLEnum(InvestmentGoal), nullable=False, default=InvestmentGoal.BALANCED_GROWTH)
    max_position_size_percentage = Column(Numeric(5, 2), nullable=False, default=Decimal('25'))

    # Loss limits
    daily_loss_limit = Column(Numeric(12, 2), nullable=True)
    weekly_loss_limit = Column(Numeric(12, 2), nullable=True)
    monthly_loss_limit = Column(Numeric(12, 2), nullable=True)

    # Sector preferences
    sector_preferences = Column(JSON, nullable=False, default=[])
    sector_exclusions = Column(JSON, nullable=False, default=[])

    # Settings
    is_active = Column(Boolean, nullable=False, default=True)
    email_notifications_enabled = Column(Boolean, nullable=False, default=True)
    sms_notifications_enabled = Column(Boolean, nullable=False, default=False)
    approval_mode_enabled = Column(Boolean, nullable=False, default=False)

    # GDPR Consent Tracking
    terms_accepted_at = Column(DateTime, nullable=True)
    privacy_accepted_at = Column(DateTime, nullable=True)
    marketing_consent = Column(Boolean, nullable=False, default=False)

    # Auto-trading
    auto_trading_enabled = Column(Boolean, nullable=False, default=False)
    watchlist = Column(JSON, nullable=False, default=[])
    trading_budget = Column(Numeric(15, 2), nullable=True)
    stop_loss_pct = Column(Numeric(5, 2), nullable=False, default=Decimal('5'))
    take_profit_pct = Column(Numeric(5, 2), nullable=False, default=Decimal('10'))
    confidence_threshold = Column(Numeric(3, 2), nullable=False, default=Decimal('0.6'))
    max_position_pct = Column(Numeric(5, 2), nullable=False, default=Decimal('20'))
    allowed_markets = Column(JSON, nullable=False, default=["US_NYSE", "US_NASDAQ", "UK_LSE", "EU_EURONEXT", "DE_XETRA", "JP_TSE", "HK_HKEX"])

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    portfolios = relationship("PortfolioORM", back_populates="user", cascade="all, delete-orphan")
    orders = relationship("OrderORM", back_populates="user", cascade="all, delete-orphan")
    positions = relationship("PositionORM", back_populates="user", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_user_email', 'email'),
        Index('idx_user_created_at', 'created_at'),
    )


class PortfolioORM(Base):
    """ORM model for Portfolio entity."""
    __tablename__ = "portfolios"

    id = Column(String(36), primary_key=True, index=True)
    user_id = Column(String(36), ForeignKey('users.id'), nullable=False, index=True)

    # Portfolio values
    total_value = Column(Numeric(15, 2), nullable=False, default=Decimal('0'))
    cash_balance = Column(Numeric(15, 2), nullable=False, default=Decimal('0'))
    invested_value = Column(Numeric(15, 2), nullable=False, default=Decimal('0'))

    # Performance metrics
    total_gain_loss = Column(Numeric(15, 2), nullable=False, default=Decimal('0'))
    total_return_percentage = Column(Numeric(5, 2), nullable=False, default=Decimal('0'))
    ytd_return_percentage = Column(Numeric(5, 2), nullable=False, default=Decimal('0'))

    # Risk metrics
    current_drawdown = Column(Numeric(5, 2), nullable=False, default=Decimal('0'))
    peak_value = Column(Numeric(15, 2), nullable=False, default=Decimal('0'))

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("UserORM", back_populates="portfolios")

    __table_args__ = (
        Index('idx_portfolio_user_id', 'user_id'),
        Index('idx_portfolio_created_at', 'created_at'),
    )


class OrderORM(Base):
    """ORM model for Order entity."""
    __tablename__ = "orders"

    id = Column(String(36), primary_key=True, index=True)
    user_id = Column(String(36), ForeignKey('users.id'), nullable=False, index=True)

    # Order details
    symbol = Column(String(10), nullable=False, index=True)
    order_type = Column(SQLEnum(OrderType), nullable=False)
    position_type = Column(SQLEnum(PositionType), nullable=False)
    quantity = Column(Numeric(12, 2), nullable=False)
    status = Column(SQLEnum(OrderStatus), nullable=False, default=OrderStatus.PENDING, index=True)

    # Pricing
    price = Column(Numeric(12, 4), nullable=True)
    limit_price = Column(Numeric(12, 4), nullable=True)
    stop_price = Column(Numeric(12, 4), nullable=True)
    commission = Column(Numeric(12, 4), nullable=True, default=Decimal('0'))

    # Execution details
    filled_quantity = Column(Numeric(12, 2), nullable=False, default=Decimal('0'))

    # Timestamps
    placed_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    executed_at = Column(DateTime, nullable=True)

    # Broker integration
    broker_order_id = Column(String(255), nullable=True)

    # Additional info
    notes = Column(String(500), nullable=True)

    # Relationships
    user = relationship("UserORM", back_populates="orders")

    __table_args__ = (
        Index('idx_order_user_id', 'user_id'),
        Index('idx_order_status', 'status'),
        Index('idx_order_symbol', 'symbol'),
        Index('idx_order_user_status', 'user_id', 'status'),
        Index('idx_order_placed_at', 'placed_at'),
    )


class PositionORM(Base):
    """ORM model for Position entity."""
    __tablename__ = "positions"

    id = Column(String(36), primary_key=True, index=True)
    user_id = Column(String(36), ForeignKey('users.id'), nullable=False, index=True)

    # Position details
    symbol = Column(String(10), nullable=False, index=True)
    quantity = Column(Numeric(12, 2), nullable=False)
    position_type = Column(SQLEnum(PositionType), nullable=False)

    # Pricing
    average_entry_price = Column(Numeric(12, 4), nullable=False)
    current_price = Column(Numeric(12, 4), nullable=True)

    # Performance
    unrealized_gain_loss = Column(Numeric(15, 2), nullable=False, default=Decimal('0'))
    realized_gain_loss = Column(Numeric(15, 2), nullable=False, default=Decimal('0'))

    # Timestamps
    opened_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    closed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("UserORM", back_populates="positions")

    __table_args__ = (
        Index('idx_position_user_id', 'user_id'),
        Index('idx_position_symbol', 'symbol'),
        Index('idx_position_user_symbol', 'user_id', 'symbol'),
        Index('idx_position_opened_at', 'opened_at'),
    )


class TradingActivityLogORM(Base):
    """ORM model for autonomous trading activity log."""
    __tablename__ = "trading_activity_log"

    id = Column(String(36), primary_key=True, index=True)
    user_id = Column(String(36), ForeignKey('users.id'), nullable=False, index=True)
    event_type = Column(String(50), nullable=False, index=True)
    symbol = Column(String(10), nullable=True)
    signal = Column(String(20), nullable=True)
    confidence = Column(Numeric(5, 4), nullable=True)
    order_id = Column(String(36), ForeignKey('orders.id'), nullable=True)
    broker_order_id = Column(String(255), nullable=True)
    quantity = Column(Numeric(12, 2), nullable=True)
    price = Column(Numeric(12, 4), nullable=True)
    message = Column(String(1000), nullable=True)
    metadata_json = Column("metadata", JSON, nullable=True)

    occurred_at = Column(DateTime, nullable=False)
    recorded_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_activity_user_id', 'user_id'),
        Index('idx_activity_event_type', 'event_type'),
        Index('idx_activity_occurred_at', 'occurred_at'),
        Index('idx_activity_user_event', 'user_id', 'event_type'),
    )


class ConversationORM(Base):
    """ORM model for Conversation entity."""
    __tablename__ = "conversations"

    id = Column(String(36), primary_key=True, index=True)
    user_id = Column(String(36), ForeignKey('users.id'), nullable=False, index=True)
    title = Column(String(500), nullable=False, default="New conversation")

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    messages = relationship("MessageORM", back_populates="conversation", cascade="all, delete-orphan",
                            order_by="MessageORM.created_at")

    __table_args__ = (
        Index('idx_conversation_user_id', 'user_id'),
        Index('idx_conversation_updated_at', 'updated_at'),
    )


class MessageORM(Base):
    """ORM model for Message entity."""
    __tablename__ = "messages"

    id = Column(String(36), primary_key=True, index=True)
    conversation_id = Column(String(36), ForeignKey('conversations.id', ondelete='CASCADE'), nullable=False, index=True)
    role = Column(String(20), nullable=False)  # user, assistant, system
    content = Column(String, nullable=False)
    metadata_json = Column("metadata", JSON, nullable=True)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    conversation = relationship("ConversationORM", back_populates="messages")

    __table_args__ = (
        Index('idx_message_conversation_id', 'conversation_id'),
        Index('idx_message_created_at', 'created_at'),
    )


class BrokerAccountORM(Base):
    """ORM model for BrokerAccount entity. API keys stored encrypted."""
    __tablename__ = "broker_accounts"

    id = Column(String(36), primary_key=True, index=True)
    user_id = Column(String(36), ForeignKey('users.id'), nullable=False, index=True)
    broker_type = Column(String(50), nullable=False)
    encrypted_api_key = Column(String(500), nullable=False)
    encrypted_secret_key = Column(String(500), nullable=False)
    paper_trading = Column(Boolean, nullable=False, default=True)
    label = Column(String(255), nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index('idx_broker_account_user_id', 'user_id'),
        Index('idx_broker_account_user_broker', 'user_id', 'broker_type', unique=True),
    )


class SavedStrategyORM(Base):
    """ORM model for SavedStrategy entity."""
    __tablename__ = "saved_strategies"

    id = Column(String(36), primary_key=True, index=True)
    user_id = Column(String(36), ForeignKey('users.id'), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(String(1000), nullable=False, default='')
    strategy_type = Column(String(50), nullable=False)
    parameters = Column(JSON, nullable=False, default={})
    symbol = Column(String(10), nullable=False, default='AAPL')
    is_public = Column(Boolean, nullable=False, default=False)
    fork_count = Column(Integer, nullable=False, default=0)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    backtest_results = relationship("BacktestResultORM", back_populates="strategy", cascade="all, delete-orphan")
    follows = relationship("StrategyFollowORM", back_populates="strategy", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_strategy_user_id', 'user_id'),
        Index('idx_strategy_public', 'is_public'),
    )


class BacktestResultORM(Base):
    """ORM model for persisted backtest results."""
    __tablename__ = "backtest_results"

    id = Column(String(36), primary_key=True, index=True)
    strategy_id = Column(String(36), ForeignKey('saved_strategies.id', ondelete='CASCADE'), nullable=False, index=True)
    user_id = Column(String(36), ForeignKey('users.id'), nullable=False, index=True)
    symbol = Column(String(10), nullable=False)
    initial_capital = Column(Numeric(15, 2), nullable=False)
    final_value = Column(Numeric(15, 2), nullable=False)
    total_return_pct = Column(Numeric(8, 2), nullable=False)
    sharpe_ratio = Column(Numeric(8, 4), nullable=False)
    max_drawdown_pct = Column(Numeric(8, 2), nullable=False)
    win_rate = Column(Numeric(5, 2), nullable=False)
    total_trades = Column(Integer, nullable=False)
    volatility = Column(Numeric(8, 2), nullable=False)
    profit_factor = Column(Numeric(8, 4), nullable=False)
    run_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    strategy = relationship("SavedStrategyORM", back_populates="backtest_results")

    __table_args__ = (
        Index('idx_backtest_strategy_id', 'strategy_id'),
        Index('idx_backtest_user_id', 'user_id'),
    )


class StrategyFollowORM(Base):
    """ORM model for strategy follow / copy trading."""
    __tablename__ = "strategy_follows"

    id = Column(String(36), primary_key=True, index=True)
    follower_user_id = Column(String(36), ForeignKey('users.id'), nullable=False, index=True)
    strategy_id = Column(String(36), ForeignKey('saved_strategies.id', ondelete='CASCADE'), nullable=False, index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    strategy = relationship("SavedStrategyORM", back_populates="follows")

    __table_args__ = (
        Index('idx_follow_user_strategy', 'follower_user_id', 'strategy_id', unique=True),
    )


class DomainEventORM(Base):
    """ORM model for Domain Events (audit trail and event sourcing)."""
    __tablename__ = "domain_events"

    id = Column(String(36), primary_key=True, index=True)
    aggregate_type = Column(String(100), nullable=False, index=True)
    aggregate_id = Column(String(36), nullable=False, index=True)
    event_type = Column(String(100), nullable=False, index=True)
    event_data = Column(JSON, nullable=False)

    user_id = Column(String(36), ForeignKey('users.id'), nullable=True, index=True)

    # Timestamps
    occurred_at = Column(DateTime, nullable=False, index=True)
    recorded_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_event_aggregate', 'aggregate_type', 'aggregate_id'),
        Index('idx_event_type', 'event_type'),
        Index('idx_event_user_id', 'user_id'),
        Index('idx_event_occurred_at', 'occurred_at'),
    )
