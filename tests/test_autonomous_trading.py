"""
Tests for AutonomousTradingService

Verifies signal processing, order placement, risk blocking, polling, and
portfolio position hydration.
"""
from __future__ import annotations

import sys
import types
import uuid
from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

# Stub out heavy third-party modules that fail on Python 3.14
for _mod in (
    "yfinance", "alpha_vantage", "alpha_vantage.timeseries",
    "finnhub", "polygon", "polygon.rest",
    "google", "google.protobuf", "google.protobuf.descriptor",
    "tensorflow", "torch", "transformers", "xgboost",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)

from src.application.services.autonomous_trading_service import (
    AutonomousTradingService,
    SIGNAL_GENERATED,
    ORDER_PLACED,
    ORDER_FILLED,
    ORDER_FAILED,
    RISK_BLOCKED,
)
from src.domain.entities.trading import (
    Order, OrderStatus, OrderType, Portfolio, Position, PositionType,
)
from src.domain.entities.user import User, RiskTolerance, InvestmentGoal
from src.domain.value_objects import Money, Symbol


@dataclass
class BrokerOrderResponse:
    broker_order_id: str
    status: str
    filled_qty: int = 0
    avg_fill_price: float = 0.0


@dataclass
class TradingSignal:
    signal: str
    confidence: float
    explanation: str
    score: float


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_user(**overrides) -> User:
    defaults = dict(
        id=str(uuid.uuid4()),
        email="test@example.com",
        first_name="Test",
        last_name="User",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        risk_tolerance=RiskTolerance.MODERATE,
        investment_goal=InvestmentGoal.BALANCED_GROWTH,
        auto_trading_enabled=True,
        watchlist=["AAPL"],
        trading_budget=Money(Decimal("10000"), "USD"),
    )
    defaults.update(overrides)
    return User(**defaults)


def _make_portfolio(user_id: str, cash: Decimal = Decimal("10000"), positions=None) -> Portfolio:
    return Portfolio(
        id=str(uuid.uuid4()),
        user_id=user_id,
        positions=positions or [],
        cash_balance=Money(cash, "USD"),
    )


def _make_position(user_id: str, symbol: str = "AAPL", qty: int = 10, price: Decimal = Decimal("150")) -> Position:
    return Position(
        id=str(uuid.uuid4()),
        user_id=user_id,
        symbol=Symbol(symbol),
        position_type=PositionType.LONG,
        quantity=qty,
        average_buy_price=Money(price, "USD"),
        current_price=Money(price, "USD"),
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )


def _make_service(**overrides) -> AutonomousTradingService:
    defaults = dict(
        user_repository=MagicMock(),
        portfolio_repository=MagicMock(),
        position_repository=MagicMock(),
        order_repository=MagicMock(),
        activity_log_repository=MagicMock(),
        ml_model_service=MagicMock(),
        broker_service=MagicMock(),
        risk_manager=MagicMock(),
        circuit_breaker=MagicMock(),
        market_data_service=MagicMock(),
        confidence_threshold=0.6,
    )
    defaults.update(overrides)
    return AutonomousTradingService(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCircuitBreaker:
    def test_cycle_respects_circuit_breaker(self):
        svc = _make_service()
        svc.circuit_breaker.is_trading_allowed.return_value = False

        svc.run_trading_cycle()

        svc.user_repo.get_auto_trading_users.assert_not_called()
        svc.broker.place_order.assert_not_called()


class TestSignalFiltering:
    def test_cycle_skips_hold_signals(self):
        user = _make_user()
        svc = _make_service()
        svc.circuit_breaker.is_trading_allowed.return_value = True
        svc.user_repo.get_auto_trading_users.return_value = [user]
        svc.portfolio_repo.get_by_user_id.return_value = _make_portfolio(user.id)
        svc.risk_manager.should_pause_trading.return_value = False
        svc.ml_service.predict_price_direction.return_value = TradingSignal(
            signal="HOLD", confidence=0.7, explanation="No trend", score=0.0
        )

        svc.run_trading_cycle()

        svc.broker.place_order.assert_not_called()

    def test_cycle_skips_low_confidence(self):
        user = _make_user()
        svc = _make_service()
        svc.circuit_breaker.is_trading_allowed.return_value = True
        svc.user_repo.get_auto_trading_users.return_value = [user]
        svc.portfolio_repo.get_by_user_id.return_value = _make_portfolio(user.id)
        svc.risk_manager.should_pause_trading.return_value = False
        svc.ml_service.predict_price_direction.return_value = TradingSignal(
            signal="BUY", confidence=0.4, explanation="Low confidence", score=0.2
        )

        svc.run_trading_cycle()

        svc.broker.place_order.assert_not_called()


class TestBuyOrders:
    def test_cycle_places_buy_order(self):
        user = _make_user()
        svc = _make_service()
        svc.circuit_breaker.is_trading_allowed.return_value = True
        svc.user_repo.get_auto_trading_users.return_value = [user]
        svc.portfolio_repo.get_by_user_id.return_value = _make_portfolio(user.id)
        svc.risk_manager.should_pause_trading.return_value = False
        svc.risk_manager.validate_order.return_value = []  # No risk errors

        svc.ml_service.predict_price_direction.return_value = TradingSignal(
            signal="BUY", confidence=0.85, explanation="Strong uptrend", score=0.8
        )
        # Current price $100 → budget $10000 → 100 shares
        from src.domain.value_objects import Price
        svc.market_data.get_current_price.return_value = Price(amount=Decimal("100"), currency="USD")
        svc.broker.place_order.return_value = BrokerOrderResponse(
            broker_order_id="broker-123", status="pending"
        )

        svc.run_trading_cycle()

        svc.broker.place_order.assert_called_once()
        placed_order = svc.broker.place_order.call_args[0][0]
        assert placed_order.quantity == 100
        assert placed_order.position_type == PositionType.LONG
        assert str(placed_order.symbol) == "AAPL"

        svc.order_repo.save.assert_called_once()
        saved_order = svc.order_repo.save.call_args[0][0]
        assert saved_order.broker_order_id == "broker-123"


class TestSellOrders:
    def test_cycle_sells_existing_position(self):
        user = _make_user()
        position = _make_position(user.id, "AAPL", qty=50)
        portfolio = _make_portfolio(user.id, positions=[position])

        svc = _make_service()
        svc.circuit_breaker.is_trading_allowed.return_value = True
        svc.user_repo.get_auto_trading_users.return_value = [user]
        svc.portfolio_repo.get_by_user_id.return_value = portfolio
        svc.risk_manager.should_pause_trading.return_value = False
        svc.ml_service.predict_price_direction.return_value = TradingSignal(
            signal="SELL", confidence=0.9, explanation="Bearish", score=-0.8
        )
        from src.domain.value_objects import Price
        svc.market_data.get_current_price.return_value = Price(amount=Decimal("150"), currency="USD")
        svc.broker.place_order.return_value = BrokerOrderResponse(
            broker_order_id="broker-sell-1", status="pending"
        )

        svc.run_trading_cycle()

        svc.broker.place_order.assert_called_once()
        placed_order = svc.broker.place_order.call_args[0][0]
        assert placed_order.position_type == PositionType.SHORT
        assert placed_order.quantity == 50

    def test_cycle_skips_sell_without_position(self):
        user = _make_user()
        portfolio = _make_portfolio(user.id, positions=[])  # No position

        svc = _make_service()
        svc.circuit_breaker.is_trading_allowed.return_value = True
        svc.user_repo.get_auto_trading_users.return_value = [user]
        svc.portfolio_repo.get_by_user_id.return_value = portfolio
        svc.risk_manager.should_pause_trading.return_value = False
        svc.ml_service.predict_price_direction.return_value = TradingSignal(
            signal="SELL", confidence=0.9, explanation="Bearish", score=-0.8
        )

        svc.run_trading_cycle()

        svc.broker.place_order.assert_not_called()


class TestRiskBlocking:
    def test_risk_manager_blocks_order(self):
        user = _make_user()
        svc = _make_service()
        svc.circuit_breaker.is_trading_allowed.return_value = True
        svc.user_repo.get_auto_trading_users.return_value = [user]
        svc.portfolio_repo.get_by_user_id.return_value = _make_portfolio(user.id)
        svc.risk_manager.should_pause_trading.return_value = False
        svc.risk_manager.validate_order.return_value = ["Position too large"]
        svc.ml_service.predict_price_direction.return_value = TradingSignal(
            signal="BUY", confidence=0.85, explanation="Uptrend", score=0.8
        )
        from src.domain.value_objects import Price
        svc.market_data.get_current_price.return_value = Price(amount=Decimal("100"), currency="USD")

        svc.run_trading_cycle()

        svc.broker.place_order.assert_not_called()
        # Verify risk-blocked event was logged
        log_calls = svc.activity_log.log_event.call_args_list
        event_types = [c.kwargs.get("event_type") or c[1].get("event_type", "") for c in log_calls]
        assert RISK_BLOCKED in event_types


class TestOrderPolling:
    def test_poll_updates_filled_order(self):
        svc = _make_service()
        order = Order(
            id=str(uuid.uuid4()),
            user_id="user-1",
            symbol=Symbol("AAPL"),
            order_type=OrderType.MARKET,
            position_type=PositionType.LONG,
            quantity=10,
            status=OrderStatus.PENDING,
            placed_at=datetime.utcnow(),
            price=Money(Decimal("150"), "USD"),
            broker_order_id="broker-456",
        )
        svc.order_repo.get_pending_with_broker_id.return_value = [order]
        svc.broker.get_order_status.return_value = "filled"
        svc.position_repo.get_by_symbol.return_value = None  # No existing position
        svc.portfolio_repo.get_by_user_id.return_value = _make_portfolio("user-1")

        svc.poll_pending_orders()

        # Order should be updated to EXECUTED
        svc.order_repo.update_order.assert_called_once()
        updated = svc.order_repo.update_order.call_args[0][0]
        assert updated.status == OrderStatus.EXECUTED
        assert updated.filled_quantity == 10

        # Position should be created
        svc.position_repo.save.assert_called_once()
        new_pos = svc.position_repo.save.call_args[0][0]
        assert new_pos.quantity == 10
        assert str(new_pos.symbol) == "AAPL"

        # Portfolio cash should be updated
        svc.portfolio_repo.update.assert_called_once()


class TestPortfolioPositionsLoaded:
    """Regression test: portfolio.positions must be hydrated from PositionRepository."""

    def test_portfolio_positions_loaded(self):
        """Verify the portfolio repo hydrates positions so sell logic can find them."""
        user = _make_user()
        position = _make_position(user.id, "AAPL", qty=20)

        # Construct a portfolio WITH a position (simulating the fix)
        portfolio = _make_portfolio(user.id, positions=[position])

        svc = _make_service()
        svc.circuit_breaker.is_trading_allowed.return_value = True
        svc.user_repo.get_auto_trading_users.return_value = [user]
        svc.portfolio_repo.get_by_user_id.return_value = portfolio
        svc.risk_manager.should_pause_trading.return_value = False
        svc.ml_service.predict_price_direction.return_value = TradingSignal(
            signal="SELL", confidence=0.9, explanation="Bearish", score=-0.8
        )
        from src.domain.value_objects import Price
        svc.market_data.get_current_price.return_value = Price(amount=Decimal("150"), currency="USD")
        svc.broker.place_order.return_value = BrokerOrderResponse(
            broker_order_id="b-sell", status="pending"
        )

        svc.run_trading_cycle()

        # Should have placed a sell since position exists
        svc.broker.place_order.assert_called_once()
        placed = svc.broker.place_order.call_args[0][0]
        assert placed.quantity == 20
