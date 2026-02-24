"""
Trading Domain Service — Order Validation Tests

Regression suite for Bug #3: Cash validation was skipped for market buy orders.
These tests verify that validate_order() correctly checks cash sufficiency
for LONG orders and skips it for SHORT orders.
"""
import pytest
from datetime import datetime
from decimal import Decimal

from src.domain.entities.trading import (
    Order, Portfolio, OrderType, PositionType, OrderStatus,
)
from src.domain.entities.user import User, RiskTolerance, InvestmentGoal
from src.domain.services.trading import DefaultTradingDomainService
from src.domain.value_objects import Money, Symbol, Price


def _money(value: str) -> Money:
    return Money(Decimal(value), "USD")


def _make_user(**overrides) -> User:
    defaults = dict(
        id="user-1",
        email="test@example.com",
        first_name="Test",
        last_name="User",
        created_at=datetime(2025, 1, 1),
        updated_at=datetime(2025, 1, 1),
        risk_tolerance=RiskTolerance.MODERATE,
        investment_goal=InvestmentGoal.BALANCED_GROWTH,
        max_position_size_percentage=Decimal("25"),
    )
    defaults.update(overrides)
    return User(**defaults)


def _make_portfolio(**overrides) -> Portfolio:
    defaults = dict(
        id="port-1",
        user_id="user-1",
        positions=[],
        cash_balance=_money("10000.00"),
        created_at=datetime(2025, 1, 1),
        updated_at=datetime(2025, 1, 1),
    )
    defaults.update(overrides)
    return Portfolio(**defaults)


def _make_order(**overrides) -> Order:
    defaults = dict(
        id="order-1",
        user_id="user-1",
        symbol=Symbol("AAPL"),
        order_type=OrderType.MARKET,
        position_type=PositionType.LONG,
        quantity=10,
        status=OrderStatus.PENDING,
        placed_at=datetime(2025, 1, 1),
        price=_money("150.00"),
    )
    defaults.update(overrides)
    return Order(**defaults)


class TestCashValidation:
    """Regression for Bug #3: LONG orders must check cash sufficiency."""

    def setup_method(self):
        self.service = DefaultTradingDomainService()

    def test_long_order_rejected_when_insufficient_cash(self):
        order = _make_order(quantity=100, price=_money("150.00"))  # $15,000
        portfolio = _make_portfolio(cash_balance=_money("500.00"))
        user = _make_user(max_position_size_percentage=Decimal("100"))
        errors = self.service.validate_order(order, user, portfolio)
        assert any("Insufficient cash" in e for e in errors)

    def test_long_order_accepted_when_sufficient_cash(self):
        order = _make_order(quantity=10, price=_money("150.00"))  # $1,500
        portfolio = _make_portfolio(cash_balance=_money("2000.00"))
        user = _make_user(max_position_size_percentage=Decimal("100"))
        errors = self.service.validate_order(order, user, portfolio)
        assert not any("Insufficient cash" in e for e in errors)

    def test_short_order_skips_cash_check(self):
        order = _make_order(
            position_type=PositionType.SHORT,
            quantity=100,
            price=_money("150.00"),
        )
        portfolio = _make_portfolio(cash_balance=_money("0.00"))
        user = _make_user(max_position_size_percentage=Decimal("100"))
        errors = self.service.validate_order(order, user, portfolio)
        assert not any("Insufficient cash" in e for e in errors)


class TestPositionSizeValidation:
    def setup_method(self):
        self.service = DefaultTradingDomainService()

    def test_position_size_percentage_exceeded(self):
        # Portfolio value = $10,000. Order = $6,000 = 60% > max 5%
        order = _make_order(quantity=40, price=_money("150.00"))
        portfolio = _make_portfolio(cash_balance=_money("10000.00"))
        user = _make_user(max_position_size_percentage=Decimal("5"))
        errors = self.service.validate_order(order, user, portfolio)
        assert any("exceeds" in e for e in errors)

    def test_position_size_percentage_within_limit(self):
        # Portfolio value = $10,000. Order = $150 = 1.5% < max 5%
        order = _make_order(quantity=1, price=_money("150.00"))
        portfolio = _make_portfolio(cash_balance=_money("10000.00"))
        user = _make_user(max_position_size_percentage=Decimal("5"))
        errors = self.service.validate_order(order, user, portfolio)
        assert not any("exceeds" in e for e in errors)


class TestOrderFieldValidation:
    def setup_method(self):
        self.service = DefaultTradingDomainService()

    def test_zero_quantity_rejected(self):
        order = _make_order(quantity=0)
        portfolio = _make_portfolio()
        user = _make_user()
        errors = self.service.validate_order(order, user, portfolio)
        assert any("Quantity must be positive" in e for e in errors)

    def test_negative_price_rejected(self):
        """Price VO itself rejects negative amounts at construction."""
        with pytest.raises(ValueError, match="cannot be negative"):
            Price(Decimal("-1"), "USD")

    def test_market_order_with_4dp_price_validates(self):
        """Price with 4dp shouldn't crash validation when quantized."""
        price_4dp = Price(Decimal("150.1234"), "USD")
        # In the real flow, the order gets a Money (2dp) via quantize.
        # Here we verify that the Price VO itself is fine with 4dp.
        assert price_4dp.amount == Decimal("150.1234")
        # And that quantizing to Money works
        money = Money(price_4dp.amount.quantize(Decimal("0.01")), "USD")
        assert money.amount == Decimal("150.12")
