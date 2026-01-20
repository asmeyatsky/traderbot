"""
Tests for Domain Entities

Tests that domain entities maintain their invariants and behave correctly.
"""
import pytest
from datetime import datetime
from decimal import Decimal
from uuid import uuid4

from src.domain.entities.user import User, RiskTolerance, InvestmentGoal
from src.domain.entities.trading import Order, Position, Portfolio, OrderType, PositionType, OrderStatus
from src.domain.value_objects import Money, Symbol, Price
from src.domain.exceptions import OrderValidationException


class TestUser:
    """Test User entity."""

    def test_user_creation(self):
        """Test creating a user."""
        user = User(
            id="user1",
            email="test@example.com",
            first_name="John",
            last_name="Doe",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            risk_tolerance=RiskTolerance.MODERATE,
            investment_goal=InvestmentGoal.BALANCED_GROWTH,
        )

        assert user.id == "user1"
        assert user.email == "test@example.com"
        assert user.risk_tolerance == RiskTolerance.MODERATE

    def test_user_immutability(self):
        """Test that user entities are immutable."""
        user = User(
            id="user1",
            email="test@example.com",
            first_name="John",
            last_name="Doe",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            risk_tolerance=RiskTolerance.MODERATE,
            investment_goal=InvestmentGoal.BALANCED_GROWTH,
        )

        # Frozen dataclass should prevent attribute changes
        with pytest.raises(Exception):
            user.email = "newemail@example.com"

    def test_user_update_risk_tolerance(self):
        """Test updating user's risk tolerance creates new instance."""
        original = User(
            id="user1",
            email="test@example.com",
            first_name="John",
            last_name="Doe",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            risk_tolerance=RiskTolerance.MODERATE,
            investment_goal=InvestmentGoal.BALANCED_GROWTH,
        )

        updated = original.update_risk_tolerance(RiskTolerance.AGGRESSIVE)

        assert original.risk_tolerance == RiskTolerance.MODERATE
        assert updated.risk_tolerance == RiskTolerance.AGGRESSIVE
        assert original is not updated

    def test_user_validation(self):
        """Test user validation."""
        user = User(
            id="user1",
            email="invalid-email",
            first_name="John",
            last_name="Doe",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            risk_tolerance=RiskTolerance.MODERATE,
            investment_goal=InvestmentGoal.BALANCED_GROWTH,
        )

        errors = user.validate()
        assert any("email" in error.lower() for error in errors)


class TestOrder:
    """Test Order entity."""

    def test_order_creation(self):
        """Test creating an order."""
        order = Order(
            id="order1",
            user_id="user1",
            symbol=Symbol("AAPL"),
            order_type=OrderType.MARKET,
            position_type=PositionType.LONG,
            quantity=100,
            status=OrderStatus.PENDING,
            placed_at=datetime.now(),
        )

        assert order.id == "order1"
        assert order.symbol == Symbol("AAPL")
        assert order.status == OrderStatus.PENDING

    def test_order_immutability(self):
        """Test that order entities are immutable."""
        order = Order(
            id="order1",
            user_id="user1",
            symbol=Symbol("AAPL"),
            order_type=OrderType.MARKET,
            position_type=PositionType.LONG,
            quantity=100,
            status=OrderStatus.PENDING,
            placed_at=datetime.now(),
        )

        with pytest.raises(Exception):
            order.quantity = 200

    def test_order_execution(self):
        """Test executing an order creates new instance."""
        order = Order(
            id="order1",
            user_id="user1",
            symbol=Symbol("AAPL"),
            order_type=OrderType.MARKET,
            position_type=PositionType.LONG,
            quantity=100,
            status=OrderStatus.PENDING,
            placed_at=datetime.now(),
            price=Money(Decimal("150.00"), "USD"),
        )

        executed = order.execute(
            execution_price=Money(Decimal("150.00"), "USD"),
            executed_at=datetime.now(),
            filled_qty=100
        )

        assert order.status == OrderStatus.PENDING
        assert executed.status == OrderStatus.EXECUTED
        assert executed.filled_quantity == 100
        assert order is not executed

    def test_order_validation(self):
        """Test order validation."""
        # Valid order
        valid_order = Order(
            id="order1",
            user_id="user1",
            symbol=Symbol("AAPL"),
            order_type=OrderType.MARKET,
            position_type=PositionType.LONG,
            quantity=100,
            status=OrderStatus.PENDING,
            placed_at=datetime.now(),
        )

        errors = valid_order.validate()
        assert len(errors) == 0

        # Invalid order (zero quantity)
        invalid_order = Order(
            id="order2",
            user_id="user1",
            symbol=Symbol("AAPL"),
            order_type=OrderType.MARKET,
            position_type=PositionType.LONG,
            quantity=0,
            status=OrderStatus.PENDING,
            placed_at=datetime.now(),
        )

        errors = invalid_order.validate()
        assert len(errors) > 0

    def test_order_with_stop_price(self):
        """Test order with stop price for stop loss orders."""
        order = Order(
            id="order1",
            user_id="user1",
            symbol=Symbol("AAPL"),
            order_type=OrderType.STOP_LOSS,
            position_type=PositionType.LONG,
            quantity=100,
            status=OrderStatus.PENDING,
            placed_at=datetime.now(),
            stop_price=Money(Decimal("140.00"), "USD"),
        )

        assert order.stop_price.amount == Decimal("140.00")


class TestPortfolio:
    """Test Portfolio entity."""

    def test_portfolio_creation(self):
        """Test creating a portfolio."""
        portfolio = Portfolio(
            id="portfolio1",
            user_id="user1",
            positions=[],
            cash_balance=Money(Decimal("5000.00"), "USD"),
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        assert portfolio.id == "portfolio1"
        assert portfolio.total_value.amount == Decimal("5000.00")

    def test_portfolio_immutability(self):
        """Test that portfolio entities are immutable."""
        portfolio = Portfolio(
            id="portfolio1",
            user_id="user1",
            positions=[],
            cash_balance=Money(Decimal("5000.00"), "USD"),
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        with pytest.raises(Exception):
            portfolio.cash_balance = Money(Decimal("15000.00"), "USD")

    def test_portfolio_update_values(self):
        """Test updating portfolio values creates new instance."""
        original = Portfolio(
            id="portfolio1",
            user_id="user1",
            positions=[],
            cash_balance=Money(Decimal("5000.00"), "USD"),
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        updated = original.update_cash_balance(Money(Decimal("4000.00"), "USD"))

        assert original.cash_balance.amount == Decimal("5000.00")
        assert updated.cash_balance.amount == Decimal("4000.00")
        assert original is not updated


class TestPosition:
    """Test Position entity."""

    def test_position_creation(self):
        """Test creating a position."""
        position = Position(
            id="position1",
            user_id="user1",
            symbol=Symbol("AAPL"),
            quantity=100,
            position_type=PositionType.LONG,
            average_buy_price=Money(Decimal("150.00"), "USD"),
            current_price=Money(Decimal("155.00"), "USD"),
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        assert position.id == "position1"
        assert position.symbol == Symbol("AAPL")
        assert position.quantity == 100

    def test_position_immutability(self):
        """Test that position entities are immutable."""
        position = Position(
            id="position1",
            user_id="user1",
            symbol=Symbol("AAPL"),
            quantity=100,
            position_type=PositionType.LONG,
            average_buy_price=Money(Decimal("150.00"), "USD"),
            current_price=Money(Decimal("155.00"), "USD"),
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        with pytest.raises(Exception):
            position.quantity = 200

    def test_position_update_price(self):
        """Test updating position price creates new instance."""
        position = Position(
            id="position1",
            user_id="user1",
            symbol=Symbol("AAPL"),
            quantity=100,
            position_type=PositionType.LONG,
            average_buy_price=Money(Decimal("150.00"), "USD"),
            current_price=Money(Decimal("155.00"), "USD"),
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        updated = position.update_price(Money(Decimal("160.00"), "USD"))

        assert position.current_price.amount == Decimal("155.00")
        assert updated.current_price.amount == Decimal("160.00")
        assert position is not updated

    def test_position_gain_loss_calculation(self):
        """Test gain/loss calculation for a position."""
        position = Position(
            id="position1",
            user_id="user1",
            symbol=Symbol("AAPL"),
            quantity=100,
            position_type=PositionType.LONG,
            average_buy_price=Money(Decimal("150.00"), "USD"),
            current_price=Money(Decimal("160.00"), "USD"),
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        # Gain should be (160 - 150) * 100 = 1000
        assert position.unrealized_pnl_amount.amount == Decimal("1000.00")
