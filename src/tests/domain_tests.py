"""
Domain Layer Tests

Comprehensive tests for domain entities, value objects, and services.
Tests focus on business logic and invariant enforcement.
"""
from __future__ import annotations

import pytest
from datetime import datetime
from decimal import Decimal

from src.domain.entities.trading import Order, Position, Portfolio, OrderStatus, OrderType, PositionType
from src.domain.entities.user import User, RiskTolerance, InvestmentGoal
from src.domain.value_objects import Money, Symbol, Price, NewsSentiment
from src.domain.exceptions import (
    OrderValidationException, InsufficientFundsException,
    InvalidUserConfiguration, DomainException
)


class TestOrderEntity:
    """Tests for Order domain entity."""

    @pytest.fixture
    def order(self):
        """Create a sample order for testing."""
        return Order(
            id="ORD-001",
            user_id="USER-001",
            symbol=Symbol("AAPL"),
            order_type=OrderType.MARKET,
            position_type=PositionType.LONG,
            quantity=100,
            status=OrderStatus.PENDING,
            placed_at=datetime.now(),
        )

    def test_order_is_immutable(self, order):
        """Test that Order instances are frozen and immutable."""
        with pytest.raises(Exception):  # FrozenInstanceError
            order.quantity = 200

    def test_order_execution(self, order):
        """Test order execution creates new instance."""
        exec_price = Price(Decimal("150.00"), "USD")
        executed_order = order.execute(exec_price, datetime.now(), 100)

        assert executed_order.status == OrderStatus.EXECUTED
        assert executed_order.filled_quantity == 100
        assert executed_order.price == exec_price
        # Original order unchanged
        assert order.status == OrderStatus.PENDING

    def test_order_cancellation(self, order):
        """Test order cancellation creates new instance."""
        cancelled_order = order.cancel()

        assert cancelled_order.status == OrderStatus.CANCELLED
        # Original order unchanged
        assert order.status == OrderStatus.PENDING

    def test_order_validation_quantity(self):
        """Test order validation catches invalid quantity."""
        invalid_order = Order(
            id="ORD-002",
            user_id="USER-001",
            symbol=Symbol("GOOGL"),
            order_type=OrderType.MARKET,
            position_type=PositionType.SHORT,
            quantity=-100,  # Invalid: negative quantity
            status=OrderStatus.PENDING,
            placed_at=datetime.now(),
        )

        errors = invalid_order.validate()
        assert len(errors) > 0
        assert "Quantity must be positive" in errors

    def test_order_is_filled(self, order):
        """Test order filled property."""
        assert not order.is_filled  # Not filled yet

        # Execute with partial fill
        partial_exec = order.execute(
            Price(Decimal("150.00"), "USD"),
            datetime.now(),
            50  # Only 50 filled out of 100
        )
        assert not partial_exec.is_filled

        # Execute with full fill
        full_exec = order.execute(
            Price(Decimal("150.00"), "USD"),
            datetime.now(),
            100
        )
        assert full_exec.is_filled


class TestPositionEntity:
    """Tests for Position domain entity."""

    @pytest.fixture
    def position(self):
        """Create a sample position for testing."""
        return Position(
            id="POS-001",
            user_id="USER-001",
            symbol=Symbol("AAPL"),
            position_type=PositionType.LONG,
            quantity=100,
            average_buy_price=Money(Decimal("150.00"), "USD"),
            current_price=Money(Decimal("160.00"), "USD"),
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

    def test_position_market_value(self, position):
        """Test position market value calculation."""
        expected_value = Decimal("100") * Decimal("160.00")
        assert position.market_value.amount == expected_value

    def test_position_total_cost(self, position):
        """Test position total cost calculation."""
        expected_cost = Decimal("100") * Decimal("150.00")
        assert position.total_cost.amount == expected_cost

    def test_position_unrealized_pnl(self, position):
        """Test position unrealized PnL calculation."""
        expected_pnl = Decimal("1000.00")  # (160 - 150) * 100
        assert position.unrealized_pnl_amount.amount == expected_pnl

    def test_position_update_price(self, position):
        """Test position price update."""
        new_price = Money(Decimal("170.00"), "USD")
        updated_position = position.update_price(new_price)

        assert updated_position.current_price == new_price
        # Original unchanged
        assert position.current_price == Money(Decimal("160.00"), "USD")

    def test_position_adjust_quantity(self, position):
        """Test position quantity adjustment."""
        new_execution_price = Money(Decimal("155.00"), "USD")
        adjusted_position = position.adjust_quantity(50, new_execution_price)

        # New quantity should be 150
        assert adjusted_position.quantity == 150
        # Average price should be recalculated


class TestUserEntity:
    """Tests for User domain entity."""

    @pytest.fixture
    def user(self):
        """Create a sample user for testing."""
        return User(
            id="USER-001",
            email="trader@example.com",
            first_name="John",
            last_name="Trader",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            risk_tolerance=RiskTolerance.MODERATE,
            investment_goal=InvestmentGoal.BALANCED_GROWTH,
        )

    def test_user_is_immutable(self, user):
        """Test that User instances are immutable."""
        with pytest.raises(Exception):  # FrozenInstanceError
            user.first_name = "Jane"

    def test_user_update_risk_tolerance(self, user):
        """Test user risk tolerance update."""
        updated_user = user.update_risk_tolerance(RiskTolerance.AGGRESSIVE)

        assert updated_user.risk_tolerance == RiskTolerance.AGGRESSIVE
        # Original unchanged
        assert user.risk_tolerance == RiskTolerance.MODERATE

    def test_user_validation_email(self):
        """Test user validation catches invalid email."""
        invalid_user = User(
            id="USER-002",
            email="invalid-email",  # Invalid email
            first_name="Jane",
            last_name="Trader",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            risk_tolerance=RiskTolerance.CONSERVATIVE,
            investment_goal=InvestmentGoal.CAPITAL_PRESERVATION,
        )

        errors = invalid_user.validate()
        assert len(errors) > 0


class TestMoneyValueObject:
    """Tests for Money value object."""

    def test_money_addition(self):
        """Test Money addition."""
        money1 = Money(Decimal("100.00"), "USD")
        money2 = Money(Decimal("50.00"), "USD")

        result = money1 + money2
        assert result.amount == Decimal("150.00")
        assert result.currency == "USD"

    def test_money_prevents_mixed_currencies(self):
        """Test Money prevents mixed currency operations."""
        usd = Money(Decimal("100.00"), "USD")
        eur = Money(Decimal("100.00"), "EUR")

        with pytest.raises(ValueError):
            usd + eur

    def test_money_multiplication(self):
        """Test Money multiplication."""
        money = Money(Decimal("100.00"), "USD")
        result = money * Decimal("2")

        assert result.amount == Decimal("200.00")

    def test_money_is_positive(self):
        """Test Money positive check."""
        positive = Money(Decimal("100.00"), "USD")
        zero = Money(Decimal("0"), "USD")
        negative = Money(Decimal("-100.00"), "USD")

        assert positive.is_positive()
        assert not zero.is_positive()
        assert negative.is_negative()


class TestSymbolValueObject:
    """Tests for Symbol value object."""

    def test_symbol_valid(self):
        """Test valid symbol creation."""
        symbol = Symbol("AAPL")
        assert str(symbol) == "AAPL"

    def test_symbol_validation(self):
        """Test symbol validation."""
        with pytest.raises(ValueError):
            Symbol("INVALID123")  # Too many characters

    def test_symbol_case_insensitive_equality(self):
        """Test symbol equality is case-insensitive."""
        symbol1 = Symbol("AAPL")
        symbol2 = Symbol("aapl")

        assert symbol1 == symbol2


class TestPriceValueObject:
    """Tests for Price value object."""

    def test_price_comparison(self):
        """Test price comparisons."""
        price1 = Price(Decimal("100.00"), "USD")
        price2 = Price(Decimal("110.00"), "USD")

        assert price1 < price2
        assert price2 > price1
        assert price1 <= price2

    def test_price_prevents_negative(self):
        """Test Price prevents negative values."""
        with pytest.raises(ValueError):
            Price(Decimal("-100.00"), "USD")

    def test_price_change_calculation(self):
        """Test price change calculation."""
        price1 = Price(Decimal("100.00"), "USD")
        price2 = Price(Decimal("110.00"), "USD")

        # 10% change
        change = price2.calculate_change(price1)
        assert change == Decimal("10")


class TestNewsSentimentValueObject:
    """Tests for NewsSentiment value object."""

    def test_sentiment_creation(self):
        """Test sentiment creation with valid values."""
        sentiment = NewsSentiment(
            score=Decimal("75"),
            confidence=Decimal("85"),
            source="VADER"
        )

        assert sentiment.is_positive()
        assert not sentiment.is_negative()

    def test_sentiment_validation(self):
        """Test sentiment validation."""
        with pytest.raises(ValueError):
            NewsSentiment(
                score=Decimal("150"),  # Invalid: > 100
                confidence=Decimal("85"),
                source="VADER"
            )

    def test_sentiment_combination(self):
        """Test sentiment combination."""
        sentiment1 = NewsSentiment(
            score=Decimal("50"),
            confidence=Decimal("80"),
            source="VADER"
        )
        sentiment2 = NewsSentiment(
            score=Decimal("70"),
            confidence=Decimal("90"),
            source="TextBlob"
        )

        combined = sentiment1.combine_with(sentiment2)
        assert combined.source == "Combined(VADER,TextBlob)"
        assert -100 <= combined.score <= 100


class TestPortfolioEntity:
    """Tests for Portfolio aggregate root."""

    @pytest.fixture
    def portfolio(self):
        """Create a sample portfolio for testing."""
        return Portfolio(
            id="PORT-001",
            user_id="USER-001",
            positions=[],
            cash_balance=Money(Decimal("10000.00"), "USD"),
        )

    def test_portfolio_total_value(self, portfolio):
        """Test portfolio total value calculation."""
        assert portfolio.total_value == portfolio.cash_balance

    def test_portfolio_add_position(self, portfolio):
        """Test adding a position to portfolio."""
        position = Position(
            id="POS-001",
            user_id="USER-001",
            symbol=Symbol("AAPL"),
            position_type=PositionType.LONG,
            quantity=100,
            average_buy_price=Money(Decimal("150.00"), "USD"),
            current_price=Money(Decimal("160.00"), "USD"),
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        updated_portfolio = portfolio.add_position(position)
        assert len(updated_portfolio.positions) == 1
        assert updated_portfolio.positions[0].symbol == Symbol("AAPL")
        # Original unchanged
        assert len(portfolio.positions) == 0


# Pytest fixtures
@pytest.fixture
def sample_money():
    """Provide sample Money for tests."""
    return Money(Decimal("1000.00"), "USD")


@pytest.fixture
def sample_symbol():
    """Provide sample Symbol for tests."""
    return Symbol("AAPL")


@pytest.fixture
def sample_price():
    """Provide sample Price for tests."""
    return Price(Decimal("150.00"), "USD")
