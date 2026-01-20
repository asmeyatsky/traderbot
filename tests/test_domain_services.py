"""
Tests for Domain Services

Tests that domain services implement business logic correctly.
"""
import pytest
from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock, MagicMock

from src.domain.entities.user import User, RiskTolerance, InvestmentGoal
from src.domain.entities.trading import Order, Portfolio, Position, OrderType, PositionType, OrderStatus
from src.domain.value_objects import Money, Symbol, Price
from src.domain.services.trading import DefaultTradingDomainService, DefaultRiskManagementDomainService


class TestDefaultTradingDomainService:
    """Test DefaultTradingDomainService."""

    @pytest.fixture
    def trading_service(self):
        """Create a trading domain service."""
        return DefaultTradingDomainService()

    @pytest.fixture
    def sample_user(self):
        """Create a sample user for testing."""
        return User(
            id="user1",
            email="test@example.com",
            first_name="John",
            last_name="Doe",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            risk_tolerance=RiskTolerance.MODERATE,
            investment_goal=InvestmentGoal.BALANCED_GROWTH,
            max_position_size_percentage=Decimal("5"),
        )

    @pytest.fixture
    def sample_portfolio(self):
        """Create a sample portfolio for testing."""
        # Create a position to give the portfolio value
        position = Position(
            id="pos1",
            user_id="user1",
            symbol=Symbol("MSFT"),
            position_type=PositionType.LONG,
            quantity=50,
            average_buy_price=Money(Decimal("100.00"), "USD"),
            current_price=Money(Decimal("100.00"), "USD"),
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        return Portfolio(
            id="portfolio1",
            user_id="user1",
            positions=[position],
            cash_balance=Money(Decimal("5000.00"), "USD"),
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

    @pytest.fixture
    def sample_order(self):
        """Create a sample order for testing that fits within position size limits."""
        # Order value: 3 shares * $150 = $450
        # Portfolio value: $10,000 (5000 cash + 5000 in positions)
        # Position size: 450/10000 = 4.5%, within 5% limit
        return Order(
            id="order1",
            user_id="user1",
            symbol=Symbol("AAPL"),
            order_type=OrderType.MARKET,
            position_type=PositionType.LONG,
            quantity=3,
            status=OrderStatus.PENDING,
            placed_at=datetime.now(),
            price=Money(Decimal("150.00"), "USD"),
        )

    def test_validate_order_valid(self, trading_service, sample_user, sample_portfolio, sample_order):
        """Test validating a valid order."""
        errors = trading_service.validate_order(sample_order, sample_user, sample_portfolio)
        assert len(errors) == 0

    def test_validate_order_insufficient_cash(self, trading_service, sample_user, sample_portfolio):
        """Test validating order with insufficient cash."""
        # Create a portfolio with very low cash
        poor_portfolio = Portfolio(
            id="portfolio1",
            user_id="user1",
            positions=[],
            cash_balance=Money(Decimal("100.00"), "USD"),
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        # Create a LIMIT order that requires more cash than available
        # (The service only checks cash for non-MARKET orders)
        expensive_order = Order(
            id="order1",
            user_id="user1",
            symbol=Symbol("AAPL"),
            order_type=OrderType.LIMIT,  # Use LIMIT to trigger cash check
            position_type=PositionType.LONG,
            quantity=1000,  # 1000 shares at $150 = $150,000
            status=OrderStatus.PENDING,
            placed_at=datetime.now(),
            price=Money(Decimal("150.00"), "USD"),
        )

        errors = trading_service.validate_order(expensive_order, sample_user, poor_portfolio)
        assert any("cash" in error.lower() for error in errors)

    def test_validate_order_position_size_exceeded(self, trading_service, sample_portfolio):
        """Test validating order that exceeds position size limit."""
        user = User(
            id="user1",
            email="test@example.com",
            first_name="John",
            last_name="Doe",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            risk_tolerance=RiskTolerance.MODERATE,
            investment_goal=InvestmentGoal.BALANCED_GROWTH,
            max_position_size_percentage=Decimal("5"),  # Max 5% of portfolio
        )

        # Create order that would be 50% of portfolio value
        oversized_order = Order(
            id="order1",
            user_id="user1",
            symbol=Symbol("AAPL"),
            order_type=OrderType.MARKET,
            position_type=PositionType.LONG,
            quantity=334,  # 334 shares * $150 = $50,100 which is ~50% of $10k portfolio
            status=OrderStatus.PENDING,
            placed_at=datetime.now(),
            price=Money(Decimal("150.00"), "USD"),
        )

        errors = trading_service.validate_order(oversized_order, user, sample_portfolio)
        assert any("position size" in error.lower() for error in errors)

    def test_calculate_position_sizing(self, trading_service, sample_user, sample_portfolio):
        """Test position sizing calculation."""
        size = trading_service.calculate_position_sizing(sample_user, sample_portfolio, Symbol("AAPL"))

        # Max position size: 5% of $10,000 = $500
        # At $100 average price: $500 / $100 = 5 shares
        # But service uses $100 average, so max shares should be 5
        assert isinstance(size, int)
        assert size > 0

    def test_execute_order(self, trading_service, sample_order):
        """Test executing an order."""
        executed = trading_service.execute_order(
            sample_order,
            Price(Decimal("150.00"), "USD")
        )

        assert executed.status == OrderStatus.EXECUTED
        assert executed.filled_quantity == sample_order.quantity


class TestDefaultRiskManagementDomainService:
    """Test DefaultRiskManagementDomainService."""

    @pytest.fixture
    def risk_service(self):
        """Create a risk management domain service."""
        return DefaultRiskManagementDomainService()

    @pytest.fixture
    def sample_user(self):
        """Create a sample user for testing."""
        return User(
            id="user1",
            email="test@example.com",
            first_name="John",
            last_name="Doe",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            risk_tolerance=RiskTolerance.MODERATE,
            investment_goal=InvestmentGoal.BALANCED_GROWTH,
            daily_loss_limit=Money(Decimal("500.00"), "USD"),
            weekly_loss_limit=Money(Decimal("1000.00"), "USD"),
            monthly_loss_limit=Money(Decimal("2000.00"), "USD"),
        )

    @pytest.fixture
    def sample_portfolio(self):
        """Create a sample portfolio for testing."""
        return Portfolio(
            id="portfolio1",
            user_id="user1",
            positions=[],
            cash_balance=Money(Decimal("5000.00"), "USD"),
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

    def test_check_portfolio_risk_limits_ok(self, risk_service, sample_user, sample_portfolio):
        """Test checking portfolio risk limits when within limits."""
        errors = risk_service.check_portfolio_risk_limits(sample_portfolio, sample_user)
        assert len(errors) == 0

    def test_should_pause_trading_when_safe(self, risk_service, sample_user, sample_portfolio):
        """Test that trading is not paused when portfolio is safe."""
        should_pause = risk_service.should_pause_trading(sample_portfolio, sample_user)
        assert should_pause is False
