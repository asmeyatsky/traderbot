"""
Tests for Repository Implementations

Tests that repositories correctly persist and retrieve domain entities.
"""
import pytest
from datetime import datetime
from decimal import Decimal

from src.domain.entities.user import User, RiskTolerance, InvestmentGoal
from src.domain.entities.trading import Order, Portfolio, Position, OrderType, PositionType, OrderStatus
from src.domain.value_objects import Money, Symbol
from src.infrastructure.orm_models import UserORM, OrderORM, PortfolioORM, PositionORM
from src.infrastructure.repositories import UserRepository, OrderRepository, PortfolioRepository, PositionRepository
from src.infrastructure.database import DatabaseManager, Base, get_database_manager


@pytest.mark.integration
class TestUserRepository:
    """Test UserRepository."""

    @pytest.fixture(autouse=True)
    def setup(self, db_manager):
        """Set up test database."""
        self.db_manager = db_manager
        self.repository = UserRepository()
        # Mock the global database manager
        import src.infrastructure.database as db_module
        db_module._db_manager = db_manager

    def test_save_and_get_user(self, sample_user):
        """Test saving and retrieving a user."""
        saved_user = self.repository.save(sample_user)

        assert saved_user.id == sample_user.id
        assert saved_user.email == sample_user.email

        retrieved = self.repository.get_by_id(sample_user.id)
        assert retrieved is not None
        assert retrieved.email == sample_user.email

    def test_get_user_by_email(self, sample_user):
        """Test retrieving a user by email."""
        self.repository.save(sample_user)

        retrieved = self.repository.get_by_email(sample_user.email)
        assert retrieved is not None
        assert retrieved.email == sample_user.email

    def test_get_nonexistent_user(self):
        """Test retrieving a non-existent user."""
        retrieved = self.repository.get_by_email("nonexistent@example.com")
        assert retrieved is None

    def test_update_user(self, sample_user):
        """Test updating a user."""
        self.repository.save(sample_user)

        updated = sample_user.update_risk_tolerance(RiskTolerance.AGGRESSIVE)
        result = self.repository.update(updated)

        assert result.risk_tolerance == RiskTolerance.AGGRESSIVE

    def test_get_all_active_users(self, sample_user):
        """Test retrieving all active users."""
        self.repository.save(sample_user)

        active = self.repository.get_all_active()
        assert len(active) > 0
        assert any(u.email == sample_user.email for u in active)

    def test_user_exists(self, sample_user):
        """Test checking if user exists."""
        self.repository.save(sample_user)

        exists = self.repository.exists(sample_user.id)
        assert exists is True

        not_exists = self.repository.exists("nonexistent")
        assert not_exists is False


@pytest.mark.integration
class TestOrderRepository:
    """Test OrderRepository."""

    @pytest.fixture(autouse=True)
    def setup(self, db_manager):
        """Set up test database."""
        self.db_manager = db_manager
        self.repository = OrderRepository()
        # Mock the global database manager
        import src.infrastructure.database as db_module
        db_module._db_manager = db_manager

    def test_save_and_get_order(self, sample_order):
        """Test saving and retrieving an order."""
        saved_order = self.repository.save(sample_order)

        assert saved_order.id == sample_order.id
        assert saved_order.symbol == sample_order.symbol

        retrieved = self.repository.get_by_id(sample_order.id)
        assert retrieved is not None
        assert retrieved.symbol == sample_order.symbol

    def test_get_orders_by_user_id(self, sample_order):
        """Test retrieving all orders for a user."""
        self.repository.save(sample_order)

        orders = self.repository.get_by_user_id(sample_order.user_id)
        assert len(orders) > 0
        assert any(o.id == sample_order.id for o in orders)

    def test_get_active_orders(self, sample_order):
        """Test retrieving active orders for a user."""
        self.repository.save(sample_order)

        active = self.repository.get_active_orders(sample_order.user_id)
        assert len(active) > 0

    def test_get_orders_by_symbol(self, sample_order):
        """Test retrieving orders by symbol."""
        self.repository.save(sample_order)

        orders = self.repository.get_by_symbol(sample_order.user_id, str(sample_order.symbol))
        assert len(orders) > 0
        assert orders[0].symbol == sample_order.symbol

    def test_get_orders_by_status(self, sample_order):
        """Test retrieving orders by status."""
        self.repository.save(sample_order)

        orders = self.repository.get_by_status(sample_order.user_id, OrderStatus.PENDING)
        assert len(orders) > 0
        assert all(o.status == OrderStatus.PENDING for o in orders)

    def test_update_order_status(self, sample_order):
        """Test updating order status."""
        self.repository.save(sample_order)

        updated = self.repository.update_status(sample_order.id, OrderStatus.EXECUTED)
        assert updated is not None
        assert updated.status == OrderStatus.EXECUTED


@pytest.mark.integration
class TestPortfolioRepository:
    """Test PortfolioRepository."""

    @pytest.fixture(autouse=True)
    def setup(self, db_manager):
        """Set up test database."""
        self.db_manager = db_manager
        self.repository = PortfolioRepository()
        # Mock the global database manager
        import src.infrastructure.database as db_module
        db_module._db_manager = db_manager

    def test_save_and_get_portfolio(self, sample_portfolio):
        """Test saving and retrieving a portfolio."""
        saved = self.repository.save(sample_portfolio)

        assert saved.id == sample_portfolio.id
        assert saved.cash_balance.amount == sample_portfolio.cash_balance.amount

        retrieved = self.repository.get_by_id(sample_portfolio.id)
        assert retrieved is not None
        assert retrieved.cash_balance.amount == sample_portfolio.cash_balance.amount

    def test_get_portfolio_by_user_id(self, sample_portfolio):
        """Test retrieving portfolio by user ID."""
        self.repository.save(sample_portfolio)

        retrieved = self.repository.get_by_user_id(sample_portfolio.user_id)
        assert retrieved is not None
        assert retrieved.user_id == sample_portfolio.user_id

    def test_update_portfolio(self, sample_portfolio):
        """Test updating a portfolio."""
        self.repository.save(sample_portfolio)

        # Update with new cash balance
        updated = sample_portfolio.update_cash_balance(Money(Decimal("4000.00"), "USD"))
        result = self.repository.update(updated)

        assert result.cash_balance.amount == Decimal("4000.00")


@pytest.mark.integration
class TestPositionRepository:
    """Test PositionRepository."""

    @pytest.fixture(autouse=True)
    def setup(self, db_manager):
        """Set up test database."""
        self.db_manager = db_manager
        self.repository = PositionRepository()
        # Mock the global database manager
        import src.infrastructure.database as db_module
        db_module._db_manager = db_manager

    def test_save_and_get_position(self, sample_position):
        """Test saving and retrieving a position."""
        saved = self.repository.save(sample_position)

        assert saved.id == sample_position.id
        assert saved.symbol == sample_position.symbol

        retrieved = self.repository.get_by_id(sample_position.id)
        assert retrieved is not None
        assert retrieved.symbol == sample_position.symbol

    def test_get_positions_by_user_id(self, sample_position):
        """Test retrieving positions for a user."""
        self.repository.save(sample_position)

        positions = self.repository.get_by_user_id(sample_position.user_id)
        assert len(positions) > 0
        assert any(p.id == sample_position.id for p in positions)

    def test_get_position_by_symbol(self, sample_position):
        """Test retrieving a position by symbol."""
        self.repository.save(sample_position)

        retrieved = self.repository.get_by_symbol(sample_position.user_id, sample_position.symbol)
        assert retrieved is not None
        assert retrieved.symbol == sample_position.symbol

    def test_get_open_positions(self, sample_position):
        """Test retrieving open positions."""
        self.repository.save(sample_position)

        # Get open positions (quantity > 0)
        positions = self.repository.get_by_user_id(sample_position.user_id)
        open_positions = [p for p in positions if p.quantity > 0]
        assert len(open_positions) > 0

    def test_update_position(self, sample_position):
        """Test updating a position."""
        self.repository.save(sample_position)

        # Update with new current price
        updated = sample_position.update_price(Money(Decimal("160.00"), "USD"))
        result = self.repository.update(updated)
        assert result.current_price.amount == Decimal("160.00")
