"""
Test Configuration and Fixtures

This module provides shared test fixtures and configuration for all tests.
"""
import pytest
import os
from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock, MagicMock

# Add the project root to the path so imports work correctly
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.domain.entities.user import User, RiskTolerance, InvestmentGoal
from src.domain.entities.trading import Order, Portfolio, Position, OrderType, PositionType, OrderStatus
from src.domain.value_objects import Money, Symbol, Price
from src.infrastructure.database import DatabaseManager, Base
from src.infrastructure.config.settings import settings


@pytest.fixture
def test_database_url():
    """Get test database URL - uses SQLite in memory."""
    return "sqlite:///:memory:"


@pytest.fixture
def db_manager(test_database_url):
    """Create a database manager for testing."""
    manager = DatabaseManager(test_database_url, echo=False)
    manager.initialize()
    # Create all tables
    Base.metadata.create_all(manager.engine)
    yield manager
    # Cleanup
    Base.metadata.drop_all(manager.engine)
    manager.close()


@pytest.fixture
def sample_user():
    """Create a sample user for testing."""
    return User(
        id="test-user-1",
        email="test@example.com",
        first_name="John",
        last_name="Doe",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        risk_tolerance=RiskTolerance.MODERATE,
        investment_goal=InvestmentGoal.BALANCED_GROWTH,
        max_position_size_percentage=Decimal("5"),
        daily_loss_limit=Money(Decimal("500.00"), "USD"),
        weekly_loss_limit=Money(Decimal("1000.00"), "USD"),
        monthly_loss_limit=Money(Decimal("2000.00"), "USD"),
    )


@pytest.fixture
def sample_portfolio():
    """Create a sample portfolio for testing."""
    return Portfolio(
        id="test-portfolio-1",
        user_id="test-user-1",
        total_value=Money(Decimal("10000.00"), "USD"),
        cash_balance=Money(Decimal("5000.00"), "USD"),
        invested_value=Money(Decimal("5000.00"), "USD"),
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )


@pytest.fixture
def sample_order():
    """Create a sample order for testing."""
    return Order(
        id="test-order-1",
        user_id="test-user-1",
        symbol=Symbol("AAPL"),
        order_type=OrderType.MARKET,
        position_type=PositionType.LONG,
        quantity=Decimal("100"),
        status=OrderStatus.PENDING,
        placed_at=datetime.now(),
        price=Money(Decimal("150.00"), "USD"),
    )


@pytest.fixture
def sample_position():
    """Create a sample position for testing."""
    return Position(
        id="test-position-1",
        user_id="test-user-1",
        symbol=Symbol("AAPL"),
        quantity=Decimal("100"),
        position_type=PositionType.LONG,
        average_entry_price=Money(Decimal("150.00"), "USD"),
        current_price=Money(Decimal("155.00"), "USD"),
        unrealized_gain_loss=Money(Decimal("500.00"), "USD"),
        opened_at=datetime.now(),
    )


@pytest.fixture
def mock_trading_execution_port():
    """Create a mock TradingExecutionPort."""
    mock = Mock()
    mock.place_order = Mock(return_value="order-123")
    mock.cancel_order = Mock(return_value=True)
    mock.get_order_status = Mock(return_value=OrderStatus.EXECUTED)
    mock.get_account_balance = Mock(return_value=Money(Decimal("10000.00"), "USD"))
    return mock


@pytest.fixture
def mock_market_data_port():
    """Create a mock MarketDataPort."""
    mock = Mock()
    mock.get_current_price = Mock(return_value=Price(Decimal("150.00"), "USD"))
    mock.get_historical_prices = Mock(return_value=[
        Price(Decimal("150.00"), "USD"),
        Price(Decimal("151.00"), "USD"),
        Price(Decimal("149.50"), "USD"),
    ])
    mock.get_market_news = Mock(return_value=[
        "AAPL stock rises on earnings",
        "Tech sector strengthens",
    ])
    return mock


@pytest.fixture
def mock_news_analysis_port():
    """Create a mock NewsAnalysisPort."""
    from src.domain.value_objects import NewsSentiment
    mock = Mock()
    mock.analyze_sentiment = Mock(return_value=NewsSentiment(score=0.8, label="POSITIVE"))
    mock.batch_analyze_sentiment = Mock(return_value=[
        NewsSentiment(score=0.8, label="POSITIVE"),
        NewsSentiment(score=0.5, label="NEUTRAL"),
    ])
    mock.extract_symbols_from_news = Mock(return_value=[Symbol("AAPL"), Symbol("MSFT")])
    return mock


# Markers for test categorization
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Mark test as a unit test")
    config.addinivalue_line("markers", "integration: Mark test as an integration test")
    config.addinivalue_line("markers", "slow: Mark test as slow running")
