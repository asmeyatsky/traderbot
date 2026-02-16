"""
Brokers Router API Tests

Tests for the /api/v1/brokers endpoints covering available brokers,
order placement, position retrieval, and account info.

Uses FastAPI TestClient with mocked DI dependencies to isolate
the presentation layer from infrastructure concerns.
"""
import pytest
from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock

from fastapi.testclient import TestClient

from src.presentation.api.main import app
from src.infrastructure.security import get_current_user
from src.presentation.api.dependencies import (
    get_user_repository,
    get_broker_adapter_manager,
)
from src.domain.entities.user import User, RiskTolerance, InvestmentGoal
from src.domain.value_objects import Money
from src.infrastructure.broker_integration import BrokerType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_overrides():
    app.dependency_overrides = {}
    yield
    app.dependency_overrides = {}


@pytest.fixture
def auth_user_id():
    return "test-user-1"


@pytest.fixture
def override_auth(auth_user_id):
    app.dependency_overrides[get_current_user] = lambda: auth_user_id


@pytest.fixture
def sample_user():
    return User(
        id="test-user-1",
        email="test@example.com",
        first_name="John",
        last_name="Doe",
        created_at=datetime(2025, 1, 1),
        updated_at=datetime(2025, 6, 1),
        risk_tolerance=RiskTolerance.MODERATE,
        investment_goal=InvestmentGoal.BALANCED_GROWTH,
    )


@pytest.fixture
def mock_broker_order():
    order = Mock()
    order.broker_order_id = "broker-ord-1"
    order.client_order_id = "client-ord-1"
    order.symbol = "AAPL"
    order.quantity = 10
    order.side = "buy"
    order.order_type = BrokerType.ALPACA  # mock; .value will be "alpaca"
    order.status = "filled"
    order.time_in_force = "day"
    order.limit_price = None
    order.stop_price = None
    order.created_at = datetime(2025, 6, 1)
    return order


@pytest.fixture
def mock_broker_position():
    pos = Mock()
    pos.symbol = "AAPL"
    pos.quantity = 100
    pos.avg_entry_price = Decimal("150.00")
    pos.current_price = Decimal("175.00")
    pos.unrealized_pnl = Decimal("2500.00")
    pos.market_value = Decimal("17500.00")
    pos.position_type = Mock(value="long")
    return pos


@pytest.fixture
def mock_account_info():
    info = Mock()
    info.account_id = "acct-1"
    info.account_number = "1234567890"
    info.account_type = "margin"
    info.buying_power = Money(Decimal("25000.00"), "USD")
    info.cash_balance = Money(Decimal("10000.00"), "USD")
    info.portfolio_value = Money(Decimal("50000.00"), "USD")
    info.day_trade_count = 2
    info.pattern_day_trader = False
    info.trading_blocked = False
    info.transfers_blocked = False
    info.account_blocked = False
    info.created_at = datetime(2025, 1, 1)
    info.updated_at = datetime(2025, 6, 1)
    return info


@pytest.fixture
def mock_user_repo(sample_user):
    repo = Mock()
    repo.get_by_id.return_value = sample_user
    app.dependency_overrides[get_user_repository] = lambda: repo
    return repo


@pytest.fixture
def mock_adapter_manager(mock_broker_order, mock_broker_position, mock_account_info):
    manager = Mock()
    manager.get_available_brokers.return_value = [BrokerType.ALPACA, BrokerType.INTERACTIVE_BROKERS]
    manager.execute_order.return_value = mock_broker_order

    broker_service = Mock()
    broker_service.get_positions.return_value = [mock_broker_position]
    manager.get_broker_service.return_value = broker_service

    manager.get_account_info.return_value = mock_account_info
    app.dependency_overrides[get_broker_adapter_manager] = lambda: manager
    return manager


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# GET /api/v1/brokers/available
# ---------------------------------------------------------------------------

class TestGetAvailableBrokers:

    def test_success(self, client, override_auth, mock_adapter_manager):
        response = client.get("/api/v1/brokers/available")
        assert response.status_code == 200
        data = response.json()
        assert data["broker_count"] == 2
        assert "alpaca" in data["available_brokers"]
        assert "interactive_brokers" in data["available_brokers"]


# ---------------------------------------------------------------------------
# POST /api/v1/brokers/{broker_type}/place-order
# ---------------------------------------------------------------------------

class TestPlaceOrder:

    def test_success(self, client, override_auth, mock_user_repo, mock_adapter_manager):
        response = client.post(
            "/api/v1/brokers/alpaca/place-order",
            params={
                "symbol": "AAPL",
                "quantity": 10,
                "side": "buy",
                "order_type": "market",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["broker_order_id"] == "broker-ord-1"
        assert data["symbol"] == "AAPL"
        assert data["broker_type"] == "alpaca"

    def test_invalid_broker_type_returns_400(self, client, override_auth, mock_user_repo, mock_adapter_manager):
        response = client.post(
            "/api/v1/brokers/nonexistent_broker/place-order",
            params={
                "symbol": "AAPL",
                "quantity": 10,
                "side": "buy",
                "order_type": "market",
            },
        )
        assert response.status_code == 400
        assert "Invalid broker type" in response.json()["detail"]

    def test_user_not_found_returns_404(self, client, override_auth, mock_adapter_manager):
        user_repo = Mock()
        user_repo.get_by_id.return_value = None
        app.dependency_overrides[get_user_repository] = lambda: user_repo

        response = client.post(
            "/api/v1/brokers/alpaca/place-order",
            params={
                "symbol": "AAPL",
                "quantity": 10,
                "side": "buy",
                "order_type": "market",
            },
        )
        assert response.status_code == 404

    def test_invalid_side_returns_422(self, client, override_auth, mock_user_repo, mock_adapter_manager):
        response = client.post(
            "/api/v1/brokers/alpaca/place-order",
            params={
                "symbol": "AAPL",
                "quantity": 10,
                "side": "invalid_side",
                "order_type": "market",
            },
        )
        assert response.status_code == 422

    def test_negative_quantity_returns_422(self, client, override_auth, mock_user_repo, mock_adapter_manager):
        response = client.post(
            "/api/v1/brokers/alpaca/place-order",
            params={
                "symbol": "AAPL",
                "quantity": -5,
                "side": "buy",
                "order_type": "market",
            },
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/v1/brokers/{broker_type}/positions
# ---------------------------------------------------------------------------

class TestGetPositions:

    def test_success(self, client, override_auth, mock_user_repo, mock_adapter_manager):
        response = client.get("/api/v1/brokers/alpaca/positions")
        assert response.status_code == 200
        data = response.json()
        assert data["position_count"] == 1
        assert data["positions"][0]["symbol"] == "AAPL"
        assert data["broker_type"] == "alpaca"

    def test_invalid_broker_type_returns_400(self, client, override_auth, mock_user_repo, mock_adapter_manager):
        response = client.get("/api/v1/brokers/bad_broker/positions")
        assert response.status_code == 400

    def test_user_not_found_returns_404(self, client, override_auth, mock_adapter_manager):
        user_repo = Mock()
        user_repo.get_by_id.return_value = None
        app.dependency_overrides[get_user_repository] = lambda: user_repo

        response = client.get("/api/v1/brokers/alpaca/positions")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/v1/brokers/{broker_type}/account-info
# ---------------------------------------------------------------------------

class TestGetAccountInfo:

    def test_success(self, client, override_auth, mock_user_repo, mock_adapter_manager):
        response = client.get("/api/v1/brokers/alpaca/account-info")
        assert response.status_code == 200
        data = response.json()
        assert data["account_id"] == "acct-1"
        assert data["buying_power"]["amount"] == 25000.0
        assert data["pattern_day_trader"] is False
        assert data["broker_type"] == "alpaca"

    def test_invalid_broker_type_returns_400(self, client, override_auth, mock_user_repo, mock_adapter_manager):
        response = client.get("/api/v1/brokers/fake_broker/account-info")
        assert response.status_code == 400

    def test_user_not_found_returns_404(self, client, override_auth, mock_adapter_manager):
        user_repo = Mock()
        user_repo.get_by_id.return_value = None
        app.dependency_overrides[get_user_repository] = lambda: user_repo

        response = client.get("/api/v1/brokers/alpaca/account-info")
        assert response.status_code == 404
