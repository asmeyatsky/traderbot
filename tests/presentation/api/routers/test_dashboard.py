"""
Dashboard Router API Tests

Tests for the /api/v1/dashboard endpoints covering overview,
allocation breakdown, and technical indicators retrieval.

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
    get_portfolio_repository,
    get_position_repository,
    get_user_repository,
    get_dashboard_analytics_service,
)
from src.domain.entities.trading import Portfolio, Position, PositionType
from src.domain.entities.user import User, RiskTolerance, InvestmentGoal
from src.domain.value_objects import Money, Symbol


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
def sample_portfolio():
    return Portfolio(
        id="port-1",
        user_id="test-user-1",
        positions=[],
        cash_balance=Money(Decimal("5000.00"), "USD"),
        created_at=datetime(2025, 1, 1),
        updated_at=datetime(2025, 6, 1),
    )


@pytest.fixture
def sample_positions():
    return [
        Position(
            id="pos-1",
            user_id="test-user-1",
            symbol=Symbol("AAPL"),
            position_type=PositionType.LONG,
            quantity=100,
            average_buy_price=Money(Decimal("150.00"), "USD"),
            current_price=Money(Decimal("175.00"), "USD"),
            created_at=datetime(2025, 1, 1),
            updated_at=datetime(2025, 6, 1),
        )
    ]


@pytest.fixture
def mock_dashboard_metrics():
    """Build a mock DashboardMetrics object with all attributes used by the router."""
    metrics = Mock()
    metrics.total_value = Money(Decimal("22500.00"), "USD")
    metrics.daily_pnl = Money(Decimal("350.00"), "USD")
    metrics.daily_pnl_percentage = Decimal("1.58")
    metrics.positions_count = 1
    metrics.active_orders_count = 0
    metrics.unrealized_pnl = Money(Decimal("2500.00"), "USD")
    metrics.realized_pnl = Money(Decimal("800.00"), "USD")
    metrics.top_gainers = [(Symbol("AAPL"), Decimal("16.67"))]
    metrics.top_losers = []
    metrics.allocation_by_sector = {"Technology": Decimal("77.78")}
    metrics.allocation_by_asset = {Symbol("AAPL"): Decimal("77.78")}
    metrics.risk_metrics = {"volatility": Decimal("0.18"), "sharpe": Decimal("1.4")}
    metrics.performance_chart_data = [
        {"date": "2025-05-01", "value": Decimal("20000.00")},
        {"date": "2025-06-01", "value": Decimal("22500.00")},
    ]

    # Technical indicators attached to dashboard metrics
    tech = Mock()
    tech.symbol = Symbol("AAPL")
    tech.sma_20 = Decimal("170.00")
    tech.sma_50 = Decimal("165.00")
    tech.ema_12 = Decimal("172.00")
    tech.ema_26 = Decimal("168.00")
    tech.rsi = Decimal("58.00")
    tech.macd = Decimal("4.00")
    tech.macd_signal = Decimal("3.50")
    tech.bollinger_upper = Decimal("185.00")
    tech.bollinger_lower = Decimal("155.00")
    tech.atr = Decimal("5.20")
    tech.calculated_at = datetime(2025, 6, 1)
    metrics.technical_indicators = [tech]

    return metrics


@pytest.fixture
def mock_repos(sample_portfolio, sample_positions, sample_user):
    portfolio_repo = Mock()
    portfolio_repo.get_by_user_id.return_value = sample_portfolio

    position_repo = Mock()
    position_repo.get_by_user_id.return_value = sample_positions

    user_repo = Mock()
    user_repo.get_by_id.return_value = sample_user

    app.dependency_overrides[get_portfolio_repository] = lambda: portfolio_repo
    app.dependency_overrides[get_position_repository] = lambda: position_repo
    app.dependency_overrides[get_user_repository] = lambda: user_repo
    return portfolio_repo, position_repo, user_repo


@pytest.fixture
def mock_dashboard_service(mock_dashboard_metrics):
    service = Mock()
    service.get_dashboard_metrics.return_value = mock_dashboard_metrics

    # Technical indicators endpoint uses a standalone call
    tech = Mock()
    tech.symbol = Symbol("AAPL")
    tech.sma_20 = Decimal("170.00")
    tech.sma_50 = Decimal("165.00")
    tech.ema_12 = Decimal("172.00")
    tech.ema_26 = Decimal("168.00")
    tech.rsi = Decimal("58.00")
    tech.macd = Decimal("4.00")
    tech.macd_signal = Decimal("3.50")
    tech.bollinger_upper = Decimal("185.00")
    tech.bollinger_lower = Decimal("155.00")
    tech.atr = Decimal("5.20")
    tech.calculated_at = datetime(2025, 6, 1)
    service.calculate_technical_indicators.return_value = tech

    app.dependency_overrides[get_dashboard_analytics_service] = lambda: service
    return service


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# GET /api/v1/dashboard/overview/{user_id}
# ---------------------------------------------------------------------------

class TestGetDashboardOverview:

    def test_success(self, client, override_auth, mock_repos, mock_dashboard_service, auth_user_id):
        response = client.get(f"/api/v1/dashboard/overview/{auth_user_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == auth_user_id
        assert data["portfolio_value"] == 22500.0
        assert data["daily_pnl"] == 350.0
        assert data["positions_count"] == 1

    def test_forbidden_when_user_id_mismatch(self, client, override_auth, mock_repos, mock_dashboard_service):
        response = client.get("/api/v1/dashboard/overview/other-user-999")
        assert response.status_code == 403

    def test_empty_dashboard_when_no_portfolio(self, client, override_auth, mock_dashboard_service, sample_user):
        portfolio_repo = Mock()
        portfolio_repo.get_by_user_id.return_value = None
        position_repo = Mock()
        position_repo.get_by_user_id.return_value = []
        user_repo = Mock()
        user_repo.get_by_id.return_value = sample_user
        app.dependency_overrides[get_portfolio_repository] = lambda: portfolio_repo
        app.dependency_overrides[get_position_repository] = lambda: position_repo
        app.dependency_overrides[get_user_repository] = lambda: user_repo

        response = client.get("/api/v1/dashboard/overview/test-user-1")
        assert response.status_code == 200
        data = response.json()
        assert data["portfolio_value"] == 0.0
        assert data["positions_count"] == 0

    def test_not_found_when_no_user(self, client, override_auth, mock_dashboard_service, sample_portfolio, sample_positions):
        portfolio_repo = Mock()
        portfolio_repo.get_by_user_id.return_value = sample_portfolio
        position_repo = Mock()
        position_repo.get_by_user_id.return_value = sample_positions
        user_repo = Mock()
        user_repo.get_by_id.return_value = None
        app.dependency_overrides[get_portfolio_repository] = lambda: portfolio_repo
        app.dependency_overrides[get_position_repository] = lambda: position_repo
        app.dependency_overrides[get_user_repository] = lambda: user_repo

        response = client.get("/api/v1/dashboard/overview/test-user-1")
        assert response.status_code == 404

    def test_response_has_expected_fields(self, client, override_auth, mock_repos, mock_dashboard_service, auth_user_id):
        response = client.get(f"/api/v1/dashboard/overview/{auth_user_id}")
        assert response.status_code == 200
        data = response.json()
        assert "portfolio_value" in data
        assert "top_performers" in data
        assert "performance_history" in data


# ---------------------------------------------------------------------------
# GET /api/v1/dashboard/allocation/{user_id}
# ---------------------------------------------------------------------------

class TestGetAllocationBreakdown:

    def test_success_asset(self, client, override_auth, mock_repos, mock_dashboard_service, auth_user_id):
        response = client.get(
            f"/api/v1/dashboard/allocation/{auth_user_id}",
            params={"breakdown_type": "asset"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == auth_user_id
        assert data["breakdown_type"] == "asset"
        assert "allocation_by_asset" in data

    def test_success_sector(self, client, override_auth, mock_repos, mock_dashboard_service, auth_user_id):
        response = client.get(
            f"/api/v1/dashboard/allocation/{auth_user_id}",
            params={"breakdown_type": "sector"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "allocation_by_sector" in data

    def test_success_both(self, client, override_auth, mock_repos, mock_dashboard_service, auth_user_id):
        response = client.get(
            f"/api/v1/dashboard/allocation/{auth_user_id}",
            params={"breakdown_type": "both"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "allocation_by_asset" in data
        assert "allocation_by_sector" in data

    def test_forbidden_when_user_id_mismatch(self, client, override_auth, mock_repos, mock_dashboard_service):
        response = client.get("/api/v1/dashboard/allocation/other-user-999")
        assert response.status_code == 403

    def test_empty_allocation_when_no_portfolio(self, client, override_auth, mock_dashboard_service, sample_user):
        portfolio_repo = Mock()
        portfolio_repo.get_by_user_id.return_value = None
        position_repo = Mock()
        position_repo.get_by_user_id.return_value = []
        user_repo = Mock()
        user_repo.get_by_id.return_value = sample_user
        app.dependency_overrides[get_portfolio_repository] = lambda: portfolio_repo
        app.dependency_overrides[get_position_repository] = lambda: position_repo
        app.dependency_overrides[get_user_repository] = lambda: user_repo

        response = client.get("/api/v1/dashboard/allocation/test-user-1")
        assert response.status_code == 200
        data = response.json()
        assert data["allocation_by_asset"] == {}
        assert data["allocation_by_sector"] == {}


# ---------------------------------------------------------------------------
# GET /api/v1/dashboard/technical-indicators/{symbol}
# ---------------------------------------------------------------------------

class TestGetTechnicalIndicators:

    def test_success(self, client, override_auth, mock_dashboard_service):
        response = client.get("/api/v1/dashboard/technical-indicators/AAPL")
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "AAPL"
        assert data["sma_20"] == 170.0
        assert data["rsi"] == 58.0
        assert data["macd"] == 4.0

    def test_custom_days_parameter(self, client, override_auth, mock_dashboard_service):
        response = client.get(
            "/api/v1/dashboard/technical-indicators/MSFT",
            params={"days": 30},
        )
        assert response.status_code == 200

    def test_service_error_returns_500(self, client, override_auth):
        service = Mock()
        service.calculate_technical_indicators.side_effect = RuntimeError("service down")
        app.dependency_overrides[get_dashboard_analytics_service] = lambda: service

        response = client.get("/api/v1/dashboard/technical-indicators/AAPL")
        assert response.status_code == 500
