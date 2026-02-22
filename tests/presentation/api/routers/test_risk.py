"""
Risk Router API Tests

Tests for the /api/v1/risk endpoints covering portfolio risk metrics,
stress testing, and correlation matrix retrieval.

Uses FastAPI TestClient with mocked DI dependencies to isolate
the presentation layer from infrastructure concerns.
"""
import pytest
from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock, MagicMock, patch

from fastapi.testclient import TestClient

from src.presentation.api.main import app
from src.infrastructure.security import get_current_user
from src.presentation.api.dependencies import (
    get_portfolio_repository,
    get_position_repository,
    get_advanced_risk_management_service,
)
from src.domain.entities.trading import Portfolio, Position, PositionType
from src.domain.value_objects import Money, Symbol


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_overrides():
    """Reset FastAPI dependency overrides before and after each test."""
    app.dependency_overrides = {}
    yield
    app.dependency_overrides = {}


@pytest.fixture
def auth_user_id():
    return "test-user-1"


@pytest.fixture
def override_auth(auth_user_id):
    """Override authentication to return a fixed user id."""
    app.dependency_overrides[get_current_user] = lambda: auth_user_id


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
def mock_risk_metrics():
    """Return a mock RiskMetrics object with all expected attributes."""
    metrics = Mock()
    metrics.value_at_risk = Money(Decimal("1200.00"), "USD")
    metrics.expected_shortfall = Money(Decimal("1500.00"), "USD")
    metrics.max_drawdown = Decimal("12.0")  # percent form (12.0%)
    metrics.volatility = Decimal("18.0")    # percent form (18.0%)
    metrics.beta = Decimal("1.05")
    metrics.sharpe_ratio = Decimal("1.45")
    metrics.sortino_ratio = Decimal("1.80")
    metrics.correlation_matrix = {"AAPL": {"AAPL": 1.0}}
    metrics.stress_test_results = {
        "2008 Financial Crisis": Money(Decimal("-2500.00"), "USD")
    }
    metrics.portfolio_at_risk = {"AAPL": Decimal("0.65")}
    return metrics


@pytest.fixture
def mock_repos(sample_portfolio, sample_positions):
    """Create mocked portfolio and position repositories."""
    portfolio_repo = Mock()
    portfolio_repo.get_by_user_id.return_value = sample_portfolio

    position_repo = Mock()
    position_repo.get_by_user_id.return_value = sample_positions

    app.dependency_overrides[get_portfolio_repository] = lambda: portfolio_repo
    app.dependency_overrides[get_position_repository] = lambda: position_repo
    return portfolio_repo, position_repo


@pytest.fixture
def mock_risk_service(mock_risk_metrics):
    """Create a mocked advanced risk management service."""
    service = Mock()
    service.calculate_portfolio_metrics.return_value = mock_risk_metrics
    service.perform_stress_test.return_value = Money(Decimal("-3000.00"), "USD")
    service.calculate_correlation_matrix.return_value = {"AAPL": {"AAPL": 1.0}}
    app.dependency_overrides[get_advanced_risk_management_service] = lambda: service
    return service


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# GET /api/v1/risk/portfolio/{user_id}
# ---------------------------------------------------------------------------

class TestGetPortfolioRiskMetrics:

    def test_success(self, client, override_auth, mock_repos, mock_risk_service, auth_user_id):
        response = client.get(f"/api/v1/risk/portfolio/{auth_user_id}")
        assert response.status_code == 200
        data = response.json()
        # Response is flat dict with fraction-based values
        assert "var_95" in data
        assert "expected_shortfall" in data
        assert "sharpe_ratio" in data
        assert data["sharpe_ratio"] == 1.45

    def test_forbidden_when_user_id_mismatch(self, client, override_auth, mock_repos, mock_risk_service):
        response = client.get("/api/v1/risk/portfolio/other-user-999")
        assert response.status_code == 403

    def test_empty_response_when_no_portfolio(self, client, override_auth, mock_risk_service):
        portfolio_repo = Mock()
        portfolio_repo.get_by_user_id.return_value = None
        position_repo = Mock()
        position_repo.get_by_user_id.return_value = []
        app.dependency_overrides[get_portfolio_repository] = lambda: portfolio_repo
        app.dependency_overrides[get_position_repository] = lambda: position_repo

        response = client.get("/api/v1/risk/portfolio/test-user-1")
        assert response.status_code == 200
        data = response.json()
        assert data["var_95"] == 0.0
        assert data["volatility"] == 0.0

    def test_custom_lookback_and_confidence(self, client, override_auth, mock_repos, mock_risk_service, auth_user_id):
        response = client.get(
            f"/api/v1/risk/portfolio/{auth_user_id}",
            params={"lookback_days": 30, "confidence_level": 99.0},
        )
        assert response.status_code == 200
        data = response.json()
        assert "var_95" in data


# ---------------------------------------------------------------------------
# POST /api/v1/risk/stress-test/{user_id}
# ---------------------------------------------------------------------------

class TestPerformStressTest:

    def test_success(self, client, override_auth, mock_repos, mock_risk_service, auth_user_id):
        response = client.post(
            f"/api/v1/risk/stress-test/{auth_user_id}",
            params={"scenario_name": "2008_crisis"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == auth_user_id
        assert data["scenario"] == "2008 Financial Crisis"

    def test_forbidden_when_user_id_mismatch(self, client, override_auth, mock_repos, mock_risk_service):
        response = client.post(
            "/api/v1/risk/stress-test/other-user-999",
            params={"scenario_name": "2008_crisis"},
        )
        assert response.status_code == 403

    def test_bad_request_unknown_scenario(self, client, override_auth, mock_repos, mock_risk_service, auth_user_id):
        response = client.post(
            f"/api/v1/risk/stress-test/{auth_user_id}",
            params={"scenario_name": "nonexistent_scenario"},
        )
        assert response.status_code == 400
        assert "Unknown scenario" in response.json()["detail"]

    def test_not_found_when_no_portfolio(self, client, override_auth, mock_risk_service):
        portfolio_repo = Mock()
        portfolio_repo.get_by_user_id.return_value = None
        position_repo = Mock()
        position_repo.get_by_user_id.return_value = []
        app.dependency_overrides[get_portfolio_repository] = lambda: portfolio_repo
        app.dependency_overrides[get_position_repository] = lambda: position_repo

        response = client.post(
            "/api/v1/risk/stress-test/test-user-1",
            params={"scenario_name": "2008_crisis"},
        )
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/v1/risk/correlation-matrix/{user_id}
# ---------------------------------------------------------------------------

class TestGetCorrelationMatrix:

    def test_success(self, client, override_auth, mock_repos, mock_risk_service, auth_user_id):
        response = client.get(f"/api/v1/risk/correlation-matrix/{auth_user_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == auth_user_id
        assert "correlation_matrix" in data
        assert data["correlation_matrix"]["AAPL"]["AAPL"] == 1.0

    def test_forbidden_when_user_id_mismatch(self, client, override_auth, mock_repos, mock_risk_service):
        response = client.get("/api/v1/risk/correlation-matrix/other-user-999")
        assert response.status_code == 403

    def test_not_found_when_no_portfolio(self, client, override_auth, mock_risk_service):
        portfolio_repo = Mock()
        portfolio_repo.get_by_user_id.return_value = None
        position_repo = Mock()
        position_repo.get_by_user_id.return_value = []
        app.dependency_overrides[get_portfolio_repository] = lambda: portfolio_repo
        app.dependency_overrides[get_position_repository] = lambda: position_repo

        response = client.get("/api/v1/risk/correlation-matrix/test-user-1")
        assert response.status_code == 404
