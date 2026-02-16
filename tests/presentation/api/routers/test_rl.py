"""
Reinforcement Learning Router API Tests

Tests for the /api/v1/rl endpoints covering algorithm listing,
agent training, evaluation, ensemble performance, and action recommendation.

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
    get_rl_agent_service,
)
from src.domain.entities.trading import Portfolio, Position, PositionType
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
def mock_rl_performance():
    perf = Mock()
    perf.accuracy = 0.72
    perf.precision = 0.68
    perf.recall = 0.75
    perf.sharpe_ratio = 1.35
    perf.max_drawdown = 0.15
    perf.annual_return = 0.22
    return perf


@pytest.fixture
def mock_rl_service(mock_rl_performance):
    service = Mock()
    service.train.return_value = True
    service.evaluate.return_value = mock_rl_performance
    service.get_action.return_value = ("BUY", Decimal("0.25"))
    app.dependency_overrides[get_rl_agent_service] = lambda: service
    return service


@pytest.fixture
def mock_repos(sample_portfolio, sample_positions):
    portfolio_repo = Mock()
    portfolio_repo.get_by_user_id.return_value = sample_portfolio
    position_repo = Mock()
    position_repo.get_by_user_id.return_value = sample_positions
    app.dependency_overrides[get_portfolio_repository] = lambda: portfolio_repo
    app.dependency_overrides[get_position_repository] = lambda: position_repo
    return portfolio_repo, position_repo


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# GET /api/v1/rl/algorithms
# ---------------------------------------------------------------------------

class TestGetAvailableAlgorithms:

    def test_success(self, client, override_auth):
        response = client.get("/api/v1/rl/algorithms")
        assert response.status_code == 200
        data = response.json()
        assert "available_algorithms" in data
        assert data["algorithm_count"] > 0
        assert "deep_q_network" in data["available_algorithms"]
        assert "ppo" in data["available_algorithms"]


# ---------------------------------------------------------------------------
# POST /api/v1/rl/agents/train/{symbol}
# ---------------------------------------------------------------------------

class TestTrainRLAgent:

    def test_success(self, client, override_auth, mock_rl_service):
        response = client.post(
            "/api/v1/rl/agents/train/AAPL",
            params={"algorithm": "dqn", "episodes": 50},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "AAPL"
        assert data["algorithm"] == "dqn"
        assert data["training_result"]["training_successful"] is True
        assert data["training_result"]["episodes_trained"] == 50

    def test_invalid_algorithm_returns_400(self, client, override_auth, mock_rl_service):
        response = client.post(
            "/api/v1/rl/agents/train/AAPL",
            params={"algorithm": "nonexistent_algo"},
        )
        assert response.status_code == 400
        assert "Invalid algorithm" in response.json()["detail"]

    def test_default_parameters(self, client, override_auth, mock_rl_service):
        response = client.post("/api/v1/rl/agents/train/MSFT")
        assert response.status_code == 200
        data = response.json()
        assert data["training_result"]["episodes_trained"] == 100  # default


# ---------------------------------------------------------------------------
# POST /api/v1/rl/agents/evaluate/{symbol}
# ---------------------------------------------------------------------------

class TestEvaluateRLAgent:

    def test_success(self, client, override_auth, mock_rl_service):
        response = client.post(
            "/api/v1/rl/agents/evaluate/AAPL",
            params={"algorithm": "ppo"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "AAPL"
        assert data["evaluation_results"]["accuracy"] == 0.72
        assert data["evaluation_results"]["sharpe_ratio"] == 1.35

    def test_invalid_algorithm_returns_400(self, client, override_auth, mock_rl_service):
        response = client.post(
            "/api/v1/rl/agents/evaluate/AAPL",
            params={"algorithm": "bad_algo"},
        )
        assert response.status_code == 400


# ---------------------------------------------------------------------------
# GET /api/v1/rl/agents/ensemble-performance
# ---------------------------------------------------------------------------

class TestGetEnsemblePerformance:

    def test_success(self, client, override_auth, mock_rl_service):
        response = client.get("/api/v1/rl/agents/ensemble-performance")
        assert response.status_code == 200
        data = response.json()
        assert "ensemble_performance" in data
        assert "agent_types" in data
        # The endpoint evaluates AAPL, MSFT, GOOGL
        assert len(data["agent_types"]) == 3

    def test_handles_service_errors_gracefully(self, client, override_auth):
        service = Mock()
        service.evaluate.side_effect = RuntimeError("no trained agent")
        app.dependency_overrides[get_rl_agent_service] = lambda: service

        response = client.get("/api/v1/rl/agents/ensemble-performance")
        assert response.status_code == 200
        data = response.json()
        # Individual failures should be captured per-symbol, not crash entire request
        for sym in ["AAPL", "MSFT", "GOOGL"]:
            assert "error" in data["ensemble_performance"][sym]


# ---------------------------------------------------------------------------
# POST /api/v1/rl/agents/get-action/{symbol}/{user_id}
# ---------------------------------------------------------------------------

class TestGetRLAction:

    def test_success(self, client, override_auth, mock_repos, mock_rl_service, auth_user_id):
        response = client.post(f"/api/v1/rl/agents/get-action/AAPL/{auth_user_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "AAPL"
        assert data["user_id"] == auth_user_id
        assert data["action"] == "BUY"
        assert data["position_size"] == 0.25

    def test_forbidden_when_user_id_mismatch(self, client, override_auth, mock_repos, mock_rl_service):
        response = client.post("/api/v1/rl/agents/get-action/AAPL/other-user-999")
        assert response.status_code == 403

    def test_works_without_portfolio(self, client, override_auth, mock_rl_service, auth_user_id):
        """When user has no portfolio yet, endpoint should still work with defaults."""
        portfolio_repo = Mock()
        portfolio_repo.get_by_user_id.return_value = None
        position_repo = Mock()
        position_repo.get_by_user_id.return_value = []
        app.dependency_overrides[get_portfolio_repository] = lambda: portfolio_repo
        app.dependency_overrides[get_position_repository] = lambda: position_repo

        response = client.post(f"/api/v1/rl/agents/get-action/AAPL/{auth_user_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["action"] == "BUY"

    def test_custom_market_regime(self, client, override_auth, mock_repos, mock_rl_service, auth_user_id):
        response = client.post(
            f"/api/v1/rl/agents/get-action/TSLA/{auth_user_id}",
            params={"market_regime": "bear"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["market_regime"] == "bear"
