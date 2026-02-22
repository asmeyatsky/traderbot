"""
Machine Learning Router API Tests

Tests for the /api/v1/ml endpoints covering price prediction, trading signals,
model performance, portfolio optimization, backtesting, and risk analysis.

Uses FastAPI TestClient with mocked DI dependencies to isolate
the presentation layer from infrastructure concerns.

Note: Several ML endpoints use container.services.<service>() directly rather than
FastAPI Depends(), so those must be patched via unittest.mock.patch on the
container attribute.
"""
import pytest
from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock

from fastapi.testclient import TestClient

from src.presentation.api.main import app
from src.infrastructure.security import get_current_user
from src.presentation.api.dependencies import (
    get_portfolio_repository,
    get_position_repository,
    get_user_repository,
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
def mock_prediction():
    pred = Mock()
    pred.signal = "BUY"
    pred.confidence = 0.85
    pred.score = 0.78
    pred.explanation = "Strong upward momentum detected"
    return pred


@pytest.fixture
def mock_ml_service(mock_prediction):
    service = Mock()
    service.predict_price_direction.return_value = mock_prediction
    service.get_model_performance.return_value = Mock(
        accuracy=0.76,
        precision=0.72,
        recall=0.80,
        sharpe_ratio=1.5,
        max_drawdown=0.12,
        annual_return=0.25,
    )
    service.retrain_model.return_value = True
    return service


@pytest.fixture
def mock_news_analyzer():
    analyzer = Mock()
    analyzer.calculate_news_impact_score.return_value = 0.35
    return analyzer


@pytest.fixture
def mock_risk_analytics_service():
    service = Mock()
    service.calculate_var.return_value = Money(Decimal("1500.00"), "USD")
    service.calculate_expected_shortfall.return_value = Money(Decimal("2000.00"), "USD")
    service.calculate_correlation_matrix.return_value = {"AAPL": {"AAPL": 1.0}}
    service.stress_test_portfolio.return_value = [
        {"name": "Market Crash", "impact": -5000.0},
    ]
    return service


@pytest.fixture
def mock_portfolio_optimization_service():
    service = Mock()
    service.optimize_portfolio.return_value = {
        "AAPL": 0.40,
        "MSFT": 0.35,
        "cash": 0.25,
    }
    return service


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
def client():
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# GET /api/v1/ml/predict/{symbol}
# ---------------------------------------------------------------------------

class TestGetPricePrediction:

    @patch("src.presentation.api.routers.ml.container")
    def test_success(self, mock_container, client, override_auth, mock_ml_service):
        mock_container.services.ml_model_service.return_value = mock_ml_service
        mock_container.services.market_data_enhancement_service.side_effect = Exception("no market data")

        response = client.get("/api/v1/ml/predict/AAPL")
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "AAPL"
        assert data["predicted_direction"] == "UP"
        assert data["confidence"] == 0.85

    @patch("src.presentation.api.routers.ml.container")
    def test_custom_lookback(self, mock_container, client, override_auth, mock_ml_service):
        mock_container.services.ml_model_service.return_value = mock_ml_service
        mock_container.services.market_data_enhancement_service.side_effect = Exception("no market data")

        response = client.get("/api/v1/ml/predict/MSFT", params={"lookback_days": 60})
        assert response.status_code == 200
        data = response.json()
        assert data["lookback_days"] == 60

    @patch("src.presentation.api.routers.ml.container")
    def test_ml_service_unavailable_returns_500(self, mock_container, client, override_auth):
        mock_container.services.ml_model_service.return_value = None

        response = client.get("/api/v1/ml/predict/AAPL")
        assert response.status_code == 500

    def test_invalid_lookback_returns_422(self, client, override_auth):
        response = client.get("/api/v1/ml/predict/AAPL", params={"lookback_days": 0})
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/v1/ml/signal/{symbol}/{user_id}
# ---------------------------------------------------------------------------

class TestGetTradingSignal:

    @patch("src.presentation.api.routers.ml.container")
    def test_success(self, mock_container, client, override_auth, mock_ml_service, mock_news_analyzer):
        mock_container.services.ml_model_service.return_value = mock_ml_service
        mock_container.services.news_impact_analyzer_service.return_value = mock_news_analyzer

        response = client.get("/api/v1/ml/signal/AAPL/test-user-1")
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "AAPL"
        assert data["original_signal"] == "BUY"
        assert "confidence" in data
        assert "risk_adjusted" in data
        assert data["user_risk_profile"] == "MODERATE"

    @patch("src.presentation.api.routers.ml.container")
    def test_custom_risk_level(self, mock_container, client, override_auth, mock_ml_service, mock_news_analyzer):
        mock_container.services.ml_model_service.return_value = mock_ml_service
        mock_container.services.news_impact_analyzer_service.return_value = mock_news_analyzer

        response = client.get(
            "/api/v1/ml/signal/AAPL/test-user-1",
            params={"risk_level": "AGGRESSIVE", "investment_goal": "MAXIMUM_RETURNS"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["user_risk_profile"] == "AGGRESSIVE"
        assert data["investment_goal"] == "MAXIMUM_RETURNS"

    @patch("src.presentation.api.routers.ml.container")
    def test_invalid_risk_level_returns_400(self, mock_container, client, override_auth, mock_ml_service, mock_news_analyzer):
        mock_container.services.ml_model_service.return_value = mock_ml_service
        mock_container.services.news_impact_analyzer_service.return_value = mock_news_analyzer

        response = client.get(
            "/api/v1/ml/signal/AAPL/test-user-1",
            params={"risk_level": "INVALID"},
        )
        assert response.status_code == 400
        assert "Invalid risk level" in response.json()["detail"]

    @patch("src.presentation.api.routers.ml.container")
    def test_invalid_investment_goal_returns_400(self, mock_container, client, override_auth, mock_ml_service, mock_news_analyzer):
        mock_container.services.ml_model_service.return_value = mock_ml_service
        mock_container.services.news_impact_analyzer_service.return_value = mock_news_analyzer

        response = client.get(
            "/api/v1/ml/signal/AAPL/test-user-1",
            params={"investment_goal": "INVALID_GOAL"},
        )
        assert response.status_code == 400
        assert "Invalid investment goal" in response.json()["detail"]

    @patch("src.presentation.api.routers.ml.container")
    def test_services_unavailable_returns_500(self, mock_container, client, override_auth):
        mock_container.services.ml_model_service.return_value = None
        mock_container.services.news_impact_analyzer_service.return_value = None

        response = client.get("/api/v1/ml/signal/AAPL/test-user-1")
        assert response.status_code == 500

    def test_unauthorized_user_returns_403(self, client, override_auth):
        """Users cannot access signals for other users."""
        response = client.get("/api/v1/ml/signal/AAPL/other-user-id")
        assert response.status_code == 403


# ---------------------------------------------------------------------------
# GET /api/v1/ml/model-performance/{model_type}
# ---------------------------------------------------------------------------

class TestGetModelPerformance:

    @patch("src.presentation.api.routers.ml.container")
    def test_success_specific_model(self, mock_container, client, override_auth, mock_ml_service):
        mock_container.services.ml_model_service.return_value = mock_ml_service

        response = client.get("/api/v1/ml/model-performance/lstm")
        assert response.status_code == 200
        data = response.json()
        assert data["model_type"] == "lstm"
        assert data["accuracy"] == 0.76
        assert data["sharpe_ratio"] == 1.5

    @patch("src.presentation.api.routers.ml.container")
    def test_success_all_models(self, mock_container, client, override_auth, mock_ml_service):
        mock_container.services.ml_model_service.return_value = mock_ml_service

        response = client.get("/api/v1/ml/model-performance/all")
        assert response.status_code == 200
        data = response.json()
        assert "model_performance" in data

    @patch("src.presentation.api.routers.ml.container")
    def test_invalid_model_type_returns_400(self, mock_container, client, override_auth, mock_ml_service):
        mock_container.services.ml_model_service.return_value = mock_ml_service

        response = client.get("/api/v1/ml/model-performance/invalid_model")
        assert response.status_code == 400
        assert "Invalid model type" in response.json()["detail"]


# ---------------------------------------------------------------------------
# POST /api/v1/ml/optimize-portfolio/{user_id}
# ---------------------------------------------------------------------------

class TestOptimizePortfolio:

    def test_unauthorized_user_returns_403(self, client, override_auth):
        """Users cannot optimize another user's portfolio."""
        response = client.post("/api/v1/ml/optimize-portfolio/other-user-id")
        assert response.status_code == 403

    @patch("src.presentation.api.routers.ml.container")
    def test_success(self, mock_container, client, override_auth, mock_repos, mock_portfolio_optimization_service):
        mock_container.services.portfolio_optimization_service.return_value = mock_portfolio_optimization_service

        response = client.post(
            "/api/v1/ml/optimize-portfolio/test-user-1",
            params={"risk_tolerance": "MODERATE", "investment_goal": "BALANCED_GROWTH"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "test-user-1"
        assert "allocation_recommendation" in data

    @patch("src.presentation.api.routers.ml.container")
    def test_invalid_risk_tolerance_returns_400(self, mock_container, client, override_auth, mock_repos, mock_portfolio_optimization_service):
        mock_container.services.portfolio_optimization_service.return_value = mock_portfolio_optimization_service

        response = client.post(
            "/api/v1/ml/optimize-portfolio/test-user-1",
            params={"risk_tolerance": "INVALID"},
        )
        assert response.status_code == 400

    @patch("src.presentation.api.routers.ml.container")
    def test_user_not_found_returns_404(self, mock_container, client, override_auth, mock_portfolio_optimization_service, sample_portfolio, sample_positions):
        mock_container.services.portfolio_optimization_service.return_value = mock_portfolio_optimization_service

        portfolio_repo = Mock()
        portfolio_repo.get_by_user_id.return_value = sample_portfolio
        position_repo = Mock()
        position_repo.get_by_user_id.return_value = sample_positions
        user_repo = Mock()
        user_repo.get_by_id.return_value = None
        app.dependency_overrides[get_portfolio_repository] = lambda: portfolio_repo
        app.dependency_overrides[get_position_repository] = lambda: position_repo
        app.dependency_overrides[get_user_repository] = lambda: user_repo

        response = client.post("/api/v1/ml/optimize-portfolio/test-user-1")
        assert response.status_code == 404

    @patch("src.presentation.api.routers.ml.container")
    def test_portfolio_not_found_returns_404(self, mock_container, client, override_auth, mock_portfolio_optimization_service, sample_user):
        mock_container.services.portfolio_optimization_service.return_value = mock_portfolio_optimization_service

        portfolio_repo = Mock()
        portfolio_repo.get_by_user_id.return_value = None
        position_repo = Mock()
        position_repo.get_by_user_id.return_value = []
        user_repo = Mock()
        user_repo.get_by_id.return_value = sample_user
        app.dependency_overrides[get_portfolio_repository] = lambda: portfolio_repo
        app.dependency_overrides[get_position_repository] = lambda: position_repo
        app.dependency_overrides[get_user_repository] = lambda: user_repo

        response = client.post("/api/v1/ml/optimize-portfolio/test-user-1")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# POST /api/v1/ml/backtest
# ---------------------------------------------------------------------------

class TestRunBacktest:

    @patch("src.presentation.api.routers.ml.container")
    @patch("src.presentation.api.routers.ml.BacktestingEngine")
    @patch("src.presentation.api.routers.ml.SMACrossoverStrategy")
    def test_success_sma(self, MockStrategy, MockEngine, mock_container, client, override_auth):
        mock_data_provider = Mock()
        mock_ml_service = Mock()
        mock_container.adapters.data_provider_service.return_value = mock_data_provider
        mock_container.services.ml_model_service.return_value = mock_ml_service

        strategy_instance = Mock()
        strategy_instance.get_strategy_name.return_value = "SMA Crossover"
        MockStrategy.return_value = strategy_instance

        backtest_result = Mock()
        backtest_result.final_portfolio_value = 12500.0
        backtest_result.total_return = 0.25
        backtest_result.annualized_return = 0.22
        backtest_result.volatility = 0.15
        backtest_result.sharpe_ratio = 1.5
        backtest_result.max_drawdown = 0.10
        backtest_result.win_rate = 0.60
        backtest_result.total_trades = 30
        backtest_result.winning_trades = 18
        backtest_result.losing_trades = 12
        backtest_result.profit_factor = 1.8
        backtest_result.trades = [
            {"date": "2024-01-15", "symbol": "AAPL", "action": "buy", "quantity": 10, "price": 150.0, "value": 1500.0, "reason": "SMA crossover"},
        ]

        engine_instance = Mock()
        engine_instance.run_backtest.return_value = backtest_result
        MockEngine.return_value = engine_instance

        response = client.post(
            "/api/v1/ml/backtest",
            params={
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "initial_capital": 10000.0,
                "symbols": ["AAPL"],
                "strategy_type": "sma",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["strategy"] == "SMA Crossover"
        assert data["total_return_pct"] == 25.0
        assert data["total_trades"] == 30

    def test_invalid_date_format_returns_400(self, client, override_auth):
        response = client.post(
            "/api/v1/ml/backtest",
            params={
                "start_date": "not-a-date",
                "end_date": "2024-12-31",
                "initial_capital": 10000.0,
                "symbols": ["AAPL"],
            },
        )
        assert response.status_code == 400

    def test_start_after_end_returns_400(self, client, override_auth):
        response = client.post(
            "/api/v1/ml/backtest",
            params={
                "start_date": "2025-01-01",
                "end_date": "2024-01-01",
                "initial_capital": 10000.0,
                "symbols": ["AAPL"],
            },
        )
        assert response.status_code == 400

    def test_invalid_strategy_type_returns_400(self, client, override_auth):
        response = client.post(
            "/api/v1/ml/backtest",
            params={
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "initial_capital": 10000.0,
                "symbols": ["AAPL"],
                "strategy_type": "invalid_strategy",
            },
        )
        assert response.status_code == 400


# ---------------------------------------------------------------------------
# GET /api/v1/ml/risk-analysis/{user_id}
# ---------------------------------------------------------------------------

class TestGetRiskAnalysis:

    def test_unauthorized_user_returns_403(self, client, override_auth):
        """Users cannot view another user's risk analysis."""
        response = client.get("/api/v1/ml/risk-analysis/other-user-id")
        assert response.status_code == 403

    @patch("src.presentation.api.routers.ml.container")
    def test_success(self, mock_container, client, override_auth, mock_repos, mock_risk_analytics_service):
        mock_container.services.risk_analytics_service.return_value = mock_risk_analytics_service

        response = client.get("/api/v1/ml/risk-analysis/test-user-1")
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "test-user-1"
        assert "risk_metrics" in data
        assert data["risk_metrics"]["value_at_risk_95"] == 1500.0

    @patch("src.presentation.api.routers.ml.container")
    def test_portfolio_not_found_returns_404(self, mock_container, client, override_auth, mock_risk_analytics_service):
        mock_container.services.risk_analytics_service.return_value = mock_risk_analytics_service

        portfolio_repo = Mock()
        portfolio_repo.get_by_user_id.return_value = None
        position_repo = Mock()
        position_repo.get_by_user_id.return_value = []
        app.dependency_overrides[get_portfolio_repository] = lambda: portfolio_repo
        app.dependency_overrides[get_position_repository] = lambda: position_repo

        response = client.get("/api/v1/ml/risk-analysis/test-user-1")
        assert response.status_code == 404

    @patch("src.presentation.api.routers.ml.container")
    def test_risk_service_unavailable_returns_500(self, mock_container, client, override_auth, mock_repos):
        mock_container.services.risk_analytics_service.return_value = None

        response = client.get("/api/v1/ml/risk-analysis/test-user-1")
        assert response.status_code == 500
