"""
Dependency Injection Container for the Trading Platform

This module defines the DI container using dependency_injector to manage
dependencies across the application following clean architecture principles.
"""
from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject

from src.infrastructure.config.settings import settings
from src.infrastructure.database import DatabaseManager
from src.infrastructure.cache import CacheManager
from src.infrastructure.security import JWTAuthenticator
from src.infrastructure.rate_limiting import RateLimiter
from src.infrastructure.event_bus import EventBus
from src.infrastructure.logging import setup_logging
from src.infrastructure.repositories import (
    UserRepository,
    OrderRepository,
    PositionRepository,
    PortfolioRepository,
)
from src.infrastructure.data_processing.ml_model_service import (
    LSTMPricePredictionService,
    XGBoostPredictionService,
    TransformerSentimentAnalysisService,
    RLTradingAgentService,
    EnsembleModelService,
    AdvancedRiskAnalyticsService,
    PortfolioOptimizationService
)
from src.infrastructure.performance_optimization import DefaultPerformanceOptimizerService
from src.infrastructure.data_processing.news_aggregation_service import (
    MarketauxNewsService,
    EnhancedNewsAggregationService,
    NewsImpactAnalyzer
)
from src.infrastructure.data_processing.backtesting_engine import (
    YahooFinanceDataProvider
)
from src.infrastructure.data_processing.ensemble_predictor import EnsemblePredictorService
from src.infrastructure.api_clients.market_data import MarketDataService
from src.infrastructure.broker_integration import (
    AlpacaBrokerService,
    BrokerIntegrationService,
    BrokerAdapterManager
)
from src.domain.entities.user import RiskTolerance
from src.domain.services.trading import DefaultTradingDomainService, DefaultRiskManagementDomainService
from src.domain.services.advanced_risk_management import DefaultAdvancedRiskManagementService
from src.domain.services.dashboard_analytics import DefaultDashboardAnalyticsService
from src.infrastructure.services.market_data_enhancement import DefaultMarketDataEnhancementService
from src.domain.services.exchange_registry import ExchangeRegistry
from src.domain.services.risk_management import RiskManager, CircuitBreakerService
from src.infrastructure.adapters.notification import LoggingNotificationAdapter
from src.infrastructure.adapters.claude_chat_adapter import ClaudeChatAdapter
from src.infrastructure.adapters.technical_analysis import PandasTATechnicalAnalysisAdapter
from src.infrastructure.adapters.stock_screener import YahooFinanceScreenerAdapter
from src.infrastructure.repositories.activity_log_repository import ActivityLogRepository
from src.infrastructure.repositories.conversation_repository import ConversationRepository
from src.infrastructure.repositories.broker_account_repository import BrokerAccountRepository
from src.infrastructure.adapters.audit_event_sink import SqlAlchemyAuditEventSink
from src.infrastructure.mcp import McpRegistry
from src.infrastructure.mcp.market_data import MarketDataMcpServer
from src.infrastructure.mcp.portfolio import PortfolioMcpServer
from src.infrastructure.mcp.research import ResearchMcpServer


def _build_mcp_registry(
    market_data_service,
    ai_model_service,
    news_analysis_service,
    portfolio_repository,
    user_repository,
    order_repository,
    technical_analysis_port,
    stock_screener,
    exchange_registry,
    ensemble_predictor,
    backtest_use_case,
):
    """Assemble the MCP registry with one server per bounded context.

    Factory function rather than a provider so each server is constructed with
    its direct collaborators and the registry carries the wiring rules in one
    place (2026 rules §3.5).
    """
    registry = McpRegistry()
    registry.register(
        MarketDataMcpServer(
            market_data_service=market_data_service,
            ai_model_service=ai_model_service,
            news_analysis_service=news_analysis_service,
            technical_analysis_port=technical_analysis_port,
            stock_screener=stock_screener,
            exchange_registry=exchange_registry,
            ensemble_predictor=ensemble_predictor,
        )
    )
    registry.register(
        PortfolioMcpServer(
            portfolio_repository=portfolio_repository,
            user_repository=user_repository,
            audit_sink=SqlAlchemyAuditEventSink(),
            order_repository=order_repository,
        )
    )
    registry.register(ResearchMcpServer(backtest_use_case=backtest_use_case))
    return registry


class RepositoryContainer(containers.DeclarativeContainer):
    """Container for repository dependencies."""

    # Core Services
    database_manager = providers.Singleton(DatabaseManager)
    cache_manager = providers.Singleton(CacheManager)

    # Repository implementations
    user_repository = providers.Factory(UserRepository)
    order_repository = providers.Factory(OrderRepository)
    position_repository = providers.Factory(PositionRepository)
    portfolio_repository = providers.Factory(PortfolioRepository)
    activity_log_repository = providers.Factory(ActivityLogRepository)
    conversation_repository = providers.Factory(ConversationRepository)
    broker_account_repository = providers.Factory(BrokerAccountRepository)


class ServiceContainer(containers.DeclarativeContainer):
    """Container for domain and application service dependencies."""

    repositories = providers.DependenciesContainer()
    adapters = providers.DependenciesContainer()

    # Core domain services
    trading_domain_service = providers.Factory(DefaultTradingDomainService)
    risk_management_service = providers.Factory(DefaultRiskManagementDomainService)

    # AI/ML Services
    lstm_model_service = providers.Factory(LSTMPricePredictionService)
    xgboost_model_service = providers.Factory(XGBoostPredictionService)
    sentiment_analysis_service = providers.Factory(TransformerSentimentAnalysisService)
    rl_agent_service = providers.Factory(RLTradingAgentService)

    # Ensemble predictor (RF + GB + SHAP)
    ensemble_predictor = providers.Singleton(EnsemblePredictorService)

    # Ensemble service combining multiple AI models
    ml_model_service = providers.Factory(
        EnsembleModelService,
        lstm_service=lstm_model_service,
        sentiment_service=sentiment_analysis_service,
        xgboost_service=xgboost_model_service,
    )

    # Advanced analytics services
    risk_analytics_service = providers.Factory(AdvancedRiskAnalyticsService)
    portfolio_optimization_service = providers.Factory(PortfolioOptimizationService)
    advanced_risk_management_service = providers.Factory(DefaultAdvancedRiskManagementService)
    dashboard_analytics_service = providers.Factory(DefaultDashboardAnalyticsService)

    # Market data enhancement backed by real API providers
    market_data_enhancement_service = providers.Singleton(
        DefaultMarketDataEnhancementService,
        market_data_provider=adapters.market_data_service,
        sentiment_service=sentiment_analysis_service,
        technical_analysis_port=adapters.technical_analysis_adapter,
    )

    # Exchange registry
    exchange_registry = providers.Singleton(ExchangeRegistry)

    # Performance optimization
    performance_optimizer_service = providers.Factory(DefaultPerformanceOptimizerService)

    # News services
    marketaux_news_service = providers.Factory(
        MarketauxNewsService,
        api_key=settings.MARKETAUX_API_KEY,
        sentiment_service=sentiment_analysis_service
    )

    news_aggregation_service = providers.Factory(
        EnhancedNewsAggregationService,
        marketaux_service=marketaux_news_service,
        sentiment_service=sentiment_analysis_service
    )

    news_impact_analyzer_service = providers.Factory(
        NewsImpactAnalyzer,
        news_aggregation_service=news_aggregation_service
    )


class AdapterContainer(containers.DeclarativeContainer):
    """Container for external adapter dependencies."""

    # Notification adapter (replaces object() placeholder)
    notification_service = providers.Singleton(LoggingNotificationAdapter)

    # Real market data service (Polygon → AlphaVantage → Finnhub → Yahoo fallback)
    market_data_service = providers.Singleton(MarketDataService)

    # Data provider for backtesting and market data
    data_provider_service = providers.Factory(YahooFinanceDataProvider)

    # Broker integration — per-user paper/live routing via factory (ADR-002).
    # Live routing is gated by EMERGENCY_HALT and ENABLE_LIVE_TRADING env vars
    # AND the daily-loss-cap tracker; never call AlpacaBrokerService directly.
    from src.infrastructure.adapters.broker_factory import BrokerServiceFactory
    from src.infrastructure.adapters.daily_loss_tracker import RedisDailyLossTracker
    daily_loss_tracker = providers.Singleton(
        RedisDailyLossTracker,
        redis_client=cache_manager.provided.client,
    )
    broker_service_factory = providers.Singleton(
        BrokerServiceFactory,
        loss_tracker=daily_loss_tracker,
    )

    # Legacy provider kept for compatibility with callers that haven't moved to
    # the factory yet. This intentionally returns a paper broker — any caller
    # that should be live-aware must be migrated to `broker_service_factory`.
    alpaca_broker_service = providers.Factory(
        AlpacaBrokerService,
        api_key=settings.ALPACA_API_KEY,
        secret_key=settings.ALPACA_SECRET_KEY,
        paper_trading=True,  # legacy callers — paper only. See broker_service_factory for per-user routing.
    )

    broker_integration_service = providers.Factory(
        BrokerIntegrationService,
        alpaca_service=alpaca_broker_service
    )

    # Technical analysis adapter
    technical_analysis_adapter = providers.Singleton(PandasTATechnicalAnalysisAdapter)

    # Stock screener adapter
    stock_screener = providers.Singleton(YahooFinanceScreenerAdapter)

    # AI Chat adapter
    claude_chat_adapter = providers.Singleton(ClaudeChatAdapter)

    # Discipline check (Phase 10.1) — per-user pre-trade AI veto.
    from src.infrastructure.adapters.discipline_check import ClaudeDisciplineCheckAdapter
    discipline_check_adapter = providers.Singleton(ClaudeDisciplineCheckAdapter)

    # Broker adapter manager
    broker_adapter_manager = providers.Singleton(BrokerAdapterManager)

    # Risk services (singletons — config injected from infrastructure settings)
    risk_manager = providers.Singleton(
        RiskManager,
        notification_service=notification_service,
        risk_limits={
            RiskTolerance.CONSERVATIVE: {
                "max_drawdown": Decimal(str(settings.RISK_CONSERVATIVE_MAX_DRAWDOWN)),
                "position_limit_percentage": Decimal(str(settings.RISK_CONSERVATIVE_POSITION_LIMIT_PCT)),
                "volatility_threshold": Decimal(str(settings.RISK_CONSERVATIVE_VOLATILITY_THRESHOLD)),
            },
            RiskTolerance.MODERATE: {
                "max_drawdown": Decimal(str(settings.RISK_MODERATE_MAX_DRAWDOWN)),
                "position_limit_percentage": Decimal(str(settings.RISK_MODERATE_POSITION_LIMIT_PCT)),
                "volatility_threshold": Decimal(str(settings.RISK_MODERATE_VOLATILITY_THRESHOLD)),
            },
            RiskTolerance.AGGRESSIVE: {
                "max_drawdown": Decimal(str(settings.RISK_AGGRESSIVE_MAX_DRAWDOWN)),
                "position_limit_percentage": Decimal(str(settings.RISK_AGGRESSIVE_POSITION_LIMIT_PCT)),
                "volatility_threshold": Decimal(str(settings.RISK_AGGRESSIVE_VOLATILITY_THRESHOLD)),
            },
        },
    )
    circuit_breaker_service = providers.Singleton(
        CircuitBreakerService,
        notification_service=notification_service,
        volatility_threshold=Decimal(str(settings.CIRCUIT_BREAKER_VOLATILITY_THRESHOLD)),
        reset_after=timedelta(minutes=settings.CIRCUIT_BREAKER_RESET_MINUTES),
    )

    # Security and infrastructure
    jwt_authenticator = providers.Singleton(JWTAuthenticator)
    rate_limiter = providers.Singleton(RateLimiter)
    event_bus = providers.Singleton(EventBus)


class UseCaseContainer(containers.DeclarativeContainer):
    """Container for use case dependencies."""

    repositories = providers.DependenciesContainer()
    services = providers.DependenciesContainer()
    adapters = providers.DependenciesContainer()

    # Use cases
    from src.application.use_cases.trading import (
        CreateOrderUseCase,
        ExecuteTradeUseCase,
        AnalyzeNewsSentimentUseCase,
        GetPortfolioPerformanceUseCase,
        GetUserPreferencesUseCase,
    )
    from src.application.use_cases.chat import ChatUseCase
    from src.application.use_cases.backtest import RunBacktestUseCase
    from src.application.use_cases.broker_account import (
        LinkBrokerAccountUseCase,
        GetBrokerAccountsUseCase,
        UpdateBrokerSettingsUseCase,
        DeleteBrokerAccountUseCase,
    )

    create_order_use_case = providers.Factory(
        CreateOrderUseCase,
        order_repository=repositories.order_repository,
        portfolio_repository=repositories.portfolio_repository,
        user_repository=repositories.user_repository,
        trading_service=services.trading_domain_service,
        market_data_service=adapters.market_data_service,
        position_repository=repositories.position_repository,
        activity_log_repository=repositories.activity_log_repository,
        broker_routing=adapters.broker_service_factory,
        discipline_check=adapters.discipline_check_adapter,
    )

    execute_trade_use_case = providers.Factory(
        ExecuteTradeUseCase,
        order_repository=repositories.order_repository,
        portfolio_repository=repositories.portfolio_repository,
        user_repository=repositories.user_repository,
        trading_service=services.trading_domain_service,
        risk_service=services.risk_analytics_service,
        market_data_service=adapters.market_data_service,
        trading_execution_service=adapters.broker_integration_service,
        notification_service=adapters.notification_service,
        ai_model_service=services.ml_model_service,
    )

    analyze_news_sentiment_use_case = providers.Factory(
        AnalyzeNewsSentimentUseCase,
        news_analysis_service=services.news_aggregation_service,
        market_data_service=adapters.market_data_service,
        portfolio_repository=repositories.portfolio_repository,
    )

    get_portfolio_performance_use_case = providers.Factory(
        GetPortfolioPerformanceUseCase,
        portfolio_repository=repositories.portfolio_repository,
        market_data_service=adapters.market_data_service,
    )

    get_user_preferences_use_case = providers.Factory(
        GetUserPreferencesUseCase,
        user_repository=repositories.user_repository,
    )

    link_broker_account_use_case = providers.Factory(
        LinkBrokerAccountUseCase,
        broker_account_repository=repositories.broker_account_repository,
    )

    get_broker_accounts_use_case = providers.Factory(
        GetBrokerAccountsUseCase,
        broker_account_repository=repositories.broker_account_repository,
    )

    update_broker_settings_use_case = providers.Factory(
        UpdateBrokerSettingsUseCase,
        broker_account_repository=repositories.broker_account_repository,
    )

    delete_broker_account_use_case = providers.Factory(
        DeleteBrokerAccountUseCase,
        broker_account_repository=repositories.broker_account_repository,
    )

    # Backtest pipeline — the use case takes a port; the adapter owns the
    # data-provider + strategy wiring (Phase 4 burn-down of ignore_imports).
    from src.infrastructure.adapters.backtest_runner import YahooBacktestRunner
    backtest_runner = providers.Singleton(YahooBacktestRunner)
    backtest_use_case = providers.Factory(
        RunBacktestUseCase, runner=backtest_runner,
    )

    mcp_registry = providers.Factory(
        _build_mcp_registry,
        market_data_service=adapters.market_data_service,
        ai_model_service=services.ml_model_service,
        news_analysis_service=services.news_aggregation_service,
        portfolio_repository=repositories.portfolio_repository,
        user_repository=repositories.user_repository,
        order_repository=repositories.order_repository,
        technical_analysis_port=adapters.technical_analysis_adapter,
        stock_screener=adapters.stock_screener,
        exchange_registry=services.exchange_registry,
        ensemble_predictor=services.ensemble_predictor,
        backtest_use_case=backtest_use_case,
    )

    chat_use_case = providers.Factory(
        ChatUseCase,
        ai_chat_port=adapters.claude_chat_adapter,
        conversation_repository=repositories.conversation_repository,
        user_repository=repositories.user_repository,
        tool_registry=mcp_registry,
    )


class Container(containers.DeclarativeContainer):
    """Main application container."""

    config = providers.Configuration()
    config.from_dict({
        'settings': settings.model_dump(),
    })

    repositories = providers.Container(RepositoryContainer)
    adapters = providers.Container(AdapterContainer)
    services = providers.Container(ServiceContainer, repositories=repositories, adapters=adapters)
    use_cases = providers.Container(UseCaseContainer,
                                   repositories=repositories,
                                   services=services,
                                   adapters=adapters)


# Create the main container instance
container = Container()


def get_container() -> Container:
    """Get the global DI container."""
    return container


# Convenience functions to access services directly from container
def ml_model_service():
    """Get the ML model service instance."""
    return container.services.ml_model_service()

def news_impact_analyzer_service():
    """Get the news impact analyzer service instance."""
    return container.services.news_impact_analyzer_service()

def risk_analytics_service():
    """Get the risk analytics service instance."""
    return container.services.risk_analytics_service()

def portfolio_optimization_service():
    """Get the portfolio optimization service instance."""
    return container.services.portfolio_optimization_service()

def data_provider_service():
    """Get the data provider service instance."""
    return container.adapters.data_provider_service()

def broker_integration_service():
    """Get the broker integration service instance."""
    return container.adapters.broker_integration_service()

def advanced_risk_management_service():
    """Get the advanced risk management service instance."""
    return container.services.advanced_risk_management_service()

def dashboard_analytics_service():
    """Get the dashboard analytics service instance."""
    return container.services.dashboard_analytics_service()

def broker_adapter_manager():
    """Get the broker adapter manager instance."""
    return container.adapters.broker_adapter_manager()
