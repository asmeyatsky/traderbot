"""
Dependency Injection Container

This module provides a centralized dependency injection container using
the dependency-injector library to manage all application dependencies.

Architectural Intent:
- Invert control to the container for dependency management
- Decouple components from their concrete implementations
- Enable easy testing with mock implementations
- Provide single source of truth for dependency configuration
"""
from __future__ import annotations

from dependency_injector import containers, providers
from src.infrastructure.config.settings import settings


class RepositoryContainer(containers.DeclarativeContainer):
    """Container for repository dependencies."""

    # Database configuration would go here
    # database = providers.Singleton(Database, url=settings.DATABASE_URL)

    # Repository implementations (will be created in next phase)
    # order_repository = providers.Singleton(OrderRepositoryImpl, db=database)
    # portfolio_repository = providers.Singleton(PortfolioRepositoryImpl, db=database)
    # user_repository = providers.Singleton(UserRepositoryImpl, db=database)
    # position_repository = providers.Singleton(PositionRepositoryImpl, db=database)


class ServiceContainer(containers.DeclarativeContainer):
    """Container for domain service dependencies."""

    repositories = providers.DependsOn(RepositoryContainer)

    # Domain services
    # trading_service = providers.Singleton(
    #     DefaultTradingDomainService
    # )
    # risk_management_service = providers.Singleton(
    #     DefaultRiskManagementDomainService
    # )
    # portfolio_optimization_service = providers.Singleton(
    #     DefaultPortfolioOptimizationDomainService
    # )
    
    # Advanced risk management service
    advanced_risk_management_service = providers.Singleton(
        "src.domain.services.advanced_risk_management.DefaultAdvancedRiskManagementService"
    )
    
    # Dashboard analytics service
    dashboard_analytics_service = providers.Singleton(
        "src.domain.services.dashboard_analytics.DefaultDashboardAnalyticsService"
    )
    
    # Market data enhancement service
    market_data_enhancement_service = providers.Singleton(
        "src.domain.services.market_data_enhancement.DefaultMarketDataEnhancementService"
    )
    
    # Performance optimizer service (infrastructure service)
    performance_optimizer_service = providers.Singleton(
        "src.infrastructure.performance_optimization.DefaultPerformanceOptimizerService"
    )
    
    # ML model service
    ml_model_service = providers.Singleton(
        "src.domain.services.ml_model_service.DefaultMLModelService"
    )
    
    # RL trading agent service
    rl_trading_agent_service = providers.Singleton(
        "src.domain.services.rl_trading_agents.MockRLAgent"
    )
    
    # ML model service
    ml_model_service = providers.Singleton(
        "src.domain.services.ml_model_service.DefaultMLModelService"
    )
    
    # RL trading agent service
    rl_trading_agent_service = providers.Singleton(
        "src.domain.services.rl_trading_agents.MockRLAgent"
    )


class AdapterContainer(containers.DeclarativeContainer):
    """Container for external adapter dependencies."""

    config = providers.Configuration()

    # Market data adapters
    # market_data_adapter = providers.Singleton(
    #     PolygonMarketDataAdapter,
    #     api_key=settings.POLYGON_API_KEY
    # )

    # News analysis adapters
    # news_analysis_adapter = providers.Singleton(
    #     NLTKNewsAnalysisAdapter
    # )

    # Trading execution adapters
    # trading_execution_adapter = providers.Singleton(
    #     AlpacaTradingExecutionAdapter,
    #     api_key=settings.ALPACA_API_KEY,
    #     secret_key=settings.ALPACA_SECRET_KEY
    # )

    # Notification adapters
    # notification_adapter = providers.Singleton(
    #     EmailNotificationAdapter
    # )


class UseCaseContainer(containers.DeclarativeContainer):
    """Container for use case dependencies."""

    repositories = providers.DependsOn(RepositoryContainer)
    services = providers.DependsOn(ServiceContainer)
    adapters = providers.DependsOn(AdapterContainer)

    # Use cases (will be created in next phase)
    # create_order_use_case = providers.Factory(
    #     CreateOrderUseCase,
    #     order_repository=repositories.order_repository,
    #     portfolio_repository=repositories.portfolio_repository,
    #     user_repository=repositories.user_repository,
    #     trading_service=services.trading_service,
    #     market_data_service=adapters.market_data_adapter
    # )

    # execute_trade_use_case = providers.Factory(
    #     ExecuteTradeUseCase,
    #     order_repository=repositories.order_repository,
    #     portfolio_repository=repositories.portfolio_repository,
    #     user_repository=repositories.user_repository,
    #     trading_service=services.trading_service,
    #     risk_service=services.risk_management_service,
    #     market_data_service=adapters.market_data_adapter,
    #     trading_execution_service=adapters.trading_execution_adapter,
    #     notification_service=adapters.notification_adapter,
    #     ai_model_service=providers.Factory(MockAIModelAdapter)
    # )

    # analyze_news_sentiment_use_case = providers.Factory(
    #     AnalyzeNewsSentimentUseCase,
    #     news_analysis_service=adapters.news_analysis_adapter,
    #     market_data_service=adapters.market_data_adapter,
    #     portfolio_repository=repositories.portfolio_repository
    # )

    # get_portfolio_performance_use_case = providers.Factory(
    #     GetPortfolioPerformanceUseCase,
    #     portfolio_repository=repositories.portfolio_repository,
    #     market_data_service=adapters.market_data_adapter
    # )

    # get_user_preferences_use_case = providers.Factory(
    #     GetUserPreferencesUseCase,
    #     user_repository=repositories.user_repository
    # )


class Container(containers.DeclarativeContainer):
    """Main application container."""

    config = providers.Configuration()
    config.from_dict({
        'settings': settings.model_dump(),
    })

    repositories = providers.Container(RepositoryContainer)
    services = providers.Container(ServiceContainer)
    adapters = providers.Container(AdapterContainer)
    use_cases = providers.Container(UseCaseContainer)


# Create global container instance
container = Container()


def get_container() -> Container:
    """Get the global DI container."""
    return container
