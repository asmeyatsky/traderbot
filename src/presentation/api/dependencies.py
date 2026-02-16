"""
FastAPI Dependency Injection Integration

This module provides FastAPI dependencies that integrate with the DI container,
following clean architecture principles by keeping infrastructure concerns
separate from the presentation layer.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Callable

from fastapi import Depends

from src.infrastructure.di_container import container, Container
from src.application.use_cases.trading import (
    CreateOrderUseCase,
    ExecuteTradeUseCase,
    AnalyzeNewsSentimentUseCase,
    GetPortfolioPerformanceUseCase,
    GetUserPreferencesUseCase,
)
from src.infrastructure.repositories import (
    OrderRepository,
    PortfolioRepository,
    UserRepository,
    PositionRepository,
)


@lru_cache(maxsize=1)
def get_container() -> Container:
    """Get the global DI container instance."""
    return container


# ============================================================================
# Repository Dependencies
# ============================================================================

def get_order_repository() -> OrderRepository:
    """Provide OrderRepository instance."""
    return container.repositories.order_repository()


def get_portfolio_repository() -> PortfolioRepository:
    """Provide PortfolioRepository instance."""
    return container.repositories.portfolio_repository()


def get_user_repository() -> UserRepository:
    """Provide UserRepository instance."""
    return container.repositories.user_repository()


def get_position_repository() -> PositionRepository:
    """Provide PositionRepository instance."""
    return container.repositories.position_repository()


# ============================================================================
# Use Case Dependencies
# ============================================================================

def get_create_order_use_case() -> CreateOrderUseCase:
    """Provide CreateOrderUseCase instance with all dependencies."""
    return container.use_cases.create_order_use_case()


def get_execute_trade_use_case() -> ExecuteTradeUseCase:
    """Provide ExecuteTradeUseCase instance with all dependencies."""
    return container.use_cases.execute_trade_use_case()


def get_analyze_news_sentiment_use_case() -> AnalyzeNewsSentimentUseCase:
    """Provide AnalyzeNewsSentimentUseCase instance with all dependencies."""
    return container.use_cases.analyze_news_sentiment_use_case()


def get_portfolio_performance_use_case() -> GetPortfolioPerformanceUseCase:
    """Provide GetPortfolioPerformanceUseCase instance with all dependencies."""
    return container.use_cases.get_portfolio_performance_use_case()


def get_user_preferences_use_case() -> GetUserPreferencesUseCase:
    """Provide GetUserPreferencesUseCase instance with all dependencies."""
    return container.use_cases.get_user_preferences_use_case()


# ============================================================================
# Service Dependencies
# ============================================================================

def get_ml_model_service():
    """Provide ML model service instance."""
    return container.services.ml_model_service()


def get_risk_analytics_service():
    """Provide risk analytics service instance."""
    return container.services.risk_analytics_service()


def get_news_aggregation_service():
    """Provide news aggregation service instance."""
    return container.services.news_aggregation_service()


def get_portfolio_optimization_service():
    """Provide portfolio optimization service instance."""
    return container.services.portfolio_optimization_service()


def get_trading_domain_service():
    """Provide trading domain service instance."""
    return container.services.trading_domain_service()


def get_risk_management_service():
    """Provide risk management service instance."""
    return container.services.risk_management_service()


# ============================================================================
# Adapter Dependencies
# ============================================================================

def get_broker_integration_service():
    """Provide broker integration service instance."""
    return container.adapters.broker_integration_service()


def get_data_provider_service():
    """Provide data provider service instance."""
    return container.adapters.data_provider_service()


def get_advanced_risk_management_service():
    """Provide advanced risk management service instance."""
    return container.services.advanced_risk_management_service()


def get_dashboard_analytics_service():
    """Provide dashboard analytics service instance."""
    return container.services.dashboard_analytics_service()


def get_broker_adapter_manager():
    """Provide broker adapter manager instance."""
    return container.adapters.broker_adapter_manager()


def get_rl_agent_service():
    """Provide RL trading agent service instance."""
    return container.services.rl_agent_service()
