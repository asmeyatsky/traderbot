"""
Enhanced Dashboard API Router

This router handles all enhanced dashboard-related endpoints following RESTful principles.
All endpoints require authentication.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import Dict, Any
import logging
from datetime import datetime
from decimal import Decimal

from src.infrastructure.security import get_current_user
from src.domain.services.dashboard_analytics import DefaultDashboardAnalyticsService
from src.domain.entities.trading import Portfolio
from src.domain.value_objects import Symbol
from src.infrastructure.repositories import PortfolioRepository, PositionRepository, UserRepository
from src.infrastructure.di_container import container
from src.presentation.api.dependencies import (
    get_portfolio_repository,
    get_position_repository,
    get_user_repository,
    get_dashboard_analytics_service,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])


def _load_portfolio_with_positions(
    user_id: str,
    portfolio_repo: PortfolioRepository,
    position_repo: PositionRepository,
) -> Portfolio | None:
    """Fetch portfolio and its positions from repositories. Returns None if no portfolio."""
    portfolio = portfolio_repo.get_by_user_id(user_id)
    if not portfolio:
        return None
    positions = position_repo.get_by_user_id(user_id)
    return Portfolio(
        id=portfolio.id,
        user_id=portfolio.user_id,
        positions=positions,
        cash_balance=portfolio.cash_balance,
        created_at=portfolio.created_at,
        updated_at=portfolio.updated_at,
    )


@router.get(
    "/overview/{user_id}",
    summary="Get comprehensive portfolio dashboard",
    responses={
        200: {"description": "Dashboard metrics retrieved successfully"},
        401: {"description": "Unauthorized"},
        404: {"description": "Portfolio not found"},
    }
)
async def get_dashboard_overview(
    user_id: str,
    include_technical: bool = Query(True, description="Include technical indicators"),
    days: int = Query(30, description="Number of days for performance chart"),
    current_user_id: str = Depends(get_current_user),
    portfolio_repo: PortfolioRepository = Depends(get_portfolio_repository),
    position_repo: PositionRepository = Depends(get_position_repository),
    user_repo: UserRepository = Depends(get_user_repository),
    dashboard_service: DefaultDashboardAnalyticsService = Depends(get_dashboard_analytics_service),
) -> Dict[str, Any]:
    """Get comprehensive dashboard overview with portfolio metrics, performance,
    technical indicators, and allocation breakdown."""
    if current_user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view this dashboard"
        )

    try:
        portfolio = _load_portfolio_with_positions(user_id, portfolio_repo, position_repo)

        # Return empty dashboard for new users with no portfolio yet
        if not portfolio:
            return {
                "user_id": user_id,
                "calculated_at": datetime.now().isoformat(),
                "portfolio_value": 0.0,
                "daily_pnl": 0.0,
                "daily_pnl_percent": 0.0,
                "total_pnl": 0.0,
                "total_pnl_percent": 0.0,
                "positions_count": 0,
                "top_performers": [],
                "worst_performers": [],
                "allocation": [],
                "performance_history": [],
            }

        user = user_repo.get_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        dashboard_metrics = dashboard_service.get_dashboard_metrics(portfolio, user)

        # Fetch real current prices for performers
        market_data_service = None
        try:
            market_data_service = container.adapters.market_data_service()
        except Exception:
            logger.warning("Market data service unavailable for dashboard prices")

        def _get_price(sym_str: str) -> float:
            if not market_data_service:
                return 0.0
            try:
                price = market_data_service.get_current_price(Symbol(sym_str))
                return float(price.amount) if price else 0.0
            except Exception:
                return 0.0

        total_pnl = float(dashboard_metrics.realized_pnl.amount) + float(dashboard_metrics.unrealized_pnl.amount)
        total_value = float(dashboard_metrics.total_value.amount)
        total_pnl_percent = (total_pnl / total_value * 100) if total_value > 0 else 0.0

        result = {
            "user_id": user_id,
            "calculated_at": datetime.now().isoformat(),
            "portfolio_value": total_value,
            "daily_pnl": float(dashboard_metrics.daily_pnl.amount),
            "daily_pnl_percent": float(dashboard_metrics.daily_pnl_percentage),
            "total_pnl": total_pnl,
            "total_pnl_percent": total_pnl_percent,
            "positions_count": dashboard_metrics.positions_count,
            "top_performers": [
                {"symbol": str(symbol), "change_percent": float(pct), "current_price": _get_price(str(symbol))}
                for symbol, pct in dashboard_metrics.top_gainers
            ],
            "worst_performers": [
                {"symbol": str(symbol), "change_percent": float(pct), "current_price": _get_price(str(symbol))}
                for symbol, pct in dashboard_metrics.top_losers
            ],
            "allocation": [
                {"name": str(symbol), "value": 0.0, "percentage": float(pct)}
                for symbol, pct in dashboard_metrics.allocation_by_asset.items()
            ],
            "performance_history": [
                {"date": item["date"], "value": float(item["value"])}
                for item in dashboard_metrics.performance_chart_data
            ],
        }

        logger.info(f"Dashboard overview retrieved for user {user_id}")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating dashboard metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to calculate dashboard metrics"
        )


@router.get(
    "/allocation/{user_id}",
    summary="Get portfolio allocation breakdown",
    responses={
        200: {"description": "Allocation data retrieved successfully"},
        401: {"description": "Unauthorized"},
        404: {"description": "Portfolio not found"},
    }
)
async def get_allocation_breakdown(
    user_id: str,
    breakdown_type: str = Query("asset", description="Type of breakdown: 'asset', 'sector', or 'both'"),
    current_user_id: str = Depends(get_current_user),
    portfolio_repo: PortfolioRepository = Depends(get_portfolio_repository),
    position_repo: PositionRepository = Depends(get_position_repository),
    user_repo: UserRepository = Depends(get_user_repository),
    dashboard_service: DefaultDashboardAnalyticsService = Depends(get_dashboard_analytics_service),
) -> Dict[str, Any]:
    """Get portfolio allocation breakdown by asset or sector."""
    if current_user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view this allocation"
        )

    try:
        portfolio = _load_portfolio_with_positions(user_id, portfolio_repo, position_repo)

        if not portfolio:
            return {
                "user_id": user_id,
                "breakdown_type": breakdown_type,
                "calculated_at": datetime.now().isoformat(),
                "allocation_by_asset": {},
                "allocation_by_sector": {},
            }

        user = user_repo.get_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        result = {
            "user_id": user_id,
            "breakdown_type": breakdown_type,
            "calculated_at": datetime.now().isoformat()
        }

        if breakdown_type in ["asset", "both"]:
            allocation_by_asset = {}
            total_value = portfolio.total_value.amount
            if total_value > 0:
                for position in portfolio.positions:
                    allocation = (position.market_value.amount / total_value) * 100
                    allocation_by_asset[str(position.symbol)] = float(allocation)
            result["allocation_by_asset"] = allocation_by_asset

        if breakdown_type in ["sector", "both"]:
            dashboard_metrics = dashboard_service.get_dashboard_metrics(portfolio, user)
            result["allocation_by_sector"] = {
                sector: float(pct)
                for sector, pct in dashboard_metrics.allocation_by_sector.items()
            }

        logger.info(f"Allocation breakdown retrieved for user {user_id}")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating allocation breakdown: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to calculate allocation breakdown"
        )


@router.get(
    "/technical-indicators/{symbol}",
    summary="Get technical indicators for a symbol",
    responses={
        200: {"description": "Technical indicators retrieved successfully"},
        401: {"description": "Unauthorized"},
        404: {"description": "Symbol not found"},
    }
)
async def get_technical_indicators(
    symbol: str,
    days: int = Query(90, description="Number of days for calculations"),
    current_user_id: str = Depends(get_current_user),
    dashboard_service: DefaultDashboardAnalyticsService = Depends(get_dashboard_analytics_service),
) -> Dict[str, Any]:
    """Get technical indicators for a specific symbol."""
    try:
        tech_indicators = dashboard_service.calculate_technical_indicators(
            Symbol(symbol), days
        )

        result = {
            "symbol": str(tech_indicators.symbol),
            "calculated_at": tech_indicators.calculated_at.isoformat() if tech_indicators.calculated_at else None,
            "days_used": days,
            "sma_20": float(tech_indicators.sma_20) if tech_indicators.sma_20 else None,
            "sma_50": float(tech_indicators.sma_50) if tech_indicators.sma_50 else None,
            "ema_12": float(tech_indicators.ema_12) if tech_indicators.ema_12 else None,
            "ema_26": float(tech_indicators.ema_26) if tech_indicators.ema_26 else None,
            "rsi": float(tech_indicators.rsi) if tech_indicators.rsi else None,
            "macd": float(tech_indicators.macd) if tech_indicators.macd else None,
            "macd_signal": float(tech_indicators.macd_signal) if tech_indicators.macd_signal else None,
            "bollinger_upper": float(tech_indicators.bollinger_upper) if tech_indicators.bollinger_upper else None,
            "bollinger_lower": float(tech_indicators.bollinger_lower) if tech_indicators.bollinger_lower else None,
            "atr": float(tech_indicators.atr) if tech_indicators.atr else None
        }

        logger.info(f"Technical indicators retrieved for symbol {symbol}")
        return result

    except Exception as e:
        logger.error(f"Error calculating technical indicators: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to calculate technical indicators"
        )
