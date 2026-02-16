"""
Risk Analytics API Router

This router handles all risk-related endpoints following RESTful principles.
All endpoints require authentication.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import Dict, Any
import logging
from datetime import datetime
from decimal import Decimal

from src.infrastructure.security import get_current_user
from src.domain.services.advanced_risk_management import (
    DefaultAdvancedRiskManagementService, RiskMetrics, StressTestScenario
)
from src.domain.entities.trading import Portfolio
from src.infrastructure.repositories import PortfolioRepository, PositionRepository
from src.presentation.api.dependencies import (
    get_portfolio_repository,
    get_position_repository,
    get_advanced_risk_management_service,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/risk", tags=["risk"])


def _load_portfolio_with_positions(
    user_id: str,
    portfolio_repo: PortfolioRepository,
    position_repo: PositionRepository,
) -> Portfolio:
    """Fetch portfolio and its positions from repositories."""
    portfolio = portfolio_repo.get_by_user_id(user_id)
    if not portfolio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Portfolio not found"
        )
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
    "/portfolio/{user_id}",
    summary="Get portfolio risk metrics",
    responses={
        200: {"description": "Risk metrics retrieved successfully"},
        401: {"description": "Unauthorized"},
        404: {"description": "Portfolio not found"},
    }
)
async def get_portfolio_risk_metrics(
    user_id: str,
    lookback_days: int = Query(252, description="Number of days to look back for calculations"),
    confidence_level: float = Query(95.0, description="Confidence level for VaR calculations"),
    current_user_id: str = Depends(get_current_user),
    portfolio_repo: PortfolioRepository = Depends(get_portfolio_repository),
    position_repo: PositionRepository = Depends(get_position_repository),
    risk_service: DefaultAdvancedRiskManagementService = Depends(get_advanced_risk_management_service),
) -> Dict[str, Any]:
    """Get comprehensive risk metrics for a user's portfolio."""
    if current_user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view this portfolio"
        )

    try:
        portfolio = _load_portfolio_with_positions(user_id, portfolio_repo, position_repo)

        risk_metrics = risk_service.calculate_portfolio_metrics(
            portfolio=portfolio,
            lookback_days=lookback_days
        )

        result = {
            "user_id": user_id,
            "calculated_at": datetime.now().isoformat(),
            "lookback_days": lookback_days,
            "confidence_level": confidence_level,
            "value_at_risk": {
                "amount": float(risk_metrics.value_at_risk.amount),
                "currency": risk_metrics.value_at_risk.currency
            } if risk_metrics.value_at_risk else None,
            "expected_shortfall": {
                "amount": float(risk_metrics.expected_shortfall.amount),
                "currency": risk_metrics.expected_shortfall.currency
            } if risk_metrics.expected_shortfall else None,
            "max_drawdown": float(risk_metrics.max_drawdown) if risk_metrics.max_drawdown else None,
            "volatility": float(risk_metrics.volatility) if risk_metrics.volatility else None,
            "beta": float(risk_metrics.beta) if risk_metrics.beta else None,
            "sharpe_ratio": float(risk_metrics.sharpe_ratio) if risk_metrics.sharpe_ratio else None,
            "sortino_ratio": float(risk_metrics.sortino_ratio) if risk_metrics.sortino_ratio else None,
            "correlation_matrix": risk_metrics.correlation_matrix,
            "stress_test_results": {
                scenario: {
                    "amount": float(result.amount) if result else 0,
                    "currency": result.currency if result else "USD"
                }
                for scenario, result in risk_metrics.stress_test_results.items()
            } if risk_metrics.stress_test_results else None,
            "risk_contribution": {
                asset: float(weight)
                for asset, weight in risk_metrics.portfolio_at_risk.items()
            } if risk_metrics.portfolio_at_risk else None
        }

        logger.info(f"Risk metrics retrieved for user {user_id}")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating risk metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to calculate risk metrics"
        )


@router.post(
    "/stress-test/{user_id}",
    summary="Perform stress test on portfolio",
    responses={
        200: {"description": "Stress test results"},
        401: {"description": "Unauthorized"},
        404: {"description": "Portfolio not found"},
    }
)
async def perform_stress_test(
    user_id: str,
    scenario_name: str = Query(..., description="Name of the stress test scenario"),
    current_user_id: str = Depends(get_current_user),
    portfolio_repo: PortfolioRepository = Depends(get_portfolio_repository),
    position_repo: PositionRepository = Depends(get_position_repository),
    risk_service: DefaultAdvancedRiskManagementService = Depends(get_advanced_risk_management_service),
) -> Dict[str, Any]:
    """Perform a stress test on the user's portfolio under specific market conditions."""
    if current_user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to test this portfolio"
        )

    try:
        # Define common stress test scenarios
        scenarios = {
            "2008_crisis": StressTestScenario(
                name="2008 Financial Crisis",
                description="Simulates conditions during 2008 financial crisis",
                market_move={
                    "SPY": Decimal("-40.0"),
                    "AAPL": Decimal("-50.0"),
                    "GOOGL": Decimal("-45.0"),
                    "TSLA": Decimal("-60.0"),
                },
                probability=Decimal("0.5")
            ),
            "mild_recession": StressTestScenario(
                name="Mild Recession",
                description="Simulates a moderate economic downturn",
                market_move={
                    "SPY": Decimal("-20.0"),
                    "AAPL": Decimal("-25.0"),
                    "GOOGL": Decimal("-22.0"),
                    "TSLA": Decimal("-35.0"),
                },
                probability=Decimal("3.0")
            ),
            "rate_hike_shock": StressTestScenario(
                name="Rate Hike Shock",
                description="Simulates rapid interest rate increases",
                market_move={
                    "SPY": Decimal("-15.0"),
                    "AAPL": Decimal("-18.0"),
                    "GOOGL": Decimal("-20.0"),
                    "TSLA": Decimal("-30.0"),
                },
                probability=Decimal("2.0")
            )
        }

        if scenario_name not in scenarios:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown scenario: {scenario_name}. Available: {list(scenarios.keys())}"
            )

        portfolio = _load_portfolio_with_positions(user_id, portfolio_repo, position_repo)

        scenario = scenarios[scenario_name]
        stress_result = risk_service.perform_stress_test(portfolio, scenario)

        result = {
            "user_id": user_id,
            "scenario": scenario.name,
            "description": scenario.description,
            "probability": float(scenario.probability),
            "portfolio_impact": {
                "amount": float(stress_result.amount),
                "currency": stress_result.currency,
                "percentage": f"{(stress_result.amount / portfolio.total_value.amount * 100):.2f}%" if portfolio.total_value.amount > 0 else "0%"
            },
            "performed_at": datetime.now().isoformat()
        }

        logger.info(f"Stress test performed for user {user_id}, scenario: {scenario_name}")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error performing stress test: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to perform stress test"
        )


@router.get(
    "/correlation-matrix/{user_id}",
    summary="Get portfolio correlation matrix",
    responses={
        200: {"description": "Correlation matrix retrieved successfully"},
        401: {"description": "Unauthorized"},
        404: {"description": "Portfolio not found"},
    }
)
async def get_correlation_matrix(
    user_id: str,
    current_user_id: str = Depends(get_current_user),
    portfolio_repo: PortfolioRepository = Depends(get_portfolio_repository),
    position_repo: PositionRepository = Depends(get_position_repository),
    risk_service: DefaultAdvancedRiskManagementService = Depends(get_advanced_risk_management_service),
) -> Dict[str, Any]:
    """Get the correlation matrix for portfolio holdings."""
    if current_user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view this portfolio"
        )

    try:
        portfolio = _load_portfolio_with_positions(user_id, portfolio_repo, position_repo)
        correlation_matrix = risk_service.calculate_correlation_matrix(portfolio)

        result = {
            "user_id": user_id,
            "correlation_matrix": correlation_matrix,
            "calculated_at": datetime.now().isoformat()
        }

        logger.info(f"Correlation matrix retrieved for user {user_id}")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating correlation matrix: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to calculate correlation matrix"
        )
