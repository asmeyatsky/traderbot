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
) -> Portfolio | None:
    """Fetch portfolio and its positions from repositories. Returns None if not found."""
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


_EMPTY_RISK_RESPONSE = {
    "var_95": 0.0,
    "var_99": 0.0,
    "expected_shortfall": 0.0,
    "max_drawdown": 0.0,
    "volatility": 0.0,
    "beta": 0.0,
    "sharpe_ratio": 0.0,
    "sortino_ratio": 0.0,
    "correlation_matrix": {},
    "stress_results": [],
}


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
        if not portfolio or not portfolio.positions:
            return _EMPTY_RISK_RESPONSE

        risk_metrics = risk_service.calculate_portfolio_metrics(
            portfolio=portfolio,
            lookback_days=lookback_days
        )

        # Normalize response for frontend RiskMetrics type.
        # Frontend multiplies these values by 100 and formats as percentage,
        # so all values must be fractions (e.g., 0.05 for 5%).
        portfolio_value = float(portfolio.total_value.amount) if portfolio.total_value.amount > 0 else 1.0

        # VaR and ES are absolute dollar amounts — convert to fraction of portfolio value
        var_95_abs = float(risk_metrics.value_at_risk.amount) if risk_metrics.value_at_risk else 0.0
        var_95_frac = var_95_abs / portfolio_value
        var_99_frac = var_95_frac * 1.3  # Approximate VaR 99% from VaR 95%
        es_abs = float(risk_metrics.expected_shortfall.amount) if risk_metrics.expected_shortfall else 0.0
        es_frac = es_abs / portfolio_value

        # max_drawdown and volatility are already in percent (e.g. 12.7 for 12.7%)
        # — convert to fraction so frontend can multiply by 100
        max_drawdown_frac = float(risk_metrics.max_drawdown) / 100.0 if risk_metrics.max_drawdown else 0.0
        volatility_frac = float(risk_metrics.volatility) / 100.0 if risk_metrics.volatility else 0.0

        # Stress results: convert absolute dollar impact to fraction of portfolio
        stress_scenario_probs = {"2008 Financial Crisis": 0.005, "COVID-19 Crash": 0.03, "Dot-com Bubble": 0.02}
        stress_results = []
        for scenario, stress_result in (risk_metrics.stress_test_results or {}).items():
            impact_abs = float(stress_result.amount) if stress_result else 0.0
            impact_frac = impact_abs / portfolio_value
            # Stress impacts are negative losses — ensure sign is negative
            if impact_frac > 0:
                impact_frac = -impact_frac
            stress_results.append({
                "scenario": scenario,
                "portfolio_impact": impact_frac,
                "probability": stress_scenario_probs.get(scenario, 0.05),
                "description": f"Impact under {scenario} scenario",
            })

        result = {
            "var_95": var_95_frac,
            "var_99": var_99_frac,
            "expected_shortfall": es_frac,
            "max_drawdown": max_drawdown_frac,
            "volatility": volatility_frac,
            "beta": float(risk_metrics.beta) if risk_metrics.beta else 0.0,
            "sharpe_ratio": float(risk_metrics.sharpe_ratio) if risk_metrics.sharpe_ratio else 0.0,
            "sortino_ratio": float(risk_metrics.sortino_ratio) if risk_metrics.sortino_ratio else 0.0,
            "correlation_matrix": risk_metrics.correlation_matrix or {},
            "stress_results": stress_results,
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
        if not portfolio:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Portfolio not found")

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
        if not portfolio:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Portfolio not found")
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
