"""
Risk Analytics API Router

This router handles all risk-related endpoints following RESTful principles.
All endpoints require authentication.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import List, Optional, Dict, Any
import logging
from decimal import Decimal

from src.infrastructure.security import get_current_user
from src.domain.services.advanced_risk_management import (
    DefaultAdvancedRiskManagementService, RiskMetrics, StressTestScenario
)
from src.infrastructure.di_container import container

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/risk", tags=["risk"])


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
) -> Dict[str, Any]:
    """
    Get comprehensive risk metrics for a user's portfolio.

    Returns:
        RiskMetrics object containing VaR, ES, volatility, correlations, etc.
    """
    if current_user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view this portfolio"
        )
    
    try:
        # Get the risk management service from DI container
        risk_service = container.advanced_risk_management_service()
        
        # In a real implementation, we would fetch the portfolio from the repository
        # For now, we'll create a mock portfolio for demonstration
        from src.domain.entities.trading import Portfolio
        from src.domain.entities.trading import Position
        from src.domain.value_objects import Symbol, Money
        from src.domain.entities.trading import PositionType
        from datetime import datetime
        
        # Mock portfolio with some positions
        mock_positions = [
            Position(
                id="pos_1",
                user_id=user_id,
                symbol=Symbol("AAPL"),
                position_type=PositionType.LONG,
                quantity=100,
                average_buy_price=Money(Decimal('150.00'), 'USD'),
                current_price=Money(Decimal('175.00'), 'USD'),
                created_at=datetime.now(),
                updated_at=datetime.now()
            ),
            Position(
                id="pos_2",
                user_id=user_id,
                symbol=Symbol("GOOGL"),
                position_type=PositionType.LONG,
                quantity=50,
                average_buy_price=Money(Decimal('2500.00'), 'USD'),
                current_price=Money(Decimal('2750.00'), 'USD'),
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
        ]
        
        mock_portfolio = Portfolio(
            id="portfolio_1",
            user_id=user_id,
            positions=mock_positions
        )
        
        # Calculate risk metrics
        risk_metrics = risk_service.calculate_portfolio_metrics(
            portfolio=mock_portfolio,
            lookback_days=lookback_days
        )
        
        # Convert to JSON-serializable format
        result = {
            "user_id": user_id,
            "calculated_at": datetime.now().isoformat(),
            "lookback_days": lookback_days,
            "confidence_level": confidence_level,
            "value_at_risk": {
                "amount": float(risk_metrics.value_at_risk.amount) if risk_metrics.value_at_risk else None,
                "currency": risk_metrics.value_at_risk.currency if risk_metrics.value_at_risk else None
            } if risk_metrics.value_at_risk else None,
            "expected_shortfall": {
                "amount": float(risk_metrics.expected_shortfall.amount) if risk_metrics.expected_shortfall else None,
                "currency": risk_metrics.expected_shortfall.currency if risk_metrics.expected_shortfall else None
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
) -> Dict[str, Any]:
    """
    Perform a stress test on the user's portfolio under specific market conditions.

    Args:
        user_id: User ID to test
        scenario_name: Name of predefined scenario to test
        current_user_id: Authenticated user ID (for authorization)

    Returns:
        Stress test results showing potential portfolio impact
    """
    if current_user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to test this portfolio"
        )
    
    try:
        risk_service = container.advanced_risk_management_service()
        
        # Define common stress test scenarios
        scenarios = {
            "2008_crisis": StressTestScenario(
                name="2008 Financial Crisis",
                description="Simulates conditions during 2008 financial crisis",
                market_move={
                    "SPY": Decimal("-40.0"),  # S&P 500 down 40%
                    "AAPL": Decimal("-50.0"),  # Tech stocks particularly hard hit
                    "GOOGL": Decimal("-45.0"),  # Growth stocks under pressure
                    "TSLA": Decimal("-60.0"),  # Speculative stocks hardest hit
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
                    "GOOGL": Decimal("-20.0"),  # Growth stocks sensitive to rates
                    "TSLA": Decimal("-30.0"),    # Highly speculative, rate-sensitive
                },
                probability=Decimal("2.0")
            )
        }
        
        if scenario_name not in scenarios:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown scenario: {scenario_name}. Available: {list(scenarios.keys())}"
            )
        
        # Create mock portfolio (same as above)
        from src.domain.entities.trading import Portfolio
        from src.domain.entities.trading import Position
        from src.domain.value_objects import Symbol, Money
        from src.domain.entities.trading import PositionType
        from datetime import datetime
        
        mock_positions = [
            Position(
                id="pos_1",
                user_id=user_id,
                symbol=Symbol("AAPL"),
                position_type=PositionType.LONG,
                quantity=100,
                average_buy_price=Money(Decimal('150.00'), 'USD'),
                current_price=Money(Decimal('175.00'), 'USD'),
                created_at=datetime.now(),
                updated_at=datetime.now()
            ),
            Position(
                id="pos_2",
                user_id=user_id,
                symbol=Symbol("GOOGL"),
                position_type=PositionType.LONG,
                quantity=50,
                average_buy_price=Money(Decimal('2500.00'), 'USD'),
                current_price=Money(Decimal('2750.00'), 'USD'),
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
        ]
        
        mock_portfolio = Portfolio(
            id="portfolio_1",
            user_id=user_id,
            positions=mock_positions
        )
        
        # Perform stress test
        scenario = scenarios[scenario_name]
        stress_result = risk_service.perform_stress_test(mock_portfolio, scenario)
        
        result = {
            "user_id": user_id,
            "scenario": scenario.name,
            "description": scenario.description,
            "probability": float(scenario.probability),
            "portfolio_impact": {
                "amount": float(stress_result.amount),
                "currency": stress_result.currency,
                "percentage": f"{(stress_result.amount / mock_portfolio.total_value.amount * 100):.2f}%" if mock_portfolio.total_value.amount > 0 else "0%"
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
) -> Dict[str, Any]:
    """
    Get the correlation matrix for portfolio holdings.

    Args:
        user_id: User ID whose portfolio to analyze
        current_user_id: Authenticated user ID (for authorization)

    Returns:
        Correlation matrix showing how portfolio holdings move together
    """
    if current_user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view this portfolio"
        )
    
    try:
        risk_service = container.advanced_risk_management_service()
        
        # Create mock portfolio
        from src.domain.entities.trading import Portfolio
        from src.domain.entities.trading import Position
        from src.domain.value_objects import Symbol, Money
        from src.domain.entities.trading import PositionType
        from datetime import datetime
        
        mock_positions = [
            Position(
                id="pos_1",
                user_id=user_id,
                symbol=Symbol("AAPL"),
                position_type=PositionType.LONG,
                quantity=100,
                average_buy_price=Money(Decimal('150.00'), 'USD'),
                current_price=Money(Decimal('175.00'), 'USD'),
                created_at=datetime.now(),
                updated_at=datetime.now()
            ),
            Position(
                id="pos_2",
                user_id=user_id,
                symbol=Symbol("GOOGL"),
                position_type=PositionType.LONG,
                quantity=50,
                average_buy_price=Money(Decimal('2500.00'), 'USD'),
                current_price=Money(Decimal('2750.00'), 'USD'),
                created_at=datetime.now(),
                updated_at=datetime.now()
            ),
            Position(
                id="pos_3",
                user_id=user_id,
                symbol=Symbol("SPY"),
                position_type=PositionType.LONG,
                quantity=200,
                average_buy_price=Money(Decimal('400.00'), 'USD'),
                current_price=Money(Decimal('450.00'), 'USD'),
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
        ]
        
        mock_portfolio = Portfolio(
            id="portfolio_1",
            user_id=user_id,
            positions=mock_positions
        )
        
        correlation_matrix = risk_service.calculate_correlation_matrix(mock_portfolio)
        
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