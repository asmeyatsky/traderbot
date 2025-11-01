"""
Advanced ML/AI Model API Router

This router handles all ML/AI model endpoints following RESTful principles.
All endpoints require authentication.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import List, Optional, Dict, Any
import logging

from src.infrastructure.security import get_current_user
from src.domain.services.ml_model_service import (
    DefaultMLModelService, MLModelType, TradingSignal, PredictionResult
)
from src.domain.value_objects import Symbol
from src.infrastructure.di_container import container

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ml", tags=["ml"])


@router.get(
    "/predict/{symbol}",
    summary="Get price prediction for a symbol",
    responses={
        200: {"description": "Price prediction retrieved successfully"},
        401: {"description": "Unauthorized"},
        404: {"description": "Symbol not found"},
    }
)
async def get_price_prediction(
    symbol: str,
    days_ahead: int = Query(1, description="Number of days ahead to predict"),
    model_type: str = Query("ensemble", description="Type of model to use"),
    current_user_id: str = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get price prediction for a symbol using ML/AI models.

    Returns:
        Price prediction with confidence and technical indicators
    """
    try:
        # Get the ML model service from DI container
        ml_service = container.ml_model_service()
        
        # Get price prediction
        prediction = ml_service.predict_price(Symbol(symbol), days_ahead)
        
        result = {
            "symbol": str(prediction.symbol),
            "predicted_price": {
                "amount": float(prediction.predicted_price.amount),
                "currency": prediction.predicted_price.currency
            },
            "confidence": float(prediction.confidence),
            "prediction_horizon": prediction.prediction_horizon,
            "model_used": prediction.model_used,
            "features_used": prediction.features_used,
            "prediction_timestamp": prediction.prediction_timestamp.isoformat(),
            "technical_indicators": prediction.technical_indicators,
            "market_regime": prediction.market_regime,
            "calculated_at": datetime.now().isoformat()
        }
        
        logger.info(f"Price prediction retrieved for symbol {symbol}")
        return result
        
    except Exception as e:
        logger.error(f"Error retrieving price prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve price prediction"
        )


@router.get(
    "/regime/{symbol}",
    summary="Get market regime detection for a symbol",
    responses={
        200: {"description": "Market regime retrieved successfully"},
        401: {"description": "Unauthorized"},
        404: {"description": "Symbol not found"},
    }
)
async def get_market_regime(
    symbol: str,
    current_user_id: str = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get current market regime detection for a symbol.

    Returns:
        Market regime with characteristics and confidence
    """
    try:
        ml_service = container.ml_model_service()
        
        # Get market regime
        regime = ml_service.detect_market_regime(Symbol(symbol))
        
        result = {
            "symbol": symbol,
            "market_regime": {
                "regime_type": regime.regime_type,
                "confidence": float(regime.confidence),
                "start_date": regime.start_date.isoformat(),
                "end_date": regime.end_date.isoformat() if regime.end_date else None,
                "characteristics": regime.characteristics
            },
            "calculated_at": datetime.now().isoformat()
        }
        
        logger.info(f"Market regime retrieved for symbol {symbol}")
        return result
        
    except Exception as e:
        logger.error(f"Error retrieving market regime: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve market regime"
        )


@router.get(
    "/signal/{symbol}/{user_id}",
    summary="Get trading signal for a symbol",
    responses={
        200: {"description": "Trading signal retrieved successfully"},
        401: {"description": "Unauthorized"},
        404: {"description": "Symbol or user not found"},
    }
)
async def get_trading_signal(
    symbol: str,
    user_id: str,
    current_user_id: str = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get trading signal for a symbol based on ML/AI analysis.

    Args:
        symbol: Stock symbol to analyze
        user_id: User ID to personalize the signal
        current_user_id: Authenticated user ID (for authorization)

    Returns:
        Trading signal with confidence level
    """
    if current_user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to get signals for this user"
        )
    
    try:
        ml_service = container.ml_model_service()
        
        # In a real implementation, we would fetch the user from the repository
        # For mock, we'll create a user with default values
        from src.domain.entities.user import User, RiskTolerance, InvestmentGoal
        from datetime import datetime
        mock_user = User(
            id=user_id,
            email="user@example.com",
            first_name="Demo",
            last_name="User",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            risk_tolerance=RiskTolerance.MODERATE,
            investment_goal=InvestmentGoal.BALANCED_GROWTH
        )
        
        # Generate trading signal
        signal, confidence = ml_service.generate_trading_signal(Symbol(symbol), mock_user)
        
        result = {
            "symbol": symbol,
            "user_id": user_id,
            "trading_signal": signal.value,
            "confidence": float(confidence),
            "generated_at": datetime.now().isoformat()
        }
        
        logger.info(f"Trading signal retrieved for symbol {symbol} and user {user_id}")
        return result
        
    except Exception as e:
        logger.error(f"Error retrieving trading signal: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve trading signal"
        )


@router.post(
    "/optimize-portfolio/{user_id}",
    summary="Optimize user's portfolio allocation",
    responses={
        200: {"description": "Portfolio optimization completed successfully"},
        401: {"description": "Unauthorized"},
        404: {"description": "User not found"},
    }
)
async def optimize_portfolio(
    user_id: str,
    include_rebalancing: bool = Query(True, description="Include rebalancing suggestions"),
    optimization_method: str = Query("mean_variance", description="Optimization method to use"),
    current_user_id: str = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Optimize portfolio allocation for a user using ML/AI models.

    Args:
        user_id: User ID whose portfolio to optimize
        include_rebalancing: Whether to include rebalancing suggestions
        optimization_method: Method to use for optimization
        current_user_id: Authenticated user ID (for authorization)

    Returns:
        Optimized portfolio allocation with risk metrics
    """
    if current_user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to optimize portfolio for this user"
        )
    
    try:
        ml_service = container.ml_model_service()
        
        # In a real implementation, we would fetch the user and portfolio from repositories
        # For mock, we'll create a mock user and portfolio
        from src.domain.entities.user import User, RiskTolerance, InvestmentGoal
        from src.domain.entities.trading import Portfolio, Position, PositionType
        from src.domain.value_objects import Money
        from datetime import datetime
        
        mock_user = User(
            id=user_id,
            email="user@example.com",
            first_name="Demo",
            last_name="User",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            risk_tolerance=RiskTolerance.MODERATE,
            investment_goal=InvestmentGoal.BALANCED_GROWTH
        )
        
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
                symbol=Symbol("MSFT"),
                position_type=PositionType.LONG,
                quantity=75,
                average_buy_price=Money(Decimal('300.00'), 'USD'),
                current_price=Money(Decimal('350.00'), 'USD'),
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
        ]
        
        mock_portfolio = Portfolio(
            id="portfolio_1",
            user_id=user_id,
            positions=mock_positions,
            cash_balance=Money(Decimal('10000.00'), 'USD')
        )
        
        # Optimize portfolio
        optimization_result = ml_service.optimize_portfolio(mock_portfolio, mock_user)
        
        result = {
            "user_id": user_id,
            "optimization_method": optimization_method,
            "allocation": optimization_result["allocation"],
            "risk_metrics": optimization_result["risk_metrics"],
            "optimization_timestamp": optimization_result["calculation_timestamp"],
            "model_used": optimization_result["optimization_method"]
        }
        
        if include_rebalancing:
            result["rebalancing_suggestions"] = optimization_result["rebalancing_suggestions"]
        
        logger.info(f"Portfolio optimization completed for user {user_id}")
        return result
        
    except Exception as e:
        logger.error(f"Error optimizing portfolio: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to optimize portfolio"
        )


@router.get(
    "/volatility-forecast/{symbol}",
    summary="Get volatility forecast for a symbol",
    responses={
        200: {"description": "Volatility forecast retrieved successfully"},
        401: {"description": "Unauthorized"},
        404: {"description": "Symbol not found"},
    }
)
async def get_volatility_forecast(
    symbol: str,
    days: int = Query(30, description="Number of days to forecast"),
    model_type: str = Query("garch", description="Type of volatility model"),
    current_user_id: str = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get volatility forecast for a symbol using ML/AI models.

    Returns:
        Volatility forecast with confidence intervals
    """
    try:
        ml_service = container.ml_model_service()
        
        # Get volatility forecast
        volatility = ml_service.forecast_volatility(Symbol(symbol), days)
        
        # Calculate confidence intervals (mock)
        base_vol = float(volatility)
        confidence_lower = base_vol * 0.8  # 80% of base volatility
        confidence_upper = base_vol * 1.2  # 120% of base volatility
        
        result = {
            "symbol": symbol,
            "volatility_forecast": base_vol,
            "volatility_percentage": base_vol * 100,  # Convert to percentage
            "forecast_days": days,
            "model_type": model_type,
            "confidence_interval": {
                "lower": confidence_lower,
                "upper": confidence_upper
            },
            "calculated_at": datetime.now().isoformat()
        }
        
        logger.info(f"Volatility forecast retrieved for symbol {symbol}")
        return result
        
    except Exception as e:
        logger.error(f"Error retrieving volatility forecast: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve volatility forecast"
        )


@router.get(
    "/model-performance/{model_type}",
    summary="Get model performance metrics",
    responses={
        200: {"description": "Model performance retrieved successfully"},
        401: {"description": "Unauthorized"},
        404: {"description": "Model type not found"},
    }
)
async def get_model_performance(
    model_type: str,
    current_user_id: str = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get performance metrics for a specific ML model type.

    Returns:
        Performance metrics and model information
    """
    try:
        ml_service = container.ml_model_service()
        
        # Validate model type
        try:
            model_enum = MLModelType(model_type.lower().replace('-', '_'))
        except ValueError:
            available_models = [m.value for m in MLModelType]
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid model type: {model_type}. Available: {available_models}"
            )
        
        # Get model performance
        performance = ml_service.get_model_performance(model_enum)
        
        result = {
            "model_type": model_type,
            "performance_metrics": {
                "accuracy": float(performance.accuracy),
                "precision": float(performance.precision),
                "recall": float(performance.recall),
                "f1_score": float(performance.f1_score),
                "sharpe_ratio": float(performance.sharpe_ratio),
                "max_drawdown": float(performance.max_drawdown),
                "backtest_return": float(performance.backtest_return)
            },
            "model_version": performance.model_version,
            "training_date": performance.training_date.isoformat(),
            "features_importance": performance.features_importance,
            "retrieved_at": datetime.now().isoformat()
        }
        
        logger.info(f"Model performance retrieved for {model_type}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving model performance: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve model performance"
        )

from datetime import datetime
from decimal import Decimal
from src.domain.entities.trading import OrderType, PositionType, OrderStatus
from src.domain.value_objects import Money