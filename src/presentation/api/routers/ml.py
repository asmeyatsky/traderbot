"""
Machine Learning API Router

This router handles all AI/ML related endpoints including model predictions,
sentiment analysis, trading signals, and model performance metrics.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime, timedelta

from src.infrastructure.security import get_current_user
from src.domain.value_objects import Symbol
from src.infrastructure.di_container import container
from src.infrastructure.data_processing.ml_model_service import (
    MLModelService, TradingSignal, ModelPerformance, 
    EnsembleModelService, AdvancedRiskAnalyticsService, 
    PortfolioOptimizationService
)
from src.infrastructure.data_processing.news_aggregation_service import NewsImpactAnalyzer
from src.infrastructure.data_processing.backtesting_engine import (
    BacktestingEngine, BacktestConfiguration, StrategyComparator,
    SMACrossoverStrategy, MLStrategy
)
from src.infrastructure.broker_integration import BrokerIntegrationService
from src.infrastructure.repositories import PortfolioRepository, PositionRepository, UserRepository
from src.presentation.api.dependencies import (
    get_portfolio_repository,
    get_position_repository,
    get_user_repository,
)


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ml", tags=["ml"])


@router.get(
    "/predict/{symbol}",
    summary="Get price prediction for a symbol",
    responses={
        200: {"description": "Price prediction with confidence"},
        400: {"description": "Invalid symbol"},
        401: {"description": "Unauthorized"},
        500: {"description": "Internal server error"},
    }
)
async def get_price_prediction(
    symbol: str,
    lookback_days: int = Query(30, ge=1, le=365, description="Number of days to look back for prediction"),
    user_id: str = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get AI-powered price prediction for a given symbol.

    Args:
        symbol: Stock symbol to predict
        lookback_days: Number of historical days to consider for prediction
        user_id: Current authenticated user ID

    Returns:
        Dictionary containing prediction details
    """
    try:
        # Validate symbol format
        try:
            symbol_obj = Symbol(symbol.upper())
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid symbol format: {e}"
            )

        # Get ML service from container
        ml_service = container.services.ml_model_service()
        if not ml_service:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="ML service not available"
            )

        logger.info(f"Getting price prediction for {symbol} for user {user_id}")

        # Get prediction from ensemble model (which combines multiple models)
        if hasattr(ml_service, 'predict_price_direction'):
            prediction = ml_service.predict_price_direction(symbol_obj, lookback_days)
        else:
            # Fallback to a specific model if ensemble not available
            from src.infrastructure.data_processing.ml_model_service import LSTMPricePredictionService
            lstm_service = LSTMPricePredictionService()
            prediction = lstm_service.predict_price_direction(symbol_obj, lookback_days)

        # Map signal to direction for frontend Prediction type
        signal_upper = prediction.signal.upper()
        if signal_upper in ("BUY", "STRONG_BUY"):
            predicted_direction = "UP"
        elif signal_upper in ("SELL", "STRONG_SELL"):
            predicted_direction = "DOWN"
        else:
            predicted_direction = "NEUTRAL"

        # Fetch current price from real market data service
        try:
            market_service = container.adapters.market_data_service()
            price = market_service.get_current_price(symbol_obj)
            current_price = float(price.amount) if price else 0.0
        except Exception as price_err:
            logger.warning(f"Failed to fetch current price for {symbol}: {price_err}")
            current_price = 0.0

        # Estimate predicted price from score and current price
        predicted_price = current_price * (1 + prediction.score * 0.05) if current_price > 0 else 0.0

        return {
            "symbol": symbol,
            "predicted_direction": predicted_direction,
            "confidence": prediction.confidence,
            "predicted_price": round(predicted_price, 2),
            "current_price": current_price,
            "score": prediction.score,
            "explanation": prediction.explanation,
            "timestamp": datetime.utcnow(),
            "lookback_days": lookback_days
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting price prediction for {symbol}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate price prediction"
        )


@router.get(
    "/signal/{symbol}/{user_id}",
    summary="Get trading signal for a symbol and user",
    responses={
        200: {"description": "Trading signal with user context"},
        400: {"description": "Invalid parameters"},
        401: {"description": "Unauthorized"},
        500: {"description": "Internal server error"},
    }
)
async def get_trading_signal(
    symbol: str,
    user_id: str,
    risk_level: str = Query("MODERATE", description="User risk tolerance level"),
    investment_goal: str = Query("BALANCED_GROWTH", description="User investment goal"),
    current_user_id: str = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get personalized trading signal based on symbol and user profile.

    Args:
        symbol: Stock symbol to analyze
        user_id: User ID for personalization
        risk_level: User's risk tolerance (CONSERVATIVE, MODERATE, AGGRESSIVE)
        investment_goal: User's investment goal

    Returns:
        Dictionary containing trading signal with user context
    """
    # Authorization check: users can only access their own signals
    if current_user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access signals for another user"
        )

    try:
        # Validate symbol
        try:
            symbol_obj = Symbol(symbol.upper())
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid symbol format: {e}"
            )

        # Validate risk level and investment goal
        valid_risk_levels = ["CONSERVATIVE", "MODERATE", "AGGRESSIVE"]
        valid_goals = ["CAPITAL_PRESERVATION", "BALANCED_GROWTH", "MAXIMUM_RETURNS"]

        if risk_level not in valid_risk_levels:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid risk level. Valid values: {valid_risk_levels}"
            )

        if investment_goal not in valid_goals:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid investment goal. Valid values: {valid_goals}"
            )

        # Get services
        ml_service = container.services.ml_model_service()
        news_analyzer = container.services.news_impact_analyzer_service()
        
        if not ml_service or not news_analyzer:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Required services not available"
            )

        logger.info(f"Getting trading signal for {symbol} for user {user_id}")

        # Get AI prediction
        prediction = ml_service.predict_price_direction(symbol_obj)

        # Get news impact
        news_impact = news_analyzer.calculate_news_impact_score(symbol_obj)

        # Adjust signal based on user profile and news impact
        adjusted_signal = prediction.signal
        adjusted_confidence = prediction.confidence
        
        # Modify signal based on user risk profile
        if risk_level == "CONSERVATIVE" and prediction.signal == "BUY":
            # Conservative users might get a HOLD instead of BUY
            if prediction.confidence < 0.7 or news_impact < 0.1:
                adjusted_signal = "HOLD"
                adjusted_confidence *= 0.8
        elif risk_level == "AGGRESSIVE" and prediction.signal == "HOLD":
            # Aggressive users might take more risks
            if news_impact > 0.3:
                adjusted_signal = "BUY"
                adjusted_confidence *= 1.1

        # Factor in news impact
        if abs(news_impact) > 0.5:  # Strong news impact
            adjusted_confidence = min(1.0, adjusted_confidence + 0.1)

        # Format news_impact as human-readable string for frontend
        if news_impact > 0.3:
            news_impact_label = "Strongly Positive"
        elif news_impact > 0.1:
            news_impact_label = "Positive"
        elif news_impact > -0.1:
            news_impact_label = "Neutral"
        elif news_impact > -0.3:
            news_impact_label = "Negative"
        else:
            news_impact_label = "Strongly Negative"

        return {
            "symbol": symbol,
            "signal": adjusted_signal,
            "confidence": adjusted_confidence,
            "news_impact": news_impact_label,
            "risk_adjusted": adjusted_signal != prediction.signal,
            "original_signal": prediction.signal,
            "news_impact_score": news_impact,
            "user_risk_profile": risk_level,
            "investment_goal": investment_goal,
            "explanation": f"Signal adjusted for {risk_level} risk profile and news impact score of {news_impact:.2f}",
            "timestamp": datetime.utcnow()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting trading signal for {symbol} and user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate trading signal"
        )


@router.get(
    "/model-performance/{model_type}",
    summary="Get model performance metrics",
    responses={
        200: {"description": "Model performance metrics"},
        400: {"description": "Invalid model type"},
        401: {"description": "Unauthorized"},
        500: {"description": "Internal server error"},
    }
)
async def get_model_performance(
    model_type: str,
    user_id: str = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get performance metrics for a specific AI model.

    Args:
        model_type: Type of model to get performance for (e.g., 'lstm', 'ensemble', 'rl')
        user_id: Current authenticated user ID

    Returns:
        Dictionary containing model performance metrics
    """
    try:
        valid_model_types = ["lstm", "ensemble", "rl", "sentiment", "all"]
        if model_type.lower() not in valid_model_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid model type. Valid values: {valid_model_types}"
            )

        ml_service = container.services.ml_model_service()
        if not ml_service:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="ML service not available"
            )

        logger.info(f"Getting performance for {model_type} model for user {user_id}")

        if model_type.lower() == "all":
            # Return performance for all models
            models = ["lstm", "sentiment", "ensemble"]
            results = {}
            for m in models:
                try:
                    performance = ml_service.get_model_performance(m)
                    results[m] = {
                        "accuracy": performance.accuracy,
                        "precision": performance.precision,
                        "recall": performance.recall,
                        "sharpe_ratio": performance.sharpe_ratio,
                        "max_drawdown": performance.max_drawdown,
                        "annual_return": performance.annual_return
                    }
                except Exception:
                    results[m] = {"error": "Performance data not available"}
            
            return {
                "model_performance": results,
                "timestamp": datetime.utcnow()
            }
        else:
            # Get specific model performance
            performance = ml_service.get_model_performance(model_type)
            
            return {
                "model_type": model_type,
                "accuracy": performance.accuracy,
                "precision": performance.precision,
                "recall": performance.recall,
                "sharpe_ratio": performance.sharpe_ratio,
                "max_drawdown": performance.max_drawdown,
                "annual_return": performance.annual_return,
                "timestamp": datetime.utcnow()
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model performance for {model_type}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get model performance"
        )


@router.post(
    "/optimize-portfolio/{user_id}",
    summary="Optimize portfolio allocation for user",
    responses={
        200: {"description": "Optimized portfolio allocation"},
        400: {"description": "Invalid parameters"},
        401: {"description": "Unauthorized"},
        500: {"description": "Internal server error"},
    }
)
async def optimize_portfolio(
    user_id: str,
    risk_tolerance: Optional[str] = Query(None, description="Risk tolerance (CONSERVATIVE, MODERATE, AGGRESSIVE)"),
    investment_goal: Optional[str] = Query(None, description="Investment goal"),
    symbols: List[str] = Query([], description="List of symbols to consider for allocation"),
    current_user_id: str = Depends(get_current_user),
    user_repo: UserRepository = Depends(get_user_repository),
    portfolio_repo: PortfolioRepository = Depends(get_portfolio_repository),
    position_repo: PositionRepository = Depends(get_position_repository),
) -> Dict[str, Any]:
    """
    Optimize user's portfolio allocation based on risk profile and goals.

    Args:
        user_id: User ID to optimize portfolio for
        risk_tolerance: User's risk tolerance level
        investment_goal: User's investment goal
        symbols: List of symbols to consider for allocation

    Returns:
        Dictionary containing optimized allocation recommendations
    """
    # Authorization check: users can only optimize their own portfolio
    if current_user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to optimize another user's portfolio"
        )

    try:
        # Validate inputs
        if risk_tolerance and risk_tolerance not in ["CONSERVATIVE", "MODERATE", "AGGRESSIVE"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid risk tolerance. Valid values: CONSERVATIVE, MODERATE, AGGRESSIVE"
            )

        if investment_goal and investment_goal not in ["CAPITAL_PRESERVATION", "BALANCED_GROWTH", "MAXIMUM_RETURNS"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid investment goal. Valid values: CAPITAL_PRESERVATION, BALANCED_GROWTH, MAXIMUM_RETURNS"
            )

        # Convert symbols to Symbol objects
        symbol_objects = []
        for sym in symbols:
            try:
                symbol_objects.append(Symbol(sym.upper()))
            except ValueError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid symbol format: {sym}"
                )

        # Get services
        portfolio_optimizer = container.services.portfolio_optimization_service()
        if not portfolio_optimizer:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Portfolio optimization service not available"
            )

        # Fetch real user and portfolio from repositories
        user = user_repo.get_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        portfolio = portfolio_repo.get_by_user_id(user_id)
        if not portfolio:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Portfolio not found"
            )

        positions = position_repo.get_by_user_id(user_id)
        from src.domain.entities.trading import Portfolio
        full_portfolio = Portfolio(
            id=portfolio.id,
            user_id=portfolio.user_id,
            positions=positions,
            cash_balance=portfolio.cash_balance,
            created_at=portfolio.created_at,
            updated_at=portfolio.updated_at,
        )

        logger.info(f"Optimizing portfolio for user {user_id} with {len(symbol_objects)} symbols")

        # Get optimization
        allocation = portfolio_optimizer.optimize_portfolio(
            user, full_portfolio, symbol_objects
        )

        return {
            "user_id": user_id,
            "risk_tolerance": risk_tolerance or "MODERATE",
            "investment_goal": investment_goal or "BALANCED_GROWTH",
            "allocation_recommendation": allocation,
            "total_allocation": sum(v for k, v in allocation.items() if k != 'cash'),
            "timestamp": datetime.utcnow()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error optimizing portfolio for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to optimize portfolio"
        )


@router.post(
    "/backtest",
    summary="Run backtest on a trading strategy",
    responses={
        200: {"description": "Backtest results"},
        400: {"description": "Invalid parameters"},
        401: {"description": "Unauthorized"},
        500: {"description": "Internal server error"},
    }
)
async def run_backtest(
    start_date: str = Query(..., description="Start date in YYYY-MM-DD format"),
    end_date: str = Query(..., description="End date in YYYY-MM-DD format"),
    initial_capital: float = Query(..., gt=0, description="Initial capital for backtest"),
    symbols: List[str] = Query(..., min_length=1, description="Symbols to backtest"),
    strategy_type: str = Query("sma", description="Type of strategy to backtest (sma, ml)"),
    user_id: str = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Run backtest on historical data with specified strategy.

    Args:
        start_date: Start date for backtest (YYYY-MM-DD format)
        end_date: End date for backtest (YYYY-MM-DD format)
        initial_capital: Initial capital to start with
        symbols: List of symbols to backtest
        strategy_type: Type of strategy to use (sma, ml)
        user_id: Current authenticated user ID

    Returns:
        Dictionary containing backtest results
    """
    try:
        from datetime import datetime
        import re

        # Validate dates
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid date format. Use YYYY-MM-DD."
            )

        if start_dt >= end_dt:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Start date must be before end date"
            )

        # Validate and convert symbols
        symbol_objects = []
        for sym in symbols:
            try:
                symbol_objects.append(Symbol(sym.upper()))
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid symbol format: {sym}"
                )

        # Validate strategy type
        if strategy_type not in ["sma", "ml"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid strategy type. Valid values: sma, ml"
            )

        # Get services
        data_provider = container.adapters.data_provider_service()
        ml_service = container.services.ml_model_service()
        
        if not data_provider:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Data provider service not available"
            )

        # Create strategy based on type
        if strategy_type == "sma":
            strategy = SMACrossoverStrategy()
        else:  # ml
            if not ml_service:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="ML service required for ML strategy"
                )
            strategy = MLStrategy(ml_service)

        # Create backtest configuration
        config = BacktestConfiguration(
            start_date=start_dt,
            end_date=end_dt,
            initial_capital=initial_capital,
            symbols=symbol_objects
        )

        logger.info(f"Running backtest for user {user_id} with {strategy_type} strategy")

        # Run backtest
        engine = BacktestingEngine(data_provider, strategy, config)
        result = engine.run_backtest()

        # Convert result to dictionary
        result_dict = {
            "strategy": strategy.get_strategy_name(),
            "period": {
                "start": start_date,
                "end": end_date
            },
            "initial_capital": initial_capital,
            "final_portfolio_value": result.final_portfolio_value,
            "total_return_pct": result.total_return * 100,
            "annualized_return_pct": result.annualized_return * 100,
            "volatility": result.volatility,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown_pct": result.max_drawdown * 100,
            "win_rate_pct": result.win_rate * 100,
            "total_trades": result.total_trades,
            "winning_trades": result.winning_trades,
            "losing_trades": result.losing_trades,
            "profit_factor": result.profit_factor,
            "results_timestamp": datetime.utcnow(),
            "trades": [
                {
                    "date": str(t.get('date', '')),
                    "symbol": t.get('symbol', ''),
                    "action": t.get('action', ''),
                    "quantity": t.get('quantity', 0),
                    "price": t.get('price', 0),
                    "value": t.get('value', 0),
                    "reason": t.get('reason', '')
                }
                for t in result.trades[:20]  # Limit to first 20 trades in response
            ] + ([{"note": f"... and {len(result.trades) - 20} more trades"}] if len(result.trades) > 20 else [])
        }

        return result_dict

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to run backtest"
        )


@router.get(
    "/risk-analysis/{user_id}",
    summary="Perform advanced risk analysis for user",
    responses={
        200: {"description": "Risk analysis results"},
        401: {"description": "Unauthorized"},
        500: {"description": "Internal server error"},
    }
)
async def get_risk_analysis(
    user_id: str,
    symbols: List[str] = Query([], description="Portfolio symbols for VaR calculation"),
    current_user_id: str = Depends(get_current_user),
    portfolio_repo: PortfolioRepository = Depends(get_portfolio_repository),
    position_repo: PositionRepository = Depends(get_position_repository),
) -> Dict[str, Any]:
    """
    Perform advanced risk analysis including VaR, Expected Shortfall, and correlation analysis.

    Args:
        user_id: User ID to analyze risk for
        symbols: List of portfolio symbols

    Returns:
        Dictionary containing risk analysis results
    """
    # Authorization check: users can only view their own risk analysis
    if current_user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view another user's risk analysis"
        )

    try:
        # Convert symbols to Symbol objects
        symbol_objects = []
        for sym in symbols:
            try:
                symbol_objects.append(Symbol(sym.upper()))
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid symbol format: {sym}"
                )

        # Get risk analytics service
        risk_service = container.services.risk_analytics_service()
        if not risk_service:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Risk analytics service not available"
            )

        # Fetch real portfolio and positions from repositories
        from src.domain.entities.trading import Portfolio
        from decimal import Decimal

        portfolio = portfolio_repo.get_by_user_id(user_id)
        if not portfolio:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Portfolio not found"
            )

        positions = position_repo.get_by_user_id(user_id)
        full_portfolio = Portfolio(
            id=portfolio.id,
            user_id=portfolio.user_id,
            positions=positions,
            cash_balance=portfolio.cash_balance,
            created_at=portfolio.created_at,
            updated_at=portfolio.updated_at,
        )

        logger.info(f"Calculating risk metrics for user {user_id} with {len(symbol_objects)} symbols")

        # Calculate VaR (Value at Risk)
        var_95 = risk_service.calculate_var(full_portfolio, confidence_level=0.95)
        var_99 = risk_service.calculate_var(full_portfolio, confidence_level=0.99)

        # Calculate Expected Shortfall
        es_95 = risk_service.calculate_expected_shortfall(full_portfolio, confidence_level=0.95)

        # Calculate correlation matrix
        correlations = risk_service.calculate_correlation_matrix(positions)

        # Define example scenarios for stress testing
        scenarios = [
            {"name": "Market Crash", "market_impact": -0.20},
            {"name": "Tech Sector Decline", "market_impact": -0.15},
            {"name": "Interest Rate Shock", "market_impact": -0.10}
        ]

        # Run stress tests
        stress_results = risk_service.stress_test_portfolio(full_portfolio, scenarios)

        return {
            "user_id": user_id,
            "risk_metrics": {
                "value_at_risk_95": float(var_95.amount),
                "value_at_risk_99": float(var_99.amount),
                "expected_shortfall_95": float(es_95.amount),
                "portfolio_volatility": 0.20,  # Mock value for now
            },
            "correlation_matrix": correlations,
            "stress_test_results": stress_results,
            "analysis_timestamp": datetime.utcnow()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error performing risk analysis for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to perform risk analysis"
        )


@router.get(
    "/retrain-model/{symbol}",
    summary="Retrain ML model for a specific symbol",
    responses={
        200: {"description": "Model retraining initiated"},
        400: {"description": "Invalid symbol"},
        401: {"description": "Unauthorized"},
        500: {"description": "Internal server error"},
    }
)
async def retrain_model(
    symbol: str,
    user_id: str = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Initiate retraining of ML models for a specific symbol with new data.

    Args:
        symbol: Symbol to retrain model for
        user_id: Current authenticated user ID

    Returns:
        Dictionary confirming retraining initiation
    """
    try:
        # Validate symbol
        try:
            symbol_obj = Symbol(symbol.upper())
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid symbol format: {e}"
            )

        # Get ML service
        ml_service = container.services.ml_model_service()
        if not ml_service:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="ML service not available"
            )

        logger.info(f"Initiating model retraining for {symbol} by user {user_id}")

        # Attempt to retrain model
        success = ml_service.retrain_model(symbol_obj)

        if success:
            return {
                "symbol": symbol,
                "status": "retraining_initiated",
                "message": f"Model retraining initiated for {symbol}",
                "retraining_successful": True,
                "timestamp": datetime.utcnow()
            }
        else:
            return {
                "symbol": symbol,
                "status": "retraining_failed",
                "message": f"Model retraining failed for {symbol}",
                "retraining_successful": False,
                "timestamp": datetime.utcnow()
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error initiating model retraining for {symbol}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initiate model retraining"
        )