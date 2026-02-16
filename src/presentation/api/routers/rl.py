"""
Reinforcement Learning Trading Agent API Router

This router handles all RL trading agent endpoints following RESTful principles.
All endpoints require authentication.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import Dict, Any
import logging
from datetime import datetime
from decimal import Decimal

from src.infrastructure.security import get_current_user
from src.domain.services.rl_trading_agents import RLAlgorithm, TradingAction
from src.domain.entities.trading import Portfolio
from src.domain.value_objects import Symbol
from src.infrastructure.repositories import PortfolioRepository, PositionRepository, UserRepository
from src.presentation.api.dependencies import (
    get_portfolio_repository,
    get_position_repository,
    get_user_repository,
    get_rl_agent_service,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/rl", tags=["rl"])


@router.get(
    "/algorithms",
    summary="Get available RL algorithms",
    responses={
        200: {"description": "Available RL algorithms retrieved successfully"},
        401: {"description": "Unauthorized"},
    }
)
async def get_available_algorithms(
    current_user_id: str = Depends(get_current_user),
) -> Dict[str, Any]:
    """Get list of available reinforcement learning algorithms."""
    try:
        algorithms = [alg.value for alg in RLAlgorithm]

        result = {
            "available_algorithms": algorithms,
            "algorithm_count": len(algorithms),
            "retrieved_at": datetime.now().isoformat()
        }

        logger.info(f"Available RL algorithms retrieved for user {current_user_id}")
        return result

    except Exception as e:
        logger.error(f"Error retrieving RL algorithms: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve RL algorithms"
        )


@router.post(
    "/agents/train/{symbol}",
    summary="Train a RL trading agent for a symbol",
    responses={
        200: {"description": "Agent training completed successfully"},
        401: {"description": "Unauthorized"},
        404: {"description": "Symbol not found"},
    }
)
async def train_rl_agent(
    symbol: str,
    algorithm: str = Query("dqn", description="RL algorithm to use"),
    episodes: int = Query(100, description="Number of training episodes"),
    initial_balance: float = Query(10000.0, description="Initial balance for training"),
    current_user_id: str = Depends(get_current_user),
    rl_service=Depends(get_rl_agent_service),
) -> Dict[str, Any]:
    """Train a reinforcement learning trading agent for a symbol."""
    try:
        # Validate algorithm — accept both enum name (dqn) and value (deep_q_network)
        normalized = algorithm.lower().replace('-', '_')
        valid_alg = None
        for alg in RLAlgorithm:
            if alg.name.lower() == normalized or alg.value == normalized:
                valid_alg = alg
                break
        if valid_alg is None:
            available_algs = [alg.name.lower() for alg in RLAlgorithm]
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid algorithm: {algorithm}. Available: {available_algs}"
            )

        training_data = [{"symbol": symbol.upper(), "episodes": episodes}]
        success = rl_service.train(training_data)

        result = {
            "symbol": symbol,
            "algorithm": algorithm,
            "training_result": {
                "episodes_trained": episodes,
                "training_successful": success,
            },
            "trained_at": datetime.now().isoformat()
        }

        logger.info(f"RL agent trained for symbol {symbol} using {algorithm}")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error training RL agent: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to train RL agent"
        )


@router.post(
    "/agents/evaluate/{symbol}",
    summary="Evaluate a RL trading agent for a symbol",
    responses={
        200: {"description": "Agent evaluation completed successfully"},
        401: {"description": "Unauthorized"},
        404: {"description": "Symbol not found"},
    }
)
async def evaluate_rl_agent(
    symbol: str,
    algorithm: str = Query("dqn", description="RL algorithm to use"),
    episodes: int = Query(10, description="Number of evaluation episodes"),
    initial_balance: float = Query(10000.0, description="Initial balance for evaluation"),
    current_user_id: str = Depends(get_current_user),
    rl_service=Depends(get_rl_agent_service),
) -> Dict[str, Any]:
    """Evaluate a reinforcement learning trading agent for a symbol."""
    try:
        # Validate algorithm — accept both enum name (dqn) and value (deep_q_network)
        normalized = algorithm.lower().replace('-', '_')
        valid_alg = None
        for alg in RLAlgorithm:
            if alg.name.lower() == normalized or alg.value == normalized:
                valid_alg = alg
                break
        if valid_alg is None:
            available_algs = [alg.name.lower() for alg in RLAlgorithm]
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid algorithm: {algorithm}. Available: {available_algs}"
            )

        evaluation_data = [{"symbol": symbol.upper()}]
        performance = rl_service.evaluate(evaluation_data)

        result = {
            "symbol": symbol,
            "algorithm": algorithm,
            "evaluation_results": {
                "accuracy": performance.accuracy,
                "precision": performance.precision,
                "recall": performance.recall,
                "sharpe_ratio": performance.sharpe_ratio,
                "max_drawdown": performance.max_drawdown,
                "annual_return": performance.annual_return,
            },
            "evaluated_at": datetime.now().isoformat()
        }

        logger.info(f"RL agent evaluated for symbol {symbol} using {algorithm}")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error evaluating RL agent: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to evaluate RL agent"
        )


@router.get(
    "/agents/ensemble-performance",
    summary="Get ensemble performance of all RL agents",
    responses={
        200: {"description": "Ensemble performance retrieved successfully"},
        401: {"description": "Unauthorized"},
    }
)
async def get_ensemble_performance(
    current_user_id: str = Depends(get_current_user),
    rl_service=Depends(get_rl_agent_service),
) -> Dict[str, Any]:
    """Get performance metrics for the RL agent."""
    try:
        # Evaluate across known symbols
        symbols = ["AAPL", "MSFT", "GOOGL"]
        performance_results = {}
        for sym in symbols:
            try:
                perf = rl_service.evaluate([{"symbol": sym}])
                performance_results[sym] = {
                    "accuracy": perf.accuracy,
                    "sharpe_ratio": perf.sharpe_ratio,
                    "max_drawdown": perf.max_drawdown,
                }
            except Exception:
                performance_results[sym] = {"error": "No trained agent available"}

        result = {
            "ensemble_performance": performance_results,
            "agent_types": list(performance_results.keys()),
            "retrieved_at": datetime.now().isoformat()
        }

        logger.info(f"Ensemble performance retrieved for user {current_user_id}")
        return result

    except Exception as e:
        logger.error(f"Error retrieving ensemble performance: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve ensemble performance"
        )


@router.post(
    "/agents/get-action/{symbol}/{user_id}",
    summary="Get action from RL agent for a symbol",
    responses={
        200: {"description": "Action retrieved successfully"},
        401: {"description": "Unauthorized"},
        404: {"description": "Symbol or user not found"},
    }
)
async def get_rl_action(
    symbol: str,
    user_id: str,
    algorithm: str = Query("ensemble", description="Algorithm to use (ensemble or specific)"),
    market_regime: str = Query("default", description="Current market regime"),
    current_user_id: str = Depends(get_current_user),
    portfolio_repo: PortfolioRepository = Depends(get_portfolio_repository),
    position_repo: PositionRepository = Depends(get_position_repository),
    rl_service=Depends(get_rl_agent_service),
) -> Dict[str, Any]:
    """Get action recommendation from RL agent for a symbol."""
    if current_user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to get actions for this user"
        )

    try:
        symbol_obj = Symbol(symbol.upper())

        # Build state from real portfolio data
        portfolio = portfolio_repo.get_by_user_id(user_id)
        positions = position_repo.get_by_user_id(user_id)

        portfolio_value = Decimal('0')
        cash_balance = Decimal('0')
        cash_percentage = 0.5
        if portfolio:
            cash_balance = portfolio.cash_balance.amount
            full_portfolio = Portfolio(
                id=portfolio.id,
                user_id=portfolio.user_id,
                positions=positions,
                cash_balance=portfolio.cash_balance,
                created_at=portfolio.created_at,
                updated_at=portfolio.updated_at,
            )
            portfolio_value = full_portfolio.total_value.amount
            if portfolio_value > 0:
                cash_percentage = float(cash_balance / portfolio_value)

        state = {
            "portfolio_value": float(portfolio_value),
            "cash_balance": float(cash_balance),
            "cash_percentage": cash_percentage,
            "market_regime": market_regime,
        }

        action, position_size = rl_service.get_action(state, symbol_obj)

        result = {
            "symbol": symbol,
            "user_id": user_id,
            "action": action,
            "position_size": float(position_size),
            "market_regime": market_regime,
            "algorithm_used": algorithm,
            "recommended_at": datetime.now().isoformat()
        }

        logger.info(f"RL action recommended for symbol {symbol} and user {user_id}")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting RL action: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get RL action recommendation"
        )
