"""
Reinforcement Learning Trading Agent API Router

This router handles all RL trading agent endpoints following RESTful principles.
All endpoints require authentication.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import List, Optional, Dict, Any
import logging

from src.infrastructure.security import get_current_user
from src.domain.services.rl_trading_agents import (
    MockRLAgent, MultiAgentRLEnsemble, RLAlgorithm, TradingAction, RLEnvironment, RLState
)
from src.domain.value_objects import Symbol
from src.infrastructure.di_container import container

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
    """
    Get list of available reinforcement learning algorithms.

    Returns:
        List of available RL algorithms
    """
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
) -> Dict[str, Any]:
    """
    Train a reinforcement learning trading agent for a symbol.

    Args:
        symbol: Stock symbol to train on
        algorithm: RL algorithm to use
        episodes: Number of training episodes
        initial_balance: Initial balance for simulation
        current_user_id: Authenticated user ID

    Returns:
        Training results with performance metrics
    """
    try:
        # Validate algorithm
        try:
            alg_enum = RLAlgorithm(algorithm.lower().replace('-', '_'))
        except ValueError:
            available_algs = [alg.value for alg in RLAlgorithm]
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid algorithm: {algorithm}. Available: {available_algs}"
            )
        
        # Create a mock environment for training
        env = RLEnvironment(
            symbol=Symbol(symbol),
            initial_balance=Decimal(str(initial_balance)),
            current_balance=Decimal(str(initial_balance)),
            positions={},
            current_price=Decimal('100.00'),  # Mock current price
            time_step=0,
            done=False,
            max_steps=1000
        )
        
        # Create and train the agent
        agent = MockRLAgent(alg_enum)
        training_result = agent.train(env, episodes, max_steps=500)
        
        result = {
            "symbol": symbol,
            "algorithm": algorithm,
            "training_result": {
                "episodes_trained": training_result.episodes_trained,
                "total_reward": float(training_result.total_reward),
                "avg_reward": float(training_result.avg_reward),
                "win_rate": float(training_result.win_rate),
                "sharpe_ratio": float(training_result.sharpe_ratio),
                "max_drawdown": float(training_result.max_drawdown),
                "final_balance": float(training_result.final_balance),
                "training_duration": str(training_result.training_duration),
                "hyperparameters": training_result.hyperparameters,
                "performance_metrics": training_result.performance_metrics
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
) -> Dict[str, Any]:
    """
    Evaluate a reinforcement learning trading agent for a symbol.

    Args:
        symbol: Stock symbol to evaluate on
        algorithm: RL algorithm to use
        episodes: Number of evaluation episodes
        initial_balance: Initial balance for simulation
        current_user_id: Authenticated user ID

    Returns:
        Evaluation results with performance metrics
    """
    try:
        # Validate algorithm
        try:
            alg_enum = RLAlgorithm(algorithm.lower().replace('-', '_'))
        except ValueError:
            available_algs = [alg.value for alg in RLAlgorithm]
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid algorithm: {algorithm}. Available: {available_algs}"
            )
        
        # Create a mock environment for evaluation
        env = RLEnvironment(
            symbol=Symbol(symbol),
            initial_balance=Decimal(str(initial_balance)),
            current_balance=Decimal(str(initial_balance)),
            positions={},
            current_price=Decimal('100.00'),  # Mock current price
            time_step=0,
            done=False,
            max_steps=1000
        )
        
        # Create and evaluate the agent
        agent = MockRLAgent(alg_enum)
        evaluation_results = agent.evaluate(env, episodes)
        
        result = {
            "symbol": symbol,
            "algorithm": algorithm,
            "evaluation_results": evaluation_results,
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
) -> Dict[str, Any]:
    """
    Get performance metrics for the ensemble of RL agents.

    Returns:
        Performance metrics for all agents in the ensemble
    """
    try:
        # Create an ensemble of agents
        ensemble = MultiAgentRLEnsemble()
        
        performance = ensemble.get_agents_performance()
        
        result = {
            "ensemble_performance": performance,
            "agent_types": list(performance.keys()),
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
) -> Dict[str, Any]:
    """
    Get action recommendation from RL agent for a symbol.

    Args:
        symbol: Stock symbol to get action for
        user_id: User ID to personalize the action
        algorithm: Algorithm to use
        market_regime: Current market regime
        current_user_id: Authenticated user ID (for authorization)

    Returns:
        Action recommendation with position size
    """
    if current_user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to get actions for this user"
        )
    
    try:
        # Create a mock state for the agent
        # In a real implementation, this would use actual market data and user portfolio
        from decimal import Decimal
        from datetime import datetime
        from src.domain.value_objects import Symbol
        from src.domain.services.rl_trading_agents import RLState, TradingAction
        
        mock_state = RLState(
            portfolio_value=Decimal('50000.00'),
            cash_balance=Decimal('20000.00'),
            position_quantities=[100, 50],  # Mock position quantities
            position_prices=[Decimal('150.00'), Decimal('2500.00')],  # Mock position prices
            market_data=[Decimal('175.00')] * 10,  # Mock market data
            technical_indicators={
                'rsi': Decimal('55.00'),
                'macd': Decimal('1.25'),
                'bb_position': Decimal('0.45'),
                'sma_ratio': Decimal('1.02')
            },
            volatility=Decimal('0.18'),
            market_regime=market_regime,
            time_step=100,
            user_risk_profile='moderate'
        )
        
        if algorithm.lower() == 'ensemble':
            # Use the ensemble of agents
            ensemble = MultiAgentRLEnsemble()
            action, position_size = ensemble.get_action(mock_state, market_regime)
        else:
            # Validate algorithm
            try:
                alg_enum = RLAlgorithm(algorithm.lower().replace('-', '_'))
            except ValueError:
                available_algs = [alg.value for alg in RLAlgorithm]
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid algorithm: {algorithm}. Available: {available_algs}"
                )
            
            # Create a single agent
            agent = MockRLAgent(alg_enum)
            action, position_size = agent.get_action(mock_state, training=False)
        
        result = {
            "symbol": symbol,
            "user_id": user_id,
            "action": action.value,
            "action_name": TradingAction(action.value).name,
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

from datetime import datetime
from decimal import Decimal