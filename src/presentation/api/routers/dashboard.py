"""
API Endpoints for User Dashboard

This module implements the API endpoints for the user dashboard
as required by the PRD (Dashboard & Portfolio Overview).
"""
from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Optional
from datetime import datetime, date

from src.application.use_cases.trading import (
    GetPortfolioPerformanceUseCase, 
    GetUserPreferencesUseCase, 
    AnalyzeNewsSentimentUseCase
)
from src.domain.entities.trading import Order, Position, Portfolio
from src.domain.entities.user import User
from src.domain.value_objects import Symbol
from src.domain.ports import (
    OrderRepositoryPort, PositionRepositoryPort, PortfolioRepositoryPort, 
    UserRepositoryPort, MarketDataPort, NewsAnalysisPort
)
from src.infrastructure.api_clients.market_data import MarketDataService


# Initialize router
router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])


# Dependency injection functions
def get_portfolio_performance_use_case() -> GetPortfolioPerformanceUseCase:
    # In a real implementation, these would be injected through a DI container
    market_data_service = MarketDataService()
    portfolio_repo = None  # Would be injected
    return GetPortfolioPerformanceUseCase(portfolio_repo, market_data_service)


def get_user_preferences_use_case() -> GetUserPreferencesUseCase:
    user_repo = None  # Would be injected
    return GetUserPreferencesUseCase(user_repo)


def get_news_sentiment_use_case() -> AnalyzeNewsSentimentUseCase:
    news_service = None  # Would be injected
    market_data_service = MarketDataService()
    portfolio_repo = None  # Would be injected
    return AnalyzeNewsSentimentUseCase(news_service, market_data_service, portfolio_repo)


@router.get("/portfolio/{user_id}")
async def get_portfolio_overview(user_id: str):
    """
    Get real-time portfolio value, P&L, and performance metrics.
    Implements FR-3.4.1 Dashboard & Portfolio Overview
    """
    use_case = get_portfolio_performance_use_case()
    
    try:
        performance_data = use_case.execute(user_id)
        return {
            "user_id": user_id,
            "timestamp": datetime.now(),
            "data": performance_data
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving portfolio data: {str(e)}"
        )


@router.get("/positions/{user_id}")
async def get_positions(user_id: str):
    """
    Get position breakdown by asset, sector, strategy.
    Implements FR-3.4.1 Dashboard & Portfolio Overview
    """
    use_case = get_portfolio_performance_use_case()
    
    try:
        performance_data = use_case.execute(user_id)
        positions = performance_data.get('positions', [])
        
        return {
            "user_id": user_id,
            "timestamp": datetime.now(),
            "positions": positions,
            "count": len(positions)
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving positions: {str(e)}"
        )


@router.get("/performance/{user_id}")
async def get_performance_metrics(user_id: str):
    """
    Get comprehensive performance metrics.
    Implements FR-3.4.3 Performance Analytics & Reporting
    """
    use_case = get_portfolio_performance_use_case()
    
    try:
        performance_data = use_case.execute(user_id)
        
        # Calculate additional performance metrics
        total_value = performance_data.get('total_value', 0)
        initial_value = 10000  # Would come from user's initial investment
        total_return = ((total_value - initial_value) / initial_value) * 100 if initial_value > 0 else 0
        
        return {
            "user_id": user_id,
            "timestamp": datetime.now(),
            "metrics": {
                "total_value": total_value,
                "total_return_pct": round(total_return, 2),
                "positions_value": performance_data.get('positions_value', 0),
                "cash_balance": performance_data.get('cash_balance', 0),
                "position_count": performance_data.get('position_count', 0)
            }
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error calculating performance metrics: {str(e)}"
        )


@router.get("/news-sentiment/{symbol}")
async def get_news_sentiment(symbol: str):
    """
    Get real-time news and sentiment affecting portfolio holdings.
    Implements US-2.1: See real-time news and sentiment affecting portfolio
    """
    try:
        symbol_obj = Symbol(symbol)
        use_case = get_news_sentiment_use_case()
        
        sentiment_results = use_case.execute(symbol_obj)
        
        return {
            "symbol": symbol,
            "timestamp": datetime.now(),
            "sentiment_data": sentiment_results
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving news sentiment: {str(e)}"
        )


@router.get("/user-preferences/{user_id}")
async def get_user_preferences(user_id: str):
    """
    Get user's risk tolerance and investment goals.
    Implements FR-1.2: User sets risk tolerance and investment goals
    """
    use_case = get_user_preferences_use_case()
    
    try:
        user = use_case.get_user_preferences(user_id)
        if not user:
            raise ValueError(f"User not found: {user_id}")
        
        return {
            "user_id": user_id,
            "preferences": {
                "risk_tolerance": user.risk_tolerance.value,
                "investment_goal": user.investment_goal.value,
                "max_position_size_percentage": float(user.max_position_size_percentage),
                "daily_loss_limit": str(user.daily_loss_limit) if user.daily_loss_limit else None,
                "weekly_loss_limit": str(user.weekly_loss_limit) if user.weekly_loss_limit else None,
                "monthly_loss_limit": str(user.monthly_loss_limit) if user.monthly_loss_limit else None,
                "sector_preferences": user.sector_preferences,
                "sector_exclusions": user.sector_exclusions,
                "approval_mode_enabled": user.approval_mode_enabled
            }
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving user preferences: {str(e)}"
        )


@router.put("/user-preferences/{user_id}/risk-tolerance")
async def update_risk_tolerance(user_id: str, risk_tolerance: str):
    """
    Update user's risk tolerance setting.
    Implements FR-1.2: User sets risk tolerance and investment goals
    """
    use_case = get_user_preferences_use_case()
    
    try:
        updated_user = use_case.update_user_risk_tolerance(user_id, risk_tolerance)
        
        return {
            "user_id": user_id,
            "updated_preferences": {
                "risk_tolerance": updated_user.risk_tolerance.value
            }
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating risk tolerance: {str(e)}"
        )


@router.put("/user-preferences/{user_id}/investment-goal")
async def update_investment_goal(user_id: str, investment_goal: str):
    """
    Update user's investment goal setting.
    Implements FR-1.2: User sets risk tolerance and investment goals
    """
    use_case = get_user_preferences_use_case()
    
    try:
        updated_user = use_case.update_user_investment_goal(user_id, investment_goal)
        
        return {
            "user_id": user_id,
            "updated_preferences": {
                "investment_goal": updated_user.investment_goal.value
            }
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating investment goal: {str(e)}"
        )


@router.get("/ai-insights/{user_id}")
async def get_ai_insights(user_id: str):
    """
    Get AI insights: news alerts, trade rationale, risk warnings.
    Implements FR-3.4.1 Dashboard & Portfolio Overview
    """
    # This would integrate with the AI model service to provide insights
    # For now, returning placeholder data
    return {
        "user_id": user_id,
        "timestamp": datetime.now(),
        "insights": {
            "news_alerts": [],
            "trade_rationale": [],
            "risk_warnings": [],
            "market_opportunities": []
        }
    }


@router.get("/trading-status/{user_id}")
async def get_trading_status(user_id: str):
    """
    Get the current trading status for the user's account.
    """
    # This would check if trading is enabled/paused for the user
    # For now, returning placeholder data
    return {
        "user_id": user_id,
        "timestamp": datetime.now(),
        "trading_status": {
            "is_enabled": True,
            "last_activity": datetime.now().isoformat(),
            "active_orders": 0,
            "todays_pnl": 0.0,
            "risk_status": "NORMAL"  # Options: NORMAL, WARNING, SUSPENDED
        }
    }