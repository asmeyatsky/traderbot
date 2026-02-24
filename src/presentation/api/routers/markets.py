"""
Markets API Router

Provides endpoints for browsing available markets and searching stocks
within each market. Markets are filtered per-user based on allowed_markets.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from typing import List, Optional
import logging

from src.infrastructure.security import get_current_user
from src.presentation.api.dependencies import get_user_repository
from src.infrastructure.repositories import UserRepository
from src.infrastructure.data.markets import get_markets_for_user, get_stocks_by_market

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/markets", tags=["markets"])


# ---------------------------------------------------------------------------
# DTOs
# ---------------------------------------------------------------------------

class MarketResponse(BaseModel):
    market_code: str
    market_name: str
    country: str
    currency: str


class MarketListResponse(BaseModel):
    markets: List[MarketResponse]


class StockResponse(BaseModel):
    symbol: str
    name: str
    sector: str


class StockListResponse(BaseModel):
    market_code: str
    stocks: List[StockResponse]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "",
    response_model=MarketListResponse,
    summary="List user's allowed markets",
    responses={
        200: {"description": "Markets retrieved"},
        401: {"description": "Unauthorized"},
    },
)
async def list_markets(
    user_id: str = Depends(get_current_user),
    user_repository: UserRepository = Depends(get_user_repository),
) -> MarketListResponse:
    """Return the list of markets the authenticated user is allowed to trade on."""
    try:
        user = user_repository.get_by_id(user_id)
        if not user:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

        markets = get_markets_for_user(user.allowed_markets)
        return MarketListResponse(
            markets=[MarketResponse(**m) for m in markets],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing markets: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list markets",
        )


@router.get(
    "/{market_code}/stocks",
    response_model=StockListResponse,
    summary="List stocks in a market",
    responses={
        200: {"description": "Stocks retrieved"},
        401: {"description": "Unauthorized"},
        403: {"description": "Market not in user's allowed list"},
        404: {"description": "Market not found"},
    },
)
async def list_stocks(
    market_code: str,
    search: Optional[str] = Query(None, min_length=1, max_length=50, description="Search filter"),
    user_id: str = Depends(get_current_user),
    user_repository: UserRepository = Depends(get_user_repository),
) -> StockListResponse:
    """Return stocks for a market, optionally filtered by a search term."""
    try:
        user = user_repository.get_by_id(user_id)
        if not user:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

        if market_code not in user.allowed_markets:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Market not in your allowed markets",
            )

        result = get_stocks_by_market(market_code, search)
        if result is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Market not found")

        return StockListResponse(
            market_code=result["market_code"],
            stocks=[StockResponse(**s) for s in result["stocks"]],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing stocks for {market_code}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list stocks",
        )
