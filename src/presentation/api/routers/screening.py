"""
Stock Screening API Router

Endpoints for screening stocks by criteria and prebuilt screens.
"""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.infrastructure.di_container import container
from src.infrastructure.security import get_current_user

router = APIRouter(prefix="/api/v1/screening", tags=["screening"])


class ScreenRequest(BaseModel):
    prebuilt_screen: Optional[str] = None
    min_change_pct: Optional[float] = None
    max_change_pct: Optional[float] = None
    min_volume: Optional[int] = None
    sectors: Optional[List[str]] = None
    limit: int = 10


class PrebuiltScreenInfo(BaseModel):
    name: str
    label: str
    description: str


@router.post("/screen")
async def screen_stocks(request: ScreenRequest, user_id: str = Depends(get_current_user)):
    """Screen stocks by criteria or prebuilt screen."""
    screener = container.adapters.stock_screener()
    return screener.screen(request.model_dump(exclude_none=True))


@router.get("/prebuilt")
async def list_prebuilt_screens(user_id: str = Depends(get_current_user)):
    """List available prebuilt screens."""
    return {
        "screens": [
            {"name": "top_gainers", "label": "Top Gainers", "description": "Stocks with highest daily gains"},
            {"name": "top_losers", "label": "Top Losers", "description": "Stocks with biggest daily losses"},
            {"name": "most_active", "label": "Most Active", "description": "Highest volume stocks today"},
            {"name": "high_momentum", "label": "High Momentum", "description": "Stocks with strong upward momentum"},
            {"name": "oversold_rsi", "label": "Oversold (RSI)", "description": "Stocks with RSI below 30"},
        ]
    }
