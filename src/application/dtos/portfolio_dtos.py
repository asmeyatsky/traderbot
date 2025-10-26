"""
Portfolio Data Transfer Objects

DTOs for portfolio information and API responses.
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class PositionResponse(BaseModel):
    """Response DTO for a position in portfolio."""

    id: str
    symbol: str
    position_type: str
    quantity: int
    average_buy_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    pnl_percentage: float
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class PortfolioResponse(BaseModel):
    """Response DTO for portfolio details."""

    id: str
    user_id: str
    total_value: float = Field(..., description="Total portfolio value (cash + positions)")
    cash_balance: float
    positions_value: float
    position_count: int
    positions: List[PositionResponse]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class PortfolioPerformanceResponse(BaseModel):
    """Response DTO for portfolio performance metrics."""

    total_value: float
    cash_balance: float
    positions_value: float
    position_count: int
    total_return_percentage: float
    daily_return_percentage: Optional[float] = None
    weekly_return_percentage: Optional[float] = None
    monthly_return_percentage: Optional[float] = None
    positions: List[PositionResponse]
    timestamp: datetime = Field(default_factory=datetime.now)


class UpdateCashBalanceRequest(BaseModel):
    """Request DTO for updating cash balance."""

    amount: float = Field(..., description="Amount to add/remove (negative for withdrawal)")
    reason: str = Field(..., max_length=200, description="Reason for the transaction")


class PortfolioAllocationResponse(BaseModel):
    """Response DTO for portfolio allocation breakdown."""

    cash_percentage: float = Field(..., ge=0, le=100)
    stocks_percentage: float = Field(..., ge=0, le=100)
    by_sector: dict[str, float] = Field(..., description="Allocation by sector")
    by_symbol: dict[str, float] = Field(..., description="Allocation by symbol")
    timestamp: datetime = Field(default_factory=datetime.now)
