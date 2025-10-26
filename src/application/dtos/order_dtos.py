"""
Order Data Transfer Objects

DTOs for order creation, updates, and API responses.
"""
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from decimal import Decimal


class CreateOrderRequest(BaseModel):
    """Request DTO for creating a new order."""

    symbol: str = Field(
        ...,
        min_length=1,
        max_length=5,
        description="Stock symbol (e.g., AAPL, GOOGL)",
        example="AAPL"
    )
    order_type: str = Field(
        ...,
        pattern="^(MARKET|LIMIT|STOP_LOSS|TRAILING_STOP)$",
        description="Type of order",
        example="MARKET"
    )
    position_type: str = Field(
        ...,
        pattern="^(LONG|SHORT)$",
        description="Position type (LONG for buy, SHORT for sell)",
        example="LONG"
    )
    quantity: int = Field(
        ...,
        gt=0,
        description="Number of shares to trade",
        example=100
    )
    limit_price: Optional[float] = Field(
        default=None,
        gt=0,
        description="Limit price (required for LIMIT orders)",
        example=150.50
    )
    stop_price: Optional[float] = Field(
        default=None,
        gt=0,
        description="Stop price (required for STOP_LOSS/TRAILING_STOP orders)",
        example=145.00
    )
    notes: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Optional notes about the order",
        example="Buy on dip"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "AAPL",
                "order_type": "MARKET",
                "position_type": "LONG",
                "quantity": 100,
                "limit_price": None,
                "stop_price": None,
                "notes": None
            }
        }


class UpdateOrderRequest(BaseModel):
    """Request DTO for updating an order."""

    quantity: Optional[int] = Field(default=None, gt=0)
    limit_price: Optional[float] = Field(default=None, gt=0)
    stop_price: Optional[float] = Field(default=None, gt=0)
    notes: Optional[str] = Field(default=None, max_length=500)


class OrderResponse(BaseModel):
    """Response DTO for order details."""

    id: str
    user_id: str
    symbol: str
    order_type: str
    position_type: str
    quantity: int
    status: str
    placed_at: datetime
    executed_at: Optional[datetime] = None
    price: Optional[float] = None
    stop_price: Optional[float] = None
    filled_quantity: int = 0
    commission: Optional[float] = None
    notes: Optional[str] = None

    class Config:
        from_attributes = True


class OrderListResponse(BaseModel):
    """Response DTO for list of orders."""

    total: int
    orders: list[OrderResponse]


class CancelOrderRequest(BaseModel):
    """Request DTO for cancelling an order."""

    reason: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Reason for cancellation"
    )
