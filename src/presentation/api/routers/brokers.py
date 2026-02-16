"""
Multi-Broker Integration API Router

This router handles all broker integration endpoints following RESTful principles.
All endpoints require authentication.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import Optional, Dict, Any
import logging
import uuid
from datetime import datetime
from decimal import Decimal

from src.infrastructure.security import get_current_user
from src.infrastructure.broker_integration import BrokerAdapterManager, BrokerType, BrokerOrder
from src.domain.entities.trading import Order, OrderType, PositionType, OrderStatus
from src.domain.value_objects import Symbol, Money
from src.infrastructure.repositories import UserRepository
from src.presentation.api.dependencies import (
    get_user_repository,
    get_broker_adapter_manager,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/brokers", tags=["brokers"])


def _get_user_or_404(user_repo: UserRepository, user_id: str):
    """Fetch user from repository or raise 404."""
    user = user_repo.get_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return user


@router.get(
    "/available",
    summary="Get available broker integrations",
    responses={
        200: {"description": "Available brokers retrieved successfully"},
        401: {"description": "Unauthorized"},
    }
)
async def get_available_brokers(
    current_user_id: str = Depends(get_current_user),
    adapter_manager: BrokerAdapterManager = Depends(get_broker_adapter_manager),
) -> Dict[str, Any]:
    """Get list of available broker integrations."""
    try:
        available_brokers = adapter_manager.get_available_brokers()

        result = {
            "available_brokers": [broker.value for broker in available_brokers],
            "broker_count": len(available_brokers),
            "retrieved_at": datetime.now().isoformat()
        }

        logger.info(f"Available brokers retrieved for user {current_user_id}")
        return result

    except Exception as e:
        logger.error(f"Error retrieving available brokers: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve available brokers"
        )


@router.post(
    "/{broker_type}/place-order",
    summary="Place an order with a specific broker",
    responses={
        200: {"description": "Order placed successfully"},
        401: {"description": "Unauthorized"},
        400: {"description": "Invalid broker type or order parameters"},
    }
)
async def place_order(
    broker_type: str,
    symbol: str = Query(..., description="Stock symbol"),
    quantity: int = Query(..., description="Number of shares", gt=0),
    side: str = Query(..., description="buy or sell", regex="^(buy|sell)$"),
    order_type: str = Query("market", description="Order type", regex="^(market|limit|stop|stop_limit|trailing_stop)$"),
    limit_price: Optional[float] = Query(None, description="Limit price for limit orders"),
    stop_price: Optional[float] = Query(None, description="Stop price for stop orders"),
    current_user_id: str = Depends(get_current_user),
    user_repo: UserRepository = Depends(get_user_repository),
    adapter_manager: BrokerAdapterManager = Depends(get_broker_adapter_manager),
) -> Dict[str, Any]:
    """Place an order with a specific broker."""
    try:
        # Validate broker type
        try:
            broker_enum = BrokerType(broker_type.lower())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid broker type: {broker_type}. Available: {[b.value for b in BrokerType]}"
            )

        user = _get_user_or_404(user_repo, current_user_id)

        # Build domain Order from request parameters
        order = Order(
            id=str(uuid.uuid4()),
            user_id=current_user_id,
            symbol=Symbol(symbol),
            order_type=OrderType(order_type.upper().replace('_', '')) if order_type.upper().replace('_', '') in [e.name for e in OrderType] else OrderType.MARKET,
            position_type=PositionType.LONG if side.lower() == 'buy' else PositionType.SHORT,
            quantity=quantity,
            status=OrderStatus.PENDING,
            placed_at=datetime.now(),
            price=Money(Decimal(str(limit_price)), 'USD') if limit_price else None,
            stop_price=Money(Decimal(str(stop_price)), 'USD') if stop_price else None,
            filled_quantity=0
        )

        broker_order = adapter_manager.execute_order(order, user, broker_enum)

        result = {
            "broker_order_id": broker_order.broker_order_id,
            "client_order_id": broker_order.client_order_id,
            "symbol": broker_order.symbol,
            "quantity": broker_order.quantity,
            "side": broker_order.side,
            "order_type": broker_order.order_type.value,
            "status": broker_order.status,
            "time_in_force": broker_order.time_in_force,
            "limit_price": float(broker_order.limit_price) if broker_order.limit_price else None,
            "stop_price": float(broker_order.stop_price) if broker_order.stop_price else None,
            "placed_at": broker_order.created_at.isoformat() if broker_order.created_at else None,
            "broker_type": broker_type
        }

        logger.info(f"Order placed with {broker_type} for user {current_user_id}")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error placing order: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to place order"
        )


@router.get(
    "/{broker_type}/positions",
    summary="Get positions from a specific broker",
    responses={
        200: {"description": "Positions retrieved successfully"},
        401: {"description": "Unauthorized"},
        400: {"description": "Invalid broker type"},
    }
)
async def get_positions(
    broker_type: str,
    current_user_id: str = Depends(get_current_user),
    user_repo: UserRepository = Depends(get_user_repository),
    adapter_manager: BrokerAdapterManager = Depends(get_broker_adapter_manager),
) -> Dict[str, Any]:
    """Get current positions from a specific broker."""
    try:
        # Validate broker type
        try:
            broker_enum = BrokerType(broker_type.lower())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid broker type: {broker_type}. Available: {[b.value for b in BrokerType]}"
            )

        user = _get_user_or_404(user_repo, current_user_id)
        positions = adapter_manager.get_broker_service(broker_enum).get_positions(user)

        result = {
            "positions": [
                {
                    "symbol": pos.symbol,
                    "quantity": pos.quantity,
                    "avg_entry_price": float(pos.avg_entry_price),
                    "current_price": float(pos.current_price),
                    "unrealized_pnl": float(pos.unrealized_pnl),
                    "market_value": float(pos.market_value),
                    "position_type": pos.position_type.value
                }
                for pos in positions
            ],
            "position_count": len(positions),
            "retrieved_at": datetime.now().isoformat(),
            "broker_type": broker_type
        }

        logger.info(f"Positions retrieved from {broker_type} for user {current_user_id}")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving positions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve positions"
        )


@router.get(
    "/{broker_type}/account-info",
    summary="Get account information from a specific broker",
    responses={
        200: {"description": "Account information retrieved successfully"},
        401: {"description": "Unauthorized"},
        400: {"description": "Invalid broker type"},
    }
)
async def get_account_info(
    broker_type: str,
    current_user_id: str = Depends(get_current_user),
    user_repo: UserRepository = Depends(get_user_repository),
    adapter_manager: BrokerAdapterManager = Depends(get_broker_adapter_manager),
) -> Dict[str, Any]:
    """Get account information from a specific broker."""
    try:
        # Validate broker type
        try:
            broker_enum = BrokerType(broker_type.lower())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid broker type: {broker_type}. Available: {[b.value for b in BrokerType]}"
            )

        user = _get_user_or_404(user_repo, current_user_id)
        account_info = adapter_manager.get_account_info(user, broker_enum)

        result = {
            "account_id": account_info.account_id,
            "account_number": account_info.account_number,
            "account_type": account_info.account_type,
            "buying_power": {
                "amount": float(account_info.buying_power.amount),
                "currency": account_info.buying_power.currency
            },
            "cash_balance": {
                "amount": float(account_info.cash_balance.amount),
                "currency": account_info.cash_balance.currency
            },
            "portfolio_value": {
                "amount": float(account_info.portfolio_value.amount),
                "currency": account_info.portfolio_value.currency
            },
            "day_trade_count": account_info.day_trade_count,
            "pattern_day_trader": account_info.pattern_day_trader,
            "trading_blocked": account_info.trading_blocked,
            "transfers_blocked": account_info.transfers_blocked,
            "account_blocked": account_info.account_blocked,
            "created_at": account_info.created_at.isoformat(),
            "updated_at": account_info.updated_at.isoformat(),
            "broker_type": broker_type
        }

        logger.info(f"Account info retrieved from {broker_type} for user {current_user_id}")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving account info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve account information"
        )
