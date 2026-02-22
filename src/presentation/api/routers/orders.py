"""
Orders API Router

This router handles all order-related endpoints following RESTful principles.
All endpoints require authentication.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Optional
import logging

from src.infrastructure.security import get_current_user
from src.application.dtos.order_dtos import (
    CreateOrderRequest, UpdateOrderRequest, OrderResponse,
    OrderListResponse, CancelOrderRequest
)
from src.presentation.api.dependencies import (
    get_create_order_use_case,
    get_order_repository,
)
from src.application.use_cases.trading import CreateOrderUseCase
from src.infrastructure.repositories import OrderRepository
from src.domain.value_objects import Symbol
from src.domain.entities.trading import OrderStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/orders", tags=["orders"])


def _order_to_response(order) -> OrderResponse:
    """Convert domain Order entity to OrderResponse DTO."""
    return OrderResponse(
        id=order.id,
        user_id=order.user_id,
        symbol=str(order.symbol),
        order_type=order.order_type.name if hasattr(order.order_type, 'name') else str(order.order_type),
        position_type=order.position_type.name if hasattr(order.position_type, 'name') else str(order.position_type),
        quantity=order.quantity,
        status=order.status.name if hasattr(order.status, 'name') else str(order.status),
        placed_at=order.placed_at,
        executed_at=order.executed_at,
        price=float(order.price.amount) if order.price else None,
        stop_price=float(order.stop_price.amount) if order.stop_price else None,
        filled_quantity=order.filled_quantity or 0,
        commission=float(order.commission.amount) if order.commission else None,
        notes=order.notes,
    )


@router.post(
    "/create",
    response_model=OrderResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new trading order",
    responses={
        201: {"description": "Order created successfully"},
        400: {"description": "Invalid order parameters"},
        401: {"description": "Unauthorized"},
        422: {"description": "Validation error"},
    }
)
async def create_order(
    request: CreateOrderRequest,
    user_id: str = Depends(get_current_user),
    use_case: CreateOrderUseCase = Depends(get_create_order_use_case),
) -> OrderResponse:
    """
    Create a new trading order.

    The system will validate the order against user constraints including:
    - Sufficient cash balance
    - Position size limits
    - Risk tolerance settings
    - Sector preferences/exclusions

    Args:
        request: Order creation request details
        user_id: Current authenticated user ID

    Returns:
        Created order details
    """
    try:
        logger.info(f"Creating order for user {user_id}: {request.symbol}")

        order = use_case.execute(
            user_id=user_id,
            symbol=Symbol(request.symbol),
            order_type=request.order_type,
            position_type=request.position_type,
            quantity=request.quantity,
            limit_price=request.limit_price,
            stop_price=request.stop_price,
        )

        if not order:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to create order"
            )

        return _order_to_response(order)

    except ValueError as e:
        logger.warning(f"Order validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating order: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create order"
        )


@router.get(
    "/{order_id}",
    response_model=OrderResponse,
    summary="Get order details",
    responses={
        200: {"description": "Order found"},
        401: {"description": "Unauthorized"},
        404: {"description": "Order not found"},
    }
)
async def get_order(
    order_id: str,
    user_id: str = Depends(get_current_user),
    order_repository: OrderRepository = Depends(get_order_repository),
) -> OrderResponse:
    """
    Get details of a specific order.

    Args:
        order_id: Order ID to retrieve
        user_id: Current authenticated user ID

    Returns:
        Order details
    """
    try:
        logger.info(f"Fetching order {order_id} for user {user_id}")

        order = order_repository.get_by_id(order_id)

        if not order:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Order {order_id} not found"
            )

        # Verify the order belongs to the user
        if order.user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this order"
            )

        return _order_to_response(order)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching order: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch order"
        )


@router.get(
    "",
    response_model=OrderListResponse,
    summary="Get user's orders",
    responses={
        200: {"description": "Orders retrieved successfully"},
        401: {"description": "Unauthorized"},
    }
)
async def get_user_orders(
    skip: int = 0,
    limit: int = 50,
    status_filter: Optional[str] = None,
    user_id: str = Depends(get_current_user),
    order_repository: OrderRepository = Depends(get_order_repository),
) -> OrderListResponse:
    """
    Get all orders for the current user with optional filtering.

    Args:
        skip: Number of orders to skip
        limit: Maximum number of orders to return
        status_filter: Filter by order status (PENDING, EXECUTED, CANCELLED, FAILED)
        user_id: Current authenticated user ID

    Returns:
        List of orders
    """
    try:
        logger.info(f"Fetching orders for user {user_id}")

        if status_filter:
            try:
                order_status = OrderStatus[status_filter.upper()]
                orders = order_repository.get_by_status(user_id, order_status)
            except KeyError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid status filter: {status_filter}"
                )
        else:
            orders = order_repository.get_by_user_id(user_id)

        # Apply pagination
        total = len(orders)
        paginated_orders = orders[skip:skip + limit]

        return OrderListResponse(
            total=total,
            orders=[_order_to_response(order) for order in paginated_orders]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching orders: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch orders"
        )


@router.put(
    "/{order_id}",
    response_model=OrderResponse,
    summary="Update an order",
    responses={
        200: {"description": "Order updated successfully"},
        400: {"description": "Invalid update parameters"},
        401: {"description": "Unauthorized"},
        404: {"description": "Order not found"},
    }
)
async def update_order(
    order_id: str,
    request: UpdateOrderRequest,
    user_id: str = Depends(get_current_user),
    order_repository: OrderRepository = Depends(get_order_repository),
) -> OrderResponse:
    """
    Update an existing order (only pending orders can be updated).

    Args:
        order_id: Order ID to update
        request: Updated order details
        user_id: Current authenticated user ID

    Returns:
        Updated order details
    """
    try:
        logger.info(f"Updating order {order_id} for user {user_id}")

        # Get the existing order
        order = order_repository.get_by_id(order_id)

        if not order:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Order {order_id} not found"
            )

        # Verify the order belongs to the user
        if order.user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to update this order"
            )

        # Only pending orders can be updated
        if order.status != OrderStatus.PENDING:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Cannot update order with status {order.status.name}"
            )

        # Create updated order using immutable pattern
        from dataclasses import replace
        from src.domain.value_objects import Money

        updates = {}
        if request.quantity is not None:
            updates['quantity'] = request.quantity
        if request.limit_price is not None:
            updates['price'] = Money(request.limit_price, 'USD')
        if request.stop_price is not None:
            updates['stop_price'] = Money(request.stop_price, 'USD')
        if request.notes is not None:
            updates['notes'] = request.notes

        if updates:
            updated_order = replace(order, **updates)
            saved_order = order_repository.save(updated_order)
            return _order_to_response(saved_order)

        return _order_to_response(order)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating order: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update order"
        )


@router.delete(
    "/{order_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Cancel an order",
    responses={
        204: {"description": "Order cancelled successfully"},
        401: {"description": "Unauthorized"},
        404: {"description": "Order not found"},
        422: {"description": "Cannot cancel executed order"},
    }
)
async def cancel_order(
    order_id: str,
    request: Optional[CancelOrderRequest] = None,
    user_id: str = Depends(get_current_user),
    order_repository: OrderRepository = Depends(get_order_repository),
) -> None:
    """
    Cancel a pending order.

    Args:
        order_id: Order ID to cancel
        request: Optional cancellation details
        user_id: Current authenticated user ID
    """
    try:
        logger.info(f"Cancelling order {order_id} for user {user_id}")

        # Get the existing order
        order = order_repository.get_by_id(order_id)

        if not order:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Order {order_id} not found"
            )

        # Verify the order belongs to the user
        if order.user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to cancel this order"
            )

        # Only pending orders can be cancelled
        if order.status not in [OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED]:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Cannot cancel order with status {order.status.name}"
            )

        # Update the order status to cancelled
        cancellation_notes = f"Cancelled by user"
        if request and request.reason:
            cancellation_notes += f": {request.reason}"

        order_repository.update_status(order_id, OrderStatus.CANCELLED)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling order: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel order"
        )
