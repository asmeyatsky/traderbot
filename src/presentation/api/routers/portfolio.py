"""
Portfolio API Router

This router handles all portfolio-related endpoints for viewing
portfolio state, performance metrics, and allocation.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from datetime import datetime
from decimal import Decimal
import logging

from src.infrastructure.security import get_current_user
from src.application.dtos.portfolio_dtos import (
    PortfolioResponse, PortfolioPerformanceResponse,
    PortfolioAllocationResponse, UpdateCashBalanceRequest,
    PositionResponse
)
from src.presentation.api.dependencies import (
    get_portfolio_repository,
    get_position_repository,
    get_portfolio_performance_use_case,
)
from src.infrastructure.repositories import PortfolioRepository, PositionRepository
from src.domain.value_objects import Money

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/portfolio", tags=["portfolio"])


def _position_to_response(position) -> PositionResponse:
    """Convert domain Position entity to PositionResponse DTO."""
    current_price = float(position.current_price.amount) if position.current_price else 0.0
    avg_buy_price = float(position.average_buy_price.amount) if hasattr(position, 'average_buy_price') and position.average_buy_price else float(position.average_entry_price.amount) if hasattr(position, 'average_entry_price') and position.average_entry_price else 0.0
    quantity = int(position.quantity) if position.quantity else 0
    market_value = current_price * quantity
    unrealized_pnl = (current_price - avg_buy_price) * quantity
    pnl_percentage = ((current_price - avg_buy_price) / avg_buy_price * 100) if avg_buy_price > 0 else 0.0

    return PositionResponse(
        id=position.id,
        symbol=str(position.symbol),
        position_type=position.position_type.name if hasattr(position.position_type, 'name') else str(position.position_type),
        quantity=quantity,
        average_buy_price=avg_buy_price,
        current_price=current_price,
        market_value=market_value,
        unrealized_pnl=unrealized_pnl,
        pnl_percentage=pnl_percentage,
        created_at=position.opened_at if hasattr(position, 'opened_at') else datetime.utcnow(),
        updated_at=position.updated_at if hasattr(position, 'updated_at') else datetime.utcnow(),
    )


def _portfolio_to_response(portfolio, positions=None) -> PortfolioResponse:
    """Convert domain Portfolio entity to PortfolioResponse DTO."""
    position_responses = []
    positions_value = 0.0

    if positions:
        for pos in positions:
            pos_response = _position_to_response(pos)
            position_responses.append(pos_response)
            positions_value += pos_response.market_value

    return PortfolioResponse(
        id=portfolio.id,
        user_id=portfolio.user_id,
        total_value=float(portfolio.total_value.amount) if portfolio.total_value else 0.0,
        cash_balance=float(portfolio.cash_balance.amount) if portfolio.cash_balance else 0.0,
        positions_value=positions_value,
        position_count=len(position_responses),
        positions=position_responses,
        created_at=portfolio.created_at,
        updated_at=portfolio.updated_at,
    )


@router.get(
    "",
    response_model=PortfolioResponse,
    summary="Get portfolio details",
    responses={
        200: {"description": "Portfolio retrieved successfully"},
        401: {"description": "Unauthorized"},
        404: {"description": "Portfolio not found"},
    }
)
async def get_portfolio(
    user_id: str = Depends(get_current_user),
    portfolio_repository: PortfolioRepository = Depends(get_portfolio_repository),
    position_repository: PositionRepository = Depends(get_position_repository),
) -> PortfolioResponse:
    """
    Get the current portfolio details including all positions and cash balance.

    Args:
        user_id: Current authenticated user ID

    Returns:
        Portfolio details with positions and valuations
    """
    try:
        logger.info(f"Fetching portfolio for user {user_id}")

        portfolio = portfolio_repository.get_by_user_id(user_id)
        if not portfolio:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Portfolio not found"
            )

        # Get positions for this portfolio
        positions = position_repository.get_by_user_id(user_id)

        return _portfolio_to_response(portfolio, positions)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching portfolio: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch portfolio"
        )


@router.get(
    "/performance",
    response_model=PortfolioPerformanceResponse,
    summary="Get portfolio performance metrics",
    responses={
        200: {"description": "Performance metrics retrieved"},
        401: {"description": "Unauthorized"},
        404: {"description": "Portfolio not found"},
    }
)
async def get_portfolio_performance(
    user_id: str = Depends(get_current_user),
    portfolio_repository: PortfolioRepository = Depends(get_portfolio_repository),
    position_repository: PositionRepository = Depends(get_position_repository),
) -> PortfolioPerformanceResponse:
    """
    Get portfolio performance metrics including returns and drawdowns.

    Returns:
        Performance metrics with daily, weekly, and monthly returns
    """
    try:
        logger.info(f"Fetching portfolio performance for user {user_id}")

        portfolio = portfolio_repository.get_by_user_id(user_id)
        if not portfolio:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Portfolio not found"
            )

        # Get positions
        positions = position_repository.get_by_user_id(user_id)
        position_responses = [_position_to_response(pos) for pos in positions] if positions else []
        positions_value = sum(pos.market_value for pos in position_responses)

        # Calculate total return percentage
        total_return_pct = float(portfolio.total_return_percentage) if hasattr(portfolio, 'total_return_percentage') and portfolio.total_return_percentage else 0.0

        return PortfolioPerformanceResponse(
            total_value=float(portfolio.total_value.amount) if portfolio.total_value else 0.0,
            cash_balance=float(portfolio.cash_balance.amount) if portfolio.cash_balance else 0.0,
            positions_value=positions_value,
            position_count=len(position_responses),
            total_return_percentage=total_return_pct,
            daily_return_percentage=None,  # Would need historical data
            weekly_return_percentage=None,  # Would need historical data
            monthly_return_percentage=None,  # Would need historical data
            positions=position_responses,
            timestamp=datetime.utcnow(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching performance: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch performance metrics"
        )


@router.get(
    "/allocation",
    response_model=PortfolioAllocationResponse,
    summary="Get portfolio allocation breakdown",
    responses={
        200: {"description": "Allocation retrieved"},
        401: {"description": "Unauthorized"},
        404: {"description": "Portfolio not found"},
    }
)
async def get_portfolio_allocation(
    user_id: str = Depends(get_current_user),
    portfolio_repository: PortfolioRepository = Depends(get_portfolio_repository),
    position_repository: PositionRepository = Depends(get_position_repository),
) -> PortfolioAllocationResponse:
    """
    Get portfolio allocation breakdown by sector and symbol.

    Returns:
        Allocation percentages for cash, stocks, and breakdown by sector
    """
    try:
        logger.info(f"Fetching allocation for user {user_id}")

        portfolio = portfolio_repository.get_by_user_id(user_id)
        if not portfolio:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Portfolio not found"
            )

        # Get positions
        positions = position_repository.get_by_user_id(user_id)

        total_value = float(portfolio.total_value.amount) if portfolio.total_value else 0.0
        cash_balance = float(portfolio.cash_balance.amount) if portfolio.cash_balance else 0.0

        if total_value <= 0:
            return PortfolioAllocationResponse(
                cash_percentage=100.0,
                stocks_percentage=0.0,
                by_sector={},
                by_symbol={},
                timestamp=datetime.utcnow(),
            )

        cash_percentage = (cash_balance / total_value) * 100
        stocks_percentage = 100.0 - cash_percentage

        # Calculate allocation by symbol
        by_symbol = {}
        if positions:
            for pos in positions:
                pos_response = _position_to_response(pos)
                symbol = str(pos.symbol)
                by_symbol[symbol] = (pos_response.market_value / total_value) * 100

        # Sector allocation would require symbol-to-sector mapping
        # For now, return empty sector allocation
        by_sector = {}

        return PortfolioAllocationResponse(
            cash_percentage=round(cash_percentage, 2),
            stocks_percentage=round(stocks_percentage, 2),
            by_sector=by_sector,
            by_symbol={k: round(v, 2) for k, v in by_symbol.items()},
            timestamp=datetime.utcnow(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching allocation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch allocation"
        )


@router.post(
    "/cash-deposit",
    response_model=PortfolioResponse,
    status_code=status.HTTP_200_OK,
    summary="Deposit cash into portfolio",
    responses={
        200: {"description": "Cash deposited successfully"},
        400: {"description": "Invalid amount"},
        401: {"description": "Unauthorized"},
    }
)
async def deposit_cash(
    request: UpdateCashBalanceRequest,
    user_id: str = Depends(get_current_user),
    portfolio_repository: PortfolioRepository = Depends(get_portfolio_repository),
    position_repository: PositionRepository = Depends(get_position_repository),
) -> PortfolioResponse:
    """
    Deposit cash into the portfolio.

    Args:
        request: Deposit amount and reason
        user_id: Current authenticated user ID

    Returns:
        Updated portfolio details
    """
    try:
        if request.amount <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Deposit amount must be positive"
            )

        logger.info(f"Processing cash deposit for user {user_id}: ${request.amount}")

        portfolio = portfolio_repository.get_by_user_id(user_id)

        from dataclasses import replace
        import uuid

        if not portfolio:
            # Auto-create portfolio for new users on first deposit
            from src.domain.entities.trading import Portfolio as PortfolioEntity
            portfolio = PortfolioEntity(
                id=str(uuid.uuid4()),
                user_id=user_id,
                positions=[],
                cash_balance=Money(Decimal(str(request.amount)), "USD"),
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            saved_portfolio = portfolio_repository.save(portfolio)
        else:
            new_cash = portfolio.cash_balance.amount + Decimal(str(request.amount))
            updated_portfolio = replace(
                portfolio,
                cash_balance=Money(new_cash, "USD"),
                updated_at=datetime.utcnow(),
            )
            saved_portfolio = portfolio_repository.update(updated_portfolio)

        # Get positions for response
        positions = position_repository.get_by_user_id(user_id)

        return _portfolio_to_response(saved_portfolio, positions)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing deposit: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process deposit"
        )


@router.post(
    "/cash-withdraw",
    response_model=PortfolioResponse,
    status_code=status.HTTP_200_OK,
    summary="Withdraw cash from portfolio",
    responses={
        200: {"description": "Cash withdrawn successfully"},
        400: {"description": "Insufficient balance"},
        401: {"description": "Unauthorized"},
    }
)
async def withdraw_cash(
    request: UpdateCashBalanceRequest,
    user_id: str = Depends(get_current_user),
    portfolio_repository: PortfolioRepository = Depends(get_portfolio_repository),
    position_repository: PositionRepository = Depends(get_position_repository),
) -> PortfolioResponse:
    """
    Withdraw cash from the portfolio.

    Args:
        request: Withdrawal amount and reason
        user_id: Current authenticated user ID

    Returns:
        Updated portfolio details
    """
    try:
        if request.amount <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Withdrawal amount must be positive"
            )

        logger.info(f"Processing cash withdrawal for user {user_id}: ${request.amount}")

        portfolio = portfolio_repository.get_by_user_id(user_id)
        if not portfolio:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Portfolio not found"
            )

        # Check sufficient balance
        withdrawal_amount = Decimal(str(request.amount))
        if portfolio.cash_balance.amount < withdrawal_amount:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Insufficient cash balance"
            )

        # Update portfolio with new cash balance
        from dataclasses import replace
        new_cash = portfolio.cash_balance.amount - withdrawal_amount

        updated_portfolio = replace(
            portfolio,
            cash_balance=Money(new_cash, "USD"),
            updated_at=datetime.utcnow(),
        )

        saved_portfolio = portfolio_repository.update(updated_portfolio)

        # Get positions for response
        positions = position_repository.get_by_user_id(user_id)

        return _portfolio_to_response(saved_portfolio, positions)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing withdrawal: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process withdrawal"
        )
