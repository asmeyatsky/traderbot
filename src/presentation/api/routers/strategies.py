"""
Strategy Marketplace and Leaderboard Router

Provides endpoints for saving strategies, viewing the leaderboard,
following strategies (copy trading), and managing saved strategies.
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
import uuid
import logging

from src.infrastructure.security import get_current_user
from src.infrastructure.database import get_database_manager
from src.infrastructure.orm_models import (
    SavedStrategyORM,
    BacktestResultORM,
    StrategyFollowORM,
    UserORM,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/strategies", tags=["strategies"])


# --- DTOs ---

class CreateStrategyRequest(BaseModel):
    name: str
    description: str = ""
    strategy_type: str
    parameters: dict = {}
    symbol: str = "AAPL"
    is_public: bool = False


class UpdateStrategyRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    is_public: Optional[bool] = None
    parameters: Optional[dict] = None
    symbol: Optional[str] = None


class SaveBacktestRequest(BaseModel):
    strategy_id: str
    symbol: str
    initial_capital: float
    final_value: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    total_trades: int
    volatility: float
    profit_factor: float


class StrategyResponse(BaseModel):
    id: str
    user_id: str
    author_name: str = ""
    name: str
    description: str
    strategy_type: str
    parameters: dict
    symbol: str
    is_public: bool
    fork_count: int
    follower_count: int = 0
    best_return_pct: Optional[float] = None
    best_sharpe: Optional[float] = None
    created_at: str
    is_following: bool = False


class BacktestResultResponse(BaseModel):
    id: str
    strategy_id: str
    symbol: str
    initial_capital: float
    final_value: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    total_trades: int
    volatility: float
    profit_factor: float
    run_at: str


class LeaderboardEntry(BaseModel):
    rank: int
    strategy_id: str
    strategy_name: str
    author_name: str
    strategy_type: str
    symbol: str
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    follower_count: int
    is_following: bool = False


# --- Endpoints ---

@router.post("", response_model=StrategyResponse)
async def create_strategy(req: CreateStrategyRequest, user_id: str = Depends(get_current_user)):
    db = get_database_manager()
    session = db._session_factory()
    try:
        strategy = SavedStrategyORM(
            id=str(uuid.uuid4()),
            user_id=user_id,
            name=req.name,
            description=req.description,
            strategy_type=req.strategy_type,
            parameters=req.parameters,
            symbol=req.symbol,
            is_public=req.is_public,
            fork_count=0,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        session.add(strategy)
        session.commit()
        session.refresh(strategy)
        return _to_response(strategy, user_id, session)
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to create strategy: {e}")
        raise HTTPException(status_code=500, detail="Failed to create strategy")
    finally:
        session.close()


@router.get("", response_model=list[StrategyResponse])
async def list_my_strategies(user_id: str = Depends(get_current_user)):
    db = get_database_manager()
    session = db._session_factory()
    try:
        strategies = session.query(SavedStrategyORM).filter(
            SavedStrategyORM.user_id == user_id
        ).order_by(SavedStrategyORM.updated_at.desc()).all()
        return [_to_response(s, user_id, session) for s in strategies]
    finally:
        session.close()


@router.get("/marketplace", response_model=list[StrategyResponse])
async def marketplace(user_id: str = Depends(get_current_user)):
    """List all public strategies (strategy marketplace)."""
    db = get_database_manager()
    session = db._session_factory()
    try:
        strategies = session.query(SavedStrategyORM).filter(
            SavedStrategyORM.is_public == True
        ).order_by(SavedStrategyORM.fork_count.desc(), SavedStrategyORM.created_at.desc()).limit(50).all()
        return [_to_response(s, user_id, session) for s in strategies]
    finally:
        session.close()


@router.get("/leaderboard", response_model=list[LeaderboardEntry])
async def leaderboard(user_id: str = Depends(get_current_user)):
    """Get the strategy leaderboard ranked by best backtest return."""
    db = get_database_manager()
    session = db._session_factory()
    try:
        from sqlalchemy import func

        # Get best backtest result per strategy (public strategies only)
        subq = session.query(
            BacktestResultORM.strategy_id,
            func.max(BacktestResultORM.total_return_pct).label('best_return'),
        ).group_by(BacktestResultORM.strategy_id).subquery()

        results = session.query(
            SavedStrategyORM,
            BacktestResultORM,
        ).join(
            subq, SavedStrategyORM.id == subq.c.strategy_id
        ).join(
            BacktestResultORM,
            (BacktestResultORM.strategy_id == SavedStrategyORM.id) &
            (BacktestResultORM.total_return_pct == subq.c.best_return)
        ).filter(
            SavedStrategyORM.is_public == True
        ).order_by(
            BacktestResultORM.total_return_pct.desc()
        ).limit(50).all()

        entries = []
        for rank, (strategy, result) in enumerate(results, 1):
            author = session.query(UserORM).filter(UserORM.id == strategy.user_id).first()
            follower_count = session.query(StrategyFollowORM).filter(
                StrategyFollowORM.strategy_id == strategy.id
            ).count()
            is_following = session.query(StrategyFollowORM).filter(
                StrategyFollowORM.follower_user_id == user_id,
                StrategyFollowORM.strategy_id == strategy.id,
            ).first() is not None

            entries.append(LeaderboardEntry(
                rank=rank,
                strategy_id=strategy.id,
                strategy_name=strategy.name,
                author_name=f"{author.first_name} {author.last_name[0]}." if author else "Unknown",
                strategy_type=strategy.strategy_type,
                symbol=strategy.symbol,
                total_return_pct=float(result.total_return_pct),
                sharpe_ratio=float(result.sharpe_ratio),
                max_drawdown_pct=float(result.max_drawdown_pct),
                win_rate=float(result.win_rate),
                follower_count=follower_count,
                is_following=is_following,
            ))
        return entries
    finally:
        session.close()


@router.get("/{strategy_id}", response_model=StrategyResponse)
async def get_strategy(strategy_id: str, user_id: str = Depends(get_current_user)):
    db = get_database_manager()
    session = db._session_factory()
    try:
        strategy = session.query(SavedStrategyORM).filter(
            SavedStrategyORM.id == strategy_id
        ).first()
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")
        if not strategy.is_public and strategy.user_id != user_id:
            raise HTTPException(status_code=403, detail="Strategy is private")
        return _to_response(strategy, user_id, session)
    finally:
        session.close()


@router.patch("/{strategy_id}", response_model=StrategyResponse)
async def update_strategy(strategy_id: str, req: UpdateStrategyRequest, user_id: str = Depends(get_current_user)):
    db = get_database_manager()
    session = db._session_factory()
    try:
        strategy = session.query(SavedStrategyORM).filter(
            SavedStrategyORM.id == strategy_id,
            SavedStrategyORM.user_id == user_id,
        ).first()
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")
        if req.name is not None:
            strategy.name = req.name
        if req.description is not None:
            strategy.description = req.description
        if req.is_public is not None:
            strategy.is_public = req.is_public
        if req.parameters is not None:
            strategy.parameters = req.parameters
        if req.symbol is not None:
            strategy.symbol = req.symbol
        strategy.updated_at = datetime.utcnow()
        session.commit()
        session.refresh(strategy)
        return _to_response(strategy, user_id, session)
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to update strategy: {e}")
        raise HTTPException(status_code=500, detail="Failed to update strategy")
    finally:
        session.close()


@router.delete("/{strategy_id}")
async def delete_strategy(strategy_id: str, user_id: str = Depends(get_current_user)):
    db = get_database_manager()
    session = db._session_factory()
    try:
        strategy = session.query(SavedStrategyORM).filter(
            SavedStrategyORM.id == strategy_id,
            SavedStrategyORM.user_id == user_id,
        ).first()
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")
        session.delete(strategy)
        session.commit()
        return {"status": "deleted"}
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail="Failed to delete strategy")
    finally:
        session.close()


@router.post("/{strategy_id}/backtest", response_model=BacktestResultResponse)
async def save_backtest_result(strategy_id: str, req: SaveBacktestRequest, user_id: str = Depends(get_current_user)):
    """Save a backtest result for a strategy."""
    db = get_database_manager()
    session = db._session_factory()
    try:
        strategy = session.query(SavedStrategyORM).filter(
            SavedStrategyORM.id == strategy_id,
            SavedStrategyORM.user_id == user_id,
        ).first()
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")

        result = BacktestResultORM(
            id=str(uuid.uuid4()),
            strategy_id=strategy_id,
            user_id=user_id,
            symbol=req.symbol,
            initial_capital=Decimal(str(req.initial_capital)),
            final_value=Decimal(str(req.final_value)),
            total_return_pct=Decimal(str(req.total_return_pct)),
            sharpe_ratio=Decimal(str(req.sharpe_ratio)),
            max_drawdown_pct=Decimal(str(req.max_drawdown_pct)),
            win_rate=Decimal(str(req.win_rate)),
            total_trades=req.total_trades,
            volatility=Decimal(str(req.volatility)),
            profit_factor=Decimal(str(req.profit_factor)),
            run_at=datetime.utcnow(),
        )
        session.add(result)
        session.commit()
        session.refresh(result)
        return _result_to_response(result)
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to save backtest result: {e}")
        raise HTTPException(status_code=500, detail="Failed to save backtest result")
    finally:
        session.close()


@router.get("/{strategy_id}/results", response_model=list[BacktestResultResponse])
async def get_backtest_results(strategy_id: str, user_id: str = Depends(get_current_user)):
    db = get_database_manager()
    session = db._session_factory()
    try:
        results = session.query(BacktestResultORM).filter(
            BacktestResultORM.strategy_id == strategy_id,
        ).order_by(BacktestResultORM.run_at.desc()).limit(20).all()
        return [_result_to_response(r) for r in results]
    finally:
        session.close()


@router.post("/{strategy_id}/follow")
async def follow_strategy(strategy_id: str, user_id: str = Depends(get_current_user)):
    """Follow (copy) a public strategy."""
    db = get_database_manager()
    session = db._session_factory()
    try:
        strategy = session.query(SavedStrategyORM).filter(
            SavedStrategyORM.id == strategy_id,
            SavedStrategyORM.is_public == True,
        ).first()
        if not strategy:
            raise HTTPException(status_code=404, detail="Public strategy not found")
        if strategy.user_id == user_id:
            raise HTTPException(status_code=400, detail="Cannot follow your own strategy")

        existing = session.query(StrategyFollowORM).filter(
            StrategyFollowORM.follower_user_id == user_id,
            StrategyFollowORM.strategy_id == strategy_id,
        ).first()
        if existing:
            return {"status": "already_following"}

        follow = StrategyFollowORM(
            id=str(uuid.uuid4()),
            follower_user_id=user_id,
            strategy_id=strategy_id,
            created_at=datetime.utcnow(),
        )
        session.add(follow)
        strategy.fork_count = strategy.fork_count + 1
        session.commit()
        return {"status": "following"}
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail="Failed to follow strategy")
    finally:
        session.close()


@router.delete("/{strategy_id}/follow")
async def unfollow_strategy(strategy_id: str, user_id: str = Depends(get_current_user)):
    """Unfollow a strategy."""
    db = get_database_manager()
    session = db._session_factory()
    try:
        follow = session.query(StrategyFollowORM).filter(
            StrategyFollowORM.follower_user_id == user_id,
            StrategyFollowORM.strategy_id == strategy_id,
        ).first()
        if follow:
            session.delete(follow)
            strategy = session.query(SavedStrategyORM).filter(SavedStrategyORM.id == strategy_id).first()
            if strategy and strategy.fork_count > 0:
                strategy.fork_count = strategy.fork_count - 1
            session.commit()
        return {"status": "unfollowed"}
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail="Failed to unfollow strategy")
    finally:
        session.close()


@router.post("/{strategy_id}/fork", response_model=StrategyResponse)
async def fork_strategy(strategy_id: str, user_id: str = Depends(get_current_user)):
    """Fork (copy) a public strategy into your own collection."""
    db = get_database_manager()
    session = db._session_factory()
    try:
        original = session.query(SavedStrategyORM).filter(
            SavedStrategyORM.id == strategy_id,
            SavedStrategyORM.is_public == True,
        ).first()
        if not original:
            raise HTTPException(status_code=404, detail="Public strategy not found")

        forked = SavedStrategyORM(
            id=str(uuid.uuid4()),
            user_id=user_id,
            name=f"{original.name} (forked)",
            description=original.description,
            strategy_type=original.strategy_type,
            parameters=original.parameters,
            symbol=original.symbol,
            is_public=False,
            fork_count=0,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        session.add(forked)
        original.fork_count = original.fork_count + 1
        session.commit()
        session.refresh(forked)
        return _to_response(forked, user_id, session)
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail="Failed to fork strategy")
    finally:
        session.close()


# --- Helpers ---

def _to_response(strategy: SavedStrategyORM, current_user_id: str, session) -> StrategyResponse:
    author = session.query(UserORM).filter(UserORM.id == strategy.user_id).first()
    follower_count = session.query(StrategyFollowORM).filter(
        StrategyFollowORM.strategy_id == strategy.id
    ).count()
    is_following = session.query(StrategyFollowORM).filter(
        StrategyFollowORM.follower_user_id == current_user_id,
        StrategyFollowORM.strategy_id == strategy.id,
    ).first() is not None

    best_result = session.query(BacktestResultORM).filter(
        BacktestResultORM.strategy_id == strategy.id
    ).order_by(BacktestResultORM.total_return_pct.desc()).first()

    return StrategyResponse(
        id=strategy.id,
        user_id=strategy.user_id,
        author_name=f"{author.first_name} {author.last_name[0]}." if author else "Unknown",
        name=strategy.name,
        description=strategy.description,
        strategy_type=strategy.strategy_type,
        parameters=strategy.parameters or {},
        symbol=strategy.symbol,
        is_public=strategy.is_public,
        fork_count=strategy.fork_count,
        follower_count=follower_count,
        best_return_pct=float(best_result.total_return_pct) if best_result else None,
        best_sharpe=float(best_result.sharpe_ratio) if best_result else None,
        created_at=strategy.created_at.isoformat() if strategy.created_at else "",
        is_following=is_following,
    )


def _result_to_response(result: BacktestResultORM) -> BacktestResultResponse:
    return BacktestResultResponse(
        id=result.id,
        strategy_id=result.strategy_id,
        symbol=result.symbol,
        initial_capital=float(result.initial_capital),
        final_value=float(result.final_value),
        total_return_pct=float(result.total_return_pct),
        sharpe_ratio=float(result.sharpe_ratio),
        max_drawdown_pct=float(result.max_drawdown_pct),
        win_rate=float(result.win_rate),
        total_trades=result.total_trades,
        volatility=float(result.volatility),
        profit_factor=float(result.profit_factor),
        run_at=result.run_at.isoformat() if result.run_at else "",
    )
