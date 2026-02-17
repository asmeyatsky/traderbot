"""
Trading Activity API Router

Provides endpoints for viewing autonomous trading activity logs.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from typing import Optional
import logging

from src.infrastructure.security import get_current_user
from src.infrastructure.repositories.activity_log_repository import ActivityLogRepository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/trading-activity", tags=["trading-activity"])


def _get_activity_log_repo() -> ActivityLogRepository:
    from src.infrastructure.di_container import container
    return container.repositories.activity_log_repository()


@router.get(
    "",
    summary="Get trading activity log",
    responses={
        200: {"description": "Activity log entries"},
        401: {"description": "Unauthorized"},
    },
)
async def get_trading_activity(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    event_type: Optional[str] = Query(None),
    user_id: str = Depends(get_current_user),
    repo: ActivityLogRepository = Depends(_get_activity_log_repo),
):
    """Paginated list of trading activity events for the current user."""
    try:
        entries = repo.get_recent_activity(
            user_id=user_id, limit=limit, skip=skip, event_type=event_type
        )
        return {"items": entries, "skip": skip, "limit": limit, "count": len(entries)}
    except Exception as e:
        logger.error(f"Error fetching trading activity: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch trading activity",
        )


@router.get(
    "/summary",
    summary="Get trading activity summary",
    responses={
        200: {"description": "Aggregate counts by event type"},
        401: {"description": "Unauthorized"},
    },
)
async def get_trading_activity_summary(
    user_id: str = Depends(get_current_user),
    repo: ActivityLogRepository = Depends(_get_activity_log_repo),
):
    """Return counts of activity events grouped by type."""
    try:
        summary = repo.get_activity_summary(user_id)
        return {"summary": summary}
    except Exception as e:
        logger.error(f"Error fetching activity summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch activity summary",
        )
