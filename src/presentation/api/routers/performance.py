"""
Performance Monitoring API Router

This router handles all performance monitoring endpoints following RESTful principles.
All endpoints require authentication.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any
from datetime import datetime
import logging

from src.infrastructure.security import get_current_user
from src.infrastructure.performance_optimization import DefaultPerformanceOptimizerService
from src.infrastructure.di_container import container

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/performance", tags=["performance"])


@router.get(
    "/metrics",
    summary="Get system performance metrics",
    responses={
        200: {"description": "Performance metrics retrieved successfully"},
        401: {"description": "Unauthorized"},
        403: {"description": "Access denied for non-admin users"},
    }
)
async def get_performance_metrics(
    current_user_id: str = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get system performance metrics including cache statistics,
    response times, and system health indicators.

    Returns:
        Performance metrics and statistics
    """
    try:
        # Get the performance optimizer service from DI container
        perf_service = container.services.performance_optimizer_service()
        
        # Calculate performance metrics
        metrics = perf_service.calculate_performance_metrics()
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics
        }
        
        logger.info(f"Performance metrics retrieved for user {current_user_id}")
        return result
        
    except Exception as e:
        logger.error(f"Error retrieving performance metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve performance metrics"
        )


@router.get(
    "/cache-stats",
    summary="Get cache performance statistics",
    responses={
        200: {"description": "Cache statistics retrieved successfully"},
        401: {"description": "Unauthorized"},
        403: {"description": "Access denied for non-admin users"},
    }
)
async def get_cache_stats(
    current_user_id: str = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get cache performance statistics including hit rates and response times.

    Returns:
        Cache statistics and performance metrics
    """
    try:
        perf_service = container.services.performance_optimizer_service()
        
        # Get cache statistics
        stats = perf_service.get_cache_stats()
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "cache_stats": {
                "hits": stats.hits,
                "misses": stats.misses,
                "hit_rate": stats.hit_rate,
                "total_requests": stats.total_requests,
                "average_response_time_ms": stats.average_response_time,
                "tier_distribution": {
                    tier.value: count 
                    for tier, count in stats.tier_distribution.items()
                }
            }
        }
        
        logger.info(f"Cache stats retrieved for user {current_user_id}")
        return result
        
    except Exception as e:
        logger.error(f"Error retrieving cache stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve cache statistics"
        )


@router.post(
    "/cache/warm/{user_id}",
    summary="Warm up user's cache",
    responses={
        200: {"description": "Cache warmed up successfully"},
        401: {"description": "Unauthorized"},
        403: {"description": "Not authorized to warm up other users' cache"},
    }
)
async def warm_cache(
    user_id: str,
    current_user_id: str = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Warm up cache with frequently accessed data for a user.

    Args:
        user_id: User ID whose cache to warm up
        current_user_id: Authenticated user ID (for authorization)

    Returns:
        Operation success status
    """
    if current_user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to warm up other users' cache"
        )
    
    try:
        perf_service = container.services.performance_optimizer_service()
        
        # Warm up the cache
        success = perf_service.warm_cache(user_id)
        
        result = {
            "user_id": user_id,
            "success": success,
            "warmed_at": datetime.now().isoformat()
        }
        
        logger.info(f"Cache warmed up for user {user_id}")
        return result
        
    except Exception as e:
        logger.error(f"Error warming cache: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to warm up cache"
        )