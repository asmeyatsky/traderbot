"""
Alternative Data Integration API Router

This router handles all alternative data integration endpoints following RESTful principles.
All endpoints require authentication.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import List, Optional, Dict, Any
import logging

from src.infrastructure.security import get_current_user
from src.infrastructure.alternative_data_integration import (
    DefaultAlternativeDataIntegrationService, AlternativeDataSource, 
    AlternativeDataInsight
)
from src.domain.value_objects import Symbol

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/alternative-data", tags=["alternative-data"])


@router.get(
    "/satellite/{symbol}",
    summary="Get satellite imagery data for a symbol",
    responses={
        200: {"description": "Satellite data retrieved successfully"},
        401: {"description": "Unauthorized"},
        404: {"description": "Symbol not found"},
    }
)
async def get_satellite_data(
    symbol: str,
    days: int = Query(30, description="Number of days of data to retrieve"),
    measurement_type: Optional[str] = Query(None, description="Filter by measurement type"),
    current_user_id: str = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get satellite imagery data for a symbol showing metrics like parking lot activity,
    building usage, or other physical indicators.

    Returns:
        Satellite data points with confidence scores
    """
    try:
        # Initialize the alternative data service
        alt_data_service = DefaultAlternativeDataIntegrationService()
        
        # Get satellite data
        sat_data = alt_data_service.get_satellite_data(Symbol(symbol), days)
        
        # Filter by measurement type if specified
        if measurement_type:
            sat_data = [d for d in sat_data if d.measurement_type.lower() == measurement_type.lower()]
        
        result = {
            "symbol": symbol,
            "measurement_type": measurement_type,
            "data_points": len(sat_data),
            "satellite_data": [
                {
                    "asset_id": point.asset_id,
                    "latitude": point.latitude,
                    "longitude": point.longitude,
                    "measurement_type": point.measurement_type,
                    "value": float(point.value),
                    "date": point.date.isoformat(),
                    "source": point.source,
                    "confidence": float(point.confidence)
                }
                for point in sat_data
            ],
            "retrieved_at": datetime.now().isoformat()
        }
        
        logger.info(f"Satellite data retrieved for symbol {symbol}")
        return result
        
    except Exception as e:
        logger.error(f"Error retrieving satellite data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve satellite data"
        )


@router.get(
    "/credit-card/{symbol}",
    summary="Get credit card transaction trends for a symbol",
    responses={
        200: {"description": "Credit card trends retrieved successfully"},
        401: {"description": "Unauthorized"},
        404: {"description": "Symbol not found"},
    }
)
async def get_credit_card_trends(
    symbol: str,
    category: Optional[str] = Query(None, description="Filter by merchant category"),
    current_user_id: str = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get credit card transaction trends for a symbol showing consumer spending patterns.

    Returns:
        Credit card trend data with confidence scores
    """
    try:
        alt_data_service = DefaultAlternativeDataIntegrationService()
        
        # Get credit card trends
        trends = alt_data_service.get_credit_card_trends(Symbol(symbol), category)
        
        result = {
            "symbol": symbol,
            "category": category,
            "trend_count": len(trends),
            "credit_card_trends": [
                {
                    "merchant_category": trend.merchant_category,
                    "geographic_region": trend.geographic_region,
                    "trend_value": float(trend.trend_value),
                    "base_period": trend.base_period,
                    "current_period": trend.current_period,
                    "data_point_count": trend.data_point_count,
                    "confidence": float(trend.confidence)
                }
                for trend in trends
            ],
            "retrieved_at": datetime.now().isoformat()
        }
        
        logger.info(f"Credit card trends retrieved for symbol {symbol}")
        return result
        
    except Exception as e:
        logger.error(f"Error retrieving credit card trends: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve credit card trends"
        )


@router.get(
    "/supply-chain/{symbol}",
    summary="Get supply chain events for a symbol",
    responses={
        200: {"description": "Supply chain events retrieved successfully"},
        401: {"description": "Unauthorized"},
        404: {"description": "Symbol not found"},
    }
)
async def get_supply_chain_events(
    symbol: str,
    days: int = Query(30, description="Number of days of events to retrieve"),
    severity: Optional[str] = Query(None, description="Filter by severity level"),
    current_user_id: str = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get supply chain events for a symbol showing disruptions, delays, or other issues.

    Returns:
        Supply chain events with impact assessments
    """
    try:
        alt_data_service = DefaultAlternativeDataIntegrationService()
        
        # Get supply chain events
        events = alt_data_service.get_supply_chain_events(Symbol(symbol), days)
        
        # Filter by severity if specified
        if severity:
            events = [e for e in events if e.severity.lower() == severity.lower()]
        
        result = {
            "symbol": symbol,
            "severity_filter": severity,
            "event_count": len(events),
            "supply_chain_events": [
                {
                    "event_id": event.event_id,
                    "company": event.company,
                    "supplier": event.supplier,
                    "event_type": event.event_type,
                    "severity": event.severity,
                    "estimated_impact": float(event.estimated_impact),
                    "start_date": event.start_date.isoformat(),
                    "estimated_resolution": event.estimated_resolution.isoformat() if event.estimated_resolution else None,
                    "source": event.source
                }
                for event in events
            ],
            "retrieved_at": datetime.now().isoformat()
        }
        
        logger.info(f"Supply chain events retrieved for symbol {symbol}")
        return result
        
    except Exception as e:
        logger.error(f"Error retrieving supply chain events: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve supply chain events"
        )


@router.get(
    "/social-sentiment/{symbol}",
    summary="Get social media sentiment for a symbol",
    responses={
        200: {"description": "Social media sentiment retrieved successfully"},
        401: {"description": "Unauthorized"},
        404: {"description": "Symbol not found"},
    }
)
async def get_social_media_sentiment(
    symbol: str,
    days: int = Query(7, description="Number of days of data to retrieve"),
    platform: Optional[str] = Query(None, description="Filter by social media platform"),
    current_user_id: str = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get social media sentiment for a symbol showing public opinion and discussion trends.

    Returns:
        Social media sentiment data with confidence scores
    """
    try:
        alt_data_service = DefaultAlternativeDataIntegrationService()
        
        # Get social media sentiment
        sentiment_data = alt_data_service.get_social_media_sentiment(Symbol(symbol), days)
        
        # Filter by platform if specified
        if platform:
            sentiment_data = [d for d in sentiment_data if d.platform.lower() == platform.lower()]
        
        # Calculate aggregate metrics
        if sentiment_data:
            avg_sentiment = sum(d.sentiment_score for d in sentiment_data) / len(sentiment_data)
            total_mentions = sum(d.mention_count for d in sentiment_data)
        else:
            avg_sentiment = 0
            total_mentions = 0
        
        result = {
            "symbol": symbol,
            "platform_filter": platform,
            "data_points": len(sentiment_data),
            "total_mentions": total_mentions,
            "average_sentiment": float(avg_sentiment),
            "social_media_sentiment": [
                {
                    "platform": data_point.platform,
                    "mention_count": data_point.mention_count,
                    "sentiment_score": float(data_point.sentiment_score),
                    "topic_category": data_point.topic_category,
                    "date": data_point.date.isoformat(),
                    "sample_size": data_point.sample_size,
                    "confidence": float(data_point.confidence)
                }
                for data_point in sentiment_data
            ],
            "retrieved_at": datetime.now().isoformat()
        }
        
        logger.info(f"Social media sentiment retrieved for symbol {symbol}")
        return result
        
    except Exception as e:
        logger.error(f"Error retrieving social media sentiment: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve social media sentiment"
        )


@router.get(
    "/esg/{symbol}",
    summary="Get ESG scores for a symbol",
    responses={
        200: {"description": "ESG scores retrieved successfully"},
        401: {"description": "Unauthorized"},
        404: {"description": "Symbol not found"},
    }
)
async def get_esg_scores(
    symbol: str,
    current_user_id: str = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get ESG (Environmental, Social, Governance) scores for a symbol.

    Returns:
        ESG scores with trend analysis
    """
    try:
        alt_data_service = DefaultAlternativeDataIntegrationService()
        
        # Get ESG scores
        esg_score = alt_data_service.get_esg_scores(Symbol(symbol))
        
        if not esg_score:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No ESG data found for symbol {symbol}"
            )
        
        result = {
            "symbol": symbol,
            "esg_scores": {
                "environmental_score": float(esg_score.environmental_score),
                "social_score": float(esg_score.social_score),
                "governance_score": float(esg_score.governance_score),
                "overall_esg_score": float(esg_score.overall_esg_score),
                "data_date": esg_score.data_date.isoformat(),
                "source": esg_score.source,
                "trend_direction": esg_score.trend_direction
            },
            "retrieved_at": datetime.now().isoformat()
        }
        
        logger.info(f"ESG scores retrieved for symbol {symbol}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving ESG scores: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve ESG scores"
        )


@router.get(
    "/insights/{symbol}",
    summary="Get alternative data insights for a symbol",
    responses={
        200: {"description": "Alternative data insights retrieved successfully"},
        401: {"description": "Unauthorized"},
        404: {"description": "Symbol not found"},
    }
)
async def get_alternative_data_insights(
    symbol: str,
    current_user_id: str = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get synthesized insights from multiple alternative data sources for a symbol.

    Returns:
        Alternative data insights with impact scores and confidence levels
    """
    try:
        alt_data_service = DefaultAlternativeDataIntegrationService()
        
        # Generate insights
        insights = alt_data_service.generate_alternative_data_insights(Symbol(symbol))
        
        result = {
            "symbol": symbol,
            "insight_count": len(insights),
            "insights": [
                {
                    "insight_type": insight.insight_type,
                    "confidence_level": float(insight.confidence_level),
                    "direction": insight.direction,
                    "impact_score": float(insight.impact_score),
                    "supporting_data": insight.supporting_data,
                    "data_sources": [src.value for src in insight.data_sources],
                    "generated_at": insight.generated_at.isoformat()
                }
                for insight in insights
            ],
            "retrieved_at": datetime.now().isoformat()
        }
        
        logger.info(f"Alternative data insights retrieved for symbol {symbol}")
        return result
        
    except Exception as e:
        logger.error(f"Error generating alternative data insights: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate alternative data insights"
        )

from datetime import datetime