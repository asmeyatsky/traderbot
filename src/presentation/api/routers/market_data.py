"""
Enhanced Market Data API Router

This router handles all enhanced market data-related endpoints following RESTful principles.
All endpoints require authentication.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import List, Optional, Dict, Any
import logging
from decimal import Decimal

from src.infrastructure.security import get_current_user
from src.domain.services.market_data_enhancement import (
    DefaultMarketDataEnhancementService, MarketDataSource, EnhancedMarketData
)
from src.infrastructure.di_container import container

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/market-data", tags=["market-data"])


@router.get(
    "/enhanced/{symbol}",
    summary="Get enhanced market data for a symbol",
    responses={
        200: {"description": "Enhanced market data retrieved successfully"},
        401: {"description": "Unauthorized"},
        404: {"description": "Symbol not found"},
    }
)
async def get_enhanced_market_data(
    symbol: str,
    include_news: bool = Query(True, description="Include news sentiment"),
    include_technical: bool = Query(True, description="Include technical signals"),
    include_economic: bool = Query(True, description="Include economic events"),
    days: int = Query(30, description="Number of days for historical data"),
    current_user_id: str = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get enhanced market data for a symbol including sentiment, volatility forecast,
    technical signals, and economic events.

    Returns:
        Enhanced market data with multiple data sources and analysis
    """
    try:
        # Get the market data enhancement service from DI container
        market_data_service = container.market_data_enhancement_service()
        
        # Get enhanced market data
        enhanced_data = market_data_service.get_enhanced_market_data(
            symbol=Symbol(symbol)
        )
        
        # Convert to JSON-serializable format
        result = {
            "symbol": str(enhanced_data.symbol),
            "current_price": {
                "amount": float(enhanced_data.current_price.amount),
                "currency": enhanced_data.current_price.currency
            },
            "volatility_forecast": float(enhanced_data.volatility_forecast) if enhanced_data.volatility_forecast else None,
            "calculated_at": datetime.now().isoformat()
        }
        
        # Include market data points if needed
        result["historical_data"] = [
            {
                "price": float(point.price.amount),
                "volume": point.volume,
                "timestamp": point.timestamp.isoformat(),
                "high": float(point.high.amount) if point.high else None,
                "low": float(point.low.amount) if point.low else None,
                "open": float(point.open.amount) if point.open else None,
                "close": float(point.close.amount) if point.close else None,
                "change": float(point.change) if point.change else None,
                "change_percent": float(point.change_percent) if point.change_percent else None
            }
            for point in enhanced_data.market_data_points[-days:]
        ]
        
        # Include news sentiment if requested
        if include_news and enhanced_data.news_sentiment:
            result["news_sentiment"] = [
                {
                    "id": article.id,
                    "title": article.title,
                    "summary": article.summary,
                    "source": article.source,
                    "published_at": article.published_at.isoformat(),
                    "symbols": [str(sym) for sym in article.symbols],
                    "sentiment": {
                        "score": float(article.sentiment.score),
                        "confidence": float(article.sentiment.confidence),
                        "source": article.sentiment.source
                    },
                    "relevance_score": float(article.relevance_score)
                }
                for article in enhanced_data.news_sentiment
            ]
        
        # Include technical signals if requested
        if include_technical and enhanced_data.technical_signals:
            result["technical_signals"] = enhanced_data.technical_signals
        
        # Include economic events if requested
        if include_economic and enhanced_data.economic_events:
            result["economic_events"] = [
                {
                    "event_id": event.event_id,
                    "event_name": event.event_name,
                    "country": event.country,
                    "date": event.date.isoformat(),
                    "impact_level": event.impact_level,
                    "forecast_value": event.forecast_value,
                    "previous_value": event.previous_value,
                    "actual_value": event.actual_value,
                    "currency": event.currency
                }
                for event in enhanced_data.economic_events
            ]
        
        logger.info(f"Enhanced market data retrieved for symbol {symbol}")
        return result
        
    except Exception as e:
        logger.error(f"Error retrieving enhanced market data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve enhanced market data"
        )


@router.get(
    "/sentiment/{symbol}",
    summary="Get news sentiment for a symbol",
    responses={
        200: {"description": "News sentiment retrieved successfully"},
        401: {"description": "Unauthorized"},
        404: {"description": "Symbol not found"},
    }
)
async def get_news_sentiment(
    symbol: str,
    days: int = Query(7, description="Number of days for news"),
    min_relevance: float = Query(70.0, description="Minimum relevance score (0-100)"),
    current_user_id: str = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get news sentiment for a symbol with filtering options.

    Returns:
        News articles with sentiment analysis
    """
    try:
        market_data_service = container.market_data_enhancement_service()
        
        # Get all news sentiment for the symbol
        all_news = market_data_service.get_news_sentiment(Symbol(symbol), days)
        
        # Filter by relevance
        filtered_news = [
            article for article in all_news 
            if float(article.relevance_score) >= min_relevance
        ]
        
        # Calculate aggregate sentiment
        if filtered_news:
            total_sentiment = sum(float(article.sentiment.score) for article in filtered_news)
            avg_sentiment = total_sentiment / len(filtered_news)
            
            positive_articles = [a for a in filtered_news if float(a.sentiment.score) > 0]
            negative_articles = [a for a in filtered_news if float(a.sentiment.score) < 0]
        else:
            avg_sentiment = 0.0
            positive_articles = []
            negative_articles = []
        
        result = {
            "symbol": symbol,
            "total_articles": len(filtered_news),
            "positive_articles": len(positive_articles),
            "negative_articles": len(negative_articles),
            "neutral_articles": len(filtered_news) - len(positive_articles) - len(negative_articles),
            "average_sentiment": avg_sentiment,
            "articles": [
                {
                    "id": article.id,
                    "title": article.title,
                    "summary": article.summary,
                    "source": article.source,
                    "published_at": article.published_at.isoformat(),
                    "sentiment": {
                        "score": float(article.sentiment.score),
                        "confidence": float(article.sentiment.confidence),
                        "source": article.sentiment.source
                    },
                    "relevance_score": float(article.relevance_score)
                }
                for article in filtered_news
            ],
            "calculated_at": datetime.now().isoformat()
        }
        
        logger.info(f"News sentiment retrieved for symbol {symbol}")
        return result
        
    except Exception as e:
        logger.error(f"Error retrieving news sentiment: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve news sentiment"
        )


@router.get(
    "/economic-calendar",
    summary="Get economic calendar events",
    responses={
        200: {"description": "Economic calendar retrieved successfully"},
        401: {"description": "Unauthorized"},
    }
)
async def get_economic_calendar(
    days_ahead: int = Query(7, description="Number of days ahead to retrieve events"),
    impact_level: Optional[str] = Query(None, description="Filter by impact level (low, medium, high)"),
    country: Optional[str] = Query(None, description="Filter by country"),
    current_user_id: str = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get economic calendar events with filtering options.

    Returns:
        Economic events with details and impact levels
    """
    try:
        market_data_service = container.market_data_enhancement_service()
        
        # Get economic events
        events = market_data_service.get_economic_calendar(days_ahead)
        
        # Apply filters
        if impact_level:
            events = [event for event in events if event.impact_level.lower() == impact_level.lower()]
        
        if country:
            events = [event for event in events if event.country.lower() == country.lower()]
        
        result = {
            "days_ahead": days_ahead,
            "total_events": len(events),
            "events": [
                {
                    "event_id": event.event_id,
                    "event_name": event.event_name,
                    "country": event.country,
                    "date": event.date.isoformat(),
                    "impact_level": event.impact_level,
                    "forecast_value": event.forecast_value,
                    "previous_value": event.previous_value,
                    "actual_value": event.actual_value,
                    "currency": event.currency
                }
                for event in events
            ],
            "calculated_at": datetime.now().isoformat()
        }
        
        logger.info(f"Economic calendar retrieved for next {days_ahead} days")
        return result
        
    except Exception as e:
        logger.error(f"Error retrieving economic calendar: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve economic calendar"
        )


@router.get(
    "/volatility-forecast/{symbol}",
    summary="Get volatility forecast for a symbol",
    responses={
        200: {"description": "Volatility forecast retrieved successfully"},
        401: {"description": "Unauthorized"},
        404: {"description": "Symbol not found"},
    }
)
async def get_volatility_forecast(
    symbol: str,
    lookback_days: int = Query(30, description="Number of days for lookback calculation"),
    current_user_id: str = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get volatility forecast for a symbol.

    Returns:
        Volatility forecast with confidence intervals
    """
    try:
        market_data_service = container.market_data_enhancement_service()
        
        # Calculate volatility forecast
        volatility = market_data_service.calculate_volatility_forecast(
            Symbol(symbol), lookback_days
        )
        
        # Calculate confidence intervals (mock implementation)
        # In production, this would be calculated using statistical models
        base_vol = float(volatility)
        confidence_lower = base_vol * 0.8  # 80% of base volatility
        confidence_upper = base_vol * 1.2  # 120% of base volatility
        
        result = {
            "symbol": symbol,
            "volatility_forecast": base_vol,
            "volatility_percentage": base_vol * 100,  # Convert to percentage
            "confidence_interval": {
                "lower": confidence_lower,
                "upper": confidence_upper
            },
            "lookback_days": lookback_days,
            "calculated_at": datetime.now().isoformat()
        }
        
        logger.info(f"Volatility forecast retrieved for symbol {symbol}")
        return result
        
    except Exception as e:
        logger.error(f"Error retrieving volatility forecast: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve volatility forecast"
        )

from datetime import datetime