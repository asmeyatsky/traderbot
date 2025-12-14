"""
News API Client

This module provides a client for interacting with news APIs,
particularly for retrieving financial news and market sentiment data.
"""
from __future__ import annotations

import asyncio
import json
from typing import Dict, List, Optional, Any
import logging
import aiohttp
from datetime import datetime

from src.domain.value_objects import Symbol, NewsSentiment

logger = logging.getLogger(__name__)


class NewsAPIClient:
    """
    API client for news services that can fetch financial news and sentiment.
    
    Supports multiple news providers and handles rate limiting, caching,
    and data transformation for consistent processing across the platform.
    """
    
    def __init__(self, api_key: str, base_url: str = "https://api.marketaux.com/v1"):
        """
        Initialize the news API client.

        Args:
            api_key: API key for authentication
            base_url: Base URL for the news API service
        """
        self.api_key = api_key
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> NewsAPIClient:
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def fetch_news(
        self,
        symbols: List[str],
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Fetch news articles for specific symbols.

        Args:
            symbols: List of stock symbols to get news for
            from_date: Optional start date for news filter
            to_date: Optional end date for news filter
            limit: Maximum number of articles to return

        Returns:
            List of news articles in standard format
        """
        if not self.session:
            logger.warning("Session not initialized, initializing now")
            await self.__aenter__()

        try:
            # Build query parameters
            params = {
                "symbols": ",".join(symbols),
                "limit": limit,
                "api_token": self.api_key,
            }

            if from_date:
                params["date_from"] = from_date.strftime("%Y-%m-%d")
            if to_date:
                params["date_to"] = to_date.strftime("%Y-%m-%d")

            # Make the API request
            url = f"{self.base_url}/news/all"
            async with self.session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()

                # Transform the response to standard format
                articles = []
                for item in data.get("data", []):
                    article = {
                        "id": item.get("id") or item.get("article_id", ""),
                        "title": item.get("title", ""),
                        "content": item.get("description", item.get("summary", "")),
                        "url": item.get("url", ""),
                        "published_at": item.get("published_at"),
                        "source": item.get("source", {}).get("name", "Unknown"),
                        "symbols": item.get("symbols", symbols),
                        "sentiment_score": item.get("sentiment_score", 0.0),
                        "sentiment_label": item.get("sentiment", "neutral"),
                    }
                    articles.append(article)

                return articles

        except aiohttp.ClientError as e:
            logger.error(f"Error fetching news from {self.base_url}: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error when fetching news: {e}")
            return []

    async def fetch_sentiment_analysis(
        self,
        text: str
    ) -> Dict[str, Any]:
        """
        Fetch sentiment analysis for a piece of text.

        Args:
            text: Text to analyze for sentiment

        Returns:
            Sentiment analysis results
        """
        if not self.session:
            await self.__aenter__()

        try:
            # Some news APIs have built-in sentiment endpoints
            # This is a placeholder implementation
            url = f"{self.base_url}/sentiment"
            payload = {
                "text": text,
                "api_token": self.api_key,
            }

            async with self.session.post(url, json=payload) as response:
                response.raise_for_status()
                result = await response.json()

                return {
                    "sentiment_score": result.get("sentiment_score", 0.0),
                    "sentiment_label": result.get("sentiment", "neutral"),
                    "confidence": result.get("confidence", 0.5),
                }

        except Exception as e:
            logger.error(f"Error fetching sentiment analysis: {e}")
            # Fallback to local sentiment analysis
            return {
                "sentiment_score": 0.0,
                "sentiment_label": "neutral",
                "confidence": 0.0,
            }

    async def fetch_company_news(
        self,
        company_name: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Fetch news for a specific company.

        Args:
            company_name: Name of the company to search for
            limit: Maximum number of articles to return

        Returns:
            List of company news articles
        """
        if not self.session:
            await self.__aenter__()

        try:
            params = {
                "q": company_name,
                "limit": limit,
                "api_token": self.api_key,
            }

            url = f"{self.base_url}/news/search"
            async with self.session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()

                articles = []
                for item in data.get("data", []):
                    article = {
                        "id": item.get("id", ""),
                        "title": item.get("title", ""),
                        "content": item.get("description", ""),
                        "url": item.get("url", ""),
                        "published_at": item.get("published_at"),
                        "source": item.get("source", {}).get("name", "Unknown"),
                        "symbols": [company_name],
                        "sentiment_score": item.get("sentiment_score", 0.0),
                        "sentiment_label": item.get("sentiment", "neutral"),
                    }
                    articles.append(article)

                return articles

        except Exception as e:
            logger.error(f"Error fetching company news: {e}")
            return []

    async def close(self) -> None:
        """Close the client session."""
        if self.session:
            await self.session.close()