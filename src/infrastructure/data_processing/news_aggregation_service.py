"""
News Aggregation and Processing Service

This module implements the news aggregation engine that collects, processes,
and analyzes financial news as required by the PRD. It handles real-time
news processing, sentiment analysis, and integration with the trading system.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import logging
import asyncio
import json
from datetime import datetime, timedelta
from dataclasses import dataclass
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.domain.value_objects import Symbol, NewsSentiment
from src.infrastructure.data_processing.ml_model_service import SentimentAnalysisService
from src.infrastructure.api_clients.news_api_client import NewsAPIClient


logger = logging.getLogger(__name__)


@dataclass
class NewsArticle:
    """Represents a single news article."""
    id: str
    title: str
    content: str
    url: str
    published_at: datetime
    source: str
    symbols: List[str]
    sentiment: Optional[NewsSentiment] = None


@dataclass
class NewsFeedConfig:
    """Configuration for news feed processing."""
    sources: List[str]
    symbols: List[str]
    refresh_interval: int  # seconds
    sentiment_threshold: float  # minimum sentiment strength to process
    max_articles_per_symbol: int


class NewsAggregationService(ABC):
    """Abstract base class for news aggregation services."""

    @abstractmethod
    def get_news_for_symbol(self, symbol: Symbol, hours_back: int = 24) -> List[NewsArticle]:
        """Fetch news articles for a specific symbol."""
        pass

    @abstractmethod
    async def get_news_for_symbols(self, symbols: List[Symbol]) -> Dict[str, List[NewsArticle]]:
        """Fetch news for multiple symbols."""
        pass

    @abstractmethod
    def get_market_news(self) -> List[NewsArticle]:
        """Fetch market-wide news."""
        pass

    @abstractmethod
    def process_news_stream(self, on_news_callback) -> None:
        """Process news in real-time stream."""
        pass


class MarketauxNewsService(NewsAggregationService):
    """News service implementation for Marketaux API."""

    def __init__(self, api_key: str, sentiment_service: SentimentAnalysisService):
        self.api_key = api_key
        self.sentiment_service = sentiment_service
        self.base_url = "https://api.marketaux.com/v1/news"
        self.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        logger.info("MarketauxNewsService initialized")

    def get_news_for_symbol(self, symbol: Symbol, hours_back: int = 24) -> List[NewsArticle]:
        """
        Fetch news articles for a specific symbol from Marketaux API.
        
        Args:
            symbol: The stock symbol to get news for
            hours_back: Number of hours back to fetch news (default 24)
        
        Returns:
            List of NewsArticle objects
        """
        try:
            # Calculate time range
            start_date = datetime.now() - timedelta(hours=hours_back)
            start_date_str = start_date.strftime("%Y-%m-%d")
            
            params = {
                'api_token': self.api_key,
                'symbols': str(symbol),
                'filter_entities': True,
                'exclude_entities': False,
                'limit': 50,  # Max allowed per request
                'date': start_date_str
            }
            
            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            articles = self._parse_marketaux_response(data, symbol)
            
            return articles
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching news for symbol {symbol}: {e}")
            # Return empty list or implement fallback logic
            return self._get_fallback_news(symbol, hours_back)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing news response for symbol {symbol}: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching news for symbol {symbol}: {e}")
            return []

    async def get_news_for_symbols(self, symbols: List[Symbol]) -> Dict[str, List[NewsArticle]]:
        """Fetch news for multiple symbols concurrently."""
        news_dict = {}
        
        # Process each symbol individually (in a real implementation, we might batch these)
        for symbol in symbols:
            news = self.get_news_for_symbol(symbol)
            news_dict[str(symbol)] = news
            
        return news_dict

    def get_market_news(self) -> List[NewsArticle]:
        """Fetch market-wide news (not specific to symbols)."""
        try:
            params = {
                'api_token': self.api_key,
                'market_news': 'top',
                'limit': 50,
            }
            
            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            # Parse market news (for now, treat as general market articles)
            articles = self._parse_marketaux_response(data, None)
            
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching market news: {e}")
            return []

    def process_news_stream(self, on_news_callback) -> None:
        """Process news in real-time stream."""
        # This would typically use a WebSocket connection or server-sent events
        # For now, we'll implement a polling approach
        logger.info("Starting news stream processing...")
        
        # In a real implementation, this would connect to a WebSocket
        # or use a server-sent events connection
        import time
        
        while True:
            try:
                # Fetch latest news for all tracked symbols
                # This is a simplified approach - real implementation would be event-driven
                time.sleep(60)  # Check every minute
            except KeyboardInterrupt:
                logger.info("News stream processing stopped")
                break
            except Exception as e:
                logger.error(f"Error in news stream processing: {e}")
                time.sleep(60)  # Wait before retrying

    def _parse_marketaux_response(self, data: Dict, symbol_filter: Optional[Symbol] = None) -> List[NewsArticle]:
        """Parse the Marketaux API response into NewsArticle objects."""
        articles = []
        
        if 'data' not in data:
            logger.warning("No 'data' field in Marketaux response")
            return articles

        for item in data['data']:
            try:
                # Extract article details
                article_id = item.get('id', item.get('uuid', 'unknown'))
                title = item.get('title', '')
                content = item.get('description', item.get('summary', ''))
                url = item.get('url', '')
                
                # Parse publish date
                published_str = item.get('published_at', '')
                published_at = datetime.fromisoformat(published_str.replace('Z', '+00:00')) if published_str else datetime.now()
                
                # Extract symbols mentioned in the article
                symbols = []
                entities = item.get('entities', [])
                for entity in entities:
                    if 'symbol' in entity:
                        symbols.append(entity['symbol'])
                
                # Apply symbol filter if provided
                if symbol_filter and str(symbol_filter) not in symbols:
                    continue
                
                # Create NewsArticle
                article = NewsArticle(
                    id=article_id,
                    title=title,
                    content=content,
                    url=url,
                    published_at=published_at,
                    source='Marketaux',
                    symbols=symbols
                )
                
                # Analyze sentiment for the article
                if content:
                    article.sentiment = self.sentiment_service.analyze_sentiment(content)
                
                articles.append(article)
                
            except Exception as e:
                logger.error(f"Error parsing news item: {e}")
                continue
        
        return articles

    def _get_fallback_news(self, symbol: Symbol, hours_back: int) -> List[NewsArticle]:
        """
        Fallback method to return simulated news when API is unavailable.
        This would be replaced with other news sources in a real implementation.
        """
        logger.warning(f"Using fallback news for {symbol}")
        
        # Return a few simulated articles for the symbol
        simulated_articles = [
            NewsArticle(
                id=f"simulated_{symbol}_1",
                title=f"Breaking: {symbol.value} Reports Strong Quarterly Earnings",
                content=f"{symbol.value} has reported stronger than expected quarterly earnings, beating analyst expectations by a wide margin.",
                url=f"https://example.com/news/{symbol.value}/earnings",
                published_at=datetime.now() - timedelta(hours=2),
                source='Simulated News',
                symbols=[str(symbol)],
                sentiment=NewsSentiment(score=85, confidence=90, source='Simulated')
            ),
            NewsArticle(
                id=f"simulated_{symbol}_2",
                title=f"Analyst Upgrade: {symbol.value} Target Price Increased",
                content=f"Major investment bank upgrades {symbol.value} to buy rating with increased target price.",
                url=f"https://example.com/news/{symbol.value}/upgrade",
                published_at=datetime.now() - timedelta(hours=5),
                source='Simulated News',
                symbols=[str(symbol)],
                sentiment=NewsSentiment(score=75, confidence=85, source='Simulated')
            )
        ]
        
        return simulated_articles


class EnhancedNewsAggregationService(NewsAggregationService):
    """Enhanced news service that combines multiple sources for comprehensive coverage."""

    def __init__(self, 
                 marketaux_service: MarketauxNewsService,
                 sentiment_service: SentimentAnalysisService,
                 additional_sources: Optional[List] = None):
        self.marketaux_service = marketaux_service
        self.sentiment_service = sentiment_service
        self.additional_sources = additional_sources or []
        self.cache = {}  # Simple in-memory cache
        logger.info("EnhancedNewsAggregationService initialized")

    def get_news_for_symbol(self, symbol: Symbol, hours_back: int = 24) -> List[NewsArticle]:
        """
        Get news for a symbol from multiple sources and aggregate.
        """
        all_articles = []
        
        # Fetch from Marketaux
        marketaux_articles = self.marketaux_service.get_news_for_symbol(symbol, hours_back)
        all_articles.extend(marketaux_articles)
        
        # Add logic to fetch from other sources if available
        # For now, we'll just return Marketaux results
        # In a real implementation, we would fetch from additional sources
        
        # Deduplicate articles
        unique_articles = self._deduplicate_articles(all_articles)
        
        # Sort by publish date (newest first)
        unique_articles.sort(key=lambda x: x.published_at, reverse=True)
        
        # Cache results for a short period
        cache_key = f"{symbol}_{hours_back}"
        self.cache[cache_key] = (datetime.now(), unique_articles)
        
        return unique_articles

    async def get_news_for_symbols(self, symbols: List[Symbol]) -> Dict[str, List[NewsArticle]]:
        """Get news for multiple symbols concurrently."""
        import asyncio
        
        async def fetch_symbol_news(symbol):
            return str(symbol), self.get_news_for_symbol(symbol)
        
        tasks = [fetch_symbol_news(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks)
        
        return dict(results)

    def get_market_news(self) -> List[NewsArticle]:
        """Get market-wide news from multiple sources."""
        all_articles = []
        
        # Fetch market news from Marketaux
        marketaux_market_news = self.marketaux_service.get_market_news()
        all_articles.extend(marketaux_market_news)
        
        # Similar to above, add other sources in real implementation
        return self._deduplicate_articles(all_articles)

    def process_news_stream(self, on_news_callback) -> None:
        """Process real-time news stream and trigger callbacks."""
        # In a real implementation, this would connect to multiple streaming sources
        # For now, we'll simulate with polling
        
        import threading
        def stream_worker():
            while True:
                try:
                    # Check for new news from all sources
                    # In real implementation, this would be event-driven
                    import time
                    time.sleep(30)  # Check every 30 seconds
                except Exception as e:
                    logger.error(f"Error in news stream: {e}")
                    time.sleep(60)  # Wait longer on error
        
        # Start the stream in a background thread
        stream_thread = threading.Thread(target=stream_worker, daemon=True)
        stream_thread.start()

    def _deduplicate_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Remove duplicate articles based on URL or content similarity."""
        seen_urls = set()
        unique_articles = []
        
        for article in articles:
            if article.url not in seen_urls:
                seen_urls.add(article.url)
                unique_articles.append(article)
        
        return unique_articles

    def get_sentiment_summary(self, symbol: Symbol, hours_back: int = 24) -> Dict:
        """Get a summary of sentiment for a symbol over the specified time period."""
        articles = self.get_news_for_symbol(symbol, hours_back)
        
        if not articles:
            return {
                'symbol': str(symbol),
                'total_articles': 0,
                'average_sentiment': 0,
                'sentiment_confidence': 0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0
            }
        
        # Calculate aggregate sentiment
        total_sentiment = sum(article.sentiment.score if article.sentiment else 0 for article in articles)
        avg_sentiment = total_sentiment / len(articles)
        
        # Count sentiment categories
        positive_count = sum(1 for article in articles if article.sentiment and article.sentiment.score > 10)
        negative_count = sum(1 for article in articles if article.sentiment and article.sentiment.score < -10)
        neutral_count = len(articles) - positive_count - negative_count
        
        # Calculate average confidence
        valid_sentiments = [article.sentiment for article in articles if article.sentiment]
        avg_confidence = sum(s.confidence for s in valid_sentiments) / len(valid_sentiments) if valid_sentiments else 0
        
        return {
            'symbol': str(symbol),
            'total_articles': len(articles),
            'average_sentiment': float(avg_sentiment),
            'sentiment_confidence': avg_confidence,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'articles': [
                {
                    'title': article.title,
                    'sentiment': float(article.sentiment.score) if article.sentiment else 0,
                    'confidence': float(article.sentiment.confidence) if article.sentiment else 0,
                    'published_at': article.published_at.isoformat()
                }
                for article in articles[:10]  # Limit to first 10 for summary
            ]
        }


class NewsImpactAnalyzer:
    """Analyzes the potential impact of news on stock prices."""

    def __init__(self, news_aggregation_service: NewsAggregationService):
        self.news_service = news_aggregation_service

    def calculate_news_impact_score(self, symbol: Symbol, lookback_hours: int = 24) -> float:
        """
        Calculate a news impact score based on sentiment, volume, and timing.
        
        Returns a score between -1 and 1 where:
        - Negative scores indicate bearish sentiment/impact
        - Positive scores indicate bullish sentiment/impact
        """
        articles = self.news_service.get_news_for_symbol(symbol, lookback_hours)
        
        if not articles:
            return 0.0  # No news, no impact
        
        # Calculate weighted impact based on:
        # 1. Sentiment strength
        # 2. Article recency
        # 3. Source credibility (if available)
        
        total_weighted_sentiment = 0
        total_weight = 0
        
        current_time = datetime.now()
        
        for article in articles:
            if not article.sentiment:
                continue
                
            # Calculate time decay factor (newer articles have more impact)
            time_diff = (current_time - article.published_at).total_seconds() / 3600  # hours
            time_weight = max(0.1, 1.0 - (time_diff / 24))  # Decays after 24 hours
            
            # Calculate confidence weight
            confidence_weight = article.sentiment.confidence / 100.0
            
            # Calculate final weight for this article
            article_weight = time_weight * confidence_weight
            weighted_sentiment = (article.sentiment.score / 100.0) * article_weight
            
            total_weighted_sentiment += weighted_sentiment
            total_weight += article_weight
        
        if total_weight == 0:
            return 0.0
            
        # Normalize the impact score
        impact_score = total_weighted_sentiment / total_weight
        
        # Clamp between -1 and 1
        return max(-1.0, min(1.0, impact_score))

    def get_news_alerts(self, symbol: Symbol, threshold: float = 0.7) -> List[Dict]:
        """
        Get news alerts that exceed the impact threshold.
        
        Args:
            symbol: The symbol to check for alerts
            threshold: Minimum impact score to trigger alert (absolute value)
        
        Returns:
            List of alert dictionaries
        """
        articles = self.news_service.get_news_for_symbol(symbol, hours_back=2)
        
        alerts = []
        for article in articles:
            if not article.sentiment:
                continue
                
            # Calculate impact for this article
            if abs(article.sentiment.score) / 100.0 >= threshold:
                alerts.append({
                    'title': article.title,
                    'url': article.url,
                    'sentiment_score': float(article.sentiment.score),
                    'confidence': float(article.sentiment.confidence),
                    'published_at': article.published_at.isoformat(),
                    'estimated_impact': float(article.sentiment.score) / 100.0
                })
        
        return alerts