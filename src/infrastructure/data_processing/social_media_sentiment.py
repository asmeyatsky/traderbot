"""
Real-Time Social Media Sentiment Analysis

This module implements real-time sentiment analysis from social media
platforms including Twitter/X, Reddit, and StockTwits for trading signals.
"""
import asyncio
import aiohttp
import tweepy  # Twitter API
import praw  # Reddit API
import requests
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime, timedelta
import re
import time
from collections import deque, defaultdict
import threading

from src.domain.value_objects import Symbol, NewsSentiment
from src.infrastructure.data_processing.advanced_nlp import AdvancedNLPProcessor, news_classifier
from src.infrastructure.config.settings import settings


class SocialMediaStreamListener:
    """
    Real-time streaming listener for social media platforms.
    """
    
    def __init__(self, nlp_processor: AdvancedNLPProcessor):
        self.nlp_processor = nlp_processor
        self.tweets_queue = deque(maxlen=1000)  # Store recent tweets
        self.reddit_posts_queue = deque(maxlen=1000)  # Store recent Reddit posts
        self.stocktwits_posts_queue = deque(maxlen=1000)  # Store recent StockTwits posts
        self.active = False
        self.streaming_thread = None
        self.callbacks = []  # Functions to call when new data arrives
    
    def add_callback(self, callback_func):
        """Add a callback function to be called when new data arrives."""
        self.callbacks.append(callback_func)
    
    def start_streaming(self, symbols: List[Symbol]):
        """Start streaming social media data."""
        self.active = True
        self.streaming_thread = threading.Thread(
            target=self._streaming_worker, 
            args=(symbols,),
            daemon=True
        )
        self.streaming_thread.start()
    
    def stop_streaming(self):
        """Stop streaming social media data."""
        self.active = False
        if self.streaming_thread:
            self.streaming_thread.join()
    
    def _streaming_worker(self, symbols: List[Symbol]):
        """Worker function that runs in background thread."""
        while self.active:
            try:
                # Fetch from different platforms
                self._fetch_twitter_data(symbols)
                self._fetch_reddit_data(symbols)
                self._fetch_stocktwits_data(symbols)
                
                # Wait before next fetch
                time.sleep(30)  # Fetch every 30 seconds
                
            except Exception as e:
                print(f"Error in social media streaming: {e}")
                time.sleep(60)  # Wait longer if there's an error
    
    def _fetch_twitter_data(self, symbols: List[Symbol]):
        """Fetch real-time data from Twitter/X."""
        try:
            # In a real implementation, we would use Twitter API
            # For now, we'll simulate data fetching
            for symbol in symbols[:3]:  # Limit for demo
                # Simulate fetching tweets mentioning the symbol
                tweets = self._simulate_twitter_data(str(symbol))
                
                for tweet in tweets:
                    self.tweets_queue.append(tweet)
                    
                    # Call callbacks with new data
                    for callback in self.callbacks:
                        callback('twitter', tweet)
                        
        except Exception as e:
            print(f"Error fetching Twitter data: {e}")
    
    def _fetch_reddit_data(self, symbols: List[Symbol]):
        """Fetch real-time data from Reddit."""
        try:
            # In a real implementation, we would use Reddit API
            for symbol in symbols[:2]:  # Limit for demo
                # Simulate fetching Reddit posts
                posts = self._simulate_reddit_data(str(symbol))
                
                for post in posts:
                    self.reddit_posts_queue.append(post)
                    
                    # Call callbacks with new data
                    for callback in self.callbacks:
                        callback('reddit', post)
                        
        except Exception as e:
            print(f"Error fetching Reddit data: {e}")
    
    def _fetch_stocktwits_data(self, symbols: List[Symbol]):
        """Fetch real-time data from StockTwits."""
        try:
            # In a real implementation, we would use StockTwits API
            for symbol in symbols[:3]:
                # Simulate fetching StockTwits posts
                posts = self._simulate_stocktwits_data(str(symbol))
                
                for post in posts:
                    self.stocktwits_posts_queue.append(post)
                    
                    # Call callbacks with new data
                    for callback in self.callbacks:
                        callback('stocktwits', post)
                        
        except Exception as e:
            print(f"Error fetching StockTwits data: {e}")
    
    def _simulate_twitter_data(self, symbol: str) -> List[Dict[str, Any]]:
        """Simulate Twitter data for demo purposes."""
        import random
        
        # Create mock tweets mentioning the symbol
        mock_tweets = [
            f"Love the upside potential in ${symbol}! The fundamentals are solid and growth prospects look great!",
            f"Considering adding ${symbol} to my portfolio. Their Q3 earnings were impressive.",
            f"Is ${symbol} overvalued? The P/E ratio seems high compared to sector peers.",
            f"Big news for ${symbol} today! Their new product launch exceeded expectations.",
            f"Downside risk for ${symbol} if interest rates continue to rise. Monitor closely.",
        ]
        
        tweets = []
        for tweet_text in random.sample(mock_tweets, min(2, len(mock_tweets))):
            tweets.append({
                'id': f"tweet_{random.randint(1000, 9999)}",
                'text': tweet_text,
                'timestamp': datetime.now(),
                'user': f"user{random.randint(1000, 9999)}",
                'retweets': random.randint(0, 100),
                'likes': random.randint(0, 500),
                'symbols': [symbol]
            })
        
        return tweets
    
    def _simulate_reddit_data(self, symbol: str) -> List[Dict[str, Any]]:
        """Simulate Reddit data for demo purposes."""
        import random
        
        mock_posts = [
            f"Daily discussion thread: What's your take on ${symbol} after today's moves?",
            f"Analysis: Why ${symbol} could be the next 10-bagger in the making",
            f"Warning signs for ${symbol}: Revenue growth slowing down?",
            f"Woke up to ${symbol} news, can't tell if it's good or bad, help!",
            f"Portfolio update: Added to my position in ${symbol} at these levels",
        ]
        
        posts = []
        for post_text in random.sample(mock_posts, min(1, len(mock_posts))):
            posts.append({
                'id': f"reddit_{random.randint(1000, 9999)}",
                'title': post_text[:50] + "..." if len(post_text) > 50 else post_text,
                'text': post_text,
                'timestamp': datetime.now(),
                'subreddit': 'wallstreetbets' if random.random() > 0.5 else 'stocks',
                'upvotes': random.randint(10, 1000),
                'comments': random.randint(5, 50),
                'symbols': [symbol]
            })
        
        return posts
    
    def _simulate_stocktwits_data(self, symbol: str) -> List[Dict[str, Any]]:
        """Simulate StockTwits data for demo purposes."""
        import random
        
        mock_posts = [
            f"${symbol} breaking above resistance! Watch for continuation.",
            f"Fade the rally in ${symbol}, volume not supporting the move.",
            f"${symbol} earnings play for next week. Positioned.",
            f"Technical analysis: ${symbol} forming a bullish pattern.",
            f"Concerned about ${symbol} after management commentary on call.",
        ]
        
        posts = []
        for post_text in random.sample(mock_posts, min(3, len(mock_posts))):
            posts.append({
                'id': f"stocktwits_{random.randint(1000, 9999)}",
                'message': post_text,
                'timestamp': datetime.now(),
                'user': f"investor{random.randint(1, 99)}",
                'likes': random.randint(0, 50),
                'symbols': [symbol],
                'sentiment': 'bullish' if random.random() > 0.5 else 'bearish'
            })
        
        return posts


class SocialMediaSentimentAnalyzer:
    """
    Analyze sentiment from social media posts for trading signals.
    """
    
    def __init__(self, nlp_processor: AdvancedNLPProcessor):
        self.nlp_processor = nlp_processor
        self.sentiment_history = defaultdict(list)  # Store sentiment history by symbol
        self.engagement_weights = {
            'likes': 0.3,
            'retweets': 0.5,
            'comments': 0.4,
            'upvotes': 0.5
        }
    
    def analyze_post_sentiment(self, post: Dict[str, Any], platform: str) -> NewsSentiment:
        """
        Analyze sentiment of a single social media post.
        """
        # Extract text based on platform
        if platform == 'twitter':
            text = post.get('text', '')
        elif platform == 'reddit':
            title = post.get('title', '')
            text_content = post.get('text', '')
            text = f"{title} {text_content}"
        elif platform == 'stocktwits':
            text = post.get('message', '')
        else:
            text = post.get('text', post.get('content', ''))
        
        # Analyze base sentiment
        base_sentiment = self.nlp_processor.analyze_sentiment_advanced(text)
        
        # Apply engagement weighting
        engagement_score = self._calculate_engagement_score(post, platform)
        
        # Adjust sentiment based on engagement
        adjusted_score = base_sentiment.score * (1 + engagement_score * 0.2)  # Up to 20% adjustment
        adjusted_confidence = min(100, base_sentiment.confidence + engagement_score * 10)
        
        return NewsSentiment(
            score=adjusted_score,
            confidence=adjusted_confidence,
            source=f"SocialMedia_{platform}"
        )
    
    def _calculate_engagement_score(self, post: Dict[str, Any], platform: str) -> float:
        """
        Calculate engagement-based weighting for a post.
        """
        engagement = 0.0
        
        # Add engagement metrics based on platform
        if platform == 'twitter':
            engagement += post.get('likes', 0) * self.engagement_weights['likes'] * 0.01
            engagement += post.get('retweets', 0) * self.engagement_weights['retweets'] * 0.01
        elif platform == 'reddit':
            engagement += post.get('upvotes', 0) * self.engagement_weights['upvotes'] * 0.001
            engagement += post.get('comments', 0) * self.engagement_weights['comments'] * 0.01
        elif platform == 'stocktwits':
            engagement += post.get('likes', 0) * self.engagement_weights['likes'] * 0.02
        
        # Cap engagement score
        return min(engagement, 2.0)  # Maximum 2x weight
    
    def aggregate_sentiment_for_symbol(self, symbol: Symbol, hours: int = 1) -> NewsSentiment:
        """
        Aggregate sentiment for a symbol over a time period.
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        relevant_sentiments = []
        
        # Collect all relevant sentiment data
        if symbol in self.sentiment_history:
            for sentiment_record in self.sentiment_history[symbol]:
                if sentiment_record['timestamp'] > cutoff_time:
                    relevant_sentiments.append(sentiment_record['sentiment'])
        
        if not relevant_sentiments:
            return NewsSentiment(score=0, confidence=0, source="NoData")
        
        # Calculate weighted average
        total_score = 0
        total_weight = 0
        total_confidence = 0
        
        for sentiment in relevant_sentiments:
            # Weight by recency (newer posts have higher weight)
            time_diff = datetime.now() - sentiment_record['timestamp']
            recency_weight = max(0.1, 1.0 - (time_diff.total_seconds() / 3600))  # Decay over hour
            
            total_score += sentiment.score * recency_weight
            total_weight += recency_weight
            total_confidence += sentiment.confidence
        
        if total_weight > 0:
            avg_score = total_score / total_weight
        else:
            avg_score = 0
            
        avg_confidence = total_confidence / len(relevant_sentiments)
        
        return NewsSentiment(
            score=avg_score,
            confidence=min(100, avg_confidence),
            source=f"Aggregated_{hours}h"
        )
    
    def process_social_media_post(self, platform: str, post: Dict[str, Any]):
        """
        Process a social media post and update sentiment history.
        """
        # Analyze sentiment
        sentiment = self.analyze_post_sentiment(post, platform)
        
        # Extract symbols from post
        post_symbols = post.get('symbols', [])
        if not post_symbols:
            # If no symbols explicitly provided, try to extract from text
            text = post.get('text', post.get('message', post.get('content', '')))
            entities = self.nlp_processor.extract_financial_entities(text)
            post_symbols = [str(s) for s in entities.symbols]
        
        # Update sentiment history for each symbol in the post
        for symbol_str in post_symbols:
            symbol = Symbol(symbol_str)
            
            sentiment_record = {
                'sentiment': sentiment,
                'timestamp': post.get('timestamp', datetime.now()),
                'platform': platform,
                'post_id': post.get('id', 'unknown')
            }
            
            self.sentiment_history[symbol].append(sentiment_record)
            
            # Keep only recent data (last 24 hours)
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.sentiment_history[symbol] = [
                record for record in self.sentiment_history[symbol]
                if record['timestamp'] > cutoff_time
            ]


class SocialMediaTradingSignalGenerator:
    """
    Generate trading signals based on social media sentiment.
    """
    
    def __init__(self, sentiment_analyzer: SocialMediaSentimentAnalyzer):
        self.sentiment_analyzer = sentiment_analyzer
        self.signal_thresholds = {
            'strong_buy': 75,
            'buy': 50,
            'strong_sell': -75,
            'sell': -50
        }
        self.min_volume_threshold = 10  # Minimum posts to generate signal
    
    def generate_trading_signals(self, symbol: Symbol) -> Dict[str, Any]:
        """
        Generate trading signals for a symbol based on social media sentiment.
        """
        # Get aggregated sentiment
        current_sentiment = self.sentiment_analyzer.aggregate_sentiment_for_symbol(symbol)
        
        # Get sentiment over different time frames
        sentiment_1h = self.sentiment_analyzer.aggregate_sentiment_for_symbol(symbol, hours=1)
        sentiment_6h = self.sentiment_analyzer.aggregate_sentiment_for_symbol(symbol, hours=6)
        sentiment_24h = self.sentiment_analyzer.aggregate_sentiment_for_symbol(symbol, hours=24)
        
        # Count recent posts for this symbol
        recent_posts_count = len([
            record for record in self.sentiment_analyzer.sentiment_history[symbol]
            if record['timestamp'] > datetime.now() - timedelta(hours=1)
        ])
        
        # Generate signal based on current sentiment and trends
        if current_sentiment.score > self.signal_thresholds['strong_buy']:
            signal = 'STRONG_BUY'
        elif current_sentiment.score > self.signal_thresholds['buy']:
            signal = 'BUY'
        elif current_sentiment.score < self.signal_thresholds['strong_sell']:
            signal = 'STRONG_SELL'
        elif current_sentiment.score < self.signal_thresholds['sell']:
            signal = 'SELL'
        else:
            signal = 'HOLD'
        
        # Adjust signal based on trend consistency
        if signal in ['BUY', 'STRONG_BUY']:
            if sentiment_1h.score > sentiment_6h.score > sentiment_24h.score:
                signal = f"STRONG_{signal}" if "STRONG" not in signal else signal
        elif signal in ['SELL', 'STRONG_SELL']:
            if sentiment_1h.score < sentiment_6h.score < sentiment_24h.score:
                signal = f"STRONG_{signal}" if "STRONG" not in signal else signal
        
        # Suppress signal if insufficient volume
        if recent_posts_count < self.min_volume_threshold:
            signal = 'HOLD'
            confidence_adj = 0.5  # Reduce confidence due to low volume
        else:
            confidence_adj = 1.0
        
        # Adjust confidence based on various factors
        adjusted_confidence = min(100, current_sentiment.confidence * confidence_adj)
        
        return {
            'symbol': symbol,
            'signal': signal,
            'confidence': adjusted_confidence,
            'current_sentiment': current_sentiment.score,
            'sentiment_trend': {
                '1h': sentiment_1h.score,
                '6h': sentiment_6h.score,
                '24h': sentiment_24h.score
            },
            'volume': recent_posts_count,
            'timestamp': datetime.now()
        }


class SocialMediaDataCollector:
    """
    Collect and process social media data from multiple platforms.
    """
    
    def __init__(self, nlp_processor: AdvancedNLPProcessor):
        self.nlp_processor = nlp_processor
        self.stream_listener = SocialMediaStreamListener(nlp_processor)
        self.sentiment_analyzer = SocialMediaSentimentAnalyzer(nlp_processor)
        self.signal_generator = SocialMediaTradingSignalGenerator(self.sentiment_analyzer)
        
        # Register callback to process incoming data
        self.stream_listener.add_callback(self._process_incoming_data)
    
    def _process_incoming_data(self, platform: str, post: Dict[str, Any]):
        """
        Process incoming social media data.
        """
        self.sentiment_analyzer.process_social_media_post(platform, post)
    
    def start_monitoring(self, symbols: List[Symbol]):
        """
        Start monitoring social media for the given symbols.
        """
        self.stream_listener.start_streaming(symbols)
    
    def stop_monitoring(self):
        """
        Stop monitoring social media.
        """
        self.stream_listener.stop_streaming()
    
    def get_current_sentiment(self, symbol: Symbol) -> NewsSentiment:
        """
        Get current aggregated sentiment for a symbol.
        """
        return self.sentiment_analyzer.aggregate_sentiment_for_symbol(symbol)
    
    def get_trading_signal(self, symbol: Symbol) -> Dict[str, Any]:
        """
        Get trading signal for a symbol based on social media.
        """
        return self.signal_generator.generate_trading_signals(symbol)
    
    def get_sentiment_trend(self, symbol: Symbol, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get sentiment trend for a symbol over time.
        """
        records = self.sentiment_analyzer.sentiment_history.get(symbol, [])
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        trend_data = []
        for record in records:
            if record['timestamp'] > cutoff_time:
                trend_data.append({
                    'timestamp': record['timestamp'],
                    'sentiment': record['sentiment'].score,
                    'confidence': record['sentiment'].confidence,
                    'platform': record['platform']
                })
        
        # Sort by timestamp
        trend_data.sort(key=lambda x: x['timestamp'])
        return trend_data


class DefaultSocialMediaService:
    """
    Default implementation of social media sentiment service.
    """
    
    def __init__(self):
        self.nlp_processor = AdvancedNLPProcessor()
        self.data_collector = SocialMediaDataCollector(self.nlp_processor)
    
    def start_social_media_monitoring(self, symbols: List[Symbol]):
        """
        Start monitoring social media for the given symbols.
        """
        self.data_collector.start_monitoring(symbols)
    
    def stop_social_media_monitoring(self):
        """
        Stop monitoring social media.
        """
        self.data_collector.stop_monitoring()
    
    def get_social_sentiment(self, symbol: Symbol) -> NewsSentiment:
        """
        Get social media sentiment for a symbol.
        """
        return self.data_collector.get_current_sentiment(symbol)
    
    def get_social_trading_signal(self, symbol: Symbol) -> Dict[str, Any]:
        """
        Get trading signal based on social media sentiment.
        """
        return self.data_collector.get_trading_signal(symbol)
    
    def get_sentiment_trend(self, symbol: Symbol, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get sentiment trend for a symbol.
        """
        return self.data_collector.get_sentiment_trend(symbol, hours)


# Initialize the social media service
social_media_service = DefaultSocialMediaService()