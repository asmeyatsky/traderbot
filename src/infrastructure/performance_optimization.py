"""
Performance Optimization and Advanced Caching Service

Implements performance optimization features and advanced caching strategies
to improve system responsiveness and reduce database load.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, List, Callable
from decimal import Decimal
import asyncio
import time
import functools
from enum import Enum

from src.domain.entities.trading import Portfolio, Position, Order
from src.domain.entities.user import User
from src.domain.value_objects import Symbol
from src.infrastructure.cache_layer import get_cache


class CacheTier(Enum):
    L1_MEMORY = "L1_MEMORY"      # Fastest: In-memory cache
    L2_REDIS = "L2_REDIS"        # Fast: Redis cache
    L3_DATABASE = "L3_DATABASE"  # Slowest: Database cache


@dataclass
class CacheStats:
    """Data class for cache statistics"""
    hits: int
    misses: int
    hit_rate: float
    total_requests: int
    average_response_time: float  # in milliseconds
    tier_distribution: Dict[CacheTier, int]


class PerformanceOptimizerService(ABC):
    """
    Abstract base class for performance optimization services.
    """
    
    @abstractmethod
    def get_cached_portfolio(self, user_id: str) -> Optional[Portfolio]:
        """Get cached portfolio for a user"""
        pass
    
    @abstractmethod
    def cache_portfolio(self, user_id: str, portfolio: Portfolio, ttl: int = 300) -> bool:
        """Cache portfolio for a user"""
        pass
    
    @abstractmethod
    def get_cached_market_data(self, symbol: Symbol) -> Optional[Any]:
        """Get cached market data for a symbol"""
        pass
    
    @abstractmethod
    def cache_market_data(self, symbol: Symbol, data: Any, ttl: int = 60) -> bool:
        """Cache market data for a symbol"""
        pass
    
    @abstractmethod
    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate overall system performance metrics"""
        pass
    
    @abstractmethod
    def warm_cache(self, user_id: str) -> bool:
        """Warm up cache with frequently accessed data"""
        pass
    
    @abstractmethod
    def get_cache_stats(self) -> CacheStats:
        """Get cache performance statistics"""
        pass


class DefaultPerformanceOptimizerService(PerformanceOptimizerService):
    """
    Default implementation of performance optimization services.
    """
    
    def __init__(self):
        self.cache = get_cache()
        self._cache_stats = CacheStats(
            hits=0,
            misses=0,
            hit_rate=0.0,
            total_requests=0,
            average_response_time=0.0,
            tier_distribution={CacheTier.L1_MEMORY: 0, CacheTier.L2_REDIS: 0, CacheTier.L3_DATABASE: 0}
        )
        self._response_times = []
    
    def _record_cache_request(self, is_hit: bool):
        """Record cache request for statistics"""
        self._cache_stats.total_requests += 1
        if is_hit:
            self._cache_stats.hits += 1
        else:
            self._cache_stats.misses += 1
        
        # Update hit rate
        if self._cache_stats.total_requests > 0:
            self._cache_stats.hit_rate = self._cache_stats.hits / self._cache_stats.total_requests
    
    def _record_response_time(self, response_time: float):
        """Record response time for statistics"""
        self._response_times.append(response_time)
        # Keep only last 1000 measurements
        if len(self._response_times) > 1000:
            self._response_times = self._response_times[-1000:]
    
    def get_cached_portfolio(self, user_id: str) -> Optional[Portfolio]:
        """
        Get cached portfolio for a user
        """
        start_time = time.time()
        
        cache_key = f"portfolio:{user_id}"
        cached_data = self.cache.get(cache_key)
        
        is_hit = cached_data is not None
        self._record_cache_request(is_hit)
        
        response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        self._record_response_time(response_time)
        
        if cached_data:
            # In a real implementation, we would deserialize the portfolio from cache
            # For now, we'll return None to indicate no cached portfolio
            return None
        else:
            return None
    
    def cache_portfolio(self, user_id: str, portfolio: Portfolio, ttl: int = 300) -> bool:
        """
        Cache portfolio for a user
        """
        cache_key = f"portfolio:{user_id}"
        
        # In a real implementation, we would serialize the portfolio before caching
        # For now, we'll just cache a simple representation
        portfolio_data = {
            "id": portfolio.id,
            "user_id": portfolio.user_id,
            "position_count": len(portfolio.positions),
            "total_value": float(portfolio.total_value.amount) if portfolio.total_value else 0.0,
            "cash_balance": float(portfolio.cash_balance.amount) if portfolio.cash_balance else 0.0,
            "updated_at": datetime.now().isoformat()
        }
        
        return self.cache.set(cache_key, portfolio_data, ttl)
    
    def get_cached_market_data(self, symbol: Symbol) -> Optional[Any]:
        """
        Get cached market data for a symbol
        """
        start_time = time.time()
        
        cache_key = f"market_data:{str(symbol)}"
        cached_data = self.cache.get(cache_key)
        
        is_hit = cached_data is not None
        self._record_cache_request(is_hit)
        
        response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        self._record_response_time(response_time)
        
        return cached_data
    
    def cache_market_data(self, symbol: Symbol, data: Any, ttl: int = 60) -> bool:
        """
        Cache market data for a symbol
        """
        cache_key = f"market_data:{str(symbol)}"
        return self.cache.set(cache_key, data, ttl)
    
    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate overall system performance metrics
        """
        # Calculate average response time
        avg_response_time = 0.0
        if self._response_times:
            avg_response_time = sum(self._response_times) / len(self._response_times)
        
        return {
            "cache_stats": {
                "hits": self._cache_stats.hits,
                "misses": self._cache_stats.misses,
                "hit_rate": round(self._cache_stats.hit_rate, 4),
                "total_requests": self._cache_stats.total_requests,
                "average_response_time_ms": round(avg_response_time, 2),
                "tier_distribution": self._cache_stats.tier_distribution
            },
            "system_performance": {
                "uptime_minutes": 0,  # Would calculate from system start time
                "active_users": 0,    # Would get from session management
                "requests_per_minute": 0,  # Would calculate from metrics
                "error_rate": 0.0     # Would calculate from error logs
            }
        }
    
    def warm_cache(self, user_id: str) -> bool:
        """
        Warm up cache with frequently accessed data
        """
        try:
            # Get portfolio data that is frequently accessed
            # This is a simplified implementation - in reality, 
            # you'd pre-load data based on user behavior patterns
            cache_keys_to_preload = [
                f"portfolio:{user_id}",
                f"user_profile:{user_id}",
                f"user_preferences:{user_id}",
                f"recent_orders:{user_id}"
            ]
            
            # Preload each key with a simple check
            for key in cache_keys_to_preload:
                # In a real implementation, we would load the actual data
                # For now, just ensure the key exists or set a default
                if not self.cache.get(key):
                    self.cache.set(key, f"preloaded:{key}", 300)  # 5 min TTL
            
            return True
        except Exception:
            return False
    
    def get_cache_stats(self) -> CacheStats:
        """
        Get cache performance statistics
        """
        # Calculate average response time
        avg_response_time = 0.0
        if self._response_times:
            avg_response_time = sum(self._response_times) / len(self._response_times)
        
        # Update the stats object
        self._cache_stats.average_response_time = avg_response_time
        return self._cache_stats


def performance_monitor(func: Callable) -> Callable:
    """
    Decorator to monitor function performance and log metrics
    """
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        func_name = func.__name__
        
        try:
            result = await func(*args, **kwargs)
            execution_time = (time.time() - start_time) * 1000  # Convert to ms
            print(f"PERFORMANCE: {func_name} executed in {execution_time:.2f}ms")
            return result
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            print(f"PERFORMANCE: {func_name} failed after {execution_time:.2f}ms: {str(e)}")
            raise
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        func_name = func.__name__
        
        try:
            result = func(*args, **kwargs)
            execution_time = (time.time() - start_time) * 1000  # Convert to ms
            print(f"PERFORMANCE: {func_name} executed in {execution_time:.2f}ms")
            return result
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            print(f"PERFORMANCE: {func_name} failed after {execution_time:.2f}ms: {str(e)}")
            raise
    
    # Return the appropriate wrapper based on whether the function is async
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


def cache_result(ttl: int = 300, key_prefix: str = "cached"):
    """
    Decorator to cache function results
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            cache = get_cache()
            # Create a unique cache key based on function name and arguments
            cache_key = f"{key_prefix}:{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache first
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                print(f"CACHE HIT: {cache_key}")
                return cached_result
            
            # Execute the function and cache the result
            result = await func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            print(f"CACHE MISS: {cache_key} (result cached)")
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            cache = get_cache()
            # Create a unique cache key based on function name and arguments
            cache_key = f"{key_prefix}:{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache first
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                print(f"CACHE HIT: {cache_key}")
                return cached_result
            
            # Execute the function and cache the result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            print(f"CACHE MISS: {cache_key} (result cached)")
            return result
        
        # Return the appropriate wrapper based on whether the function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class QueryOptimizer:
    """
    Class to optimize database queries and reduce execution time
    """
    
    @staticmethod
    def batch_queries(queries: List[Callable], batch_size: int = 10) -> List[Any]:
        """
        Execute multiple queries in batches to optimize performance
        """
        results = []
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]
            batch_results = [query() for query in batch]
            results.extend(batch_results)
        return results
    
    @staticmethod
    def optimize_joins(query_func: Callable) -> Callable:
        """
        Decorator to optimize query joins
        """
        @functools.wraps(query_func)
        def wrapper(*args, **kwargs):
            # In a real implementation, this would analyze and optimize the query
            # For now, just execute the query
            return query_func(*args, **kwargs)
        return wrapper