"""
Caching Layer with Redis

This module provides a caching abstraction for frequently accessed data,
reducing database load and improving response times.

Following decorator pattern for easy integration with existing code.
"""
from __future__ import annotations

import redis
import json
import logging
from functools import wraps
from typing import Optional, Any, Callable
from datetime import timedelta

from src.infrastructure.config.settings import settings

logger = logging.getLogger(__name__)


class CacheManager:
    """Redis-based cache manager for application data."""

    def __init__(self, redis_url: str):
        """
        Initialize cache manager.

        Args:
            redis_url: Redis connection URL
        """
        self.redis_url = redis_url
        self.client = None
        self._connect()

    def _connect(self) -> None:
        """Connect to Redis."""
        try:
            from redis import Redis
            self.client = Redis.from_url(self.redis_url, decode_responses=True)
            self.client.ping()
            logger.info("Connected to Redis cache")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.client = None

    def is_connected(self) -> bool:
        """Check if cache is connected."""
        if not self.client:
            return False
        try:
            self.client.ping()
            return True
        except Exception:
            return False

    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        if not self.client:
            return None

        try:
            value = self.client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.warning(f"Cache get error for key {key}: {e}")
            return None

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Set a value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds

        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            return False

        try:
            json_value = json.dumps(value)
            if ttl:
                self.client.setex(key, ttl, json_value)
            else:
                self.client.set(key, json_value)
            return True
        except Exception as e:
            logger.warning(f"Cache set error for key {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """
        Delete a value from cache.

        Args:
            key: Cache key

        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            return False

        try:
            self.client.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Cache delete error for key {key}: {e}")
            return False

    def clear(self, pattern: str = "*") -> int:
        """
        Clear cache keys matching pattern.

        Args:
            pattern: Key pattern to match

        Returns:
            Number of keys deleted
        """
        if not self.client:
            return 0

        try:
            keys = self.client.keys(pattern)
            if keys:
                return self.client.delete(*keys)
            return 0
        except Exception as e:
            logger.warning(f"Cache clear error for pattern {pattern}: {e}")
            return 0

    def get_many(self, keys: list[str]) -> dict[str, Any]:
        """
        Get multiple values from cache.

        Args:
            keys: List of cache keys

        Returns:
            Dictionary of key-value pairs
        """
        if not self.client:
            return {}

        try:
            values = self.client.mget(keys)
            result = {}
            for key, value in zip(keys, values):
                if value:
                    result[key] = json.loads(value)
            return result
        except Exception as e:
            logger.warning(f"Cache mget error: {e}")
            return {}

    def set_many(
        self,
        data: dict[str, Any],
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Set multiple values in cache.

        Args:
            data: Dictionary of key-value pairs
            ttl: Time to live in seconds

        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            return False

        try:
            for key, value in data.items():
                self.set(key, value, ttl)
            return True
        except Exception as e:
            logger.warning(f"Cache mset error: {e}")
            return False


def cache_key(*parts: str) -> str:
    """
    Generate a cache key from parts.

    Args:
        parts: Key components

    Returns:
        Generated cache key
    """
    return ":".join(str(p) for p in parts)


def cached(
    ttl: int = 300,  # 5 minutes default
    key_prefix: Optional[str] = None,
    condition: Optional[Callable] = None,
):
    """
    Decorator for caching function results.

    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache key
        condition: Optional condition function to decide whether to cache

    Usage:
        @cached(ttl=600)
        def get_user_portfolio(user_id: str):
            # Expensive operation
            pass

        @cached(ttl=300, key_prefix="symbol_price")
        def get_current_price(symbol: str):
            # API call
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Build cache key
            key_parts = [key_prefix or func.__name__]
            key_parts.extend(str(arg) for arg in args if not hasattr(arg, '__call__'))
            key_parts.extend(f"{k}={v}" for k, v in kwargs.items())
            cache_k = cache_key(*key_parts)

            # Try to get from cache
            cache_mgr = get_cache_manager()
            cached_result = cache_mgr.get(cache_k)
            if cached_result is not None:
                logger.debug(f"Cache hit for {cache_k}")
                return cached_result

            # Call function
            result = await func(*args, **kwargs)

            # Check condition and cache result
            if condition is None or condition(result):
                cache_mgr.set(cache_k, result, ttl)
                logger.debug(f"Cached result for {cache_k} with TTL {ttl}s")

            return result

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Build cache key
            key_parts = [key_prefix or func.__name__]
            key_parts.extend(str(arg) for arg in args if not hasattr(arg, '__call__'))
            key_parts.extend(f"{k}={v}" for k, v in kwargs.items())
            cache_k = cache_key(*key_parts)

            # Try to get from cache
            cache_mgr = get_cache_manager()
            cached_result = cache_mgr.get(cache_k)
            if cached_result is not None:
                logger.debug(f"Cache hit for {cache_k}")
                return cached_result

            # Call function
            result = func(*args, **kwargs)

            # Check condition and cache result
            if condition is None or condition(result):
                cache_mgr.set(cache_k, result, ttl)
                logger.debug(f"Cached result for {cache_k} with TTL {ttl}s")

            return result

        # Determine if function is async
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def initialize_cache() -> CacheManager:
    """Initialize the global cache manager."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager(settings.REDIS_URL)
    return _cache_manager


def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance."""
    if _cache_manager is None:
        return initialize_cache()
    return _cache_manager
