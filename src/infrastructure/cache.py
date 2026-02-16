"""
Caching Layer with DiskCache

This module provides a caching abstraction for frequently accessed data,
reducing database load and improving response times.

Following decorator pattern for easy integration with existing code.
"""

from __future__ import annotations

import json
import logging
from functools import wraps
from typing import Optional, Any, Callable
from datetime import timedelta

logger = logging.getLogger(__name__)


class CacheManager:
    """DiskCache-based cache manager for application data."""

    def __init__(self, cache_dir: str = "./cache"):
        """
        Initialize cache manager.

        Args:
            cache_dir: Directory for cache storage
        """
        self.cache_dir = cache_dir
        self.client = None
        self._connect()

    def _connect(self) -> None:
        """Connect to DiskCache."""
        try:
            from diskcache import FanoutCache

            self.client = FanoutCache(self.cache_dir, statistics=True)
            logger.info("Connected to DiskCache")
        except Exception as e:
            logger.error(f"Failed to initialize DiskCache: {e}")
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
        return self.client is not None

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
            return value
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
            expire = ttl if ttl else None
            if ttl:
                self.client.set(key, value, expire=ttl)
            else:
                self.client.set(key, value)
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
            return self.client.delete(key)
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
            # DiskCache doesn't support pattern matching like Redis
            # For simplicity, clear all cache when pattern is "*"
            if pattern == "*":
                self.client.clear()
                return 1
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
            result = {}
            for key in keys:
                value = self.client.get(key)
                if value is not None:
                    result[key] = value
            return result
        except Exception as e:
            logger.warning(f"Cache get_many error: {e}")
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
            expire = ttl if ttl else None
            for key, value in data.items():
                if ttl:
                    self.client.set(key, value, expire=ttl)
                else:
                    self.client.set(key, value)
            return True
        except Exception as e:
            logger.warning(f"Cache set_many error: {e}")
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
            key_parts.extend(str(arg) for arg in args if not hasattr(arg, "__call__"))
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
            key_parts.extend(str(arg) for arg in args if not hasattr(arg, "__call__"))
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
        _cache_manager = CacheManager("./cache")
    return _cache_manager


def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance."""
    if _cache_manager is None:
        return initialize_cache()
    return _cache_manager
