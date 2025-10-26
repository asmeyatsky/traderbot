"""
Caching Layer Implementation

Provides a multi-level caching system with Redis as primary cache
and in-memory cache as fallback.

Architectural Intent:
- Abstract caching interface with multiple implementations
- Reduces database load for frequently accessed data
- Improves response times for common queries
- Automatic cache invalidation strategies
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Optional, Callable, TypeVar, Generic
from functools import wraps
import hashlib

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CacheBackend(ABC, Generic[T]):
    """Abstract base class for cache backends."""

    @abstractmethod
    def get(self, key: str) -> Optional[T]:
        """Get a value from cache."""
        pass

    @abstractmethod
    def set(self, key: str, value: T, ttl: int = 300) -> None:
        """Set a value in cache with optional TTL in seconds."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a value from cache."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if a key exists in cache."""
        pass

    @abstractmethod
    def get_ttl(self, key: str) -> Optional[int]:
        """Get remaining TTL for a key in seconds."""
        pass


class InMemoryCache(CacheBackend[T]):
    """
    In-memory cache implementation using a simple dictionary.

    Useful for development and fallback cache.
    """

    def __init__(self):
        """Initialize in-memory cache."""
        self._cache: dict[str, tuple[T, Optional[datetime]]] = {}

    def get(self, key: str) -> Optional[T]:
        """Get a value from memory cache."""
        if key not in self._cache:
            return None

        value, expiry = self._cache[key]

        # Check expiration
        if expiry and datetime.utcnow() > expiry:
            del self._cache[key]
            return None

        return value

    def set(self, key: str, value: T, ttl: int = 300) -> None:
        """Set a value in memory cache."""
        expiry = datetime.utcnow() + timedelta(seconds=ttl) if ttl else None
        self._cache[key] = (value, expiry)
        logger.debug(f"Cache SET: {key} (TTL: {ttl}s)")

    def delete(self, key: str) -> bool:
        """Delete a value from memory cache."""
        if key in self._cache:
            del self._cache[key]
            logger.debug(f"Cache DELETE: {key}")
            return True
        return False

    def clear(self) -> None:
        """Clear all memory cache entries."""
        self._cache.clear()
        logger.info("Memory cache cleared")

    def exists(self, key: str) -> bool:
        """Check if key exists in memory cache."""
        if key not in self._cache:
            return False

        value, expiry = self._cache[key]
        if expiry and datetime.utcnow() > expiry:
            del self._cache[key]
            return False

        return True

    def get_ttl(self, key: str) -> Optional[int]:
        """Get remaining TTL for a key."""
        if key not in self._cache:
            return None

        value, expiry = self._cache[key]
        if not expiry:
            return -1  # No expiry

        remaining = expiry - datetime.utcnow()
        return max(0, int(remaining.total_seconds()))

    def size(self) -> int:
        """Get number of items in cache."""
        return len(self._cache)


class RedisCache(CacheBackend[T]):
    """
    Redis-based cache implementation.

    Provides distributed caching for multi-instance deployments.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        """
        Initialize Redis cache.

        Args:
            redis_url: Redis connection URL
        """
        try:
            import redis
            from redis import Redis
            self.redis_client: Optional[Redis] = Redis.from_url(redis_url, decode_responses=True)
            # Test connection
            self.redis_client.ping()
            self.available = True
            logger.info(f"Redis cache connected: {redis_url}")
        except ImportError:
            logger.warning("redis package not installed, Redis cache unavailable")
            self.redis_client = None
            self.available = False
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            self.redis_client = None
            self.available = False

    def get(self, key: str) -> Optional[T]:
        """Get a value from Redis cache."""
        if not self.available or not self.redis_client:
            return None

        try:
            value = self.redis_client.get(key)
            if value:
                logger.debug(f"Cache HIT: {key}")
                return json.loads(value)
            logger.debug(f"Cache MISS: {key}")
            return None
        except Exception as e:
            logger.error(f"Error getting from Redis cache: {e}")
            return None

    def set(self, key: str, value: T, ttl: int = 300) -> None:
        """Set a value in Redis cache."""
        if not self.available or not self.redis_client:
            return

        try:
            self.redis_client.setex(key, ttl, json.dumps(value))
            logger.debug(f"Cache SET: {key} (TTL: {ttl}s)")
        except Exception as e:
            logger.error(f"Error setting in Redis cache: {e}")

    def delete(self, key: str) -> bool:
        """Delete a value from Redis cache."""
        if not self.available or not self.redis_client:
            return False

        try:
            result = self.redis_client.delete(key)
            logger.debug(f"Cache DELETE: {key}")
            return result > 0
        except Exception as e:
            logger.error(f"Error deleting from Redis cache: {e}")
            return False

    def clear(self) -> None:
        """Clear all Redis cache entries."""
        if not self.available or not self.redis_client:
            return

        try:
            self.redis_client.flushdb()
            logger.info("Redis cache cleared")
        except Exception as e:
            logger.error(f"Error clearing Redis cache: {e}")

    def exists(self, key: str) -> bool:
        """Check if key exists in Redis cache."""
        if not self.available or not self.redis_client:
            return False

        try:
            return self.redis_client.exists(key) > 0
        except Exception as e:
            logger.error(f"Error checking Redis cache: {e}")
            return False

    def get_ttl(self, key: str) -> Optional[int]:
        """Get remaining TTL for a key in Redis."""
        if not self.available or not self.redis_client:
            return None

        try:
            ttl = self.redis_client.ttl(key)
            if ttl == -2:
                return None  # Key doesn't exist
            if ttl == -1:
                return -1  # No expiry
            return ttl
        except Exception as e:
            logger.error(f"Error getting TTL from Redis: {e}")
            return None


class HybridCache(CacheBackend[T]):
    """
    Hybrid cache using both Redis and in-memory cache.

    - Redis for distributed cache
    - In-memory for local performance
    - Automatic fallback if Redis unavailable
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        """Initialize hybrid cache."""
        self.memory_cache = InMemoryCache()
        self.redis_cache = RedisCache(redis_url)

    def get(self, key: str) -> Optional[T]:
        """Get from memory cache first, fallback to Redis."""
        # Try memory cache first (fastest)
        value = self.memory_cache.get(key)
        if value is not None:
            logger.debug(f"Cache HIT (memory): {key}")
            return value

        # Fallback to Redis
        value = self.redis_cache.get(key)
        if value is not None:
            # Populate memory cache
            self.memory_cache.set(key, value, ttl=60)
            return value

        logger.debug(f"Cache MISS: {key}")
        return None

    def set(self, key: str, value: T, ttl: int = 300) -> None:
        """Set in both memory and Redis cache."""
        self.memory_cache.set(key, value, ttl=min(ttl, 300))  # Keep memory cache short
        self.redis_cache.set(key, value, ttl=ttl)

    def delete(self, key: str) -> bool:
        """Delete from both caches."""
        self.memory_cache.delete(key)
        return self.redis_cache.delete(key)

    def clear(self) -> None:
        """Clear both caches."""
        self.memory_cache.clear()
        self.redis_cache.clear()

    def exists(self, key: str) -> bool:
        """Check if key exists in either cache."""
        return self.memory_cache.exists(key) or self.redis_cache.exists(key)

    def get_ttl(self, key: str) -> Optional[int]:
        """Get TTL from Redis (authoritative)."""
        return self.redis_cache.get_ttl(key)


def cache_decorator(
    ttl: int = 300,
    cache: Optional[CacheBackend] = None,
) -> Callable:
    """
    Decorator for caching function results.

    Args:
        ttl: Time to live in seconds
        cache: Cache backend to use (defaults to in-memory)

    Usage:
        @cache_decorator(ttl=600)
        def expensive_function(user_id: str):
            return user_repository.get_by_id(user_id)
    """
    if cache is None:
        cache = InMemoryCache()

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Generate cache key from function name and arguments
            key_parts = [func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
            cache_key = hashlib.md5(":".join(key_parts).encode()).hexdigest()

            # Try to get from cache
            cached = cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache HIT: {func.__name__}")
                return cached

            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl=ttl)
            return result

        return wrapper

    return decorator


# Global cache instance
_cache_instance: Optional[CacheBackend] = None


def get_cache() -> CacheBackend:
    """Get the global cache instance."""
    global _cache_instance
    if _cache_instance is None:
        from src.infrastructure.config.settings import settings
        _cache_instance = HybridCache(settings.REDIS_URL)
    return _cache_instance


def set_cache(cache: CacheBackend) -> None:
    """Set the global cache instance."""
    global _cache_instance
    _cache_instance = cache
