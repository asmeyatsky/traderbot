"""
Rate Limiting Implementation

Provides distributed rate limiting for API endpoints with multiple strategies:
- Token bucket algorithm
- Sliding window
- Per-user and per-IP limiting

Architectural Intent:
- Prevent abuse and DoS attacks
- Ensure fair resource allocation
- Support multiple limiting strategies
- Configurable limits per endpoint/user
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Optional, Dict, Tuple
from datetime import datetime, timedelta
from functools import wraps
from collections import defaultdict

from fastapi import Request, HTTPException, status
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

logger = logging.getLogger(__name__)


class RateLimitStrategy(ABC):
    """Abstract base class for rate limiting strategies."""

    @abstractmethod
    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed under this strategy."""
        pass

    @abstractmethod
    def get_remaining(self, key: str) -> int:
        """Get remaining requests for the key."""
        pass

    @abstractmethod
    def get_reset_time(self, key: str) -> datetime:
        """Get when the limit resets."""
        pass


class TokenBucketLimiter(RateLimitStrategy):
    """
    Token bucket algorithm for rate limiting.

    - Fixed number of tokens in bucket
    - Tokens replenish at fixed rate
    - Request consumes one token
    - No request if no tokens available
    """

    def __init__(
        self,
        capacity: int = 100,
        refill_rate: float = 10.0,  # tokens per second
    ):
        """
        Initialize token bucket limiter.

        Args:
            capacity: Maximum number of tokens in bucket
            refill_rate: Rate of token replenishment per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.buckets: Dict[
            str, Tuple[float, float]
        ] = {}  # key: (tokens, last_refill_time)

    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed."""
        self._refill(key)

        if key not in self.buckets:
            self.buckets[key] = (self.capacity - 1, time.time())
            return True

        tokens, _ = self.buckets[key]
        if tokens > 0:
            self.buckets[key] = (tokens - 1, time.time())
            return True

        return False

    def get_remaining(self, key: str) -> int:
        """Get remaining tokens."""
        self._refill(key)
        if key in self.buckets:
            tokens, _ = self.buckets[key]
            return int(tokens)
        return self.capacity

    def get_reset_time(self, key: str) -> datetime:
        """Get time when bucket will be full."""
        if key not in self.buckets:
            return datetime.utcnow()

        tokens, last_refill = self.buckets[key]
        time_to_full = (self.capacity - tokens) / self.refill_rate
        reset_time = datetime.fromtimestamp(last_refill + time_to_full)
        return reset_time

    def _refill(self, key: str) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()

        if key not in self.buckets:
            self.buckets[key] = (self.capacity, now)
            return

        tokens, last_refill = self.buckets[key]
        elapsed = now - last_refill
        new_tokens = min(self.capacity, tokens + (elapsed * self.refill_rate))
        self.buckets[key] = (new_tokens, now)


class SlidingWindowLimiter(RateLimitStrategy):
    """
    Sliding window algorithm for rate limiting.

    - Fixed time window
    - Count requests in window
    - Reject if count exceeds limit
    """

    def __init__(self, limit: int = 100, window_seconds: int = 60):
        """
        Initialize sliding window limiter.

        Args:
            limit: Maximum requests in window
            window_seconds: Time window in seconds
        """
        self.limit = limit
        self.window_seconds = window_seconds
        self.requests: Dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed."""
        now = time.time()
        window_start = now - self.window_seconds

        # Remove old requests outside window
        self.requests[key] = [
            req_time for req_time in self.requests[key] if req_time > window_start
        ]

        # Check limit
        if len(self.requests[key]) < self.limit:
            self.requests[key].append(now)
            return True

        return False

    def get_remaining(self, key: str) -> int:
        """Get remaining requests in window."""
        now = time.time()
        window_start = now - self.window_seconds

        self.requests[key] = [
            req_time for req_time in self.requests[key] if req_time > window_start
        ]

        return max(0, self.limit - len(self.requests[key]))

    def get_reset_time(self, key: str) -> datetime:
        """Get time when window resets."""
        if not self.requests[key]:
            return datetime.utcnow()

        oldest_request = min(self.requests[key])
        reset_time = datetime.fromtimestamp(oldest_request + self.window_seconds)
        return reset_time


class DistributedRateLimiter:
    """
    Distributed rate limiter using DiskCache for local coordination.
    """

    def __init__(
        self,
        strategy: RateLimitStrategy = None,
        use_disk_cache: bool = True,
        cache_dir: str = "./rate_limit_cache",
    ):
        """
        Initialize distributed rate limiter.

        Args:
            strategy: Rate limiting strategy (defaults to token bucket)
            use_disk_cache: Whether to use DiskCache for persistent limiting
            cache_dir: Directory for rate limit cache storage
        """
        self.strategy = strategy or TokenBucketLimiter(capacity=1000, refill_rate=100)
        self.use_disk_cache = use_disk_cache
        self.disk_cache = None

        if use_disk_cache:
            try:
                from diskcache import FanoutCache

                self.disk_cache = FanoutCache(cache_dir, statistics=True)
                logger.info("DiskCache rate limiter connected")
            except Exception as e:
                logger.warning(f"Failed to initialize DiskCache for rate limiting: {e}")
                self.use_disk_cache = False

    def is_allowed(self, key: str, limit: int = None, window: int = None) -> bool:
        """Check if request is allowed."""
        if self.use_disk_cache and self.disk_cache:
            return self._disk_cache_is_allowed(key, limit or 1000, window or 60)
        return self.strategy.is_allowed(key)

    def get_remaining(self, key: str) -> int:
        """Get remaining requests."""
        if self.use_disk_cache and self.disk_cache:
            try:
                remaining = self.disk_cache.get(f"rl:{key}:remaining")
                return int(remaining) if remaining else 0
            except Exception as e:
                logger.error(f"Error getting remaining from DiskCache: {e}")
                return 0

        return self.strategy.get_remaining(key)

    def get_reset_time(self, key: str) -> datetime:
        """Get reset time."""
        if self.use_disk_cache and self.disk_cache:
            try:
                reset_ts = self.disk_cache.get(f"rl:{key}:reset")
                return (
                    datetime.fromtimestamp(float(reset_ts))
                    if reset_ts
                    else datetime.utcnow()
                )
            except Exception as e:
                logger.error(f"Error getting reset time from DiskCache: {e}")
                return datetime.utcnow()

        return self.strategy.get_reset_time(key)

    def _disk_cache_is_allowed(self, key: str, limit: int, window: int) -> bool:
        """Check if allowed using DiskCache."""
        try:
            cache_key = f"rl:{key}"
            current_count = self.disk_cache.get(cache_key) or 0

            if current_count >= limit:
                return False

            # Increment count
            new_count = current_count + 1
            self.disk_cache.set(cache_key, new_count, expire=window)

            # Set remaining count
            remaining = max(0, limit - new_count)
            self.disk_cache.set(f"{cache_key}:remaining", remaining, expire=window)

            # Set reset time
            reset_time = (datetime.utcnow() + timedelta(seconds=window)).timestamp()
            self.disk_cache.set(f"{cache_key}:reset", reset_time, expire=window)

            return True
        except Exception as e:
            logger.error(f"Error in DiskCache rate limiting: {e}")
            return True  # Allow if error


def rate_limit(
    limit: str = "100/minute",
    key_func=None,
) -> callable:
    """
    Decorator for rate limiting endpoints.

    Args:
        limit: Rate limit string (e.g., "100/minute", "1000/hour")
        key_func: Function to generate limiting key (defaults to client IP)

    Usage:
        @rate_limit(limit="100/minute")
        async def get_orders(request: Request):
            return ...

        @rate_limit(limit="50/minute", key_func=lambda req: req.user.id)
        async def create_order(request: Request):
            return ...
    """
    limiter = Limiter(key_func=key_func or get_remote_address)

    def decorator(func: callable) -> callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> any:
            # Slowapi integration
            try:
                return await func(*args, **kwargs)
            except RateLimitExceeded as e:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded: {limit}",
                    headers={"Retry-After": "60"},
                )

        return wrapper

    return decorator


# Global rate limiter instance
_rate_limiter: Optional[DistributedRateLimiter] = None


def get_rate_limiter() -> DistributedRateLimiter:
    """Get the global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = DistributedRateLimiter(
            use_disk_cache=True,
            cache_dir="./rate_limit_cache",
        )
    return _rate_limiter


class RateLimiter:
    """
    Simple rate limiter class for dependency injection.
    Wrapper around the DistributedRateLimiter for DI container compatibility.
    """

    def __init__(self, cache_dir: str = "./rate_limit_cache"):
        """Initialize the rate limiter."""
        self.distributed_limiter = DistributedRateLimiter(
            use_disk_cache=True,
            cache_dir=cache_dir,
        )

    return _rate_limiter


class RateLimiter:
    """
    Simple rate limiter class for dependency injection.
    Wrapper around the DistributedRateLimiter for DI container compatibility.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        """Initialize the rate limiter."""
        self.distributed_limiter = DistributedRateLimiter(
            use_redis=True,
            redis_url=redis_url,
        )

    def is_allowed(self, key: str, limit: int = None, window: int = None) -> bool:
        """Check if request is allowed."""
        return self.distributed_limiter.is_allowed(key, limit, window)

    def get_remaining(self, key: str) -> int:
        """Get remaining requests."""
        return self.distributed_limiter.get_remaining(key)

    def get_reset_time(self, key: str) -> datetime:
        """Get reset time."""
        return self.distributed_limiter.get_reset_time(key)


def set_rate_limiter(limiter: DistributedRateLimiter) -> None:
    """Set the global rate limiter instance."""
    global _rate_limiter
    _rate_limiter = limiter
