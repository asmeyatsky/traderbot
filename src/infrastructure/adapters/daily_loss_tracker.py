"""
Redis-backed daily loss tracker.

Architectural Intent:
- Key `loss:{user_id}:{YYYY-MM-DD}` holds the cumulative realised loss total
  in USD cents (integer — avoids Decimal<->string ping-pong).
- Keys expire at the next UTC midnight so the tracker self-cleans.
- A null Redis client (test/local) falls back to an in-process dict so the
  application still boots without Redis — the cap check becomes best-effort.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, Optional

from src.domain.ports.daily_loss_tracker import DailyLossTrackerPort

logger = logging.getLogger(__name__)


def _today_utc() -> date:
    return datetime.now(timezone.utc).date()


def _seconds_until_utc_midnight() -> int:
    now = datetime.now(timezone.utc)
    tomorrow = datetime.combine(
        now.date() + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc
    )
    return max(int((tomorrow - now).total_seconds()), 60)


class RedisDailyLossTracker(DailyLossTrackerPort):
    """Redis-backed tracker. Falls back to in-process dict when client is None."""

    def __init__(self, redis_client=None):
        self._redis = redis_client
        self._memory: Dict[str, int] = {}  # user_id_date → cents

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _key(user_id: str, day: date) -> str:
        return f"loss:{user_id}:{day.isoformat()}"

    @staticmethod
    def _to_cents(usd: Decimal) -> int:
        return int((usd * Decimal(100)).to_integral_value())

    @staticmethod
    def _from_cents(cents: int) -> Decimal:
        return (Decimal(cents) / Decimal(100)).quantize(Decimal("0.01"))

    # -- port ------------------------------------------------------------

    def record_loss(self, user_id: str, loss_usd: Decimal) -> Decimal:
        if loss_usd <= 0:
            return self.today_loss(user_id)

        cents = self._to_cents(loss_usd)
        key = self._key(user_id, _today_utc())

        if self._redis is not None:
            try:
                new_total = self._redis.incrby(key, cents)
                # Set expiry on first write of the day — INCRBY creates the
                # key if absent, but doesn't set TTL.
                self._redis.expire(key, _seconds_until_utc_midnight())
                return self._from_cents(int(new_total))
            except Exception:  # noqa: BLE001 — fall back to memory on Redis hiccup
                logger.exception(
                    "daily_loss_tracker redis failure for user=%s; using memory fallback",
                    user_id,
                )

        self._memory[key] = self._memory.get(key, 0) + cents
        return self._from_cents(self._memory[key])

    def today_loss(self, user_id: str) -> Decimal:
        key = self._key(user_id, _today_utc())

        if self._redis is not None:
            try:
                raw = self._redis.get(key)
                if raw is not None:
                    return self._from_cents(int(raw))
            except Exception:  # noqa: BLE001
                logger.exception("daily_loss_tracker read failure user=%s", user_id)

        return self._from_cents(self._memory.get(key, 0))

    def reset_today(self, user_id: str) -> None:
        key = self._key(user_id, _today_utc())

        if self._redis is not None:
            try:
                self._redis.delete(key)
            except Exception:  # noqa: BLE001
                logger.exception("daily_loss_tracker reset failure user=%s", user_id)

        self._memory.pop(key, None)
