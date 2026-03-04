"""
Exchange Domain Entity

Architectural Intent:
- Frozen dataclass representing a stock exchange with trading hours
- is_open(), next_open(), next_close() use zoneinfo for timezone-aware logic
- No infrastructure dependencies
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Optional
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class Exchange:
    """Represents a stock exchange with trading session hours."""
    code: str            # e.g. "NYSE", "LSE", "TSE"
    name: str            # e.g. "New York Stock Exchange"
    timezone: str        # IANA timezone, e.g. "America/New_York"
    open_time: time      # Local open time
    close_time: time     # Local close time
    country: str
    currency: str
    suffix: str = ""     # Yahoo Finance ticker suffix, e.g. ".L" for LSE

    def _now_local(self) -> datetime:
        return datetime.now(ZoneInfo(self.timezone))

    def is_open(self) -> bool:
        """Check if the exchange is currently in a trading session (weekday + within hours)."""
        now = self._now_local()
        if now.weekday() >= 5:  # Saturday or Sunday
            return False
        return self.open_time <= now.time() <= self.close_time

    def next_open(self) -> datetime:
        """Return the next opening datetime in UTC."""
        now = self._now_local()
        candidate = now.replace(hour=self.open_time.hour, minute=self.open_time.minute, second=0, microsecond=0)

        # If already past today's open or it's a weekend, move to next weekday
        if now.time() >= self.open_time or now.weekday() >= 5:
            candidate += timedelta(days=1)

        while candidate.weekday() >= 5:
            candidate += timedelta(days=1)

        return candidate.astimezone(ZoneInfo("UTC"))

    def next_close(self) -> datetime:
        """Return the next closing datetime in UTC."""
        now = self._now_local()
        candidate = now.replace(hour=self.close_time.hour, minute=self.close_time.minute, second=0, microsecond=0)

        if now.time() >= self.close_time or now.weekday() >= 5:
            candidate += timedelta(days=1)

        while candidate.weekday() >= 5:
            candidate += timedelta(days=1)

        return candidate.astimezone(ZoneInfo("UTC"))
