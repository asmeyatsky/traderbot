"""
Exchange Registry Domain Service

Architectural Intent:
- Hardcoded registry of major global exchanges
- Provides get_all_statuses() for the chat tool
- No infrastructure dependencies
"""
from __future__ import annotations

from datetime import time
from typing import Any, Dict, List

from src.domain.entities.exchange import Exchange


_EXCHANGES: List[Exchange] = [
    Exchange(
        code="NYSE", name="New York Stock Exchange",
        timezone="America/New_York", open_time=time(9, 30), close_time=time(16, 0),
        country="US", currency="USD",
    ),
    Exchange(
        code="NASDAQ", name="NASDAQ",
        timezone="America/New_York", open_time=time(9, 30), close_time=time(16, 0),
        country="US", currency="USD",
    ),
    Exchange(
        code="LSE", name="London Stock Exchange",
        timezone="Europe/London", open_time=time(8, 0), close_time=time(16, 30),
        country="GB", currency="GBP", suffix=".L",
    ),
    Exchange(
        code="TSE", name="Tokyo Stock Exchange",
        timezone="Asia/Tokyo", open_time=time(9, 0), close_time=time(15, 0),
        country="JP", currency="JPY", suffix=".T",
    ),
    Exchange(
        code="HKEX", name="Hong Kong Stock Exchange",
        timezone="Asia/Hong_Kong", open_time=time(9, 30), close_time=time(16, 0),
        country="HK", currency="HKD", suffix=".HK",
    ),
    Exchange(
        code="EURONEXT", name="Euronext Paris",
        timezone="Europe/Paris", open_time=time(9, 0), close_time=time(17, 30),
        country="FR", currency="EUR", suffix=".PA",
    ),
    Exchange(
        code="XETRA", name="XETRA (Frankfurt)",
        timezone="Europe/Berlin", open_time=time(9, 0), close_time=time(17, 30),
        country="DE", currency="EUR", suffix=".DE",
    ),
    Exchange(
        code="ASX", name="Australian Securities Exchange",
        timezone="Australia/Sydney", open_time=time(10, 0), close_time=time(16, 0),
        country="AU", currency="AUD", suffix=".AX",
    ),
]


class ExchangeRegistry:
    """Registry of supported exchanges with market-hours logic."""

    def __init__(self):
        self._exchanges = {ex.code: ex for ex in _EXCHANGES}

    def get(self, code: str) -> Exchange | None:
        return self._exchanges.get(code.upper())

    def list_all(self) -> List[Exchange]:
        return list(self._exchanges.values())

    def get_all_statuses(self) -> Dict[str, Any]:
        """Return JSON-serialisable status for every exchange (used by chat tool)."""
        statuses = []
        for ex in self._exchanges.values():
            is_open = ex.is_open()
            statuses.append({
                "code": ex.code,
                "name": ex.name,
                "country": ex.country,
                "currency": ex.currency,
                "is_open": is_open,
                "next_open": ex.next_open().isoformat() if not is_open else None,
                "next_close": ex.next_close().isoformat() if is_open else None,
            })
        return {"exchanges": statuses}
