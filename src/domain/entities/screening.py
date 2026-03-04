"""
Stock Screening Domain Entities

Architectural Intent:
- Frozen dataclasses for screen results and criteria
- PrebuiltScreen enum for named screens (top_gainers, top_losers, etc.)
- No infrastructure dependencies
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class PrebuiltScreen(Enum):
    TOP_GAINERS = "top_gainers"
    TOP_LOSERS = "top_losers"
    MOST_ACTIVE = "most_active"
    HIGH_MOMENTUM = "high_momentum"
    OVERSOLD_RSI = "oversold_rsi"


@dataclass(frozen=True)
class ScreenResult:
    """A single stock result from a screening operation."""
    symbol: str
    name: str
    price: float
    change_pct: float
    volume: int
    market_cap: Optional[float] = None
    sector: Optional[str] = None
    rsi: Optional[float] = None


@dataclass(frozen=True)
class ScreenCriteria:
    """User-specified screening criteria."""
    prebuilt_screen: Optional[PrebuiltScreen] = None
    min_change_pct: Optional[float] = None
    max_change_pct: Optional[float] = None
    min_volume: Optional[int] = None
    sectors: Optional[List[str]] = None
    limit: int = 10
