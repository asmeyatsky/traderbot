"""
Stock Screening Domain Service

Architectural Intent:
- Defines the StockScreenerPort ABC
- Pure domain logic for filtering and sorting screen results
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from src.domain.entities.screening import PrebuiltScreen, ScreenCriteria, ScreenResult


class StockScreenerPort(ABC):
    """Port for fetching stock screening data from market data providers."""

    @abstractmethod
    def fetch_screen_data(self) -> List[ScreenResult]:
        """Fetch base data for all tracked tickers."""
        pass

    def screen(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        High-level screen method: fetches data, applies criteria, returns JSON-serialisable dict.
        """
        criteria = _parse_criteria(args)
        all_results = self.fetch_screen_data()
        filtered = _apply_criteria(all_results, criteria)
        return {
            "screen": criteria.prebuilt_screen.value if criteria.prebuilt_screen else "custom",
            "count": len(filtered),
            "results": [
                {
                    "symbol": r.symbol,
                    "name": r.name,
                    "price": r.price,
                    "change_pct": r.change_pct,
                    "volume": r.volume,
                    "market_cap": r.market_cap,
                    "sector": r.sector,
                    "rsi": r.rsi,
                }
                for r in filtered
            ],
        }


def _parse_criteria(args: Dict[str, Any]) -> ScreenCriteria:
    prebuilt = None
    if "prebuilt_screen" in args and args["prebuilt_screen"]:
        prebuilt = PrebuiltScreen(args["prebuilt_screen"])
    return ScreenCriteria(
        prebuilt_screen=prebuilt,
        min_change_pct=args.get("min_change_pct"),
        max_change_pct=args.get("max_change_pct"),
        min_volume=args.get("min_volume"),
        sectors=args.get("sectors"),
        limit=args.get("limit", 10),
    )


def _apply_criteria(results: List[ScreenResult], criteria: ScreenCriteria) -> List[ScreenResult]:
    """Filter and sort results based on criteria / prebuilt screen."""
    filtered = list(results)

    # Apply filters
    if criteria.min_change_pct is not None:
        filtered = [r for r in filtered if r.change_pct >= criteria.min_change_pct]
    if criteria.max_change_pct is not None:
        filtered = [r for r in filtered if r.change_pct <= criteria.max_change_pct]
    if criteria.min_volume is not None:
        filtered = [r for r in filtered if r.volume >= criteria.min_volume]
    if criteria.sectors:
        sector_set = {s.lower() for s in criteria.sectors}
        filtered = [r for r in filtered if r.sector and r.sector.lower() in sector_set]

    # Prebuilt screen sorting
    if criteria.prebuilt_screen == PrebuiltScreen.TOP_GAINERS:
        filtered.sort(key=lambda r: r.change_pct, reverse=True)
    elif criteria.prebuilt_screen == PrebuiltScreen.TOP_LOSERS:
        filtered.sort(key=lambda r: r.change_pct)
    elif criteria.prebuilt_screen == PrebuiltScreen.MOST_ACTIVE:
        filtered.sort(key=lambda r: r.volume, reverse=True)
    elif criteria.prebuilt_screen == PrebuiltScreen.HIGH_MOMENTUM:
        filtered.sort(key=lambda r: r.change_pct, reverse=True)
        filtered = [r for r in filtered if r.change_pct > 0]
    elif criteria.prebuilt_screen == PrebuiltScreen.OVERSOLD_RSI:
        filtered = [r for r in filtered if r.rsi is not None and r.rsi < 30]
        filtered.sort(key=lambda r: r.rsi or 100)

    return filtered[: criteria.limit]
