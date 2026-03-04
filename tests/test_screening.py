"""
Tests for Stock Screening Domain Logic

Tests cover:
- ScreenCriteria parsing
- Filtering by change%, volume, sector
- Prebuilt screen sorting (top_gainers, top_losers, most_active, oversold_rsi)
"""
import pytest

from src.domain.entities.screening import PrebuiltScreen, ScreenCriteria, ScreenResult
from src.domain.services.screening import _apply_criteria, _parse_criteria


def _sample_results():
    return [
        ScreenResult(symbol="AAPL", name="Apple", price=150.0, change_pct=2.5, volume=1_000_000, sector="Technology"),
        ScreenResult(symbol="MSFT", name="Microsoft", price=400.0, change_pct=-1.2, volume=800_000, sector="Technology"),
        ScreenResult(symbol="XOM", name="Exxon", price=110.0, change_pct=0.3, volume=500_000, sector="Energy"),
        ScreenResult(symbol="JNJ", name="J&J", price=160.0, change_pct=-3.0, volume=300_000, sector="Healthcare", rsi=28.0),
        ScreenResult(symbol="TSLA", name="Tesla", price=250.0, change_pct=5.1, volume=2_000_000, sector="Consumer Cyclical"),
    ]


class TestParseCriteria:
    def test_parses_prebuilt(self):
        c = _parse_criteria({"prebuilt_screen": "top_gainers", "limit": 5})
        assert c.prebuilt_screen == PrebuiltScreen.TOP_GAINERS
        assert c.limit == 5

    def test_defaults(self):
        c = _parse_criteria({})
        assert c.prebuilt_screen is None
        assert c.limit == 10


class TestApplyCriteria:
    def test_top_gainers(self):
        criteria = ScreenCriteria(prebuilt_screen=PrebuiltScreen.TOP_GAINERS, limit=3)
        results = _apply_criteria(_sample_results(), criteria)
        assert results[0].symbol == "TSLA"
        assert len(results) == 3

    def test_top_losers(self):
        criteria = ScreenCriteria(prebuilt_screen=PrebuiltScreen.TOP_LOSERS, limit=2)
        results = _apply_criteria(_sample_results(), criteria)
        assert results[0].symbol == "JNJ"

    def test_most_active(self):
        criteria = ScreenCriteria(prebuilt_screen=PrebuiltScreen.MOST_ACTIVE, limit=2)
        results = _apply_criteria(_sample_results(), criteria)
        assert results[0].symbol == "TSLA"

    def test_oversold_rsi(self):
        criteria = ScreenCriteria(prebuilt_screen=PrebuiltScreen.OVERSOLD_RSI, limit=10)
        results = _apply_criteria(_sample_results(), criteria)
        assert len(results) == 1
        assert results[0].symbol == "JNJ"

    def test_high_momentum(self):
        criteria = ScreenCriteria(prebuilt_screen=PrebuiltScreen.HIGH_MOMENTUM, limit=10)
        results = _apply_criteria(_sample_results(), criteria)
        assert all(r.change_pct > 0 for r in results)

    def test_min_change_pct(self):
        criteria = ScreenCriteria(min_change_pct=1.0, limit=10)
        results = _apply_criteria(_sample_results(), criteria)
        assert all(r.change_pct >= 1.0 for r in results)

    def test_max_change_pct(self):
        criteria = ScreenCriteria(max_change_pct=0.0, limit=10)
        results = _apply_criteria(_sample_results(), criteria)
        assert all(r.change_pct <= 0.0 for r in results)

    def test_min_volume(self):
        criteria = ScreenCriteria(min_volume=900_000, limit=10)
        results = _apply_criteria(_sample_results(), criteria)
        assert all(r.volume >= 900_000 for r in results)

    def test_sector_filter(self):
        criteria = ScreenCriteria(sectors=["Energy"], limit=10)
        results = _apply_criteria(_sample_results(), criteria)
        assert all(r.sector == "Energy" for r in results)

    def test_limit(self):
        criteria = ScreenCriteria(limit=2)
        results = _apply_criteria(_sample_results(), criteria)
        assert len(results) == 2
