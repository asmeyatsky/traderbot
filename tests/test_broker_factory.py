"""
Tests for BrokerServiceFactory — Phase 6 kill-switch and daily-loss routing.

Covers the four gates in `BrokerServiceFactory.for_user`:
1. EMERGENCY_HALT=true → always paper (even for LIVE users)
2. ENABLE_LIVE_TRADING unset/false → always paper
3. user.trading_mode == PAPER → always paper
4. user.trading_mode == LIVE + loss cap breached → refused
   user.trading_mode == LIVE + within cap → live broker

The factory's internal broker instances are mocked out so the tests don't
hit the network — the assertions are on which branch was taken, not what
any real Alpaca endpoint returns.
"""
from __future__ import annotations

import os
from dataclasses import replace
from datetime import datetime
from decimal import Decimal
from unittest.mock import patch

import pytest

from src.domain.entities.user import (
    InvestmentGoal, RiskTolerance, TradingMode, User,
)
from src.infrastructure.adapters.broker_factory import (
    BrokerServiceFactory,
    DailyLossCapBreachedError,
    LiveTradingHaltedError,
)


def _user(**overrides) -> User:
    defaults = dict(
        id="user-1",
        email="a@b.com",
        first_name="A",
        last_name="B",
        created_at=datetime(2026, 1, 1),
        updated_at=datetime(2026, 1, 1),
        risk_tolerance=RiskTolerance.MODERATE,
        investment_goal=InvestmentGoal.BALANCED_GROWTH,
        trading_mode=TradingMode.PAPER,
    )
    defaults.update(overrides)
    return User(**defaults)


@pytest.fixture
def env_clean(monkeypatch):
    """Ensure neither flag is set unless the test sets it explicitly."""
    monkeypatch.delenv("EMERGENCY_HALT", raising=False)
    monkeypatch.delenv("ENABLE_LIVE_TRADING", raising=False)
    return monkeypatch


@pytest.fixture
def factory():
    """Factory with its two broker slots pre-filled by sentinels so we can
    tell which branch was taken without touching AlpacaBrokerService.__init__
    (which would try to hit the network).
    """
    f = BrokerServiceFactory(loss_tracker=None)
    f._paper_broker = "PAPER_SENTINEL"  # type: ignore[assignment]
    f._live_broker = "LIVE_SENTINEL"  # type: ignore[assignment]
    return f


class TestEmergencyHalt:
    def test_emergency_halt_forces_paper_for_live_user(self, env_clean, factory):
        env_clean.setenv("EMERGENCY_HALT", "true")
        env_clean.setenv("ENABLE_LIVE_TRADING", "true")
        user = _user(trading_mode=TradingMode.LIVE)
        assert factory.for_user(user) == "PAPER_SENTINEL"

    def test_emergency_halt_blocks_assert_live_allowed(self, env_clean, factory):
        env_clean.setenv("EMERGENCY_HALT", "true")
        with pytest.raises(LiveTradingHaltedError, match="EMERGENCY_HALT"):
            factory.assert_live_trading_allowed()


class TestFeatureFlag:
    def test_flag_off_forces_paper(self, env_clean, factory):
        # Neither flag set → default off → paper.
        user = _user(trading_mode=TradingMode.LIVE)
        assert factory.for_user(user) == "PAPER_SENTINEL"

    def test_flag_false_string_forces_paper(self, env_clean, factory):
        env_clean.setenv("ENABLE_LIVE_TRADING", "false")
        user = _user(trading_mode=TradingMode.LIVE)
        assert factory.for_user(user) == "PAPER_SENTINEL"

    def test_flag_on_but_user_paper_stays_paper(self, env_clean, factory):
        env_clean.setenv("ENABLE_LIVE_TRADING", "true")
        user = _user(trading_mode=TradingMode.PAPER)
        assert factory.for_user(user) == "PAPER_SENTINEL"

    def test_flag_on_and_user_live_goes_live(self, env_clean, factory):
        env_clean.setenv("ENABLE_LIVE_TRADING", "true")
        user = _user(trading_mode=TradingMode.LIVE)
        assert factory.for_user(user) == "LIVE_SENTINEL"


class TestDailyLossCap:
    def _factory_with_tracker(self, today_loss: Decimal):
        class _StubTracker:
            def today_loss(self, _user_id):
                return today_loss

            def record_loss(self, _user_id, _amount):
                pass

        f = BrokerServiceFactory(loss_tracker=_StubTracker())
        f._paper_broker = "PAPER_SENTINEL"  # type: ignore[assignment]
        f._live_broker = "LIVE_SENTINEL"  # type: ignore[assignment]
        return f

    def test_under_cap_routes_to_live(self, env_clean):
        env_clean.setenv("ENABLE_LIVE_TRADING", "true")
        f = self._factory_with_tracker(today_loss=Decimal("50"))
        user = _user(
            trading_mode=TradingMode.LIVE,
            daily_loss_cap_usd=Decimal("100"),
        )
        assert f.for_user(user) == "LIVE_SENTINEL"

    def test_at_cap_refuses(self, env_clean):
        env_clean.setenv("ENABLE_LIVE_TRADING", "true")
        f = self._factory_with_tracker(today_loss=Decimal("100"))
        user = _user(
            trading_mode=TradingMode.LIVE,
            daily_loss_cap_usd=Decimal("100"),
        )
        with pytest.raises(DailyLossCapBreachedError, match="Daily loss cap"):
            f.for_user(user)

    def test_no_cap_no_tracker_allows_live(self, env_clean, factory):
        """User without a cap bypasses the check entirely."""
        env_clean.setenv("ENABLE_LIVE_TRADING", "true")
        user = _user(trading_mode=TradingMode.LIVE, daily_loss_cap_usd=None)
        assert factory.for_user(user) == "LIVE_SENTINEL"


class TestAssertLiveAllowed:
    def test_happy_path_returns_none(self, env_clean, factory):
        env_clean.setenv("ENABLE_LIVE_TRADING", "true")
        # Should simply return — no exception
        assert factory.assert_live_trading_allowed() is None

    def test_flag_off_raises(self, env_clean, factory):
        with pytest.raises(LiveTradingHaltedError, match="ENABLE_LIVE_TRADING"):
            factory.assert_live_trading_allowed()
