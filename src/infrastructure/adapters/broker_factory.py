"""
Broker service factory — per-user routing to paper or live Alpaca.

Architectural Intent (ADR-002):
- The hardcoded `paper_trading=True` previously at di_container.py:217 is the
  single most-dangerous flag in the codebase. This factory replaces it with a
  per-user decision routed through two platform-level kill switches:

    1. ENABLE_LIVE_TRADING=false  → every user forced to paper
    2. EMERGENCY_HALT=true         → every user forced to paper AND live
                                      broker calls refused hard

- A user whose `trading_mode` is LIVE still gets paper if either switch is off.
  The switches fail SAFE (to paper) — an unset env is treated as off.
"""
from __future__ import annotations

import logging
import os

from decimal import Decimal
from typing import Optional

from src.domain.entities.user import TradingMode, User
from src.domain.ports.broker_routing import BrokerRoutingPort
from src.domain.ports.daily_loss_tracker import DailyLossTrackerPort
from src.infrastructure.broker_integration import (
    AlpacaBrokerService,
    BrokerAuthenticationError,
    TradingExecutionPort,
)
from src.infrastructure.config.settings import settings

logger = logging.getLogger(__name__)


class LiveTradingHaltedError(BrokerAuthenticationError):
    """Raised when live-order routing is refused by a kill switch."""


class DailyLossCapBreachedError(BrokerAuthenticationError):
    """Raised when the user's realised loss today already exceeds their cap."""


def _flag_on(name: str) -> bool:
    return os.environ.get(name, "").lower() == "true"


class BrokerServiceFactory(BrokerRoutingPort):
    """Single source of truth for "paper or live" at the adapter boundary.

    Callers (order use case, chat MCP) should NEVER construct `AlpacaBrokerService`
    directly — always go through `for_user(...)`. This keeps the kill-switch
    and audit logic in one place.
    """

    def __init__(
        self,
        loss_tracker: Optional[DailyLossTrackerPort] = None,
    ) -> None:
        self._api_key = settings.ALPACA_API_KEY
        self._secret_key = settings.ALPACA_SECRET_KEY
        self._loss_tracker = loss_tracker
        # Lazy singletons: constructing a broker is cheap but we reuse the
        # HTTPS session for connection pooling.
        self._paper_broker: AlpacaBrokerService | None = None
        self._live_broker: AlpacaBrokerService | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def for_user(self, user: User) -> TradingExecutionPort:
        """Return the broker service appropriate for this user.

        Routing rules (in order):
        1. EMERGENCY_HALT=true → always paper, and we also log loudly.
        2. ENABLE_LIVE_TRADING not 'true' → always paper.
        3. user.trading_mode == PAPER → paper.
        4. Daily-loss cap breached → refuse with DailyLossCapBreachedError.
        5. Otherwise → live.

        The daily-loss check raises rather than silently routing to paper —
        we want the caller to see the breach, flip the user back to paper
        explicitly, and surface the state to the UI rather than hide it.
        """
        if _flag_on("EMERGENCY_HALT"):
            logger.warning(
                "broker_routed_to_paper reason=emergency_halt user_id=%s",
                user.id,
            )
            return self._get_paper()

        if not _flag_on("ENABLE_LIVE_TRADING"):
            return self._get_paper()

        if user.trading_mode == TradingMode.PAPER:
            return self._get_paper()

        # User is in live mode — enforce the cap.
        self._assert_under_daily_loss_cap(user)

        logger.info("broker_routed_to_live user_id=%s", user.id)
        return self._get_live()

    def _assert_under_daily_loss_cap(self, user: User) -> None:
        """Raise DailyLossCapBreachedError if today's loss exceeds user's cap.

        If the tracker isn't wired (None), we fail-open — paper users and
        tests can still work. Live prod wires the Redis-backed tracker via DI.
        """
        if self._loss_tracker is None or user.daily_loss_cap_usd is None:
            return
        today_loss = self._loss_tracker.today_loss(user.id)
        if today_loss >= user.daily_loss_cap_usd:
            logger.warning(
                "daily_loss_cap_breached user_id=%s today_loss=%s cap=%s",
                user.id, today_loss, user.daily_loss_cap_usd,
            )
            raise DailyLossCapBreachedError(
                f"Daily loss cap of ${user.daily_loss_cap_usd} reached "
                f"(today: ${today_loss}). Live trading refused; flip back to "
                f"paper and re-attest when ready."
            )

    def for_paper(self) -> TradingExecutionPort:
        """Explicit paper broker — for background jobs that should never go live."""
        return self._get_paper()

    def assert_live_trading_allowed(self) -> None:
        """Raise LiveTradingHaltedError if either platform switch blocks live trading.

        Useful for routers that want to refuse a user request BEFORE touching
        any state (e.g. enable-live-mode).
        """
        if _flag_on("EMERGENCY_HALT"):
            raise LiveTradingHaltedError(
                "EMERGENCY_HALT is active — live trading halted platform-wide."
            )
        if not _flag_on("ENABLE_LIVE_TRADING"):
            raise LiveTradingHaltedError(
                "Live trading is disabled (ENABLE_LIVE_TRADING != 'true')."
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_paper(self) -> AlpacaBrokerService:
        if self._paper_broker is None:
            self._paper_broker = AlpacaBrokerService(
                api_key=self._api_key,
                secret_key=self._secret_key,
                paper_trading=True,
            )
        return self._paper_broker

    def _get_live(self) -> AlpacaBrokerService:
        if self._live_broker is None:
            self._live_broker = AlpacaBrokerService(
                api_key=self._api_key,
                secret_key=self._secret_key,
                paper_trading=False,
            )
        return self._live_broker
