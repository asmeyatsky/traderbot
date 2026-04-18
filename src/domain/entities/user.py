"""
User Domain Entity

This module contains the user domain entity for the trading platform,
following DDD principles and clean architecture patterns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Optional
from enum import Enum
from src.domain.value_objects import Money


class RiskTolerance(Enum):
    CONSERVATIVE = "CONSERVATIVE"  # max 10% drawdown
    MODERATE = "MODERATE"           # max 15% drawdown
    AGGRESSIVE = "AGGRESSIVE"      # max 25% drawdown


class InvestmentGoal(Enum):
    CAPITAL_PRESERVATION = "CAPITAL_PRESERVATION"
    BALANCED_GROWTH = "BALANCED_GROWTH"
    MAXIMUM_RETURNS = "MAXIMUM_RETURNS"


class TradingMode(Enum):
    """Per-user trading mode (ADR-002).

    `PAPER` routes every order to Alpaca paper — no real money. `LIVE` requires
    the full enable-live-mode flow (KYC + TOTP + daily loss cap + risk
    acknowledgement) and every order carries a TOTP re-challenge.
    """
    PAPER = "paper"
    LIVE = "live"


@dataclass(frozen=True)
class User:
    """
    User Domain Entity
    
    Represents a user of the trading platform with their preferences and settings.
    
    Architectural Intent:
    - Encapsulates all user-related business rules and preferences
    - Maintains user data invariants
    - Immutable to prevent accidental state changes
    """
    id: str
    email: str
    first_name: str
    last_name: str
    created_at: datetime
    updated_at: datetime
    risk_tolerance: RiskTolerance
    investment_goal: InvestmentGoal
    max_position_size_percentage: Decimal = Decimal('25')  # Max 25% per stock
    daily_loss_limit: Optional[Money] = None
    weekly_loss_limit: Optional[Money] = None
    monthly_loss_limit: Optional[Money] = None
    sector_preferences: List[str] = field(default_factory=list)  # List of preferred sectors
    sector_exclusions: List[str] = field(default_factory=list)   # List of excluded sectors
    is_active: bool = True
    email_notifications_enabled: bool = True
    sms_notifications_enabled: bool = False
    approval_mode_enabled: bool = False  # Requires approval for trades
    terms_accepted_at: Optional[datetime] = None
    privacy_accepted_at: Optional[datetime] = None
    marketing_consent: bool = False
    auto_trading_enabled: bool = False
    watchlist: List[str] = field(default_factory=list)
    trading_budget: Optional[Money] = None
    stop_loss_pct: Decimal = Decimal('5')
    take_profit_pct: Decimal = Decimal('10')
    confidence_threshold: Decimal = Decimal('0.6')
    max_position_pct: Decimal = Decimal('20')
    allowed_markets: List[str] = field(default_factory=lambda: ["US_NYSE", "US_NASDAQ", "UK_LSE", "EU_EURONEXT", "DE_XETRA", "JP_TSE", "HK_HKEX"])

    # ── Live-trading state (ADR-002) ────────────────────────────────────
    # Defaults keep every existing user in paper mode until they explicitly
    # complete the enable-live-mode flow. All fields below are None for
    # paper-mode users.
    trading_mode: TradingMode = TradingMode.PAPER
    daily_loss_cap_usd: Optional[Decimal] = None
    kyc_attestation_hash: Optional[str] = None
    totp_secret_encrypted: Optional[str] = None
    live_mode_enabled_at: Optional[datetime] = None

    def enable_live_mode(
        self,
        kyc_attestation_hash: str,
        daily_loss_cap_usd: Decimal,
        totp_secret_encrypted: str,
    ) -> 'User':
        """Return a new User with live mode enabled.

        Caller is responsible for having validated all four gates (KYC
        attestation, TOTP setup, daily loss cap set, risk acknowledgement).
        The entity enforces the shape, not the policy — the policy lives in
        the use case / router that assembles these arguments.
        """
        from dataclasses import replace
        if daily_loss_cap_usd <= 0:
            raise ValueError("daily_loss_cap_usd must be positive")
        if not kyc_attestation_hash:
            raise ValueError("kyc_attestation_hash required")
        if not totp_secret_encrypted:
            raise ValueError("totp_secret_encrypted required")
        return replace(
            self,
            trading_mode=TradingMode.LIVE,
            kyc_attestation_hash=kyc_attestation_hash,
            daily_loss_cap_usd=daily_loss_cap_usd,
            totp_secret_encrypted=totp_secret_encrypted,
            live_mode_enabled_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

    def revert_to_paper(self) -> 'User':
        """Flip back to paper mode (safe anytime — no gates).

        Does NOT wipe TOTP secret or KYC hash; those persist so a subsequent
        re-enable is a TOTP challenge away rather than a full KYC re-do.
        """
        from dataclasses import replace
        return replace(
            self,
            trading_mode=TradingMode.PAPER,
            updated_at=datetime.now(timezone.utc),
        )

    def update_risk_tolerance(self, new_risk_tolerance: RiskTolerance) -> 'User':
        """Update user's risk tolerance and return new instance"""
        from dataclasses import replace
        return replace(self, risk_tolerance=new_risk_tolerance, updated_at=datetime.now(timezone.utc))
    
    def update_investment_goal(self, new_investment_goal: InvestmentGoal) -> 'User':
        """Update user's investment goal and return new instance"""
        from dataclasses import replace
        return replace(self, investment_goal=new_investment_goal, updated_at=datetime.now(timezone.utc))
    
    def update_loss_limits(
        self,
        daily_limit: Optional[Money] = None,
        weekly_limit: Optional[Money] = None,
        monthly_limit: Optional[Money] = None
    ) -> 'User':
        """Update user's loss limits and return new instance"""
        from dataclasses import replace
        return replace(
            self,
            daily_loss_limit=daily_limit or self.daily_loss_limit,
            weekly_loss_limit=weekly_limit or self.weekly_loss_limit,
            monthly_loss_limit=monthly_limit or self.monthly_loss_limit,
            updated_at=datetime.now(timezone.utc),
        )
    
    def update_sector_preferences(self, preferred_sectors: List[str], excluded_sectors: List[str]) -> 'User':
        """Update user's sector preferences and return new instance"""
        from dataclasses import replace
        return replace(
            self,
            sector_preferences=preferred_sectors,
            sector_exclusions=excluded_sectors,
            updated_at=datetime.now(timezone.utc),
        )
    
    def toggle_approval_mode(self, enabled: bool) -> 'User':
        """Toggle approval mode and return new instance"""
        from dataclasses import replace
        return replace(self, approval_mode_enabled=enabled, updated_at=datetime.now(timezone.utc))
    
    def validate(self) -> List[str]:
        """Validate the user and return a list of validation errors"""
        errors = []
        
        if not self.email or '@' not in self.email:
            errors.append("Invalid email address")
        
        if self.max_position_size_percentage <= 0 or self.max_position_size_percentage > 100:
            errors.append("Max position size must be between 0 and 100 percent")
        
        return errors