"""
User Domain Entity

This module contains the user domain entity for the trading platform,
following DDD principles and clean architecture patterns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
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
    allowed_markets: List[str] = field(default_factory=lambda: ["US_NYSE", "US_NASDAQ"])

    def update_risk_tolerance(self, new_risk_tolerance: RiskTolerance) -> 'User':
        """Update user's risk tolerance and return new instance"""
        from dataclasses import replace
        return replace(self, risk_tolerance=new_risk_tolerance, updated_at=datetime.now())
    
    def update_investment_goal(self, new_investment_goal: InvestmentGoal) -> 'User':
        """Update user's investment goal and return new instance"""
        from dataclasses import replace
        return replace(self, investment_goal=new_investment_goal, updated_at=datetime.now())
    
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
            updated_at=datetime.now(),
        )
    
    def update_sector_preferences(self, preferred_sectors: List[str], excluded_sectors: List[str]) -> 'User':
        """Update user's sector preferences and return new instance"""
        from dataclasses import replace
        return replace(
            self,
            sector_preferences=preferred_sectors,
            sector_exclusions=excluded_sectors,
            updated_at=datetime.now(),
        )
    
    def toggle_approval_mode(self, enabled: bool) -> 'User':
        """Toggle approval mode and return new instance"""
        from dataclasses import replace
        return replace(self, approval_mode_enabled=enabled, updated_at=datetime.now())
    
    def validate(self) -> List[str]:
        """Validate the user and return a list of validation errors"""
        errors = []
        
        if not self.email or '@' not in self.email:
            errors.append("Invalid email address")
        
        if self.max_position_size_percentage <= 0 or self.max_position_size_percentage > 100:
            errors.append("Max position size must be between 0 and 100 percent")
        
        return errors