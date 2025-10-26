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
    MODERATE = "MODERATIVE"        # max 15% drawdown  
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
    max_position_size_percentage: Decimal = Decimal('5')  # Max 5% per stock
    daily_loss_limit: Optional[Money] = None
    weekly_loss_limit: Optional[Money] = None
    monthly_loss_limit: Optional[Money] = None
    sector_preferences: List[str] = field(default_factory=list)  # List of preferred sectors
    sector_exclusions: List[str] = field(default_factory=list)   # List of excluded sectors
    is_active: bool = True
    email_notifications_enabled: bool = True
    sms_notifications_enabled: bool = False
    approval_mode_enabled: bool = False  # Requires approval for trades
    
    def update_risk_tolerance(self, new_risk_tolerance: RiskTolerance) -> 'User':
        """Update user's risk tolerance and return new instance"""
        return User(
            id=self.id,
            email=self.email,
            first_name=self.first_name,
            last_name=self.last_name,
            created_at=self.created_at,
            updated_at=datetime.now(),
            risk_tolerance=new_risk_tolerance,
            investment_goal=self.investment_goal,
            max_position_size_percentage=self.max_position_size_percentage,
            daily_loss_limit=self.daily_loss_limit,
            weekly_loss_limit=self.weekly_loss_limit,
            monthly_loss_limit=self.monthly_loss_limit,
            sector_preferences=self.sector_preferences,
            sector_exclusions=self.sector_exclusions,
            is_active=self.is_active,
            email_notifications_enabled=self.email_notifications_enabled,
            sms_notifications_enabled=self.sms_notifications_enabled,
            approval_mode_enabled=self.approval_mode_enabled
        )
    
    def update_investment_goal(self, new_investment_goal: InvestmentGoal) -> 'User':
        """Update user's investment goal and return new instance"""
        return User(
            id=self.id,
            email=self.email,
            first_name=self.first_name,
            last_name=self.last_name,
            created_at=self.created_at,
            updated_at=datetime.now(),
            risk_tolerance=self.risk_tolerance,
            investment_goal=new_investment_goal,
            max_position_size_percentage=self.max_position_size_percentage,
            daily_loss_limit=self.daily_loss_limit,
            weekly_loss_limit=self.weekly_loss_limit,
            monthly_loss_limit=self.monthly_loss_limit,
            sector_preferences=self.sector_preferences,
            sector_exclusions=self.sector_exclusions,
            is_active=self.is_active,
            email_notifications_enabled=self.email_notifications_enabled,
            sms_notifications_enabled=self.sms_notifications_enabled,
            approval_mode_enabled=self.approval_mode_enabled
        )
    
    def update_loss_limits(
        self, 
        daily_limit: Optional[Money] = None,
        weekly_limit: Optional[Money] = None,
        monthly_limit: Optional[Money] = None
    ) -> 'User':
        """Update user's loss limits and return new instance"""
        return User(
            id=self.id,
            email=self.email,
            first_name=self.first_name,
            last_name=self.last_name,
            created_at=self.created_at,
            updated_at=datetime.now(),
            risk_tolerance=self.risk_tolerance,
            investment_goal=self.investment_goal,
            max_position_size_percentage=self.max_position_size_percentage,
            daily_loss_limit=daily_limit or self.daily_loss_limit,
            weekly_loss_limit=weekly_limit or self.weekly_loss_limit,
            monthly_loss_limit=monthly_limit or self.monthly_loss_limit,
            sector_preferences=self.sector_preferences,
            sector_exclusions=self.sector_exclusions,
            is_active=self.is_active,
            email_notifications_enabled=self.email_notifications_enabled,
            sms_notifications_enabled=self.sms_notifications_enabled,
            approval_mode_enabled=self.approval_mode_enabled
        )
    
    def update_sector_preferences(self, preferred_sectors: List[str], excluded_sectors: List[str]) -> 'User':
        """Update user's sector preferences and return new instance"""
        return User(
            id=self.id,
            email=self.email,
            first_name=self.first_name,
            last_name=self.last_name,
            created_at=self.created_at,
            updated_at=datetime.now(),
            risk_tolerance=self.risk_tolerance,
            investment_goal=self.investment_goal,
            max_position_size_percentage=self.max_position_size_percentage,
            daily_loss_limit=self.daily_loss_limit,
            weekly_loss_limit=self.weekly_loss_limit,
            monthly_loss_limit=self.monthly_loss_limit,
            sector_preferences=preferred_sectors,
            sector_exclusions=excluded_sectors,
            is_active=self.is_active,
            email_notifications_enabled=self.email_notifications_enabled,
            sms_notifications_enabled=self.sms_notifications_enabled,
            approval_mode_enabled=self.approval_mode_enabled
        )
    
    def toggle_approval_mode(self, enabled: bool) -> 'User':
        """Toggle approval mode and return new instance"""
        return User(
            id=self.id,
            email=self.email,
            first_name=self.first_name,
            last_name=self.last_name,
            created_at=self.created_at,
            updated_at=datetime.now(),
            risk_tolerance=self.risk_tolerance,
            investment_goal=self.investment_goal,
            max_position_size_percentage=self.max_position_size_percentage,
            daily_loss_limit=self.daily_loss_limit,
            weekly_loss_limit=self.weekly_loss_limit,
            monthly_loss_limit=self.monthly_loss_limit,
            sector_preferences=self.sector_preferences,
            sector_exclusions=self.sector_exclusions,
            is_active=self.is_active,
            email_notifications_enabled=self.email_notifications_enabled,
            sms_notifications_enabled=self.sms_notifications_enabled,
            approval_mode_enabled=enabled
        )
    
    def validate(self) -> List[str]:
        """Validate the user and return a list of validation errors"""
        errors = []
        
        if not self.email or '@' not in self.email:
            errors.append("Invalid email address")
        
        if self.max_position_size_percentage <= 0 or self.max_position_size_percentage > 100:
            errors.append("Max position size must be between 0 and 100 percent")
        
        return errors