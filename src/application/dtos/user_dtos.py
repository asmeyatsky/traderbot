"""
User Data Transfer Objects

DTOs for user management and API responses.
"""
from pydantic import BaseModel, Field, EmailStr, field_validator
from typing import Optional, List
from datetime import datetime


class CreateUserRequest(BaseModel):
    """Request DTO for creating a new user."""

    email: EmailStr = Field(..., description="User email address")
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=8, max_length=128)
    risk_tolerance: str = Field(
        ...,
        pattern="^(CONSERVATIVE|MODERATE|AGGRESSIVE)$",
        description="Risk tolerance level"
    )
    investment_goal: str = Field(
        ...,
        pattern="^(CAPITAL_PRESERVATION|BALANCED_GROWTH|MAXIMUM_RETURNS)$",
        description="Investment goal"
    )


class UpdateUserRequest(BaseModel):
    """Request DTO for updating user settings."""

    first_name: Optional[str] = Field(None, min_length=1, max_length=100)
    last_name: Optional[str] = Field(None, min_length=1, max_length=100)
    risk_tolerance: Optional[str] = Field(
        None,
        pattern="^(CONSERVATIVE|MODERATE|AGGRESSIVE)$"
    )
    investment_goal: Optional[str] = Field(
        None,
        pattern="^(CAPITAL_PRESERVATION|BALANCED_GROWTH|MAXIMUM_RETURNS)$"
    )
    email_notifications_enabled: Optional[bool] = None
    sms_notifications_enabled: Optional[bool] = None
    approval_mode_enabled: Optional[bool] = None
    allowed_markets: Optional[List[str]] = None


class UpdateRiskSettingsRequest(BaseModel):
    """Request DTO for updating risk settings."""

    daily_loss_limit: Optional[float] = Field(None, gt=0)
    weekly_loss_limit: Optional[float] = Field(None, gt=0)
    monthly_loss_limit: Optional[float] = Field(None, gt=0)
    max_position_size_percentage: Optional[float] = Field(None, ge=0.1, le=100)


class UpdateSectorPreferencesRequest(BaseModel):
    """Request DTO for updating sector preferences."""

    preferred_sectors: List[str] = Field(default_factory=list)
    excluded_sectors: List[str] = Field(default_factory=list)


class UpdateDisciplineRulesRequest(BaseModel):
    """Request DTO for updating discipline rules and trading philosophy (Phase 10.1).

    Both fields are independently optional so the client can update either
    without round-tripping the other. Setting `discipline_rules` to an empty
    list clears all rules; setting `trading_philosophy` to an empty string or
    None clears the philosophy.
    """

    discipline_rules: Optional[List[str]] = Field(
        default=None,
        description=(
            "Free-form user-written rules; each is evaluated independently "
            "by the pre-trade AI veto. Max 50 rules, 500 chars each."
        ),
        max_length=50,
    )
    trading_philosophy: Optional[str] = Field(
        default=None,
        max_length=4000,
        description=(
            "Paragraph describing your trading approach. Used as additional "
            "context by the pre-trade AI veto for ambiguous orders."
        ),
    )

    @field_validator("discipline_rules")
    @classmethod
    def validate_rule_shape(cls, v):
        if v is None:
            return v
        cleaned = []
        for rule in v:
            if not isinstance(rule, str):
                raise ValueError("Every rule must be a string")
            stripped = rule.strip()
            if not stripped:
                continue  # silently drop blanks
            if len(stripped) > 500:
                raise ValueError("Each rule is limited to 500 characters")
            cleaned.append(stripped)
        return cleaned


class UserResponse(BaseModel):
    """Response DTO for user details."""

    id: str
    email: str
    first_name: str
    last_name: str
    risk_tolerance: str
    investment_goal: str
    max_position_size_percentage: float
    daily_loss_limit: Optional[float] = None
    weekly_loss_limit: Optional[float] = None
    monthly_loss_limit: Optional[float] = None
    sector_preferences: List[str]
    sector_exclusions: List[str]
    is_active: bool
    email_notifications_enabled: bool
    sms_notifications_enabled: bool
    approval_mode_enabled: bool
    allowed_markets: List[str]
    # Phase 10.1 — discipline coach state. Defaults keep existing users at
    # zero rules / no philosophy, so the check is a no-op until they opt in.
    discipline_rules: List[str] = []
    trading_philosophy: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class UserPreferencesResponse(BaseModel):
    """Response DTO for user preferences."""

    risk_tolerance: str
    investment_goal: str
    max_position_size_percentage: float
    daily_loss_limit: Optional[float]
    weekly_loss_limit: Optional[float]
    monthly_loss_limit: Optional[float]
    sector_preferences: List[str]
    sector_exclusions: List[str]
    email_notifications_enabled: bool
    sms_notifications_enabled: bool
    approval_mode_enabled: bool


class UpdateAutoTradingRequest(BaseModel):
    """Request DTO for updating auto-trading settings."""

    enabled: Optional[bool] = None
    watchlist: Optional[List[str]] = Field(
        None, max_length=50, description="List of ticker symbols (max 50)"
    )
    trading_budget: Optional[float] = Field(None, gt=0, description="Trading budget in USD (must be positive)")
    stop_loss_pct: Optional[float] = Field(None, ge=1, le=25, description="Stop-loss percentage (1-25%)")
    take_profit_pct: Optional[float] = Field(None, ge=5, le=50, description="Take-profit percentage (5-50%)")
    confidence_threshold: Optional[float] = Field(None, ge=0.5, le=0.95, description="Confidence threshold (0.5-0.95)")
    max_position_pct: Optional[float] = Field(None, ge=5, le=50, description="Max position size as % of budget (5-50%)")

    @field_validator("watchlist")
    @classmethod
    def validate_watchlist_symbols(cls, v):
        if v is not None:
            import re
            pattern = re.compile(r"^[A-Z]{1,5}$")
            validated = []
            for symbol in v:
                upper = symbol.upper()
                if not pattern.match(upper):
                    raise ValueError(f"Invalid ticker symbol: {symbol}. Must be 1-5 uppercase letters.")
                validated.append(upper)
            return validated
        return v


class ChangePasswordRequest(BaseModel):
    """Request DTO for changing password."""

    current_password: str = Field(..., min_length=8)
    new_password: str = Field(..., min_length=8, max_length=128)


class LoginRequest(BaseModel):
    """Request DTO for user login."""

    email: EmailStr
    password: str = Field(..., min_length=8)


class LoginResponse(BaseModel):
    """Response DTO for login."""

    access_token: str
    token_type: str = "bearer"
    user: UserResponse
