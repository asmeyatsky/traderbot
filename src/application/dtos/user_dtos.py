"""
User Data Transfer Objects

DTOs for user management and API responses.
"""
from pydantic import BaseModel, Field, EmailStr
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


class ChangePasswordRequest(BaseModel):
    """Request DTO for changing password."""

    current_password: str = Field(..., min_length=8)
    new_password: str = Field(..., min_length=8, max_length=128)


class LoginRequest(BaseModel):
    """Request DTO for user login."""

    email: EmailStr
    password: str = Field(..., min_length=1)


class LoginResponse(BaseModel):
    """Response DTO for login."""

    access_token: str
    token_type: str = "bearer"
    user: UserResponse
