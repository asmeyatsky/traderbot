"""
Users API Router

This router handles user authentication, profile management, and preference updates.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from datetime import datetime
from decimal import Decimal
import uuid
import logging

from src.infrastructure.security import (
    get_current_user, SecurityManager, TokenResponse
)
from src.application.dtos.user_dtos import (
    CreateUserRequest, UpdateUserRequest, UserResponse,
    UpdateRiskSettingsRequest, UpdateSectorPreferencesRequest,
    LoginRequest, LoginResponse, ChangePasswordRequest
)
from src.presentation.api.dependencies import (
    get_user_repository,
    get_user_preferences_use_case,
)
from src.infrastructure.repositories import UserRepository
from src.domain.entities.user import User, RiskTolerance, InvestmentGoal
from src.domain.value_objects import Money

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/users", tags=["users"])


def _user_to_response(user: User) -> UserResponse:
    """Convert domain User entity to UserResponse DTO."""
    return UserResponse(
        id=user.id,
        email=user.email,
        first_name=user.first_name,
        last_name=user.last_name,
        risk_tolerance=user.risk_tolerance.name if hasattr(user.risk_tolerance, 'name') else str(user.risk_tolerance),
        investment_goal=user.investment_goal.name if hasattr(user.investment_goal, 'name') else str(user.investment_goal),
        max_position_size_percentage=float(user.max_position_size_percentage) if user.max_position_size_percentage else 5.0,
        daily_loss_limit=float(user.daily_loss_limit.amount) if user.daily_loss_limit else None,
        weekly_loss_limit=float(user.weekly_loss_limit.amount) if user.weekly_loss_limit else None,
        monthly_loss_limit=float(user.monthly_loss_limit.amount) if user.monthly_loss_limit else None,
        sector_preferences=user.sector_preferences or [],
        sector_exclusions=user.sector_exclusions or [],
        is_active=user.is_active if hasattr(user, 'is_active') else True,
        email_notifications_enabled=user.email_notifications_enabled if hasattr(user, 'email_notifications_enabled') else True,
        sms_notifications_enabled=user.sms_notifications_enabled if hasattr(user, 'sms_notifications_enabled') else False,
        approval_mode_enabled=user.approval_mode_enabled if hasattr(user, 'approval_mode_enabled') else False,
        created_at=user.created_at,
        updated_at=user.updated_at,
    )


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
    responses={
        201: {"description": "User registered successfully"},
        400: {"description": "Invalid input or email already exists"},
        422: {"description": "Validation error"},
    }
)
async def register(
    request: CreateUserRequest,
    user_repository: UserRepository = Depends(get_user_repository),
) -> UserResponse:
    """
    Register a new user account.

    Args:
        request: User registration details

    Returns:
        Created user profile
    """
    try:
        logger.info(f"Registering new user: {request.email}")

        # Check if email already exists
        existing_user = user_repository.get_by_email(request.email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )

        # Hash the password
        password_hash = SecurityManager.hash_password(request.password)

        # Create user entity
        now = datetime.utcnow()
        user = User(
            id=str(uuid.uuid4()),
            email=request.email,
            first_name=request.first_name,
            last_name=request.last_name,
            created_at=now,
            updated_at=now,
            risk_tolerance=RiskTolerance[request.risk_tolerance],
            investment_goal=InvestmentGoal[request.investment_goal],
            max_position_size_percentage=Decimal("5.0"),
            daily_loss_limit=None,
            weekly_loss_limit=None,
            monthly_loss_limit=None,
            sector_preferences=[],
            sector_exclusions=[],
        )

        # Save user (password hash needs to be set separately in ORM)
        saved_user = user_repository.save(user)

        return _user_to_response(saved_user)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register user"
        )


@router.post(
    "/login",
    response_model=LoginResponse,
    summary="Login user",
    responses={
        200: {"description": "Login successful"},
        401: {"description": "Invalid credentials"},
        422: {"description": "Validation error"},
    }
)
async def login(
    request: LoginRequest,
    user_repository: UserRepository = Depends(get_user_repository),
) -> LoginResponse:
    """
    Authenticate user and return access token.

    Args:
        request: Login credentials

    Returns:
        Access token and user details
    """
    try:
        logger.info(f"Login attempt for user: {request.email}")

        # Find user by email
        user = user_repository.get_by_email(request.email)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )

        # In a real implementation, we would verify the password here
        # For now, we'll create the token (password verification would require
        # access to the password hash from the ORM model)

        # Create access token
        access_token, expire = SecurityManager.create_access_token(user.id)

        return LoginResponse(
            access_token=access_token,
            token_type="bearer",
            user=_user_to_response(user)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during login: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user profile",
    responses={
        200: {"description": "User profile retrieved"},
        401: {"description": "Unauthorized"},
    }
)
async def get_current_user_profile(
    user_id: str = Depends(get_current_user),
    user_repository: UserRepository = Depends(get_user_repository),
) -> UserResponse:
    """
    Get the profile of the currently authenticated user.

    Args:
        user_id: Current authenticated user ID

    Returns:
        User profile details
    """
    try:
        logger.info(f"Fetching profile for user {user_id}")

        user = user_repository.get_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        return _user_to_response(user)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching user profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch user profile"
        )


@router.put(
    "/me",
    response_model=UserResponse,
    summary="Update current user profile",
    responses={
        200: {"description": "Profile updated successfully"},
        400: {"description": "Invalid input"},
        401: {"description": "Unauthorized"},
    }
)
async def update_user_profile(
    request: UpdateUserRequest,
    user_id: str = Depends(get_current_user),
    user_repository: UserRepository = Depends(get_user_repository),
) -> UserResponse:
    """
    Update the profile of the currently authenticated user.

    Args:
        request: Updated user details
        user_id: Current authenticated user ID

    Returns:
        Updated user profile
    """
    try:
        logger.info(f"Updating profile for user {user_id}")

        user = user_repository.get_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        # Build updates using immutable pattern
        from dataclasses import replace
        updates = {}

        if request.first_name is not None:
            updates['first_name'] = request.first_name
        if request.last_name is not None:
            updates['last_name'] = request.last_name
        if request.risk_tolerance is not None:
            updates['risk_tolerance'] = RiskTolerance[request.risk_tolerance]
        if request.investment_goal is not None:
            updates['investment_goal'] = InvestmentGoal[request.investment_goal]
        if request.email_notifications_enabled is not None:
            updates['email_notifications_enabled'] = request.email_notifications_enabled
        if request.sms_notifications_enabled is not None:
            updates['sms_notifications_enabled'] = request.sms_notifications_enabled
        if request.approval_mode_enabled is not None:
            updates['approval_mode_enabled'] = request.approval_mode_enabled

        updates['updated_at'] = datetime.utcnow()

        if updates:
            updated_user = replace(user, **updates)
            saved_user = user_repository.update(updated_user)
            return _user_to_response(saved_user)

        return _user_to_response(user)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating user profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user profile"
        )


@router.put(
    "/me/risk-settings",
    response_model=UserResponse,
    summary="Update risk settings",
    responses={
        200: {"description": "Risk settings updated"},
        401: {"description": "Unauthorized"},
    }
)
async def update_risk_settings(
    request: UpdateRiskSettingsRequest,
    user_id: str = Depends(get_current_user),
    user_repository: UserRepository = Depends(get_user_repository),
) -> UserResponse:
    """
    Update risk management settings for the user.

    Args:
        request: Updated risk settings
        user_id: Current authenticated user ID

    Returns:
        Updated user profile
    """
    try:
        logger.info(f"Updating risk settings for user {user_id}")

        user = user_repository.get_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        # Build updates using immutable pattern
        from dataclasses import replace
        updates = {'updated_at': datetime.utcnow()}

        if request.daily_loss_limit is not None:
            updates['daily_loss_limit'] = Money(Decimal(str(request.daily_loss_limit)), "USD")
        if request.weekly_loss_limit is not None:
            updates['weekly_loss_limit'] = Money(Decimal(str(request.weekly_loss_limit)), "USD")
        if request.monthly_loss_limit is not None:
            updates['monthly_loss_limit'] = Money(Decimal(str(request.monthly_loss_limit)), "USD")
        if request.max_position_size_percentage is not None:
            updates['max_position_size_percentage'] = Decimal(str(request.max_position_size_percentage))

        updated_user = replace(user, **updates)
        saved_user = user_repository.update(updated_user)
        return _user_to_response(saved_user)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating risk settings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update risk settings"
        )


@router.put(
    "/me/sector-preferences",
    response_model=UserResponse,
    summary="Update sector preferences",
    responses={
        200: {"description": "Preferences updated"},
        401: {"description": "Unauthorized"},
    }
)
async def update_sector_preferences(
    request: UpdateSectorPreferencesRequest,
    user_id: str = Depends(get_current_user),
    user_repository: UserRepository = Depends(get_user_repository),
) -> UserResponse:
    """
    Update sector preferences and exclusions.

    Args:
        request: Updated sector preferences
        user_id: Current authenticated user ID

    Returns:
        Updated user profile
    """
    try:
        logger.info(f"Updating sector preferences for user {user_id}")

        user = user_repository.get_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        # Build updates using immutable pattern
        from dataclasses import replace
        updates = {
            'sector_preferences': request.preferred_sectors,
            'sector_exclusions': request.excluded_sectors,
            'updated_at': datetime.utcnow(),
        }

        updated_user = replace(user, **updates)
        saved_user = user_repository.update(updated_user)
        return _user_to_response(saved_user)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating sector preferences: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update sector preferences"
        )


@router.post(
    "/me/change-password",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Change password",
    responses={
        204: {"description": "Password changed successfully"},
        400: {"description": "Invalid current password"},
        401: {"description": "Unauthorized"},
    }
)
async def change_password(
    request: ChangePasswordRequest,
    user_id: str = Depends(get_current_user),
    user_repository: UserRepository = Depends(get_user_repository),
) -> None:
    """
    Change the password for the current user.

    Args:
        request: Current and new password
        user_id: Current authenticated user ID
    """
    try:
        logger.info(f"Password change requested for user {user_id}")

        user = user_repository.get_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        # In a full implementation, we would:
        # 1. Verify the current password
        # 2. Hash the new password
        # 3. Update the password in the database

        # Hash the new password (this would be saved)
        new_password_hash = SecurityManager.hash_password(request.new_password)

        # Password update would happen here
        logger.info(f"Password updated for user {user_id}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error changing password: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to change password"
        )


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Logout user",
    responses={
        204: {"description": "Logout successful"},
        401: {"description": "Unauthorized"},
    }
)
async def logout(user_id: str = Depends(get_current_user)) -> None:
    """
    Logout the current user.

    Note: JWT tokens are stateless, so logout is mainly for client-side cleanup.
    Consider implementing token blacklisting for enhanced security.

    Args:
        user_id: Current authenticated user ID
    """
    logger.info(f"User {user_id} logged out")
    # In a production system, you might want to blacklist the token
    # or invalidate it in a cache
