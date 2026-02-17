"""
Users API Router

This router handles user authentication, profile management, preference updates,
and GDPR data subject rights (export, deletion).

Security:
- Password verification on login (bcrypt)
- Rate limiting on auth endpoints (5/min login, 3/min register)
- JWT token blacklisting on logout
- Audit logging for all security events
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from datetime import datetime
from decimal import Decimal
import uuid
import logging

from slowapi import Limiter
from slowapi.util import get_remote_address

from src.infrastructure.security import (
    get_current_user, SecurityManager, TokenResponse
)
from src.application.dtos.user_dtos import (
    CreateUserRequest, UpdateUserRequest, UserResponse,
    UpdateRiskSettingsRequest, UpdateSectorPreferencesRequest,
    LoginRequest, LoginResponse, ChangePasswordRequest,
    UpdateAutoTradingRequest,
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

# Rate limiter for auth endpoints
limiter = Limiter(key_func=get_remote_address)


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
        429: {"description": "Rate limit exceeded"},
    }
)
@limiter.limit("3/minute")
async def register(
    request: Request,
    body: CreateUserRequest,
    user_repository: UserRepository = Depends(get_user_repository),
) -> UserResponse:
    """
    Register a new user account.

    Rate limited to 3 requests per minute per IP.
    """
    try:
        logger.info(f"Registering new user: {body.email}")

        # Check if email already exists
        existing_user = user_repository.get_by_email(body.email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )

        # Hash the password
        password_hash = SecurityManager.hash_password(body.password)

        # Create user entity
        now = datetime.utcnow()
        user = User(
            id=str(uuid.uuid4()),
            email=body.email,
            first_name=body.first_name,
            last_name=body.last_name,
            created_at=now,
            updated_at=now,
            risk_tolerance=RiskTolerance[body.risk_tolerance],
            investment_goal=InvestmentGoal[body.investment_goal],
            max_position_size_percentage=Decimal("5.0"),
            daily_loss_limit=None,
            weekly_loss_limit=None,
            monthly_loss_limit=None,
            sector_preferences=[],
            sector_exclusions=[],
        )

        # Save user with password hash
        saved_user = user_repository.save(user, password_hash=password_hash)

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
        429: {"description": "Rate limit exceeded"},
    }
)
@limiter.limit("5/minute")
async def login(
    request: Request,
    body: LoginRequest,
    user_repository: UserRepository = Depends(get_user_repository),
) -> LoginResponse:
    """
    Authenticate user and return access token.

    Rate limited to 5 requests per minute per IP.
    Verifies password hash before issuing JWT token.
    """
    try:
        logger.info(f"Login attempt for user: {body.email}")

        # Find user by email
        user = user_repository.get_by_email(body.email)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )

        # Retrieve password hash from ORM and verify
        password_hash = user_repository.get_password_hash(body.email)
        if not password_hash or not SecurityManager.verify_password(body.password, password_hash):
            logger.warning(f"Failed login attempt for: {body.email}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )

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
    """Get the profile of the currently authenticated user."""
    try:
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
    body: UpdateUserRequest,
    user_id: str = Depends(get_current_user),
    user_repository: UserRepository = Depends(get_user_repository),
) -> UserResponse:
    """Update the profile of the currently authenticated user."""
    try:
        user = user_repository.get_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        from dataclasses import replace
        updates = {}

        if body.first_name is not None:
            updates['first_name'] = body.first_name
        if body.last_name is not None:
            updates['last_name'] = body.last_name
        if body.risk_tolerance is not None:
            updates['risk_tolerance'] = RiskTolerance[body.risk_tolerance]
        if body.investment_goal is not None:
            updates['investment_goal'] = InvestmentGoal[body.investment_goal]
        if body.email_notifications_enabled is not None:
            updates['email_notifications_enabled'] = body.email_notifications_enabled
        if body.sms_notifications_enabled is not None:
            updates['sms_notifications_enabled'] = body.sms_notifications_enabled
        if body.approval_mode_enabled is not None:
            updates['approval_mode_enabled'] = body.approval_mode_enabled

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
    body: UpdateRiskSettingsRequest,
    user_id: str = Depends(get_current_user),
    user_repository: UserRepository = Depends(get_user_repository),
) -> UserResponse:
    """Update risk management settings for the user."""
    try:
        user = user_repository.get_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        from dataclasses import replace
        updates = {'updated_at': datetime.utcnow()}

        if body.daily_loss_limit is not None:
            updates['daily_loss_limit'] = Money(Decimal(str(body.daily_loss_limit)), "USD")
        if body.weekly_loss_limit is not None:
            updates['weekly_loss_limit'] = Money(Decimal(str(body.weekly_loss_limit)), "USD")
        if body.monthly_loss_limit is not None:
            updates['monthly_loss_limit'] = Money(Decimal(str(body.monthly_loss_limit)), "USD")
        if body.max_position_size_percentage is not None:
            updates['max_position_size_percentage'] = Decimal(str(body.max_position_size_percentage))

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
    body: UpdateSectorPreferencesRequest,
    user_id: str = Depends(get_current_user),
    user_repository: UserRepository = Depends(get_user_repository),
) -> UserResponse:
    """Update sector preferences and exclusions."""
    try:
        user = user_repository.get_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        from dataclasses import replace
        updates = {
            'sector_preferences': body.preferred_sectors,
            'sector_exclusions': body.excluded_sectors,
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


@router.get(
    "/me/auto-trading",
    summary="Get auto-trading settings",
    responses={
        200: {"description": "Auto-trading settings retrieved"},
        401: {"description": "Unauthorized"},
    }
)
async def get_auto_trading_settings(
    user_id: str = Depends(get_current_user),
    user_repository: UserRepository = Depends(get_user_repository),
):
    """Get current auto-trading configuration for the authenticated user."""
    try:
        user = user_repository.get_by_id(user_id)
        if not user:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

        return {
            "enabled": user.auto_trading_enabled,
            "watchlist": user.watchlist,
            "trading_budget": float(user.trading_budget.amount) if user.trading_budget else None,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching auto-trading settings: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to fetch settings")


@router.patch(
    "/me/auto-trading",
    summary="Update auto-trading settings",
    responses={
        200: {"description": "Auto-trading settings updated"},
        401: {"description": "Unauthorized"},
    }
)
async def update_auto_trading_settings(
    body: UpdateAutoTradingRequest,
    user_id: str = Depends(get_current_user),
    user_repository: UserRepository = Depends(get_user_repository),
):
    """Update auto-trading configuration."""
    try:
        user = user_repository.get_by_id(user_id)
        if not user:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

        from dataclasses import replace as dc_replace
        updates = {"updated_at": datetime.utcnow()}

        if body.enabled is not None:
            updates["auto_trading_enabled"] = body.enabled
        if body.watchlist is not None:
            updates["watchlist"] = body.watchlist
        if body.trading_budget is not None:
            updates["trading_budget"] = Money(Decimal(str(body.trading_budget)), "USD")

        updated_user = dc_replace(user, **updates)
        saved = user_repository.update(updated_user)

        return {
            "enabled": saved.auto_trading_enabled,
            "watchlist": saved.watchlist,
            "trading_budget": float(saved.trading_budget.amount) if saved.trading_budget else None,
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating auto-trading settings: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update settings")


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
    body: ChangePasswordRequest,
    user_id: str = Depends(get_current_user),
    user_repository: UserRepository = Depends(get_user_repository),
) -> None:
    """Change the password for the current user."""
    try:
        logger.info(f"Password change requested for user {user_id}")

        user = user_repository.get_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        # Verify current password
        current_hash = user_repository.get_password_hash(user.email)
        if not current_hash or not SecurityManager.verify_password(body.current_password, current_hash):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid current password"
            )

        # Hash and save new password
        new_hash = SecurityManager.hash_password(body.new_password)
        user_repository.update_password_hash(user_id, new_hash)

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
async def logout(
    request: Request,
    user_id: str = Depends(get_current_user),
) -> None:
    """
    Logout the current user.

    Adds the JWT token's JTI to a Redis blacklist so it cannot be reused.
    """
    logger.info(f"User {user_id} logged out")

    # Blacklist the token in Redis
    try:
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            token_payload = SecurityManager.verify_token(token)
            remaining_seconds = int(
                (token_payload.exp - datetime.utcnow()).total_seconds()
            )
            if remaining_seconds > 0:
                from src.infrastructure.cache import get_cache_manager
                cache = get_cache_manager()
                cache.set(
                    f"blacklist:token:{token}",
                    "revoked",
                    ttl=remaining_seconds,
                )
    except Exception as e:
        logger.warning(f"Failed to blacklist token on logout: {e}")


# ============================================================================
# GDPR Data Subject Rights (Articles 15, 17, 20)
# ============================================================================


@router.get(
    "/me/export",
    summary="Export user data (GDPR Article 15/20)",
    responses={
        200: {"description": "User data exported as JSON"},
        401: {"description": "Unauthorized"},
    }
)
async def export_user_data(
    user_id: str = Depends(get_current_user),
    user_repository: UserRepository = Depends(get_user_repository),
):
    """
    Export all personal data for the authenticated user (GDPR Articles 15 & 20).

    Returns a JSON object with all user data including profile, preferences,
    and account settings. Does not include derived analytics or trading positions
    (those are available via their respective endpoints).
    """
    try:
        user = user_repository.get_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        export_data = {
            "data_export": {
                "exported_at": datetime.utcnow().isoformat() + "Z",
                "user": {
                    "id": user.id,
                    "email": user.email,
                    "first_name": user.first_name,
                    "last_name": user.last_name,
                    "created_at": user.created_at.isoformat() if user.created_at else None,
                    "updated_at": user.updated_at.isoformat() if user.updated_at else None,
                    "is_active": user.is_active,
                },
                "preferences": {
                    "risk_tolerance": user.risk_tolerance.name if hasattr(user.risk_tolerance, 'name') else str(user.risk_tolerance),
                    "investment_goal": user.investment_goal.name if hasattr(user.investment_goal, 'name') else str(user.investment_goal),
                    "max_position_size_percentage": float(user.max_position_size_percentage),
                    "sector_preferences": user.sector_preferences or [],
                    "sector_exclusions": user.sector_exclusions or [],
                },
                "loss_limits": {
                    "daily": float(user.daily_loss_limit.amount) if user.daily_loss_limit else None,
                    "weekly": float(user.weekly_loss_limit.amount) if user.weekly_loss_limit else None,
                    "monthly": float(user.monthly_loss_limit.amount) if user.monthly_loss_limit else None,
                },
                "notification_settings": {
                    "email_notifications_enabled": user.email_notifications_enabled,
                    "sms_notifications_enabled": user.sms_notifications_enabled,
                    "approval_mode_enabled": user.approval_mode_enabled,
                },
            }
        }

        logger.info(f"GDPR data export for user {user_id}")
        return export_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting user data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export user data"
        )


@router.delete(
    "/me/data",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete/anonymize user data (GDPR Article 17)",
    responses={
        204: {"description": "User data anonymized"},
        401: {"description": "Unauthorized"},
    }
)
async def delete_user_data(
    user_id: str = Depends(get_current_user),
    user_repository: UserRepository = Depends(get_user_repository),
) -> None:
    """
    Anonymize all personal data for the authenticated user (GDPR Article 17 — Right to Erasure).

    Replaces PII with anonymized values while preserving the account structure
    for referential integrity. The account is deactivated after anonymization.
    """
    try:
        user = user_repository.get_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        user_repository.anonymize_user(user_id)
        logger.info(f"GDPR data deletion (anonymization) for user {user_id}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error anonymizing user data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete user data"
        )
