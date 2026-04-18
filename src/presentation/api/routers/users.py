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
from typing import Optional
import uuid
import logging

from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.infrastructure.security import (
    get_current_user, SecurityManager, TokenResponse
)
from src.application.dtos.user_dtos import (
    CreateUserRequest, UpdateUserRequest, UserResponse,
    UpdateRiskSettingsRequest, UpdateSectorPreferencesRequest,
    LoginRequest, LoginResponse, ChangePasswordRequest,
    UpdateAutoTradingRequest, UpdateDisciplineRulesRequest,
)
from src.presentation.api.dependencies import (
    get_user_repository,
    get_user_preferences_use_case,
    get_portfolio_repository,
)
from src.infrastructure.repositories import UserRepository, PortfolioRepository
from src.domain.entities.user import User, RiskTolerance, InvestmentGoal
from src.domain.entities.trading import Portfolio
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
        allowed_markets=user.allowed_markets if hasattr(user, 'allowed_markets') else ["US_NYSE", "US_NASDAQ", "UK_LSE", "EU_EURONEXT", "DE_XETRA", "JP_TSE", "HK_HKEX"],
        discipline_rules=getattr(user, 'discipline_rules', []) or [],
        trading_philosophy=getattr(user, 'trading_philosophy', None),
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
    portfolio_repository: PortfolioRepository = Depends(get_portfolio_repository),
) -> UserResponse:
    """
    Register a new user account.

    Rate limited to 3 requests per minute per IP.
    """
    try:
        logger.info(f"Registering new user: {body.email}")

        # Check if email already exists (generic message to prevent enumeration)
        existing_user = user_repository.get_by_email(body.email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Registration failed. Please check your input and try again."
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
            max_position_size_percentage=Decimal("25"),
            daily_loss_limit=None,
            weekly_loss_limit=None,
            monthly_loss_limit=None,
            sector_preferences=[],
            sector_exclusions=[],
        )

        # Save user with password hash
        saved_user = user_repository.save(user, password_hash=password_hash)

        # Auto-create portfolio with $10,000 paper-trading cash
        portfolio = Portfolio(
            id=str(uuid.uuid4()),
            user_id=saved_user.id,
            positions=[],
            created_at=now,
            updated_at=now,
            cash_balance=Money(Decimal('10000'), 'USD'),
        )
        portfolio_repository.save(portfolio)

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
    Emits an audit event on every outcome (success or failure) per 2026 rules §4.
    """
    from src.infrastructure.adapters.audit_event_sink import audit_event_sink
    from src.domain.ports.audit_event_sink import AuditEvent
    from src.infrastructure.observability import get_correlation_id
    from datetime import datetime as _dt
    import uuid as _uuid

    client_ip = (
        request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        or (request.client.host if request.client else "unknown")
    )
    user_agent = request.headers.get("user-agent", "")
    correlation_id = get_correlation_id() or None

    def _emit(action: str, actor: str | None, aggregate_id: str, payload: dict) -> None:
        try:
            audit_event_sink.append(
                AuditEvent(
                    id=str(_uuid.uuid4()),
                    actor_user_id=actor,
                    action=action,
                    aggregate_type="user",
                    aggregate_id=aggregate_id,
                    before_hash=None,
                    after_hash=None,
                    payload_json=payload,
                    occurred_at=_dt.utcnow(),
                    correlation_id=correlation_id,
                    client_ip=client_ip,
                    user_agent=user_agent,
                )
            )
        except Exception:  # noqa: BLE001 — audit must never break login
            logger.exception("audit event emission failed for %s", action)

    try:
        logger.info(f"Login attempt for user: {body.email}")

        # Find user by email
        user = user_repository.get_by_email(body.email)
        if not user:
            _emit("LoginFailed", None, body.email, {"reason": "unknown_email"})
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )

        # Retrieve password hash from ORM and verify
        password_hash = user_repository.get_password_hash(body.email)
        if not password_hash or not SecurityManager.verify_password(body.password, password_hash):
            logger.warning(f"Failed login attempt for: {body.email}")
            _emit("LoginFailed", user.id, body.email, {"reason": "invalid_credentials"})
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )

        # Create access token
        access_token, expire = SecurityManager.create_access_token(user.id)
        _emit("LoginSucceeded", user.id, user.id, {"email": body.email})

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
        if body.allowed_markets is not None:
            updates['allowed_markets'] = body.allowed_markets

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
            "stop_loss_pct": float(user.stop_loss_pct),
            "take_profit_pct": float(user.take_profit_pct),
            "confidence_threshold": float(user.confidence_threshold),
            "max_position_pct": float(user.max_position_pct),
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
        if body.stop_loss_pct is not None:
            updates["stop_loss_pct"] = Decimal(str(body.stop_loss_pct))
        if body.take_profit_pct is not None:
            updates["take_profit_pct"] = Decimal(str(body.take_profit_pct))
        if body.confidence_threshold is not None:
            updates["confidence_threshold"] = Decimal(str(body.confidence_threshold))
        if body.max_position_pct is not None:
            updates["max_position_pct"] = Decimal(str(body.max_position_pct))

        updated_user = dc_replace(user, **updates)
        saved = user_repository.update(updated_user)

        return {
            "enabled": saved.auto_trading_enabled,
            "watchlist": saved.watchlist,
            "trading_budget": float(saved.trading_budget.amount) if saved.trading_budget else None,
            "stop_loss_pct": float(saved.stop_loss_pct),
            "take_profit_pct": float(saved.take_profit_pct),
            "confidence_threshold": float(saved.confidence_threshold),
            "max_position_pct": float(saved.max_position_pct),
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
        429: {"description": "Rate limit exceeded"},
    }
)
@limiter.limit("3/minute")
async def change_password(
    request: Request,
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
        429: {"description": "Rate limit exceeded"},
    }
)
@limiter.limit("2/hour")
async def export_user_data(
    request: Request,
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


# ============================================================================
# Discipline coach — rules + philosophy (Phase 10.1)
# ============================================================================


@router.put(
    "/me/discipline",
    response_model=UserResponse,
    summary="Update discipline rules and trading philosophy",
    responses={
        200: {"description": "Rules updated"},
        401: {"description": "Unauthorized"},
        422: {"description": "Validation error (rule length, count)"},
    },
)
async def update_discipline_rules(
    body: UpdateDisciplineRulesRequest,
    user_id: str = Depends(get_current_user),
    user_repository: UserRepository = Depends(get_user_repository),
) -> UserResponse:
    """Update the user's discipline rules and/or trading philosophy.

    Either field is optional — only the fields present in the request body
    are updated. Setting `discipline_rules` to `[]` explicitly clears all
    rules. Setting `trading_philosophy` to an empty string clears it.
    """
    user = user_repository.get_by_id(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    from dataclasses import replace
    updates = {"updated_at": datetime.utcnow()}
    if body.discipline_rules is not None:
        updates["discipline_rules"] = body.discipline_rules
    if body.trading_philosophy is not None:
        # Treat the empty string as an explicit clear — the UI sends "" when
        # the user deletes the textarea content, and None when they haven't
        # touched the field at all.
        updates["trading_philosophy"] = body.trading_philosophy.strip() or None

    updated = replace(user, **updates)
    saved = user_repository.update(updated)
    return _user_to_response(saved)


# ============================================================================
# Live Trading — per-user mode flip (ADR-002)
# ============================================================================


class EnableLiveModeRequest(BaseModel):
    """All four gates in one request. Any missing field → 400."""
    # KYC attestation — the user confirms they are the account holder, over 18,
    # and trading their own money. Frontend collects, hashes, and sends.
    kyc_attestation_payload: str = Field(..., min_length=20, max_length=4096)
    # The daily-loss cap in USD. At launch the UI restricts this to ≤ 1000.
    daily_loss_cap_usd: float = Field(..., gt=0, le=10000)
    # The user's 6-digit TOTP code from their authenticator app, proving they
    # can reproduce the secret they were shown during TOTP enrollment.
    totp_code: str = Field(..., pattern=r"^\d{6}$")
    # Verbatim risk acknowledgement — MUST match the phrase defined below.
    risk_acknowledgement: str = Field(...)


REQUIRED_RISK_PHRASE = "I understand I will lose real money."


class TotpEnrollmentResponse(BaseModel):
    secret: str  # plaintext — shown once during enrollment
    provisioning_uri: str
    issuer: str = "TraderBot"


class LiveModeStatusResponse(BaseModel):
    trading_mode: str
    daily_loss_cap_usd: Optional[float] = None
    live_mode_enabled_at: Optional[datetime] = None
    has_totp: bool


@router.post(
    "/me/totp/enroll",
    response_model=TotpEnrollmentResponse,
    summary="Generate a TOTP secret for live-trading 2FA",
    responses={
        200: {"description": "TOTP secret generated; show QR code to user"},
        401: {"description": "Unauthorized"},
    },
)
@limiter.limit("5/minute")
async def enroll_totp(
    request: Request,
    user_id: str = Depends(get_current_user),
    user_repository: UserRepository = Depends(get_user_repository),
) -> TotpEnrollmentResponse:
    """Start TOTP enrollment. The plaintext secret is returned ONCE for the
    client to render as a QR code; only the encrypted form is stored."""
    from src.infrastructure.services.totp_service import (
        generate_totp_secret,
        provisioning_uri,
    )

    user = user_repository.get_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    plaintext, encrypted = generate_totp_secret()
    # Persist the encrypted secret so we can verify the user's code in the
    # enable-live-mode step. Overwrites any previous enrollment.
    user_repository.update_totp_secret(user_id, encrypted)

    return TotpEnrollmentResponse(
        secret=plaintext,
        provisioning_uri=provisioning_uri(plaintext, user.email),
    )


@router.post(
    "/me/enable-live-mode",
    response_model=LiveModeStatusResponse,
    summary="Switch the authenticated user from paper to live trading",
    responses={
        200: {"description": "Live mode enabled"},
        400: {"description": "Validation failed (KYC, TOTP, or phrase mismatch)"},
        401: {"description": "Unauthorized"},
        403: {"description": "ENABLE_LIVE_TRADING is off at the platform level"},
    },
)
@limiter.limit("3/minute")
async def enable_live_mode(
    request: Request,
    body: EnableLiveModeRequest,
    user_id: str = Depends(get_current_user),
    user_repository: UserRepository = Depends(get_user_repository),
) -> LiveModeStatusResponse:
    """Flip the user from paper to live (ADR-002).

    Gates (all must pass, in order):
    1. Platform feature flag `ENABLE_LIVE_TRADING=true`.
    2. User has enrolled TOTP (/me/totp/enroll first).
    3. TOTP code verifies.
    4. KYC attestation payload is non-trivial.
    5. Daily-loss cap is in range (already Pydantic-validated).
    6. Risk acknowledgement phrase matches exactly.
    Emits a `LiveModeEnabled` audit event regardless of success or failure.
    """
    from src.infrastructure.adapters.audit_event_sink import audit_event_sink
    from src.domain.ports.audit_event_sink import AuditEvent, hash_state
    from src.infrastructure.services.totp_service import verify_totp
    from src.infrastructure.observability import get_correlation_id
    from decimal import Decimal as _Dec
    import hashlib as _hashlib
    import uuid as _uuid
    import os as _os

    client_ip = (
        request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        or (request.client.host if request.client else "unknown")
    )
    user_agent = request.headers.get("user-agent", "")
    correlation_id = get_correlation_id() or None

    def _emit(action: str, payload: dict) -> None:
        try:
            audit_event_sink.append(
                AuditEvent(
                    id=str(_uuid.uuid4()),
                    actor_user_id=user_id,
                    action=action,
                    aggregate_type="user",
                    aggregate_id=user_id,
                    before_hash=None,
                    after_hash=hash_state(payload),
                    payload_json=payload,
                    occurred_at=datetime.utcnow(),
                    correlation_id=correlation_id,
                    client_ip=client_ip,
                    user_agent=user_agent,
                )
            )
        except Exception:  # noqa: BLE001
            logger.exception("audit emission failed for %s", action)

    # Gate 1: platform-level kill switch
    if _os.getenv("ENABLE_LIVE_TRADING", "false").lower() != "true":
        _emit("LiveModeEnableRejected", {"reason": "feature_flag_off"})
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Live trading is currently disabled at the platform level.",
        )

    user = user_repository.get_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Gate 2 + 3: TOTP enrolled + verifies
    if not user.totp_secret_encrypted:
        _emit("LiveModeEnableRejected", {"reason": "totp_not_enrolled"})
        raise HTTPException(status_code=400, detail="Enrol TOTP first (POST /users/me/totp/enroll).")
    if not verify_totp(user.totp_secret_encrypted, body.totp_code):
        _emit("LiveModeEnableRejected", {"reason": "totp_invalid"})
        raise HTTPException(status_code=400, detail="Invalid TOTP code.")

    # Gate 6: exact phrase match
    if body.risk_acknowledgement != REQUIRED_RISK_PHRASE:
        _emit("LiveModeEnableRejected", {"reason": "risk_phrase_mismatch"})
        raise HTTPException(
            status_code=400,
            detail=f'Risk acknowledgement must be exactly: "{REQUIRED_RISK_PHRASE}"',
        )

    # Gate 4: non-trivial KYC payload → hash it for the audit trail
    kyc_payload = body.kyc_attestation_payload.strip()
    kyc_hash = _hashlib.sha256(kyc_payload.encode("utf-8")).hexdigest()

    updated = user.enable_live_mode(
        kyc_attestation_hash=kyc_hash,
        daily_loss_cap_usd=_Dec(str(body.daily_loss_cap_usd)),
        totp_secret_encrypted=user.totp_secret_encrypted,
    )
    user_repository.save_live_mode_state(updated)
    _emit(
        "LiveModeEnabled",
        {
            "daily_loss_cap_usd": body.daily_loss_cap_usd,
            "kyc_hash_prefix": kyc_hash[:16],
        },
    )

    return LiveModeStatusResponse(
        trading_mode=updated.trading_mode.value,
        daily_loss_cap_usd=float(updated.daily_loss_cap_usd),
        live_mode_enabled_at=updated.live_mode_enabled_at,
        has_totp=True,
    )


@router.post(
    "/me/disable-live-mode",
    response_model=LiveModeStatusResponse,
    summary="Revert the authenticated user from live to paper (no gate)",
)
async def disable_live_mode(
    request: Request,
    user_id: str = Depends(get_current_user),
    user_repository: UserRepository = Depends(get_user_repository),
) -> LiveModeStatusResponse:
    """Immediate, always-allowed flip back to paper mode. ADR-002 mandates this
    remains free — never impose a cooldown on the off-ramp from real money."""
    from src.infrastructure.adapters.audit_event_sink import audit_event_sink
    from src.domain.ports.audit_event_sink import AuditEvent
    from src.infrastructure.observability import get_correlation_id
    import uuid as _uuid

    user = user_repository.get_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    updated = user.revert_to_paper()
    user_repository.save_live_mode_state(updated)

    try:
        audit_event_sink.append(
            AuditEvent(
                id=str(_uuid.uuid4()),
                actor_user_id=user_id,
                action="LiveModeDisabled",
                aggregate_type="user",
                aggregate_id=user_id,
                before_hash=None,
                after_hash=None,
                payload_json={"reason": "user_requested"},
                occurred_at=datetime.utcnow(),
                correlation_id=get_correlation_id() or None,
            )
        )
    except Exception:  # noqa: BLE001
        logger.exception("audit emission failed for LiveModeDisabled")

    return LiveModeStatusResponse(
        trading_mode=updated.trading_mode.value,
        daily_loss_cap_usd=(
            float(updated.daily_loss_cap_usd)
            if updated.daily_loss_cap_usd is not None else None
        ),
        live_mode_enabled_at=updated.live_mode_enabled_at,
        has_totp=bool(updated.totp_secret_encrypted),
    )


@router.get(
    "/me/live-mode-status",
    response_model=LiveModeStatusResponse,
    summary="Report whether the user is in paper or live mode",
)
async def live_mode_status(
    user_id: str = Depends(get_current_user),
    user_repository: UserRepository = Depends(get_user_repository),
) -> LiveModeStatusResponse:
    user = user_repository.get_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return LiveModeStatusResponse(
        trading_mode=user.trading_mode.value,
        daily_loss_cap_usd=(
            float(user.daily_loss_cap_usd)
            if user.daily_loss_cap_usd is not None else None
        ),
        live_mode_enabled_at=user.live_mode_enabled_at,
        has_totp=bool(user.totp_secret_encrypted),
    )
