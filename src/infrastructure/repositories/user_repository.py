"""
User Repository Implementation

Implements the UserRepositoryPort for user persistence operations.
Includes password hash management and GDPR anonymization support.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional
from sqlalchemy.orm import Session
import logging
import uuid

from src.domain.entities.user import User, RiskTolerance, InvestmentGoal
from src.domain.value_objects import Money
from src.domain.ports import UserRepositoryPort
from src.infrastructure.orm_models import UserORM
from src.infrastructure.repositories.base_repository import BaseRepository
from src.infrastructure.database import get_database_manager

logger = logging.getLogger(__name__)


class UserRepository(BaseRepository[User, UserORM], UserRepositoryPort):
    """
    Repository for User entity persistence.

    Implements UserRepositoryPort and provides all user-related persistence operations,
    including password hash management and GDPR anonymization.
    """

    def __init__(self):
        """Initialize UserRepository with UserORM model class."""
        super().__init__(UserORM)

    def _to_domain_entity(self, orm_obj: UserORM) -> User:
        """Convert UserORM to domain User entity."""
        if not orm_obj:
            return None

        return User(
            id=orm_obj.id,
            email=orm_obj.email,
            first_name=orm_obj.first_name,
            last_name=orm_obj.last_name,
            created_at=orm_obj.created_at,
            updated_at=orm_obj.updated_at,
            risk_tolerance=orm_obj.risk_tolerance,
            investment_goal=orm_obj.investment_goal,
            max_position_size_percentage=orm_obj.max_position_size_percentage,
            daily_loss_limit=Money(orm_obj.daily_loss_limit, "USD") if orm_obj.daily_loss_limit else None,
            weekly_loss_limit=Money(orm_obj.weekly_loss_limit, "USD") if orm_obj.weekly_loss_limit else None,
            monthly_loss_limit=Money(orm_obj.monthly_loss_limit, "USD") if orm_obj.monthly_loss_limit else None,
            sector_preferences=orm_obj.sector_preferences or [],
            sector_exclusions=orm_obj.sector_exclusions or [],
            is_active=orm_obj.is_active,
            email_notifications_enabled=orm_obj.email_notifications_enabled,
            sms_notifications_enabled=orm_obj.sms_notifications_enabled,
            approval_mode_enabled=orm_obj.approval_mode_enabled,
            terms_accepted_at=orm_obj.terms_accepted_at,
            privacy_accepted_at=orm_obj.privacy_accepted_at,
            marketing_consent=orm_obj.marketing_consent,
            auto_trading_enabled=orm_obj.auto_trading_enabled,
            watchlist=orm_obj.watchlist or [],
            trading_budget=Money(orm_obj.trading_budget, "USD") if orm_obj.trading_budget else None,
            stop_loss_pct=orm_obj.stop_loss_pct,
            take_profit_pct=orm_obj.take_profit_pct,
            confidence_threshold=orm_obj.confidence_threshold,
            max_position_pct=orm_obj.max_position_pct,
        )

    def _to_orm_model(self, entity: User, password_hash: str = "") -> UserORM:
        """Convert User domain entity to UserORM."""
        return UserORM(
            id=entity.id,
            email=entity.email,
            first_name=entity.first_name,
            last_name=entity.last_name,
            password_hash=password_hash,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
            risk_tolerance=entity.risk_tolerance,
            investment_goal=entity.investment_goal,
            max_position_size_percentage=entity.max_position_size_percentage,
            daily_loss_limit=entity.daily_loss_limit.amount if entity.daily_loss_limit else None,
            weekly_loss_limit=entity.weekly_loss_limit.amount if entity.weekly_loss_limit else None,
            monthly_loss_limit=entity.monthly_loss_limit.amount if entity.monthly_loss_limit else None,
            sector_preferences=entity.sector_preferences,
            sector_exclusions=entity.sector_exclusions,
            is_active=entity.is_active,
            email_notifications_enabled=entity.email_notifications_enabled,
            sms_notifications_enabled=entity.sms_notifications_enabled,
            approval_mode_enabled=entity.approval_mode_enabled,
            terms_accepted_at=entity.terms_accepted_at,
            privacy_accepted_at=entity.privacy_accepted_at,
            marketing_consent=entity.marketing_consent,
            auto_trading_enabled=entity.auto_trading_enabled,
            watchlist=entity.watchlist,
            trading_budget=entity.trading_budget.amount if entity.trading_budget else None,
            stop_loss_pct=entity.stop_loss_pct,
            take_profit_pct=entity.take_profit_pct,
            confidence_threshold=entity.confidence_threshold,
            max_position_pct=entity.max_position_pct,
        )

    def save(self, entity: User, password_hash: str = "") -> User:
        """
        Save a new user with their password hash.

        Args:
            entity: User domain entity
            password_hash: Bcrypt-hashed password

        Returns:
            Saved User domain entity
        """
        from src.domain.exceptions import DomainException

        session = self._get_session()
        try:
            orm_obj = self._to_orm_model(entity, password_hash=password_hash)
            session.add(orm_obj)
            session.commit()
            session.refresh(orm_obj)
            return self._to_domain_entity(orm_obj)
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save user: {e}")
            raise DomainException(f"Failed to save entity: {str(e)}")
        finally:
            session.close()

    def get_by_email(self, email: str) -> Optional[User]:
        """Retrieve a user by email address."""
        db_manager = get_database_manager()
        session = db_manager._session_factory()
        try:
            orm_obj = session.query(UserORM).filter(
                UserORM.email == email
            ).first()
            return self._to_domain_entity(orm_obj) if orm_obj else None
        except Exception as e:
            logger.error(f"Failed to get user by email: {e}")
            return None
        finally:
            session.close()

    def get_password_hash(self, email: str) -> Optional[str]:
        """
        Retrieve the password hash for a user by email.

        This is separated from get_by_email to keep password hashes
        out of the domain entity (which has no password_hash field).

        Args:
            email: User's email address

        Returns:
            Bcrypt password hash string, or None if user not found
        """
        db_manager = get_database_manager()
        session = db_manager._session_factory()
        try:
            orm_obj = session.query(UserORM).filter(
                UserORM.email == email
            ).first()
            return orm_obj.password_hash if orm_obj else None
        except Exception as e:
            logger.error(f"Failed to get password hash: {e}")
            return None
        finally:
            session.close()

    def update_password_hash(self, user_id: str, new_hash: str) -> None:
        """
        Update the password hash for a user.

        Args:
            user_id: User ID
            new_hash: New bcrypt password hash
        """
        db_manager = get_database_manager()
        session = db_manager._session_factory()
        try:
            orm_obj = session.query(UserORM).filter(
                UserORM.id == user_id
            ).first()
            if orm_obj:
                orm_obj.password_hash = new_hash
                orm_obj.updated_at = datetime.utcnow()
                session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to update password hash: {e}")
            raise
        finally:
            session.close()

    def get_all_active(self) -> list[User]:
        """Retrieve all active users."""
        db_manager = get_database_manager()
        session = db_manager._session_factory()
        try:
            orm_objs = session.query(UserORM).filter(
                UserORM.is_active == True
            ).all()
            return [self._to_domain_entity(orm_obj) for orm_obj in orm_objs]
        except Exception as e:
            logger.error(f"Failed to get active users: {e}")
            return []
        finally:
            session.close()

    def get_auto_trading_users(self) -> list[User]:
        """Retrieve all active users with auto-trading enabled."""
        db_manager = get_database_manager()
        session = db_manager._session_factory()
        try:
            orm_objs = session.query(UserORM).filter(
                UserORM.is_active == True,
                UserORM.auto_trading_enabled == True,
            ).all()
            return [self._to_domain_entity(orm_obj) for orm_obj in orm_objs]
        except Exception as e:
            logger.error(f"Failed to get auto-trading users: {e}")
            return []
        finally:
            session.close()

    def update(self, entity: User) -> User:
        """Update an existing user."""
        db_manager = get_database_manager()
        session = db_manager._session_factory()
        try:
            orm_obj = session.query(UserORM).filter(
                UserORM.id == entity.id
            ).first()

            if not orm_obj:
                raise ValueError(f"User with ID {entity.id} not found")

            # Update fields (preserve password_hash — not in domain entity)
            orm_obj.email = entity.email
            orm_obj.first_name = entity.first_name
            orm_obj.last_name = entity.last_name
            orm_obj.risk_tolerance = entity.risk_tolerance
            orm_obj.investment_goal = entity.investment_goal
            orm_obj.max_position_size_percentage = entity.max_position_size_percentage
            orm_obj.daily_loss_limit = entity.daily_loss_limit.amount if entity.daily_loss_limit else None
            orm_obj.weekly_loss_limit = entity.weekly_loss_limit.amount if entity.weekly_loss_limit else None
            orm_obj.monthly_loss_limit = entity.monthly_loss_limit.amount if entity.monthly_loss_limit else None
            orm_obj.sector_preferences = entity.sector_preferences
            orm_obj.sector_exclusions = entity.sector_exclusions
            orm_obj.is_active = entity.is_active
            orm_obj.email_notifications_enabled = entity.email_notifications_enabled
            orm_obj.sms_notifications_enabled = entity.sms_notifications_enabled
            orm_obj.approval_mode_enabled = entity.approval_mode_enabled
            orm_obj.terms_accepted_at = entity.terms_accepted_at
            orm_obj.privacy_accepted_at = entity.privacy_accepted_at
            orm_obj.marketing_consent = entity.marketing_consent
            orm_obj.auto_trading_enabled = entity.auto_trading_enabled
            orm_obj.watchlist = entity.watchlist
            orm_obj.trading_budget = entity.trading_budget.amount if entity.trading_budget else None
            orm_obj.stop_loss_pct = entity.stop_loss_pct
            orm_obj.take_profit_pct = entity.take_profit_pct
            orm_obj.confidence_threshold = entity.confidence_threshold
            orm_obj.max_position_pct = entity.max_position_pct
            orm_obj.updated_at = datetime.utcnow()

            session.commit()
            session.refresh(orm_obj)
            return self._to_domain_entity(orm_obj)
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to update user: {e}")
            raise
        finally:
            session.close()

    def anonymize_user(self, user_id: str) -> None:
        """
        Anonymize a user's personal data for GDPR Article 17 (Right to Erasure).

        Replaces PII with anonymized values and deactivates the account.
        Preserves referential integrity by keeping the user record.

        Args:
            user_id: User ID to anonymize
        """
        db_manager = get_database_manager()
        session = db_manager._session_factory()
        try:
            orm_obj = session.query(UserORM).filter(
                UserORM.id == user_id
            ).first()

            if not orm_obj:
                raise ValueError(f"User with ID {user_id} not found")

            # Replace PII with anonymized values
            anon_id = uuid.uuid4().hex[:8]
            orm_obj.email = f"deleted-{anon_id}@anonymized.local"
            orm_obj.first_name = "DELETED"
            orm_obj.last_name = "USER"
            orm_obj.password_hash = ""
            orm_obj.sector_preferences = []
            orm_obj.sector_exclusions = []
            orm_obj.is_active = False
            orm_obj.email_notifications_enabled = False
            orm_obj.sms_notifications_enabled = False
            orm_obj.updated_at = datetime.utcnow()

            session.commit()
            logger.info(f"User {user_id} anonymized for GDPR compliance")
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to anonymize user {user_id}: {e}")
            raise
        finally:
            session.close()
