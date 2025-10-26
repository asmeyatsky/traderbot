"""
User Repository Implementation

Implements the UserRepositoryPort for user persistence operations.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional
from sqlalchemy.orm import Session
import logging

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

    Implements UserRepositoryPort and provides all user-related persistence operations.
    """

    def __init__(self):
        """Initialize UserRepository with UserORM model class."""
        super().__init__(UserORM)

    def _to_domain_entity(self, orm_obj: UserORM) -> User:
        """
        Convert UserORM to domain User entity.

        Args:
            orm_obj: UserORM instance

        Returns:
            User domain entity
        """
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
        )

    def _to_orm_model(self, entity: User) -> UserORM:
        """
        Convert User domain entity to UserORM.

        Args:
            entity: User domain entity

        Returns:
            UserORM instance
        """
        return UserORM(
            id=entity.id,
            email=entity.email,
            first_name=entity.first_name,
            last_name=entity.last_name,
            password_hash="",  # Password should be set separately
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
        )

    def get_by_email(self, email: str) -> Optional[User]:
        """
        Retrieve a user by email address.

        Args:
            email: User's email address

        Returns:
            User domain entity or None if not found
        """
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

    def get_all_active(self) -> list[User]:
        """
        Retrieve all active users.

        Returns:
            List of active User domain entities
        """
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

    def update(self, entity: User) -> User:
        """
        Update an existing user.

        Args:
            entity: Updated User domain entity

        Returns:
            Updated User domain entity
        """
        db_manager = get_database_manager()
        session = db_manager._session_factory()
        try:
            orm_obj = session.query(UserORM).filter(
                UserORM.id == entity.id
            ).first()

            if not orm_obj:
                raise ValueError(f"User with ID {entity.id} not found")

            # Update fields
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
