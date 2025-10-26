"""
Portfolio Repository Implementation

Implements the PortfolioRepositoryPort for portfolio persistence operations.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional
from decimal import Decimal
import logging

from src.domain.entities.trading import Portfolio
from src.domain.value_objects import Money
from src.domain.ports import PortfolioRepositoryPort
from src.infrastructure.orm_models import PortfolioORM
from src.infrastructure.repositories.base_repository import BaseRepository
from src.infrastructure.database import get_database_manager

logger = logging.getLogger(__name__)


class PortfolioRepository(BaseRepository[Portfolio, PortfolioORM], PortfolioRepositoryPort):
    """
    Repository for Portfolio entity persistence.

    Implements PortfolioRepositoryPort and provides all portfolio-related persistence operations.
    """

    def __init__(self):
        """Initialize PortfolioRepository with PortfolioORM model class."""
        super().__init__(PortfolioORM)

    def _to_domain_entity(self, orm_obj: PortfolioORM) -> Portfolio:
        """
        Convert PortfolioORM to domain Portfolio entity.

        Args:
            orm_obj: PortfolioORM instance

        Returns:
            Portfolio domain entity
        """
        if not orm_obj:
            return None

        return Portfolio(
            id=orm_obj.id,
            user_id=orm_obj.user_id,
            total_value=Money(orm_obj.total_value, "USD"),
            cash_balance=Money(orm_obj.cash_balance, "USD"),
            invested_value=Money(orm_obj.invested_value, "USD"),
            total_gain_loss=Money(orm_obj.total_gain_loss, "USD"),
            total_return_percentage=orm_obj.total_return_percentage,
            ytd_return_percentage=orm_obj.ytd_return_percentage,
            current_drawdown=orm_obj.current_drawdown,
            peak_value=Money(orm_obj.peak_value, "USD"),
            created_at=orm_obj.created_at,
            updated_at=orm_obj.updated_at,
        )

    def _to_orm_model(self, entity: Portfolio) -> PortfolioORM:
        """
        Convert Portfolio domain entity to PortfolioORM.

        Args:
            entity: Portfolio domain entity

        Returns:
            PortfolioORM instance
        """
        return PortfolioORM(
            id=entity.id,
            user_id=entity.user_id,
            total_value=entity.total_value.amount,
            cash_balance=entity.cash_balance.amount,
            invested_value=entity.invested_value.amount,
            total_gain_loss=entity.total_gain_loss.amount,
            total_return_percentage=entity.total_return_percentage,
            ytd_return_percentage=entity.ytd_return_percentage,
            current_drawdown=entity.current_drawdown,
            peak_value=entity.peak_value.amount,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
        )

    def get_by_user_id(self, user_id: str) -> Optional[Portfolio]:
        """
        Retrieve a portfolio by user ID.

        Args:
            user_id: User ID

        Returns:
            Portfolio domain entity or None if not found
        """
        db_manager = get_database_manager()
        session = db_manager._session_factory()
        try:
            orm_obj = session.query(PortfolioORM).filter(
                PortfolioORM.user_id == user_id
            ).first()
            return self._to_domain_entity(orm_obj) if orm_obj else None
        except Exception as e:
            logger.error(f"Failed to get portfolio for user {user_id}: {e}")
            return None
        finally:
            session.close()

    def update(self, entity: Portfolio) -> Portfolio:
        """
        Update an existing portfolio.

        Args:
            entity: Updated Portfolio domain entity

        Returns:
            Updated Portfolio domain entity
        """
        db_manager = get_database_manager()
        session = db_manager._session_factory()
        try:
            orm_obj = session.query(PortfolioORM).filter(
                PortfolioORM.id == entity.id
            ).first()

            if not orm_obj:
                raise ValueError(f"Portfolio with ID {entity.id} not found")

            # Update fields
            orm_obj.total_value = entity.total_value.amount
            orm_obj.cash_balance = entity.cash_balance.amount
            orm_obj.invested_value = entity.invested_value.amount
            orm_obj.total_gain_loss = entity.total_gain_loss.amount
            orm_obj.total_return_percentage = entity.total_return_percentage
            orm_obj.ytd_return_percentage = entity.ytd_return_percentage
            orm_obj.current_drawdown = entity.current_drawdown
            orm_obj.peak_value = entity.peak_value.amount
            orm_obj.updated_at = datetime.utcnow()

            session.commit()
            session.refresh(orm_obj)
            return self._to_domain_entity(orm_obj)
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to update portfolio: {e}")
            raise
        finally:
            session.close()

    def get_by_user_id_for_update(self, user_id: str) -> Optional[Portfolio]:
        """
        Retrieve a portfolio by user ID with row-level locking for updates.

        This method is useful for avoiding race conditions during concurrent updates.

        Args:
            user_id: User ID

        Returns:
            Portfolio domain entity or None if not found
        """
        db_manager = get_database_manager()
        session = db_manager._session_factory()
        try:
            from sqlalchemy import select
            orm_obj = session.query(PortfolioORM).filter(
                PortfolioORM.user_id == user_id
            ).with_for_update().first()
            return self._to_domain_entity(orm_obj) if orm_obj else None
        except Exception as e:
            logger.error(f"Failed to get portfolio for update for user {user_id}: {e}")
            return None
        finally:
            session.close()
