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

    Note: The Portfolio domain entity uses computed properties for total_value and positions_value,
    so we store the cash_balance and reconstruct the Portfolio without positions (positions are
    stored in a separate repository).
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

        # Quantize decimal values to 2 decimal places for Money compatibility
        def quantize_money(value):
            if value is None:
                return Decimal("0")
            return Decimal(str(value)).quantize(Decimal("0.01"))

        # Portfolio entity has total_value as a computed property, not a field
        # We store positions separately, so we create Portfolio with empty positions
        return Portfolio(
            id=orm_obj.id,
            user_id=orm_obj.user_id,
            positions=[],  # Positions are loaded separately
            cash_balance=Money(quantize_money(orm_obj.cash_balance), "USD"),
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
        # Calculate values from the entity's computed properties
        total_value = entity.total_value.amount
        positions_value = entity.positions_value.amount

        return PortfolioORM(
            id=entity.id,
            user_id=entity.user_id,
            total_value=total_value,
            cash_balance=entity.cash_balance.amount,
            invested_value=positions_value,
            total_gain_loss=Decimal("0"),  # Would need historical data to compute
            total_return_percentage=Decimal("0"),
            ytd_return_percentage=Decimal("0"),
            current_drawdown=Decimal("0"),
            peak_value=total_value,
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

            # Update fields - compute values from entity's properties
            total_value = entity.total_value.amount
            positions_value = entity.positions_value.amount

            orm_obj.total_value = total_value
            orm_obj.cash_balance = entity.cash_balance.amount
            orm_obj.invested_value = positions_value
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
