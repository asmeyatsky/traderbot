"""
Position Repository Implementation

Implements the PositionRepositoryPort for position persistence operations.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional, List
from decimal import Decimal
import logging

from src.domain.entities.trading import Position, PositionType
from src.domain.value_objects import Symbol, Money
from src.domain.ports import PositionRepositoryPort
from src.infrastructure.orm_models import PositionORM
from src.infrastructure.repositories.base_repository import BaseRepository
from src.infrastructure.database import get_database_manager

logger = logging.getLogger(__name__)


class PositionRepository(BaseRepository[Position, PositionORM], PositionRepositoryPort):
    """
    Repository for Position entity persistence.

    Implements PositionRepositoryPort and provides all position-related persistence operations.
    """

    def __init__(self):
        """Initialize PositionRepository with PositionORM model class."""
        super().__init__(PositionORM)

    def _to_domain_entity(self, orm_obj: PositionORM) -> Position:
        """
        Convert PositionORM to domain Position entity.

        Args:
            orm_obj: PositionORM instance

        Returns:
            Position domain entity
        """
        if not orm_obj:
            return None

        return Position(
            id=orm_obj.id,
            user_id=orm_obj.user_id,
            symbol=Symbol(orm_obj.symbol),
            quantity=orm_obj.quantity,
            position_type=orm_obj.position_type,
            average_entry_price=Money(orm_obj.average_entry_price, "USD"),
            current_price=Money(orm_obj.current_price, "USD") if orm_obj.current_price else None,
            unrealized_gain_loss=Money(orm_obj.unrealized_gain_loss, "USD"),
            realized_gain_loss=Money(orm_obj.realized_gain_loss, "USD"),
            opened_at=orm_obj.opened_at,
            closed_at=orm_obj.closed_at,
        )

    def _to_orm_model(self, entity: Position) -> PositionORM:
        """
        Convert Position domain entity to PositionORM.

        Args:
            entity: Position domain entity

        Returns:
            PositionORM instance
        """
        return PositionORM(
            id=entity.id,
            user_id=entity.user_id,
            symbol=str(entity.symbol),
            quantity=entity.quantity,
            position_type=entity.position_type,
            average_entry_price=entity.average_entry_price.amount,
            current_price=entity.current_price.amount if entity.current_price else None,
            unrealized_gain_loss=entity.unrealized_gain_loss.amount,
            realized_gain_loss=entity.realized_gain_loss.amount,
            opened_at=entity.opened_at,
            closed_at=entity.closed_at,
        )

    def get_by_user_id(self, user_id: str) -> List[Position]:
        """
        Retrieve all positions for a user.

        Args:
            user_id: User ID

        Returns:
            List of Position domain entities
        """
        db_manager = get_database_manager()
        session = db_manager._session_factory()
        try:
            orm_objs = session.query(PositionORM).filter(
                PositionORM.user_id == user_id,
                PositionORM.closed_at == None
            ).order_by(PositionORM.opened_at.desc()).all()
            return [self._to_domain_entity(orm_obj) for orm_obj in orm_objs]
        except Exception as e:
            logger.error(f"Failed to get positions for user {user_id}: {e}")
            return []
        finally:
            session.close()

    def get_by_symbol(self, user_id: str, symbol: Symbol) -> Optional[Position]:
        """
        Retrieve a specific position by user and symbol.

        Args:
            user_id: User ID
            symbol: Stock symbol

        Returns:
            Position domain entity or None if not found
        """
        db_manager = get_database_manager()
        session = db_manager._session_factory()
        try:
            orm_obj = session.query(PositionORM).filter(
                PositionORM.user_id == user_id,
                PositionORM.symbol == str(symbol),
                PositionORM.closed_at == None
            ).first()
            return self._to_domain_entity(orm_obj) if orm_obj else None
        except Exception as e:
            logger.error(f"Failed to get position for user {user_id} and symbol {symbol}: {e}")
            return None
        finally:
            session.close()

    def get_closed_positions(self, user_id: str) -> List[Position]:
        """
        Retrieve all closed positions for a user.

        Args:
            user_id: User ID

        Returns:
            List of closed Position domain entities
        """
        db_manager = get_database_manager()
        session = db_manager._session_factory()
        try:
            orm_objs = session.query(PositionORM).filter(
                PositionORM.user_id == user_id,
                PositionORM.closed_at != None
            ).order_by(PositionORM.closed_at.desc()).all()
            return [self._to_domain_entity(orm_obj) for orm_obj in orm_objs]
        except Exception as e:
            logger.error(f"Failed to get closed positions for user {user_id}: {e}")
            return []
        finally:
            session.close()

    def update(self, entity: Position) -> Position:
        """
        Update an existing position.

        Args:
            entity: Updated Position domain entity

        Returns:
            Updated Position domain entity
        """
        db_manager = get_database_manager()
        session = db_manager._session_factory()
        try:
            orm_obj = session.query(PositionORM).filter(
                PositionORM.id == entity.id
            ).first()

            if not orm_obj:
                raise ValueError(f"Position with ID {entity.id} not found")

            orm_obj.quantity = entity.quantity
            orm_obj.current_price = entity.current_price.amount if entity.current_price else None
            orm_obj.unrealized_gain_loss = entity.unrealized_gain_loss.amount
            orm_obj.realized_gain_loss = entity.realized_gain_loss.amount
            orm_obj.closed_at = entity.closed_at
            orm_obj.updated_at = datetime.utcnow()

            session.commit()
            session.refresh(orm_obj)
            return self._to_domain_entity(orm_obj)
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to update position: {e}")
            raise
        finally:
            session.close()
