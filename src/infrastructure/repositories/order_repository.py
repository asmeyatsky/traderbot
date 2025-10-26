"""
Order Repository Implementation

Implements the OrderRepositoryPort for order persistence operations.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional, List
from decimal import Decimal
import logging

from src.domain.entities.trading import Order, OrderType, PositionType, OrderStatus
from src.domain.value_objects import Symbol, Money
from src.domain.ports import OrderRepositoryPort
from src.infrastructure.orm_models import OrderORM
from src.infrastructure.repositories.base_repository import BaseRepository
from src.infrastructure.database import get_database_manager

logger = logging.getLogger(__name__)


class OrderRepository(BaseRepository[Order, OrderORM], OrderRepositoryPort):
    """
    Repository for Order entity persistence.

    Implements OrderRepositoryPort and provides all order-related persistence operations.
    """

    def __init__(self):
        """Initialize OrderRepository with OrderORM model class."""
        super().__init__(OrderORM)

    def _to_domain_entity(self, orm_obj: OrderORM) -> Order:
        """
        Convert OrderORM to domain Order entity.

        Args:
            orm_obj: OrderORM instance

        Returns:
            Order domain entity
        """
        if not orm_obj:
            return None

        return Order(
            id=orm_obj.id,
            user_id=orm_obj.user_id,
            symbol=Symbol(orm_obj.symbol),
            order_type=orm_obj.order_type,
            position_type=orm_obj.position_type,
            quantity=orm_obj.quantity,
            status=orm_obj.status,
            placed_at=orm_obj.placed_at,
            executed_at=orm_obj.executed_at,
            price=Money(orm_obj.price, "USD") if orm_obj.price else None,
            stop_price=Money(orm_obj.stop_price, "USD") if orm_obj.stop_price else None,
            filled_quantity=orm_obj.filled_quantity,
            commission=Money(orm_obj.commission or Decimal('0'), "USD"),
            notes=orm_obj.notes,
        )

    def _to_orm_model(self, entity: Order) -> OrderORM:
        """
        Convert Order domain entity to OrderORM.

        Args:
            entity: Order domain entity

        Returns:
            OrderORM instance
        """
        return OrderORM(
            id=entity.id,
            user_id=entity.user_id,
            symbol=str(entity.symbol),
            order_type=entity.order_type,
            position_type=entity.position_type,
            quantity=entity.quantity,
            status=entity.status,
            placed_at=entity.placed_at,
            executed_at=entity.executed_at,
            price=entity.price.amount if entity.price else None,
            limit_price=None,  # Handle if needed
            stop_price=entity.stop_price.amount if entity.stop_price else None,
            filled_quantity=entity.filled_quantity,
            commission=entity.commission.amount if entity.commission else Decimal('0'),
            notes=entity.notes,
        )

    def get_by_user_id(self, user_id: str) -> List[Order]:
        """
        Retrieve all orders for a user.

        Args:
            user_id: User ID

        Returns:
            List of Order domain entities
        """
        db_manager = get_database_manager()
        session = db_manager._session_factory()
        try:
            orm_objs = session.query(OrderORM).filter(
                OrderORM.user_id == user_id
            ).order_by(OrderORM.placed_at.desc()).all()
            return [self._to_domain_entity(orm_obj) for orm_obj in orm_objs]
        except Exception as e:
            logger.error(f"Failed to get orders for user {user_id}: {e}")
            return []
        finally:
            session.close()

    def get_active_orders(self, user_id: str) -> List[Order]:
        """
        Retrieve all active (non-executed, non-cancelled) orders for a user.

        Args:
            user_id: User ID

        Returns:
            List of active Order domain entities
        """
        db_manager = get_database_manager()
        session = db_manager._session_factory()
        try:
            orm_objs = session.query(OrderORM).filter(
                OrderORM.user_id == user_id,
                OrderORM.status.in_([OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED])
            ).order_by(OrderORM.placed_at.desc()).all()
            return [self._to_domain_entity(orm_obj) for orm_obj in orm_objs]
        except Exception as e:
            logger.error(f"Failed to get active orders for user {user_id}: {e}")
            return []
        finally:
            session.close()

    def get_by_symbol(self, user_id: str, symbol: str) -> List[Order]:
        """
        Retrieve all orders for a user and symbol.

        Args:
            user_id: User ID
            symbol: Stock symbol

        Returns:
            List of Order domain entities
        """
        db_manager = get_database_manager()
        session = db_manager._session_factory()
        try:
            orm_objs = session.query(OrderORM).filter(
                OrderORM.user_id == user_id,
                OrderORM.symbol == str(symbol)
            ).order_by(OrderORM.placed_at.desc()).all()
            return [self._to_domain_entity(orm_obj) for orm_obj in orm_objs]
        except Exception as e:
            logger.error(f"Failed to get orders for user {user_id} and symbol {symbol}: {e}")
            return []
        finally:
            session.close()

    def get_by_status(self, user_id: str, status: OrderStatus) -> List[Order]:
        """
        Retrieve all orders for a user with a specific status.

        Args:
            user_id: User ID
            status: Order status

        Returns:
            List of Order domain entities
        """
        db_manager = get_database_manager()
        session = db_manager._session_factory()
        try:
            orm_objs = session.query(OrderORM).filter(
                OrderORM.user_id == user_id,
                OrderORM.status == status
            ).order_by(OrderORM.placed_at.desc()).all()
            return [self._to_domain_entity(orm_obj) for orm_obj in orm_objs]
        except Exception as e:
            logger.error(f"Failed to get orders with status {status}: {e}")
            return []
        finally:
            session.close()

    def update_status(self, order_id: str, new_status: OrderStatus) -> Optional[Order]:
        """
        Update the status of an order.

        Args:
            order_id: Order ID
            new_status: New order status

        Returns:
            Updated Order domain entity or None if not found
        """
        db_manager = get_database_manager()
        session = db_manager._session_factory()
        try:
            orm_obj = session.query(OrderORM).filter(
                OrderORM.id == order_id
            ).first()

            if not orm_obj:
                return None

            orm_obj.status = new_status
            if new_status == OrderStatus.EXECUTED:
                orm_obj.executed_at = datetime.utcnow()

            session.commit()
            session.refresh(orm_obj)
            return self._to_domain_entity(orm_obj)
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to update order status: {e}")
            raise
        finally:
            session.close()
