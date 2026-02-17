"""
Activity Log Repository

Persists autonomous trading activity events for auditability and user visibility.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Dict, Any
from decimal import Decimal
import logging
import uuid

from src.infrastructure.orm_models import TradingActivityLogORM
from src.infrastructure.database import get_database_manager

logger = logging.getLogger(__name__)


class ActivityLogRepository:
    """Repository for trading activity log persistence."""

    def log_event(
        self,
        user_id: str,
        event_type: str,
        message: str,
        symbol: Optional[str] = None,
        signal: Optional[str] = None,
        confidence: Optional[float] = None,
        order_id: Optional[str] = None,
        broker_order_id: Optional[str] = None,
        quantity: Optional[int] = None,
        price: Optional[Decimal] = None,
        metadata: Optional[Dict[str, Any]] = None,
        occurred_at: Optional[datetime] = None,
    ) -> None:
        """Persist an activity log entry."""
        db_manager = get_database_manager()
        session = db_manager._session_factory()
        try:
            now = datetime.utcnow()
            orm_obj = TradingActivityLogORM(
                id=str(uuid.uuid4()),
                user_id=user_id,
                event_type=event_type,
                symbol=symbol,
                signal=signal,
                confidence=Decimal(str(confidence)) if confidence is not None else None,
                order_id=order_id,
                broker_order_id=broker_order_id,
                quantity=quantity,
                price=price,
                message=message,
                metadata_json=metadata,
                occurred_at=occurred_at or now,
                recorded_at=now,
            )
            session.add(orm_obj)
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to log activity event: {e}")
        finally:
            session.close()

    def get_recent_activity(
        self, user_id: str, limit: int = 50, skip: int = 0, event_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return latest activity entries for a user."""
        db_manager = get_database_manager()
        session = db_manager._session_factory()
        try:
            query = session.query(TradingActivityLogORM).filter(
                TradingActivityLogORM.user_id == user_id
            )
            if event_type:
                query = query.filter(TradingActivityLogORM.event_type == event_type)

            orm_objs = (
                query
                .order_by(TradingActivityLogORM.occurred_at.desc())
                .offset(skip)
                .limit(limit)
                .all()
            )
            return [self._to_dict(obj) for obj in orm_objs]
        except Exception as e:
            logger.error(f"Failed to get activity for user {user_id}: {e}")
            return []
        finally:
            session.close()

    def get_activity_summary(self, user_id: str) -> Dict[str, int]:
        """Return counts of activity events grouped by event_type."""
        db_manager = get_database_manager()
        session = db_manager._session_factory()
        try:
            from sqlalchemy import func
            rows = (
                session.query(
                    TradingActivityLogORM.event_type,
                    func.count(TradingActivityLogORM.id),
                )
                .filter(TradingActivityLogORM.user_id == user_id)
                .group_by(TradingActivityLogORM.event_type)
                .all()
            )
            return {event_type: count for event_type, count in rows}
        except Exception as e:
            logger.error(f"Failed to get activity summary for user {user_id}: {e}")
            return {}
        finally:
            session.close()

    @staticmethod
    def _to_dict(orm_obj: TradingActivityLogORM) -> Dict[str, Any]:
        return {
            "id": orm_obj.id,
            "user_id": orm_obj.user_id,
            "event_type": orm_obj.event_type,
            "symbol": orm_obj.symbol,
            "signal": orm_obj.signal,
            "confidence": float(orm_obj.confidence) if orm_obj.confidence else None,
            "order_id": orm_obj.order_id,
            "broker_order_id": orm_obj.broker_order_id,
            "quantity": float(orm_obj.quantity) if orm_obj.quantity else None,
            "price": float(orm_obj.price) if orm_obj.price else None,
            "message": orm_obj.message,
            "metadata": orm_obj.metadata_json,
            "occurred_at": orm_obj.occurred_at.isoformat() if orm_obj.occurred_at else None,
            "recorded_at": orm_obj.recorded_at.isoformat() if orm_obj.recorded_at else None,
        }
