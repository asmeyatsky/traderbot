"""
Event Store Implementation

Provides persistence of domain events to the database for audit trail and event sourcing.
"""
from __future__ import annotations

from typing import List, Type
from datetime import datetime
import json
import logging
import uuid

from src.domain.events import DomainEvent
from src.infrastructure.event_bus import EventStore
from src.infrastructure.orm_models import DomainEventORM
from src.infrastructure.database import get_database_manager

logger = logging.getLogger(__name__)


class DatabaseEventStore(EventStore):
    """
    Event store implementation using a relational database.

    Persists all domain events for audit trail and event sourcing capabilities.
    """

    def __init__(self):
        """Initialize the database event store."""
        pass

    async def append(self, event: DomainEvent) -> None:
        """
        Append an event to the event store.

        Args:
            event: Domain event to store

        Raises:
            Exception: If storage fails
        """
        self.append_sync(event)

    def append_sync(self, event: DomainEvent) -> None:
        """
        Append an event synchronously to the database.

        Args:
            event: Domain event to store

        Raises:
            Exception: If storage fails
        """
        db_manager = get_database_manager()
        session = db_manager._session_factory()

        try:
            # Convert event to JSON-serializable dict
            event_data = self._serialize_event(event)

            # Create ORM model
            orm_event = DomainEventORM(
                id=str(uuid.uuid4()),
                aggregate_type=self._get_aggregate_type(event),
                aggregate_id=event.aggregate_id,
                event_type=event.__class__.__name__,
                event_data=event_data,
                occurred_at=event.occurred_at,
                recorded_at=datetime.utcnow(),
            )

            session.add(orm_event)
            session.commit()
            logger.debug(f"Event {event.__class__.__name__} stored for aggregate {event.aggregate_id}")

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to store event: {e}")
            raise
        finally:
            session.close()

    async def get_events(
        self,
        aggregate_id: str,
        event_type: Type[DomainEvent] | None = None,
    ) -> List[DomainEvent]:
        """
        Get all events for an aggregate.

        Args:
            aggregate_id: ID of the aggregate
            event_type: Optional specific event type to filter

        Returns:
            List of domain events
        """
        db_manager = get_database_manager()
        session = db_manager._session_factory()

        try:
            query = session.query(DomainEventORM).filter(
                DomainEventORM.aggregate_id == aggregate_id
            )

            if event_type:
                query = query.filter(
                    DomainEventORM.event_type == event_type.__name__
                )

            orm_events = query.order_by(DomainEventORM.occurred_at).all()

            events = []
            for orm_event in orm_events:
                try:
                    event = self._deserialize_event(orm_event.event_data, orm_event.event_type)
                    events.append(event)
                except Exception as e:
                    logger.error(f"Failed to deserialize event {orm_event.event_type}: {e}")

            return events

        except Exception as e:
            logger.error(f"Failed to retrieve events for aggregate {aggregate_id}: {e}")
            return []
        finally:
            session.close()

    async def get_all_events(
        self,
        since: datetime | None = None,
    ) -> List[DomainEvent]:
        """
        Get all events since a specific time.

        Args:
            since: Optional start time filter

        Returns:
            List of domain events
        """
        db_manager = get_database_manager()
        session = db_manager._session_factory()

        try:
            query = session.query(DomainEventORM)

            if since:
                query = query.filter(DomainEventORM.occurred_at >= since)

            orm_events = query.order_by(DomainEventORM.occurred_at).all()

            events = []
            for orm_event in orm_events:
                try:
                    event = self._deserialize_event(orm_event.event_data, orm_event.event_type)
                    events.append(event)
                except Exception as e:
                    logger.error(f"Failed to deserialize event {orm_event.event_type}: {e}")

            return events

        except Exception as e:
            logger.error(f"Failed to retrieve events: {e}")
            return []
        finally:
            session.close()

    def _serialize_event(self, event: DomainEvent) -> dict:
        """
        Convert a domain event to a JSON-serializable dictionary.

        Args:
            event: Domain event to serialize

        Returns:
            Dictionary representation of the event
        """
        event_dict = {}

        for field_name, field_value in event.__dict__.items():
            if field_name.startswith('_'):
                continue

            if isinstance(field_value, datetime):
                event_dict[field_name] = field_value.isoformat()
            elif hasattr(field_value, '__dict__'):
                # Handle value objects
                if hasattr(field_value, 'amount') and hasattr(field_value, 'currency'):
                    # Money value object
                    event_dict[field_name] = {
                        'amount': str(field_value.amount),
                        'currency': field_value.currency
                    }
                elif hasattr(field_value, 'value'):
                    # Symbol or other simple value objects
                    event_dict[field_name] = str(field_value)
                else:
                    event_dict[field_name] = str(field_value)
            else:
                try:
                    json.dumps(field_value)  # Test if serializable
                    event_dict[field_name] = field_value
                except (TypeError, ValueError):
                    event_dict[field_name] = str(field_value)

        return event_dict

    def _deserialize_event(self, event_data: dict, event_type_name: str) -> DomainEvent:
        """
        Reconstruct a domain event from stored data.

        Args:
            event_data: Event data dictionary
            event_type_name: Name of the event class

        Returns:
            Reconstructed domain event
        """
        # This is a simplified implementation
        # In a real system, you'd need to properly reconstruct all value objects
        from src.domain import events as events_module

        event_class = getattr(events_module, event_type_name)

        # Reconstruct the event (simplified)
        # In production, handle all value object reconstructions
        return event_class(**event_data)

    def _get_aggregate_type(self, event: DomainEvent) -> str:
        """
        Determine the aggregate type from the event.

        Args:
            event: Domain event

        Returns:
            Aggregate type name
        """
        event_class_name = event.__class__.__name__

        if 'Order' in event_class_name:
            return 'Order'
        elif 'Position' in event_class_name:
            return 'Position'
        elif 'Portfolio' in event_class_name:
            return 'Portfolio'
        elif 'User' in event_class_name:
            return 'User'
        else:
            return 'Other'
