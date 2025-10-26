"""
Event Bus Implementation

This module implements the event bus for publishing and handling domain events.
Follows the observer pattern for loose coupling between event publishers and handlers.

Architectural Intent:
- Central hub for event publication and dispatch
- Event handlers subscribe to specific event types
- Enables asynchronous event processing
- Provides audit trail via event persistence
"""
from __future__ import annotations

from typing import Callable, List, Dict, Type, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import asyncio
from datetime import datetime

from src.domain.events import DomainEvent

logger = logging.getLogger(__name__)


class EventHandler(ABC):
    """
    Abstract base class for domain event handlers.

    Subclasses should implement handle() to process specific event types.
    """

    @abstractmethod
    async def handle(self, event: DomainEvent) -> None:
        """
        Handle a domain event.

        Args:
            event: The domain event to handle

        Raises:
            Exception: If event handling fails
        """
        pass

    @abstractmethod
    def can_handle(self, event: DomainEvent) -> bool:
        """
        Check if this handler can process the given event.

        Args:
            event: The domain event to check

        Returns:
            True if this handler can process the event, False otherwise
        """
        pass


class EventBus:
    """
    Event bus for publishing and dispatching domain events.

    Manages subscriptions, dispatching events to handlers, and event persistence.
    Supports both synchronous and asynchronous event handling.
    """

    def __init__(self):
        """Initialize the event bus."""
        self._handlers: Dict[Type[DomainEvent], List[EventHandler]] = {}
        self._subscribers: Dict[Type[DomainEvent], List[Callable]] = {}
        self._event_store: EventStore | None = None
        self._is_publishing = False

    def set_event_store(self, event_store: EventStore) -> None:
        """
        Set the event store for persisting events.

        Args:
            event_store: Event store implementation
        """
        self._event_store = event_store

    def subscribe(
        self,
        event_type: Type[DomainEvent],
        handler: EventHandler | Callable,
    ) -> None:
        """
        Subscribe a handler to an event type.

        Args:
            event_type: Type of domain event to handle
            handler: Event handler (EventHandler instance or callable)
        """
        if isinstance(handler, EventHandler):
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            self._handlers[event_type].append(handler)
            logger.info(f"Subscribed {handler.__class__.__name__} to {event_type.__name__}")
        else:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(handler)
            logger.info(f"Subscribed callable to {event_type.__name__}")

    def unsubscribe(
        self,
        event_type: Type[DomainEvent],
        handler: EventHandler | Callable,
    ) -> None:
        """
        Unsubscribe a handler from an event type.

        Args:
            event_type: Type of domain event
            handler: Event handler to remove
        """
        if isinstance(handler, EventHandler):
            if event_type in self._handlers:
                self._handlers[event_type].remove(handler)
        else:
            if event_type in self._subscribers:
                self._subscribers[event_type].remove(handler)

    async def publish(self, event: DomainEvent) -> None:
        """
        Publish a domain event to all subscribed handlers.

        Persists the event to the event store if configured, then dispatches
        to all subscribed handlers asynchronously.

        Args:
            event: Domain event to publish
        """
        try:
            self._is_publishing = True

            # Persist the event
            if self._event_store:
                await self._event_store.append(event)

            # Dispatch to handlers
            event_type = type(event)

            # Handle EventHandler subscribers
            if event_type in self._handlers:
                tasks = []
                for handler in self._handlers[event_type]:
                    try:
                        task = handler.handle(event)
                        if asyncio.iscoroutine(task):
                            tasks.append(task)
                        else:
                            await task
                    except Exception as e:
                        logger.error(f"Error in event handler {handler.__class__.__name__}: {e}")

                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

            # Handle callable subscribers
            if event_type in self._subscribers:
                tasks = []
                for subscriber in self._subscribers[event_type]:
                    try:
                        result = subscriber(event)
                        if asyncio.iscoroutine(result):
                            tasks.append(result)
                        else:
                            await result if result else None
                    except Exception as e:
                        logger.error(f"Error in event subscriber: {e}")

                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

            logger.info(f"Published event {event.__class__.__name__} to {event.aggregate_id}")

        except Exception as e:
            logger.error(f"Error publishing event {event.__class__.__name__}: {e}")
            raise
        finally:
            self._is_publishing = False

    def publish_sync(self, event: DomainEvent) -> None:
        """
        Publish a domain event synchronously (blocking).

        Args:
            event: Domain event to publish
        """
        try:
            self._is_publishing = True

            # Persist the event
            if self._event_store:
                self._event_store.append_sync(event)

            # Dispatch to handlers
            event_type = type(event)

            if event_type in self._handlers:
                for handler in self._handlers[event_type]:
                    try:
                        handler.handle(event)
                    except Exception as e:
                        logger.error(f"Error in event handler {handler.__class__.__name__}: {e}")

            if event_type in self._subscribers:
                for subscriber in self._subscribers[event_type]:
                    try:
                        subscriber(event)
                    except Exception as e:
                        logger.error(f"Error in event subscriber: {e}")

            logger.info(f"Published event {event.__class__.__name__} to {event.aggregate_id}")

        except Exception as e:
            logger.error(f"Error publishing event {event.__class__.__name__}: {e}")
            raise
        finally:
            self._is_publishing = False

    def is_publishing(self) -> bool:
        """Check if event bus is currently publishing events."""
        return self._is_publishing

    def get_subscribers_count(self, event_type: Type[DomainEvent]) -> int:
        """Get the number of subscribers for an event type."""
        count = len(self._handlers.get(event_type, []))
        count += len(self._subscribers.get(event_type, []))
        return count


class EventStore(ABC):
    """
    Abstract base class for event storage/persistence.

    Implementations should persist events for audit trail and event sourcing.
    """

    @abstractmethod
    async def append(self, event: DomainEvent) -> None:
        """
        Append an event to the event store.

        Args:
            event: Domain event to store

        Raises:
            Exception: If storage fails
        """
        pass

    @abstractmethod
    def append_sync(self, event: DomainEvent) -> None:
        """
        Append an event synchronously.

        Args:
            event: Domain event to store
        """
        pass

    @abstractmethod
    async def get_events(
        self,
        aggregate_id: str,
        event_type: Type[DomainEvent] | None = None,
    ) -> List[DomainEvent]:
        """
        Get events for an aggregate.

        Args:
            aggregate_id: ID of the aggregate
            event_type: Optional specific event type to filter

        Returns:
            List of events
        """
        pass

    @abstractmethod
    async def get_all_events(
        self,
        since: datetime | None = None,
    ) -> List[DomainEvent]:
        """
        Get all events since a specific time.

        Args:
            since: Optional start time filter

        Returns:
            List of all events
        """
        pass


# Global event bus instance
_event_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """Get the global event bus instance."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


def set_event_bus(bus: EventBus) -> None:
    """Set the global event bus instance."""
    global _event_bus
    _event_bus = bus


def reset_event_bus() -> None:
    """Reset the global event bus (useful for testing)."""
    global _event_bus
    _event_bus = None
