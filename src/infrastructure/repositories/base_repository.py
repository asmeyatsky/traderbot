"""
Base Repository Class

Provides common functionality for all repository implementations.
"""
from __future__ import annotations

from typing import TypeVar, Generic, Optional, List
from sqlalchemy.orm import Session
import logging

from src.infrastructure.database import get_db_session
from src.domain.exceptions import DomainException

logger = logging.getLogger(__name__)

# Type variables
Entity = TypeVar('Entity')
ORM_Model = TypeVar('ORM_Model')


class BaseRepository(Generic[Entity, ORM_Model]):
    """
    Base repository providing common CRUD operations.

    Architectural Intent:
    - Provides shared functionality for all repositories
    - Handles session management and error handling
    - Translates between domain entities and ORM models
    """

    def __init__(self, orm_model_class: type[ORM_Model]):
        """
        Initialize repository with ORM model class.

        Args:
            orm_model_class: The SQLAlchemy ORM model class
        """
        self.orm_model_class = orm_model_class

    def _get_session(self) -> Session:
        """
        Get a database session.

        Returns:
            SQLAlchemy session object
        """
        from src.infrastructure.database import get_database_manager
        db_manager = get_database_manager()
        return db_manager._session_factory()

    def _to_domain_entity(self, orm_obj: ORM_Model) -> Entity:
        """
        Convert ORM model to domain entity.
        Should be overridden by subclasses.

        Args:
            orm_obj: ORM model instance

        Returns:
            Domain entity instance
        """
        raise NotImplementedError("Subclasses must implement _to_domain_entity")

    def _to_orm_model(self, entity: Entity) -> ORM_Model:
        """
        Convert domain entity to ORM model.
        Should be overridden by subclasses.

        Args:
            entity: Domain entity instance

        Returns:
            ORM model instance
        """
        raise NotImplementedError("Subclasses must implement _to_orm_model")

    def save(self, entity: Entity) -> Entity:
        """
        Save or update an entity.

        Args:
            entity: Domain entity to save

        Returns:
            Saved entity with any generated IDs

        Raises:
            DomainException: If save fails
        """
        session = self._get_session()
        try:
            orm_obj = self._to_orm_model(entity)
            session.add(orm_obj)
            session.commit()
            session.refresh(orm_obj)
            return self._to_domain_entity(orm_obj)
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save {self.orm_model_class.__name__}: {e}")
            raise DomainException(f"Failed to save entity: {str(e)}")
        finally:
            session.close()

    def get_by_id(self, entity_id: str) -> Optional[Entity]:
        """
        Retrieve entity by ID.

        Args:
            entity_id: Entity ID

        Returns:
            Domain entity or None if not found
        """
        session = self._get_session()
        try:
            orm_obj = session.query(self.orm_model_class).filter(
                self.orm_model_class.id == entity_id
            ).first()
            return self._to_domain_entity(orm_obj) if orm_obj else None
        except Exception as e:
            logger.error(f"Failed to get {self.orm_model_class.__name__} by ID: {e}")
            return None
        finally:
            session.close()

    def delete(self, entity_id: str) -> bool:
        """
        Delete an entity by ID.

        Args:
            entity_id: Entity ID

        Returns:
            True if deleted, False if not found

        Raises:
            DomainException: If delete fails
        """
        session = self._get_session()
        try:
            orm_obj = session.query(self.orm_model_class).filter(
                self.orm_model_class.id == entity_id
            ).first()

            if not orm_obj:
                return False

            session.delete(orm_obj)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to delete {self.orm_model_class.__name__}: {e}")
            raise DomainException(f"Failed to delete entity: {str(e)}")
        finally:
            session.close()

    def exists(self, entity_id: str) -> bool:
        """
        Check if entity exists.

        Args:
            entity_id: Entity ID

        Returns:
            True if exists, False otherwise
        """
        session = self._get_session()
        try:
            return session.query(
                session.query(self.orm_model_class).filter(
                    self.orm_model_class.id == entity_id
                ).exists()
            ).scalar()
        except Exception as e:
            logger.error(f"Failed to check existence: {e}")
            return False
        finally:
            session.close()
