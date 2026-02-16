"""
Database Connection Management

This module handles database configuration, connection pooling,
and session management for SQLAlchemy ORM.

Following clean architecture principles for data persistence.
"""
from __future__ import annotations

import os

from sqlalchemy import create_engine, event
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from sqlalchemy.pool import QueuePool, NullPool
from contextlib import contextmanager
from typing import Generator, Optional
import logging

from src.infrastructure.config.settings import settings

logger = logging.getLogger(__name__)

# Declarative base for ORM models
Base = declarative_base()


class DatabaseManager:
    """Manages database connections and sessions."""

    def __init__(self, database_url: str, echo: bool = False):
        """
        Initialize database manager.

        Args:
            database_url: Database connection URL
            echo: Whether to log SQL statements
        """
        self.database_url = database_url
        self.echo = echo
        self.engine = None
        self.async_engine = None
        self._session_factory = None
        self._async_session_factory = None

    def initialize(self) -> None:
        """Initialize database engines and session factories."""
        try:
            # Determine if using async driver
            is_async = "postgresql+asyncpg" in self.database_url or \
                      "mysql+aiomysql" in self.database_url

            if is_async:
                self._initialize_async()
            else:
                self._initialize_sync()

            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    def _initialize_sync(self) -> None:
        """Initialize synchronous database engine and session factory."""
        # Enforce SSL for non-development environments
        connect_args = {}
        environment = os.getenv("ENVIRONMENT", "development")
        if environment != "development" and "postgresql" in self.database_url:
            connect_args["sslmode"] = "require"

        self.engine = create_engine(
            self.database_url,
            poolclass=QueuePool,
            pool_size=20,
            max_overflow=40,
            pool_recycle=3600,  # Recycle connections after 1 hour
            pool_pre_ping=True,  # Test connections before using them
            echo=self.echo,
            connect_args=connect_args,
        )

        self._session_factory = sessionmaker(
            bind=self.engine,
            expire_on_commit=False,
        )

        # Add event listeners for connection pool
        @event.listens_for(self.engine, "connect")
        def receive_connect(dbapi_conn, connection_record):
            # Set connection timeouts
            if "postgresql" in self.database_url:
                dbapi_conn.set_isolation_level(0)

    def _initialize_async(self) -> None:
        """Initialize asynchronous database engine and session factory."""
        self.async_engine = create_async_engine(
            self.database_url,
            echo=self.echo,
            pool_size=20,
            max_overflow=40,
            pool_recycle=3600,
            pool_pre_ping=True,
        )

        self._async_session_factory = sessionmaker(
            bind=self.async_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    async def create_tables(self) -> None:
        """Create all tables in the database."""
        if self.async_engine:
            async with self.async_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
        elif self.engine:
            Base.metadata.create_all(self.engine)

    async def drop_tables(self) -> None:
        """Drop all tables from the database."""
        if self.async_engine:
            async with self.async_engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
        elif self.engine:
            Base.metadata.drop_all(self.engine)

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Get a synchronous database session.

        Usage:
            with db_manager.get_session() as session:
                user = session.query(User).get(user_id)
        """
        if not self._session_factory:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()

    async def get_async_session(self):
        """
        Get an asynchronous database session.

        Usage:
            async with db_manager.get_async_session() as session:
                user = await session.get(User, user_id)
        """
        if not self._async_session_factory:
            raise RuntimeError("Async database not initialized. Call initialize() first.")

        session = self._async_session_factory()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Async database session error: {e}")
            raise
        finally:
            await session.close()

    def close(self) -> None:
        """Close database connections."""
        if self.engine:
            self.engine.dispose()

    async def aclose(self) -> None:
        """Close async database connections."""
        if self.async_engine:
            await self.async_engine.dispose()

    def health_check(self) -> bool:
        """Check if database connection is healthy."""
        try:
            from sqlalchemy import text
            with self.get_session() as session:
                session.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def initialize_database() -> DatabaseManager:
    """Initialize and return the global database manager."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager(
            database_url=settings.DATABASE_URL,
            echo=(settings.ENVIRONMENT == "development")
        )
        _db_manager.initialize()
    return _db_manager


def get_database_manager() -> DatabaseManager:
    """Get the global database manager instance."""
    if _db_manager is None:
        raise RuntimeError("Database not initialized. Call initialize_database() first.")
    return _db_manager


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Get a database session from the global manager."""
    db_manager = get_database_manager()
    with db_manager.get_session() as session:
        yield session
