"""
Repository Implementations

This module contains the concrete implementations of repository ports defined in the domain layer.
Repositories provide an abstraction for data persistence and follow the repository pattern.

Architectural Intent:
- Repositories bridge domain entities and infrastructure (database)
- They implement the port interfaces defined in the domain layer
- All database access should go through repositories
- Repositories are responsible for translating between domain entities and ORM models
"""
from .base_repository import BaseRepository
from .user_repository import UserRepository
from .order_repository import OrderRepository
from .position_repository import PositionRepository
from .portfolio_repository import PortfolioRepository

__all__ = [
    "BaseRepository",
    "UserRepository",
    "OrderRepository",
    "PositionRepository",
    "PortfolioRepository",
]
