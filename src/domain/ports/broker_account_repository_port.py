"""
Broker Account Repository Port

Architectural Intent:
- Defines the interface for broker account persistence
- Infrastructure layer implements encryption/decryption of API keys
- Domain layer remains unaware of storage or encryption mechanism
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from src.domain.entities.broker_account import BrokerAccount, BrokerType


class BrokerAccountRepositoryPort(ABC):
    """
    Port for broker account persistence.

    Implementations must handle encryption of API keys at rest.
    The domain entity always holds decrypted values in memory.
    """

    @abstractmethod
    def save(self, account: BrokerAccount) -> BrokerAccount:
        """Save or update a broker account. Keys are encrypted at rest."""
        pass

    @abstractmethod
    def get_by_id(self, account_id: str) -> Optional[BrokerAccount]:
        """Get a broker account by ID with decrypted keys."""
        pass

    @abstractmethod
    def get_by_user_and_broker(
        self, user_id: str, broker_type: BrokerType
    ) -> Optional[BrokerAccount]:
        """Get a user's broker account for a specific broker type."""
        pass

    @abstractmethod
    def get_by_user(self, user_id: str) -> List[BrokerAccount]:
        """Get all broker accounts for a user."""
        pass

    @abstractmethod
    def delete(self, account_id: str) -> bool:
        """Delete a broker account. Returns True if deleted."""
        pass
