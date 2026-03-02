"""
Broker Account Domain Entity

Architectural Intent:
- Models per-user broker account linking following DDD principles
- BrokerAccount is an entity with identity (user_id + broker_type)
- API keys are stored encrypted at the infrastructure layer
- This entity holds decrypted keys only in memory during use
- Supports paper vs live trading modes per user

Key Design Decisions:
1. Immutable entity (frozen dataclass) to prevent accidental state corruption
2. Encryption/decryption is an infrastructure concern handled by the repository
3. BrokerType reuses the existing enum from broker_integration module
4. paper_trading flag controls routing to paper vs live Alpaca endpoints
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
from typing import Optional


class BrokerType(str, Enum):
    """Supported broker types for user account linking."""
    ALPACA = "alpaca"


@dataclass(frozen=True)
class BrokerAccount:
    """
    Broker Account Entity

    Represents a user's linked brokerage account with encrypted API credentials.

    Invariants:
    - A user can have at most one account per broker_type
    - API key and secret key must not be empty
    - paper_trading defaults to True for safety
    """
    id: str
    user_id: str
    broker_type: BrokerType
    api_key: str
    secret_key: str
    paper_trading: bool = True
    label: Optional[str] = None
    is_active: bool = True
    created_at: datetime = None  # type: ignore[assignment]
    updated_at: datetime = None  # type: ignore[assignment]

    def __post_init__(self):
        if not self.api_key or not self.api_key.strip():
            raise ValueError("API key must not be empty")
        if not self.secret_key or not self.secret_key.strip():
            raise ValueError("Secret key must not be empty")

    def switch_to_live(self) -> BrokerAccount:
        """Switch from paper to live trading. Returns new instance."""
        return replace(self, paper_trading=False, updated_at=datetime.utcnow())

    def switch_to_paper(self) -> BrokerAccount:
        """Switch from live to paper trading. Returns new instance."""
        return replace(self, paper_trading=True, updated_at=datetime.utcnow())

    def deactivate(self) -> BrokerAccount:
        """Deactivate the broker account. Returns new instance."""
        return replace(self, is_active=False, updated_at=datetime.utcnow())

    def update_keys(self, api_key: str, secret_key: str) -> BrokerAccount:
        """Update API credentials. Returns new instance."""
        return replace(
            self,
            api_key=api_key,
            secret_key=secret_key,
            updated_at=datetime.utcnow(),
        )
