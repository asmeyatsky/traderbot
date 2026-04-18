"""
Broker Account Use Cases

Architectural Intent:
- Orchestrates broker account CRUD operations
- Validates user exists before linking
- Enforces one-account-per-broker constraint
- Creates per-user AlpacaBrokerService instances with user's own keys
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import List, Optional

from src.domain.entities.broker_account import BrokerAccount, BrokerType
from src.domain.ports.broker_account_repository_port import BrokerAccountRepositoryPort

logger = logging.getLogger(__name__)


class LinkBrokerAccountUseCase:
    """Link a brokerage account with user-provided API keys."""

    def __init__(
        self,
        broker_account_repository: BrokerAccountRepositoryPort,
    ):
        self._repo = broker_account_repository

    def execute(
        self,
        user_id: str,
        broker_type: str,
        api_key: str,
        secret_key: str,
        paper_trading: bool = True,
        label: Optional[str] = None,
    ) -> BrokerAccount:
        broker = BrokerType(broker_type)

        # Check for existing account — update if exists
        existing = self._repo.get_by_user_and_broker(user_id, broker)
        if existing:
            updated = existing.update_keys(api_key, secret_key)
            updated = BrokerAccount(
                id=updated.id,
                user_id=updated.user_id,
                broker_type=updated.broker_type,
                api_key=api_key,
                secret_key=secret_key,
                paper_trading=paper_trading,
                label=label or updated.label,
                is_active=True,
                created_at=updated.created_at,
                updated_at=datetime.utcnow(),
            )
            self._repo.save(updated)
            logger.info(f"Updated broker account {broker_type} for user {user_id}")
            return updated

        now = datetime.utcnow()
        account = BrokerAccount(
            id=str(uuid.uuid4()),
            user_id=user_id,
            broker_type=broker,
            api_key=api_key,
            secret_key=secret_key,
            paper_trading=paper_trading,
            label=label,
            is_active=True,
            created_at=now,
            updated_at=now,
        )
        self._repo.save(account)
        logger.info(f"Linked new broker account {broker_type} for user {user_id}")
        return account


class GetBrokerAccountsUseCase:
    """Retrieve user's linked broker accounts."""

    def __init__(self, broker_account_repository: BrokerAccountRepositoryPort):
        self._repo = broker_account_repository

    def execute(self, user_id: str) -> List[BrokerAccount]:
        return self._repo.get_by_user(user_id)


class UpdateBrokerSettingsUseCase:
    """Update paper/live toggle or deactivate a broker account."""

    def __init__(self, broker_account_repository: BrokerAccountRepositoryPort):
        self._repo = broker_account_repository

    def execute(
        self,
        account_id: str,
        user_id: str,
        paper_trading: Optional[bool] = None,
        is_active: Optional[bool] = None,
    ) -> Optional[BrokerAccount]:
        account = self._repo.get_by_id(account_id)
        if not account or account.user_id != user_id:
            return None

        if paper_trading is not None:
            account = account.switch_to_paper() if paper_trading else account.switch_to_live()
        if is_active is not None and not is_active:
            account = account.deactivate()

        self._repo.save(account)
        return account


class DeleteBrokerAccountUseCase:
    """Unlink a broker account."""

    def __init__(self, broker_account_repository: BrokerAccountRepositoryPort):
        self._repo = broker_account_repository

    def execute(self, account_id: str, user_id: str) -> bool:
        account = self._repo.get_by_id(account_id)
        if not account or account.user_id != user_id:
            return False
        return self._repo.delete(account_id)


# GetUserBrokerServiceUseCase removed 2026-04 — it had zero callers and its
# only reason to exist was to construct an AlpacaBrokerService from a stored
# BrokerAccount. If that flow is needed later, put the factory in
# `src/infrastructure/adapters/broker_factory.py` alongside the system-level
# factory rather than reintroducing this cross-layer import.
