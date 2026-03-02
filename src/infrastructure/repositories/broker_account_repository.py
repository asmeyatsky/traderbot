"""
Broker Account Repository Implementation

Architectural Intent:
- Implements BrokerAccountRepositoryPort using SQLAlchemy ORM
- Handles Fernet encryption/decryption of API keys at rest
- Encryption key derived from JWT_SECRET_KEY via HKDF for key isolation
- Domain entity always holds decrypted values in memory
"""
from __future__ import annotations

import base64
import logging
from datetime import datetime
from typing import List, Optional

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from src.domain.entities.broker_account import BrokerAccount, BrokerType
from src.domain.ports.broker_account_repository_port import BrokerAccountRepositoryPort
from src.infrastructure.database import get_db_session
from src.infrastructure.orm_models import BrokerAccountORM

logger = logging.getLogger(__name__)


def _derive_fernet_key(secret: str) -> bytes:
    """Derive a Fernet-compatible key from JWT_SECRET_KEY using HKDF."""
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=b"broker-account-keys",
        info=b"traderbot-broker-encryption",
    )
    key_bytes = hkdf.derive(secret.encode())
    return base64.urlsafe_b64encode(key_bytes)


class BrokerAccountRepository(BrokerAccountRepositoryPort):
    """SQLAlchemy implementation with Fernet-encrypted API keys."""

    def __init__(self):
        from src.infrastructure.config.settings import settings
        fernet_key = _derive_fernet_key(settings.JWT_SECRET_KEY)
        self._fernet = Fernet(fernet_key)

    def _encrypt(self, plaintext: str) -> str:
        return self._fernet.encrypt(plaintext.encode()).decode()

    def _decrypt(self, ciphertext: str) -> str:
        return self._fernet.decrypt(ciphertext.encode()).decode()

    def _to_entity(self, orm: BrokerAccountORM) -> BrokerAccount:
        return BrokerAccount(
            id=orm.id,
            user_id=orm.user_id,
            broker_type=BrokerType(orm.broker_type),
            api_key=self._decrypt(orm.encrypted_api_key),
            secret_key=self._decrypt(orm.encrypted_secret_key),
            paper_trading=orm.paper_trading,
            label=orm.label,
            is_active=orm.is_active,
            created_at=orm.created_at,
            updated_at=orm.updated_at,
        )

    def _to_orm(self, entity: BrokerAccount) -> BrokerAccountORM:
        return BrokerAccountORM(
            id=entity.id,
            user_id=entity.user_id,
            broker_type=entity.broker_type.value,
            encrypted_api_key=self._encrypt(entity.api_key),
            encrypted_secret_key=self._encrypt(entity.secret_key),
            paper_trading=entity.paper_trading,
            label=entity.label,
            is_active=entity.is_active,
            created_at=entity.created_at or datetime.utcnow(),
            updated_at=entity.updated_at or datetime.utcnow(),
        )

    def save(self, account: BrokerAccount) -> BrokerAccount:
        session = get_db_session()
        try:
            existing = session.query(BrokerAccountORM).filter_by(id=account.id).first()
            if existing:
                existing.broker_type = account.broker_type.value
                existing.encrypted_api_key = self._encrypt(account.api_key)
                existing.encrypted_secret_key = self._encrypt(account.secret_key)
                existing.paper_trading = account.paper_trading
                existing.label = account.label
                existing.is_active = account.is_active
                existing.updated_at = datetime.utcnow()
            else:
                orm = self._to_orm(account)
                session.add(orm)
            session.commit()
            return account
        except Exception:
            session.rollback()
            raise

    def get_by_id(self, account_id: str) -> Optional[BrokerAccount]:
        session = get_db_session()
        orm = session.query(BrokerAccountORM).filter_by(id=account_id).first()
        return self._to_entity(orm) if orm else None

    def get_by_user_and_broker(
        self, user_id: str, broker_type: BrokerType
    ) -> Optional[BrokerAccount]:
        session = get_db_session()
        orm = (
            session.query(BrokerAccountORM)
            .filter_by(user_id=user_id, broker_type=broker_type.value)
            .first()
        )
        return self._to_entity(orm) if orm else None

    def get_by_user(self, user_id: str) -> List[BrokerAccount]:
        session = get_db_session()
        orms = (
            session.query(BrokerAccountORM)
            .filter_by(user_id=user_id)
            .order_by(BrokerAccountORM.created_at)
            .all()
        )
        return [self._to_entity(orm) for orm in orms]

    def delete(self, account_id: str) -> bool:
        session = get_db_session()
        try:
            orm = session.query(BrokerAccountORM).filter_by(id=account_id).first()
            if not orm:
                return False
            session.delete(orm)
            session.commit()
            return True
        except Exception:
            session.rollback()
            raise
