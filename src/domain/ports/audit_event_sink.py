"""
Audit Event Sink — port.

Architectural Intent:
- 2026 rules §4: every write emits an audit event (actor, action, before/after
  hash, append-only, separate IAM).
- Domain defines this port; infrastructure implements against the
  `audit_events` table. Application use cases depend on this port only.
"""
from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class AuditEvent:
    """Immutable security-audit record.

    `before_hash` / `after_hash` are sha256 hex digests of the JSON-serialised
    before/after states (when applicable). A hash — not the payload — keeps the
    audit table small and PII-light while still providing tamper evidence.
    """
    id: str
    actor_user_id: Optional[str]
    action: str
    aggregate_type: str
    aggregate_id: str
    before_hash: Optional[str]
    after_hash: Optional[str]
    payload_json: Mapping[str, Any]
    occurred_at: datetime
    correlation_id: Optional[str] = None
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None


def hash_state(state: Any) -> str:
    """Deterministic sha256 of a JSON-serialisable state snapshot.

    Used to compute `before_hash` / `after_hash`. Falls back to repr() for
    non-JSON-serialisable objects — still deterministic as long as the object's
    repr is stable, which is true for frozen dataclasses.
    """
    try:
        serialised = json.dumps(state, sort_keys=True, default=str)
    except TypeError:
        serialised = repr(state)
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


class AuditEventSink(ABC):
    """Port for appending audit events. Implementations MUST be append-only."""

    @abstractmethod
    def append(self, event: AuditEvent) -> None:
        """Persist an audit event. Must never update or delete existing rows."""
        raise NotImplementedError
