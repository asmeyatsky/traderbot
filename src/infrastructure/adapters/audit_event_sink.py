"""
SQLAlchemy-backed audit event sink.

Architectural Intent:
- Implements `AuditEventSink` against the `audit_events` table created by
  migration 008.
- Insert-only — no update/delete methods exist here; in production the
  connection uses a role with INSERT grant only (see migration 008).
"""
from __future__ import annotations

import logging
from typing import Optional

from src.domain.ports.audit_event_sink import AuditEvent, AuditEventSink
from src.infrastructure.database import get_database_manager
from src.infrastructure.orm_models import AuditEventORM

logger = logging.getLogger(__name__)


class SqlAlchemyAuditEventSink(AuditEventSink):
    """Persists audit events to Postgres/SQLite via SQLAlchemy."""

    def append(self, event: AuditEvent) -> None:
        db_manager = get_database_manager()
        session = db_manager._session_factory()
        try:
            orm = AuditEventORM(
                id=event.id,
                actor_user_id=event.actor_user_id,
                action=event.action,
                aggregate_type=event.aggregate_type,
                aggregate_id=event.aggregate_id,
                before_hash=event.before_hash,
                after_hash=event.after_hash,
                payload_json=dict(event.payload_json),
                occurred_at=event.occurred_at,
                correlation_id=event.correlation_id,
                client_ip=event.client_ip,
                user_agent=event.user_agent,
            )
            session.add(orm)
            session.commit()
        except Exception:
            session.rollback()
            # Never let an audit failure bubble up into the business path — log
            # it loudly and keep the user's request flowing. Audit gaps are
            # discovered via the Prometheus counter `audit_event_sink_errors`
            # (wired in Phase 5 once test infrastructure lands).
            logger.exception(
                "Failed to append audit event action=%s aggregate=%s:%s",
                event.action, event.aggregate_type, event.aggregate_id,
            )
        finally:
            session.close()


# Module-level singleton — cheap to construct, no state. Wired into DI in
# src/infrastructure/di_container.py (Phase 6 will replace with container
# binding once tests demand alternate implementations).
audit_event_sink: Optional[AuditEventSink] = SqlAlchemyAuditEventSink()
