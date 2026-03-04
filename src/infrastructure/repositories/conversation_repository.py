"""
Conversation Repository Implementation

Implements ConversationRepositoryPort for conversation persistence.

Architectural Intent:
- Translates between domain Conversation/Message entities and ORM models
- Handles session management via BaseRepository pattern
- Messages are eagerly loaded with conversations
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import List, Optional

from src.domain.entities.conversation import (
    Conversation,
    Message,
    MessageRole,
    TradeAction,
    TradeActionType,
)
from sqlalchemy.orm import joinedload

from src.domain.ports.conversation_repository_port import ConversationRepositoryPort
from src.infrastructure.database import get_database_manager
from src.infrastructure.orm_models import ConversationORM, MessageORM

logger = logging.getLogger(__name__)


class ConversationRepository(ConversationRepositoryPort):
    """Repository for Conversation entity persistence."""

    def _to_trade_action(self, data: dict) -> TradeAction:
        return TradeAction(
            symbol=data["symbol"],
            action=TradeActionType(data["action"]),
            quantity=data["quantity"],
            reasoning=data["reasoning"],
            confidence=data["confidence"],
            executed=data.get("executed", False),
        )

    def _message_to_domain(self, orm: MessageORM) -> Message:
        metadata = orm.metadata_json or {}
        trade_actions_data = metadata.pop("trade_actions", [])
        trade_actions = [self._to_trade_action(ta) for ta in trade_actions_data]

        return Message(
            id=orm.id,
            conversation_id=orm.conversation_id,
            role=MessageRole(orm.role),
            content=orm.content,
            created_at=orm.created_at,
            metadata=metadata,
            trade_actions=trade_actions,
        )

    def _message_to_orm(self, message: Message) -> MessageORM:
        metadata = dict(message.metadata)
        if message.trade_actions:
            metadata["trade_actions"] = [
                {
                    "symbol": ta.symbol,
                    "action": ta.action.value,
                    "quantity": ta.quantity,
                    "reasoning": ta.reasoning,
                    "confidence": ta.confidence,
                    "executed": ta.executed,
                }
                for ta in message.trade_actions
            ]

        return MessageORM(
            id=message.id,
            conversation_id=message.conversation_id,
            role=message.role.value,
            content=message.content,
            metadata_json=metadata if metadata else None,
            created_at=message.created_at,
        )

    def _to_domain(self, orm: ConversationORM, include_messages: bool = True) -> Conversation:
        messages = []
        if include_messages and orm.messages:
            messages = sorted(
                [self._message_to_domain(m) for m in orm.messages],
                key=lambda m: m.created_at,
            )

        return Conversation(
            id=orm.id,
            user_id=orm.user_id,
            title=orm.title,
            created_at=orm.created_at,
            updated_at=orm.updated_at,
            messages=messages,
        )

    def save(self, conversation: Conversation) -> Conversation:
        db_manager = get_database_manager()
        session = db_manager._session_factory()
        try:
            orm = ConversationORM(
                id=conversation.id,
                user_id=conversation.user_id,
                title=conversation.title,
                created_at=conversation.created_at,
                updated_at=conversation.updated_at,
            )
            session.merge(orm)
            session.commit()
            result = session.query(ConversationORM).filter(
                ConversationORM.id == conversation.id
            ).first()
            return self._to_domain(result)
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save conversation: {e}")
            raise
        finally:
            session.close()

    def get_by_id(
        self,
        conversation_id: str,
        user_id: str | None = None,
        message_limit: int = 200,
    ) -> Optional[Conversation]:
        db_manager = get_database_manager()
        session = db_manager._session_factory()
        try:
            query = (
                session.query(ConversationORM)
                .options(joinedload(ConversationORM.messages))
                .filter(ConversationORM.id == conversation_id)
            )
            if user_id is not None:
                query = query.filter(ConversationORM.user_id == user_id)
            orm = query.first()
            if orm and len(orm.messages) > message_limit:
                orm.messages = sorted(
                    orm.messages, key=lambda m: m.created_at
                )[-message_limit:]
            return self._to_domain(orm) if orm else None
        except Exception as e:
            logger.error(f"Failed to get conversation {conversation_id}: {e}")
            return None
        finally:
            session.close()

    def get_by_user_id(
        self, user_id: str, limit: int = 50, offset: int = 0
    ) -> List[Conversation]:
        db_manager = get_database_manager()
        session = db_manager._session_factory()
        try:
            orms = (
                session.query(ConversationORM)
                .filter(ConversationORM.user_id == user_id)
                .order_by(ConversationORM.updated_at.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )
            return [self._to_domain(orm, include_messages=False) for orm in orms]
        except Exception as e:
            logger.error(f"Failed to get conversations for user {user_id}: {e}")
            return []
        finally:
            session.close()

    def add_message(self, message: Message) -> Message:
        db_manager = get_database_manager()
        session = db_manager._session_factory()
        try:
            orm = self._message_to_orm(message)
            session.add(orm)

            # Update conversation's updated_at
            conv = session.query(ConversationORM).filter(
                ConversationORM.id == message.conversation_id
            ).first()
            if conv:
                conv.updated_at = datetime.utcnow()

            session.commit()
            session.refresh(orm)
            return self._message_to_domain(orm)
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to add message: {e}")
            raise
        finally:
            session.close()

    def delete(self, conversation_id: str) -> bool:
        db_manager = get_database_manager()
        session = db_manager._session_factory()
        try:
            orm = session.query(ConversationORM).filter(
                ConversationORM.id == conversation_id
            ).first()
            if not orm:
                return False
            session.delete(orm)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to delete conversation {conversation_id}: {e}")
            raise
        finally:
            session.close()
