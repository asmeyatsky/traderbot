"""
Conversation Domain Entities

Architectural Intent:
- Models the chat conversation lifecycle following DDD principles
- Conversation is the aggregate root containing Messages
- TradeAction is a value object representing AI-recommended trades
- All entities are immutable to prevent accidental state corruption
- Messages support metadata for tool calls and trade actions
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class MessageRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class TradeActionType(Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True)
class TradeAction:
    """
    Trade Action Value Object

    Represents an AI-recommended trade that requires user confirmation.

    Invariants:
    - Quantity must be positive
    - Confidence is 0-1 scale
    - executed is False until user explicitly confirms
    """
    symbol: str
    action: TradeActionType
    quantity: int
    reasoning: str
    confidence: float
    executed: bool = False

    def __post_init__(self):
        if self.quantity <= 0:
            raise ValueError("Trade quantity must be positive")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0 and 1")

    def mark_executed(self) -> TradeAction:
        return replace(self, executed=True)


@dataclass(frozen=True)
class Message:
    """
    Message Entity

    Represents a single message in a conversation.

    Invariants:
    - Content must not be empty for user messages
    - Role must be a valid MessageRole
    - Metadata stores tool calls, trade actions, and other structured data
    """
    id: str
    conversation_id: str
    role: MessageRole
    content: str
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    trade_actions: List[TradeAction] = field(default_factory=list)

    def with_trade_actions(self, actions: List[TradeAction]) -> Message:
        return replace(self, trade_actions=actions)


@dataclass(frozen=True)
class Conversation:
    """
    Conversation Aggregate Root

    Invariants:
    - A conversation must belong to a user
    - Messages are ordered by created_at
    - Title is auto-generated from first user message if not provided
    """
    id: str
    user_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    messages: List[Message] = field(default_factory=list)

    def add_message(self, message: Message) -> Conversation:
        new_messages = list(self.messages)
        new_messages.append(message)
        return replace(
            self,
            messages=new_messages,
            updated_at=datetime.utcnow(),
        )

    def update_title(self, title: str) -> Conversation:
        return replace(self, title=title, updated_at=datetime.utcnow())

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def last_message(self) -> Optional[Message]:
        return self.messages[-1] if self.messages else None
