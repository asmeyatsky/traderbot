"""
Conversation Repository Port

Architectural Intent:
- Defines the interface for conversation persistence
- Domain layer depends on this abstraction
- Supports CRUD operations for conversations and messages
"""
from abc import ABC, abstractmethod
from typing import List, Optional

from src.domain.entities.conversation import Conversation, Message


class ConversationRepositoryPort(ABC):
    """
    Port for conversation persistence operations.

    Architectural Intent:
    - Defines contract for storing and retrieving conversations
    - Domain layer depends on this abstraction, not concrete implementation
    - Enables dependency inversion principle
    """

    @abstractmethod
    def save(self, conversation: Conversation) -> Conversation:
        """Save a conversation (create or update)."""
        pass

    @abstractmethod
    def get_by_id(self, conversation_id: str) -> Optional[Conversation]:
        """Retrieve a conversation by its ID, including messages."""
        pass

    @abstractmethod
    def get_by_user_id(
        self, user_id: str, limit: int = 50, offset: int = 0
    ) -> List[Conversation]:
        """Retrieve conversations for a user, ordered by updated_at desc."""
        pass

    @abstractmethod
    def add_message(self, message: Message) -> Message:
        """Add a message to a conversation."""
        pass

    @abstractmethod
    def delete(self, conversation_id: str) -> bool:
        """Delete a conversation and its messages."""
        pass
