"""
Tests for the Chat Feature

Covers:
- Domain: Conversation and Message entity tests (immutability, creation)
- Application: ChatUseCase tests with mocked dependencies
- Presentation: Chat router API endpoint tests
"""
import json
import pytest
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, MagicMock, patch

from src.domain.entities.conversation import (
    Conversation,
    Message,
    MessageRole,
    TradeAction,
    TradeActionType,
)
from src.domain.ports.ai_chat_port import (
    ChatMessage,
    ChatStreamEvent,
    ToolCall,
    ToolDefinition,
    ToolResult,
    UserContext,
)
from src.application.use_cases.chat import ChatUseCase, SYSTEM_PROMPT


# ============================================================================
# Domain Tests
# ============================================================================


class TestTradeAction:
    """Test TradeAction value object."""

    def test_create_valid_trade_action(self):
        ta = TradeAction(
            symbol="AAPL",
            action=TradeActionType.BUY,
            quantity=10,
            reasoning="Strong momentum",
            confidence=0.85,
        )
        assert ta.symbol == "AAPL"
        assert ta.action == TradeActionType.BUY
        assert ta.quantity == 10
        assert ta.executed is False

    def test_trade_action_immutability(self):
        ta = TradeAction(
            symbol="AAPL",
            action=TradeActionType.BUY,
            quantity=10,
            reasoning="Test",
            confidence=0.5,
        )
        with pytest.raises(AttributeError):
            ta.symbol = "MSFT"

    def test_mark_executed_returns_new_instance(self):
        ta = TradeAction(
            symbol="AAPL",
            action=TradeActionType.BUY,
            quantity=10,
            reasoning="Test",
            confidence=0.5,
        )
        executed = ta.mark_executed()
        assert executed.executed is True
        assert ta.executed is False
        assert executed is not ta

    def test_invalid_quantity_raises(self):
        with pytest.raises(ValueError, match="positive"):
            TradeAction(
                symbol="AAPL",
                action=TradeActionType.BUY,
                quantity=0,
                reasoning="Test",
                confidence=0.5,
            )

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            TradeAction(
                symbol="AAPL",
                action=TradeActionType.BUY,
                quantity=10,
                reasoning="Test",
                confidence=1.5,
            )


class TestMessage:
    """Test Message entity."""

    def test_create_message(self):
        msg = Message(
            id="msg-1",
            conversation_id="conv-1",
            role=MessageRole.USER,
            content="Hello",
            created_at=datetime.utcnow(),
        )
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"
        assert msg.trade_actions == []
        assert msg.metadata == {}

    def test_message_immutability(self):
        msg = Message(
            id="msg-1",
            conversation_id="conv-1",
            role=MessageRole.USER,
            content="Hello",
            created_at=datetime.utcnow(),
        )
        with pytest.raises(AttributeError):
            msg.content = "Changed"

    def test_with_trade_actions(self):
        msg = Message(
            id="msg-1",
            conversation_id="conv-1",
            role=MessageRole.ASSISTANT,
            content="I recommend buying AAPL",
            created_at=datetime.utcnow(),
        )
        ta = TradeAction(
            symbol="AAPL",
            action=TradeActionType.BUY,
            quantity=10,
            reasoning="Strong",
            confidence=0.8,
        )
        updated = msg.with_trade_actions([ta])
        assert len(updated.trade_actions) == 1
        assert msg.trade_actions == []  # Original unchanged


class TestConversation:
    """Test Conversation aggregate root."""

    def test_create_conversation(self):
        conv = Conversation(
            id="conv-1",
            user_id="user-1",
            title="Test Chat",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        assert conv.message_count == 0
        assert conv.last_message is None

    def test_add_message(self):
        conv = Conversation(
            id="conv-1",
            user_id="user-1",
            title="Test",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        msg = Message(
            id="msg-1",
            conversation_id="conv-1",
            role=MessageRole.USER,
            content="Hello",
            created_at=datetime.utcnow(),
        )
        updated = conv.add_message(msg)
        assert updated.message_count == 1
        assert conv.message_count == 0  # Original unchanged
        assert updated is not conv

    def test_update_title(self):
        conv = Conversation(
            id="conv-1",
            user_id="user-1",
            title="Old Title",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        updated = conv.update_title("New Title")
        assert updated.title == "New Title"
        assert conv.title == "Old Title"

    def test_conversation_immutability(self):
        conv = Conversation(
            id="conv-1",
            user_id="user-1",
            title="Test",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        with pytest.raises(AttributeError):
            conv.title = "Changed"


# ============================================================================
# Application Layer Tests
# ============================================================================


class TestChatUseCase:
    """Test ChatUseCase with mocked dependencies."""

    def _make_use_case(self):
        self.mock_ai = AsyncMock()
        self.mock_conv_repo = Mock()
        self.mock_user_repo = Mock()
        # Post-Phase 4: the use case only talks to the tool registry, not
        # individual services. Tool-level behaviour is covered in
        # test_mcp_framework.py.
        self.mock_tool_registry = Mock()

        return ChatUseCase(
            ai_chat_port=self.mock_ai,
            conversation_repository=self.mock_conv_repo,
            user_repository=self.mock_user_repo,
            tool_registry=self.mock_tool_registry,
        )

    def test_create_conversation(self):
        uc = self._make_use_case()
        expected = Conversation(
            id="conv-1",
            user_id="user-1",
            title="My Chat",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        self.mock_conv_repo.save.return_value = expected

        result = uc.create_conversation("user-1", "My Chat")

        self.mock_conv_repo.save.assert_called_once()
        assert result.user_id == "user-1"
        assert result.title == "My Chat"

    def test_get_conversation(self):
        uc = self._make_use_case()
        expected = Conversation(
            id="conv-1",
            user_id="user-1",
            title="Test",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        self.mock_conv_repo.get_by_id.return_value = expected

        result = uc.get_conversation("conv-1")

        assert result.id == "conv-1"
        self.mock_conv_repo.get_by_id.assert_called_once_with("conv-1")

    def test_get_conversation_not_found(self):
        uc = self._make_use_case()
        self.mock_conv_repo.get_by_id.return_value = None

        result = uc.get_conversation("nonexistent")

        assert result is None

    def test_delete_conversation(self):
        uc = self._make_use_case()
        self.mock_conv_repo.delete.return_value = True

        result = uc.delete_conversation("conv-1")

        assert result is True
        self.mock_conv_repo.delete.assert_called_once_with("conv-1")

    def test_get_user_conversations(self):
        uc = self._make_use_case()
        convs = [
            Conversation(
                id=f"conv-{i}",
                user_id="user-1",
                title=f"Chat {i}",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            for i in range(3)
        ]
        self.mock_conv_repo.get_by_user_id.return_value = convs

        result = uc.get_user_conversations("user-1", limit=10, offset=0)

        assert len(result) == 3
        self.mock_conv_repo.get_by_user_id.assert_called_once_with("user-1", 10, 0)

    # Tool-level behaviour moved to MCP servers (Phase 4). See
    # tests/test_mcp_framework.py for per-tool input validation, round-trip,
    # and audit-emission tests.

    # Remaining tool-level tests removed — coverage moved to
    # tests/test_mcp_framework.py where each MCP server is exercised directly.


# ============================================================================
# Presentation Layer Tests
# ============================================================================


class TestChatRouter:
    """Test chat API endpoint DTOs and response formatting."""

    def test_conversation_entity_serialization(self):
        """Test that conversation entities serialize correctly for API responses."""
        conv = Conversation(
            id="conv-1",
            user_id="test-user-1",
            title="Test Chat",
            created_at=datetime(2026, 3, 2, 12, 0, 0),
            updated_at=datetime(2026, 3, 2, 12, 0, 0),
        )

        # Verify the entity exposes the fields the router DTO needs
        assert conv.id == "conv-1"
        assert conv.user_id == "test-user-1"
        assert conv.title == "Test Chat"
        assert conv.message_count == 0

    def test_message_role_values_match_api_contract(self):
        """Verify MessageRole enum values match API response strings."""
        assert MessageRole.USER.value == "user"
        assert MessageRole.ASSISTANT.value == "assistant"
        assert MessageRole.SYSTEM.value == "system"

    def test_conversation_with_messages_structure(self):
        """Test that conversation messages serialize correctly."""
        msg = Message(
            id="msg-1",
            conversation_id="conv-1",
            role=MessageRole.USER,
            content="Hello AI",
            created_at=datetime(2026, 3, 2, 12, 0, 0),
        )
        conv = Conversation(
            id="conv-1",
            user_id="test-user-1",
            title="Test",
            created_at=datetime(2026, 3, 2, 12, 0, 0),
            updated_at=datetime(2026, 3, 2, 12, 0, 0),
            messages=[msg],
        )

        assert conv.message_count == 1
        assert conv.messages[0].role.value == "user"
        assert conv.messages[0].content == "Hello AI"

    def test_trade_action_serialization(self):
        """Test that trade actions in messages serialize for API response."""
        ta = TradeAction(
            symbol="AAPL",
            action=TradeActionType.BUY,
            quantity=10,
            reasoning="Bullish signal",
            confidence=0.85,
        )
        msg = Message(
            id="msg-1",
            conversation_id="conv-1",
            role=MessageRole.ASSISTANT,
            content="I recommend buying AAPL",
            created_at=datetime(2026, 3, 2, 12, 0, 0),
            trade_actions=[ta],
        )

        assert len(msg.trade_actions) == 1
        assert msg.trade_actions[0].symbol == "AAPL"
        assert msg.trade_actions[0].action.value == "BUY"
        assert msg.trade_actions[0].executed is False
        assert msg.trade_actions[0].confidence == 0.85
