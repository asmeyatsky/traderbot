"""
AI Chat Port

Architectural Intent:
- Defines the interface for AI chat generation services
- Domain layer depends on this abstraction, not concrete AI provider
- Supports streaming responses via async iterator
- Tool definitions enable the AI to call backend services
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional


@dataclass(frozen=True)
class ChatMessage:
    """Message format for the AI chat port."""
    role: str  # "user", "assistant", "system"
    content: str


@dataclass(frozen=True)
class ToolDefinition:
    """Definition of a tool the AI can call."""
    name: str
    description: str
    input_schema: Dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    """A tool call made by the AI."""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    """Result of executing a tool call."""
    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass(frozen=True)
class ChatStreamEvent:
    """
    Event emitted during chat streaming.

    Types:
    - text_delta: partial text content
    - tool_call: AI wants to call a tool
    - tool_result: result of a tool execution
    - done: stream complete
    - error: an error occurred
    """
    type: str  # text_delta, tool_call, tool_result, done, error
    content: str = ""
    tool_call: Optional[ToolCall] = None
    tool_result: Optional[ToolResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UserContext:
    """Context about the user for the AI to personalize responses."""
    user_id: str
    risk_tolerance: str
    investment_goal: str
    portfolio_summary: Optional[str] = None


class AIChatPort(ABC):
    """
    Port for AI chat generation.

    Architectural Intent:
    - Abstracts the AI provider (Claude, GPT, etc.)
    - Supports streaming for real-time chat experience
    - Tool calling enables the AI to interact with backend services
    """

    @abstractmethod
    async def generate_response(
        self,
        messages: List[ChatMessage],
        tools: List[ToolDefinition],
        user_context: UserContext,
        system_prompt: str,
    ) -> AsyncIterator[ChatStreamEvent]:
        """
        Generate a streaming AI response.

        Args:
            messages: Conversation history
            tools: Available tools the AI can call
            user_context: User context for personalization
            system_prompt: System-level instructions

        Yields:
            ChatStreamEvent instances as the response streams
        """
        pass
