"""
Tool Registry — port.

Architectural Intent:
- Application use cases (ChatUseCase) depend on this port, not on the MCP
  implementation in infrastructure.
- `ToolCallOutcome` is a domain-level value object so the use case never sees
  infrastructure types.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional

from src.domain.ports.ai_chat_port import ToolDefinition


@dataclass(frozen=True)
class ToolCallOutcome:
    """Result of dispatching a tool call through the registry.

    `payload` is the JSON-serialisable body returned to the AI; caller is
    responsible for stringifying if needed.
    """
    payload: Mapping[str, Any]
    is_error: bool = False
    error_message: Optional[str] = None


class ToolRegistryPort(ABC):
    """Abstracts the set of tools the AI can call.

    2026 rules §3.5 land on this port: each underlying MCP server owns one
    bounded context. The registry aggregates them so the chat use case stays
    context-agnostic.
    """

    @abstractmethod
    def tool_definitions(self) -> List[ToolDefinition]:
        """Return all tools the AI may call, in Claude-API shape."""

    @abstractmethod
    async def call_tool(
        self, tool_name: str, args: Mapping[str, Any], actor_user_id: str
    ) -> ToolCallOutcome:
        """Dispatch a tool call and return its outcome."""
