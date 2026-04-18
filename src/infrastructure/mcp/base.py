"""
MCP framework — abstract server + registry.

Architectural Intent:
- `McpServer` is the abstract bounded-context server. Each concrete server
  owns one context (market_data, portfolio, research).
- `McpRegistry` aggregates servers, validates tool input/output against JSON
  Schema, and dispatches tool calls to the right server.
- Callers (ChatUseCase) interact only with the registry — they never see or
  reach into an individual server. This keeps the chat use case free of
  bounded-context knowledge.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from jsonschema import Draft7Validator, ValidationError

from src.domain.ports.ai_chat_port import ToolDefinition
from src.domain.ports.tool_registry import ToolCallOutcome, ToolRegistryPort

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class McpTool:
    """A tool exposed by an MCP server.

    `is_write` separates tools (writes, 2026 rules §3.5) from resources (reads).
    Writes are expected to emit audit events — enforced in each server's
    implementation, not the framework.
    """
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Optional[Dict[str, Any]] = None
    is_write: bool = False


@dataclass(frozen=True)
class McpResource:
    """A resource (read-only) exposed by an MCP server.

    URIs follow `<scheme>://<path>` where the scheme names the resource type
    (e.g. `price://AAPL`). The server maps the URI to a handler.
    """
    uri_template: str
    name: str
    description: str


@dataclass(frozen=True)
class McpToolResult:
    """Result of a tool invocation — internal to this module.

    The registry translates this to `ToolCallOutcome` (a domain-level type)
    before returning to the application layer.
    """
    payload: Mapping[str, Any]
    is_error: bool = False
    error_message: Optional[str] = None


# ---------------------------------------------------------------------------
# Server base class
# ---------------------------------------------------------------------------


class McpServer(ABC):
    """Abstract MCP server — one bounded context per subclass.

    Subclasses register tools in `_tools` and implement `_dispatch`. The base
    class validates input against the tool's JSON Schema before dispatch and
    validates the payload against `output_schema` (if present) before return.
    """

    # Short identifier used in logs and registry conflict resolution.
    context: str = "unnamed"

    def __init__(self) -> None:
        self._tools: Dict[str, McpTool] = {}
        self._resources: Dict[str, McpResource] = {}
        self._validators: Dict[str, Draft7Validator] = {}
        self._output_validators: Dict[str, Draft7Validator] = {}
        self._register()

    # -- subclass hooks ----------------------------------------------------

    @abstractmethod
    def _register(self) -> None:
        """Populate `self._tools` and `self._resources`. Called from __init__."""

    @abstractmethod
    async def _dispatch(
        self, tool_name: str, args: Mapping[str, Any], actor_user_id: str
    ) -> Mapping[str, Any]:
        """Invoke the named tool. Must return a JSON-serialisable mapping.

        Raising is allowed — the registry converts exceptions to `is_error`.
        """

    # -- helpers for subclasses -------------------------------------------

    def _add_tool(self, tool: McpTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name!r} already registered in {self.context}")
        self._tools[tool.name] = tool
        self._validators[tool.name] = Draft7Validator(tool.input_schema)
        if tool.output_schema is not None:
            self._output_validators[tool.name] = Draft7Validator(tool.output_schema)

    def _add_resource(self, resource: McpResource) -> None:
        self._resources[resource.uri_template] = resource

    # -- framework API (called by the registry) ---------------------------

    def list_tools(self) -> List[McpTool]:
        return list(self._tools.values())

    def list_resources(self) -> List[McpResource]:
        return list(self._resources.values())

    def has_tool(self, tool_name: str) -> bool:
        return tool_name in self._tools

    async def invoke(
        self, tool_name: str, args: Mapping[str, Any], actor_user_id: str
    ) -> McpToolResult:
        """Validate, dispatch, validate output, wrap in result."""
        if tool_name not in self._tools:
            return McpToolResult(
                payload={},
                is_error=True,
                error_message=f"Unknown tool {tool_name!r} in {self.context}",
            )

        # Input validation — reject malformed args with the first error.
        validator = self._validators[tool_name]
        input_errors = sorted(validator.iter_errors(dict(args)), key=lambda e: e.path)
        if input_errors:
            first = input_errors[0]
            logger.warning(
                "mcp_input_validation_failed context=%s tool=%s error=%s",
                self.context, tool_name, first.message,
            )
            return McpToolResult(
                payload={},
                is_error=True,
                error_message=f"Invalid input for {tool_name}: {first.message}",
            )

        try:
            payload = await self._dispatch(tool_name, args, actor_user_id)
        except Exception as exc:  # noqa: BLE001 — wrap and surface
            logger.exception(
                "mcp_dispatch_error context=%s tool=%s", self.context, tool_name,
            )
            return McpToolResult(
                payload={},
                is_error=True,
                error_message=f"Error executing {tool_name}: {exc}",
            )

        # Output validation — strict when schema is declared, best-effort log otherwise.
        output_validator = self._output_validators.get(tool_name)
        if output_validator is not None:
            try:
                output_validator.validate(dict(payload))
            except ValidationError as exc:
                logger.error(
                    "mcp_output_validation_failed context=%s tool=%s error=%s",
                    self.context, tool_name, exc.message,
                )
                return McpToolResult(
                    payload={},
                    is_error=True,
                    error_message=(
                        f"{tool_name} returned invalid output — refusing to pass to "
                        f"the AI: {exc.message}"
                    ),
                )

        return McpToolResult(payload=payload)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class McpRegistry(ToolRegistryPort):
    """Aggregates MCP servers and routes tool calls to the right one.

    Implements `ToolRegistryPort` so the application layer stays unaware of the
    MCP-specific types. Tool names are unique across ALL servers (enforced at
    registration).
    """

    def __init__(self) -> None:
        self._servers: List[McpServer] = []
        self._tool_owner: Dict[str, McpServer] = {}

    def register(self, server: McpServer) -> None:
        for tool in server.list_tools():
            if tool.name in self._tool_owner:
                owner = self._tool_owner[tool.name]
                raise ValueError(
                    f"Tool {tool.name!r} already owned by {owner.context!r}; "
                    f"cannot also register in {server.context!r}"
                )
            self._tool_owner[tool.name] = server
        self._servers.append(server)
        logger.info(
            "mcp_server_registered context=%s tools=%d resources=%d",
            server.context, len(server.list_tools()), len(server.list_resources()),
        )

    def tool_definitions(self) -> List[ToolDefinition]:
        """Render all tools in the shape the AIChatPort expects."""
        defs: List[ToolDefinition] = []
        for server in self._servers:
            for tool in server.list_tools():
                defs.append(
                    ToolDefinition(
                        name=tool.name,
                        description=tool.description,
                        input_schema=tool.input_schema,
                    )
                )
        return defs

    async def call_tool(
        self, tool_name: str, args: Mapping[str, Any], actor_user_id: str
    ) -> ToolCallOutcome:
        """Dispatch a tool call to its owning server."""
        server = self._tool_owner.get(tool_name)
        if server is None:
            return ToolCallOutcome(
                payload={},
                is_error=True,
                error_message=f"Unknown tool {tool_name!r}",
            )
        result = await server.invoke(tool_name, args, actor_user_id)
        return ToolCallOutcome(
            payload=result.payload,
            is_error=result.is_error,
            error_message=result.error_message,
        )
