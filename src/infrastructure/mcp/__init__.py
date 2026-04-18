"""
MCP (Model Context Protocol) servers.

Architectural Intent:
- 2026 rules §3.5: "One MCP server per bounded context. Tools = writes. Resources = reads."
- Three servers live here: market_data (all reads), portfolio (reads + writes),
  research (reads + writes). Each owns one bounded context and contains zero
  business logic — they wrap domain ports + services and validate input/output
  against JSON Schema at the boundary.
- The registry aggregates tools from every server and is the single entry point
  the chat use case talks to.
- Implementation is in-process (not separate stdio subprocesses) because we're
  running single-operator on one EC2. The bounded-context discipline is what the
  rule protects — the process model is an implementation choice. An ADR can
  capture this if/when we split into separate processes.
"""
from src.infrastructure.mcp.base import McpRegistry, McpServer, McpTool, McpResource

__all__ = ["McpRegistry", "McpServer", "McpTool", "McpResource"]
