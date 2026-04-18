"""
Research MCP server — bounded context for backtesting and strategy research.

Layer: infrastructure
Ports used: backtest use case (injected; Phase 4 keeps it loose — Phase 6 will
            tighten the contract as live trading fleshes out the strategy ORM)
MCP integration: 1 tool (run_backtest) for now; save_strategy / fork_strategy
                 live in the dedicated strategies router today and will fold
                 into this server when the chat flow needs them (Phase 7).

Starting narrow on purpose — a single read-adjacent tool keeps the bounded
context's contract small and provable ahead of launch. Widening it later is a
simpler change than shrinking it after the API is in use.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

from src.infrastructure.mcp.base import McpServer, McpTool


class ResearchMcpServer(McpServer):
    """Strategy research and backtesting."""

    context = "research"

    def __init__(self, backtest_use_case: Optional[Any] = None) -> None:
        self._backtest_use_case = backtest_use_case
        super().__init__()

    def _register(self) -> None:
        self._add_tool(
            McpTool(
                name="run_backtest",
                description=(
                    "Run a backtest of a trading strategy "
                    "(sma_crossover, rsi_mean_reversion, momentum) on historical data."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "strategy": {
                            "type": "string",
                            "enum": ["sma_crossover", "rsi_mean_reversion", "momentum"],
                        },
                        "symbol": {"type": "string"},
                        "start_date": {
                            "type": "string",
                            "description": "Start date (YYYY-MM-DD). Default: 1 year ago.",
                        },
                        "end_date": {
                            "type": "string",
                            "description": "End date (YYYY-MM-DD). Default: today.",
                        },
                        "initial_capital": {
                            "type": "number",
                            "description": "Starting capital in USD (default: 10000)",
                            "minimum": 0,
                        },
                    },
                    "required": ["strategy", "symbol"],
                },
            )
        )

    async def _dispatch(
        self, tool_name: str, args: Mapping[str, Any], actor_user_id: str
    ) -> Mapping[str, Any]:
        if tool_name != "run_backtest":
            raise AssertionError(f"unreachable: tool {tool_name!r}")
        if self._backtest_use_case is None:
            raise RuntimeError("Backtesting not available")
        result = self._backtest_use_case.run(dict(args))
        # Normalise to a mapping: the backtest use case returns a dict or a
        # small result dataclass. Be defensive — MCP output must be JSON-ish.
        if hasattr(result, "__dict__"):
            return {"result": vars(result)}
        return {"result": result}
