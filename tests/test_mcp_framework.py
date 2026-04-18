"""
MCP framework tests — base registry + servers.

Covers (2026 rules §5 — MCP schema-compliance + round-trip):
- Every tool exposes a valid JSON Schema.
- Tool names are unique across all servers.
- Input validation rejects malformed args without reaching dispatch.
- Successful dispatch returns ToolCallOutcome with a JSON-serialisable payload.
- Errors from dispatch are wrapped, not leaked.
"""
from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from unittest.mock import MagicMock

import pytest
from jsonschema import Draft7Validator

from src.domain.ports.tool_registry import ToolCallOutcome
from src.domain.ports.audit_event_sink import AuditEvent, AuditEventSink
from src.infrastructure.mcp import McpRegistry, McpServer, McpTool
from src.infrastructure.mcp.market_data import MarketDataMcpServer
from src.infrastructure.mcp.portfolio import PortfolioMcpServer
from src.infrastructure.mcp.research import ResearchMcpServer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _InMemorySink(AuditEventSink):
    """Audit sink that captures events in-memory for assertions."""

    def __init__(self):
        self.events: list[AuditEvent] = []

    def append(self, event: AuditEvent) -> None:
        self.events.append(event)


def _make_market_data_server() -> MarketDataMcpServer:
    market_data = MagicMock()
    market_data.get_current_price.return_value = MagicMock(amount=Decimal("150.00"), currency="USD")
    market_data.get_market_news.return_value = ["Headline A", "Headline B"]
    ai_model = MagicMock()
    ai_model.predict_price_movement.return_value = 0.02
    ai_model.get_trading_signal.return_value = "BUY"
    news = MagicMock()
    news.batch_analyze_sentiment.return_value = [MagicMock(score=Decimal("3"))]
    return MarketDataMcpServer(
        market_data_service=market_data,
        ai_model_service=ai_model,
        news_analysis_service=news,
    )


def _make_portfolio_server(sink: AuditEventSink | None = None) -> PortfolioMcpServer:
    portfolio_repo = MagicMock()
    portfolio_repo.get_by_user_id.return_value = None
    user_repo = MagicMock()
    return PortfolioMcpServer(
        portfolio_repository=portfolio_repo,
        user_repository=user_repo,
        audit_sink=sink or _InMemorySink(),
    )


def _make_research_server() -> ResearchMcpServer:
    return ResearchMcpServer(backtest_use_case=None)


def _build_full_registry(sink: AuditEventSink | None = None) -> McpRegistry:
    registry = McpRegistry()
    registry.register(_make_market_data_server())
    registry.register(_make_portfolio_server(sink))
    registry.register(_make_research_server())
    return registry


# ---------------------------------------------------------------------------
# Schema compliance — every tool's input_schema must be a valid JSON Schema
# ---------------------------------------------------------------------------


@pytest.mark.infrastructure
def test_every_tool_declares_a_valid_json_schema():
    registry = _build_full_registry()
    for td in registry.tool_definitions():
        # Draft7Validator.check_schema raises on an invalid schema.
        Draft7Validator.check_schema(td.input_schema)
        assert td.input_schema.get("type") == "object", (
            f"{td.name} must have input_schema with type: object"
        )


@pytest.mark.infrastructure
def test_tool_names_are_unique_across_servers():
    registry = _build_full_registry()
    names = [td.name for td in registry.tool_definitions()]
    assert len(names) == len(set(names)), f"duplicate tool names: {names}"


@pytest.mark.infrastructure
def test_registry_rejects_duplicate_tool_registration():
    registry = McpRegistry()
    registry.register(_make_portfolio_server())
    with pytest.raises(ValueError, match="already owned"):
        # The market_data server has no overlap, but registering the same
        # portfolio server twice triggers the conflict check.
        registry.register(_make_portfolio_server())


# ---------------------------------------------------------------------------
# Input validation — malformed args rejected before dispatch
# ---------------------------------------------------------------------------


@pytest.mark.infrastructure
def test_missing_required_arg_returns_error_without_dispatch():
    server = _make_portfolio_server()
    # cancel_order requires `order_id`; omit it.
    outcome = asyncio.run(server.invoke("cancel_order", {}, actor_user_id="u1"))
    assert outcome.is_error
    assert "order_id" in (outcome.error_message or "")


@pytest.mark.infrastructure
def test_invalid_enum_value_rejected():
    server = _make_portfolio_server()
    bad_args = {"symbol": "AAPL", "action": "HOLD", "quantity": 1, "reasoning": "x"}
    outcome = asyncio.run(server.invoke("place_order", bad_args, actor_user_id="u1"))
    assert outcome.is_error
    assert "HOLD" in (outcome.error_message or "")


@pytest.mark.infrastructure
def test_negative_quantity_rejected_by_schema():
    server = _make_portfolio_server()
    outcome = asyncio.run(
        server.invoke(
            "place_order",
            {"symbol": "AAPL", "action": "BUY", "quantity": 0, "reasoning": "x"},
            actor_user_id="u1",
        )
    )
    assert outcome.is_error


# ---------------------------------------------------------------------------
# Round-trip — successful dispatch returns JSON-serialisable payload
# ---------------------------------------------------------------------------


@pytest.mark.infrastructure
def test_market_data_get_stock_price_round_trip():
    server = _make_market_data_server()
    outcome = asyncio.run(server.invoke("get_stock_price", {"symbol": "aapl"}, "u1"))
    assert not outcome.is_error
    assert outcome.payload["symbol"] == "AAPL"
    assert outcome.payload["price"] == 150.0
    # JSON-serialisable is a framework contract: the ChatUseCase calls json.dumps on it.
    json.dumps(dict(outcome.payload))


@pytest.mark.infrastructure
def test_portfolio_empty_round_trip():
    server = _make_portfolio_server()
    outcome = asyncio.run(server.invoke("get_portfolio", {}, "user-with-no-portfolio"))
    assert not outcome.is_error
    assert outcome.payload["num_positions"] == 0
    assert outcome.payload["positions"] == []


@pytest.mark.infrastructure
def test_place_order_emits_audit_event():
    sink = _InMemorySink()
    server = _make_portfolio_server(sink)
    outcome = asyncio.run(
        server.invoke(
            "place_order",
            {"symbol": "AAPL", "action": "BUY", "quantity": 10, "reasoning": "because"},
            actor_user_id="user-123",
        )
    )
    assert not outcome.is_error
    assert outcome.payload["status"] == "pending_confirmation"
    assert len(sink.events) == 1
    event = sink.events[0]
    assert event.action == "TradeRecommendationCreated"
    assert event.actor_user_id == "user-123"
    assert event.after_hash is not None
    # before_hash is None because there is no prior state for a new recommendation
    assert event.before_hash is None


# ---------------------------------------------------------------------------
# Error wrapping — exceptions from dispatch never escape raw
# ---------------------------------------------------------------------------


class _BoomServer(McpServer):
    context = "boom"

    def _register(self) -> None:
        self._add_tool(
            McpTool(
                name="boom",
                description="always raises",
                input_schema={"type": "object", "properties": {}},
            )
        )

    async def _dispatch(self, tool_name, args, actor_user_id):
        raise RuntimeError("kaboom")


@pytest.mark.infrastructure
def test_dispatch_exception_wrapped_as_error_outcome():
    registry = McpRegistry()
    registry.register(_BoomServer())
    outcome = asyncio.run(registry.call_tool("boom", {}, "u1"))
    assert outcome.is_error
    assert "kaboom" in (outcome.error_message or "")
    assert isinstance(outcome, ToolCallOutcome)


@pytest.mark.infrastructure
def test_unknown_tool_returns_error():
    registry = _build_full_registry()
    outcome = asyncio.run(registry.call_tool("totally_fake_tool", {}, "u1"))
    assert outcome.is_error
    assert "Unknown tool" in (outcome.error_message or "")
