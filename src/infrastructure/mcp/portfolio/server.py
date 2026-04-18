"""
Portfolio MCP server — bounded context for positions, orders, and
trade-recommendation writes.

Layer: infrastructure
Ports used: PortfolioRepositoryPort, OrderRepository (legacy port type),
            AuditEventSink
MCP integration: 3 read tools + 2 write tools
Stack choice: in-process — see src/infrastructure/mcp/__init__.py

Writes (`place_order`, `cancel_order`) emit audit events per 2026 rules §4.
`place_order` does NOT execute against the broker — it creates a trade
recommendation that the UI surfaces to the user for explicit confirmation.
Real-money execution is gated behind the Phase 6 live-mode flow (ADR-002).
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Mapping, Optional

from src.domain.ports import PortfolioRepositoryPort, UserRepositoryPort
from src.domain.ports.audit_event_sink import AuditEvent, AuditEventSink, hash_state
from src.domain.value_objects import Symbol
from src.infrastructure.mcp.base import McpServer, McpTool

logger = logging.getLogger(__name__)


class PortfolioMcpServer(McpServer):
    """Portfolio reads + trade-recommendation writes."""

    context = "portfolio"

    def __init__(
        self,
        portfolio_repository: PortfolioRepositoryPort,
        user_repository: UserRepositoryPort,
        audit_sink: AuditEventSink,
        order_repository: Optional[Any] = None,
    ) -> None:
        self._portfolios = portfolio_repository
        self._users = user_repository
        self._audit = audit_sink
        self._orders = order_repository
        super().__init__()

    def _register(self) -> None:
        # --- reads ---------------------------------------------------
        self._add_tool(
            McpTool(
                name="get_portfolio",
                description=(
                    "Get the user's current portfolio including positions, "
                    "cash balance, and P&L."
                ),
                input_schema={"type": "object", "properties": {}},
            )
        )
        self._add_tool(
            McpTool(
                name="get_orders",
                description=(
                    "Get the user's recent orders, optionally filtered by "
                    "status (PENDING, EXECUTED, CANCELLED)."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["PENDING", "EXECUTED", "CANCELLED", "ALL"],
                        }
                    },
                },
            )
        )
        self._add_tool(
            McpTool(
                name="get_position_details",
                description="Get detailed information about a specific position the user holds.",
                input_schema={
                    "type": "object",
                    "properties": {"symbol": {"type": "string"}},
                    "required": ["symbol"],
                },
            )
        )
        # --- writes --------------------------------------------------
        self._add_tool(
            McpTool(
                name="place_order",
                description=(
                    "Recommend a trade to the user. This creates a trade "
                    "recommendation card that the user must explicitly confirm. "
                    "Never call this without explaining the reasoning first."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"},
                        "action": {"type": "string", "enum": ["BUY", "SELL"]},
                        "quantity": {"type": "integer", "minimum": 1},
                        "reasoning": {"type": "string", "minLength": 1},
                    },
                    "required": ["symbol", "action", "quantity", "reasoning"],
                },
                is_write=True,
            )
        )
        self._add_tool(
            McpTool(
                name="cancel_order",
                description="Cancel a pending order by its order ID.",
                input_schema={
                    "type": "object",
                    "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"],
                },
                is_write=True,
            )
        )

    async def _dispatch(
        self, tool_name: str, args: Mapping[str, Any], actor_user_id: str
    ) -> Mapping[str, Any]:
        if tool_name == "get_portfolio":
            return self._get_portfolio(actor_user_id)
        if tool_name == "get_orders":
            return self._get_orders(args, actor_user_id)
        if tool_name == "get_position_details":
            return self._get_position_details(args, actor_user_id)
        if tool_name == "place_order":
            return self._place_order(args, actor_user_id)
        if tool_name == "cancel_order":
            return self._cancel_order(args, actor_user_id)
        raise AssertionError(f"unreachable: tool {tool_name!r}")

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def _get_portfolio(self, user_id: str) -> Mapping[str, Any]:
        portfolio = self._portfolios.get_by_user_id(user_id)
        if portfolio is None:
            return {
                "total_value": 0,
                "cash_balance": 0,
                "positions_value": 0,
                "num_positions": 0,
                "positions": [],
            }
        positions_data = [
            {
                "symbol": str(pos.symbol),
                "quantity": pos.quantity,
                "avg_price": float(pos.average_buy_price.amount),
                "current_price": float(pos.current_price.amount),
                "unrealized_pnl": float(pos.unrealized_pnl_amount.amount),
            }
            for pos in portfolio.positions
        ]
        return {
            "total_value": float(portfolio.total_value.amount),
            "cash_balance": float(portfolio.cash_balance.amount),
            "positions_value": float(portfolio.positions_value.amount),
            "num_positions": len(portfolio.positions),
            "positions": positions_data,
        }

    def _get_orders(self, args: Mapping[str, Any], user_id: str) -> Mapping[str, Any]:
        if self._orders is None:
            raise RuntimeError("Order service not available")
        status_filter = args.get("status", "ALL")
        orders = self._orders.get_by_user_id(user_id)
        if status_filter != "ALL":
            orders = [o for o in orders if o.status.value == status_filter]
        orders_data = [
            {
                "id": o.id,
                "symbol": str(o.symbol),
                "order_type": o.order_type.value,
                "position_type": o.position_type.value,
                "quantity": o.quantity,
                "status": o.status.value,
                "price": float(o.price.amount) if o.price else None,
                "placed_at": o.placed_at.isoformat() if o.placed_at else None,
            }
            for o in orders[:20]
        ]
        return {"orders": orders_data, "total": len(orders)}

    def _get_position_details(
        self, args: Mapping[str, Any], user_id: str
    ) -> Mapping[str, Any]:
        portfolio = self._portfolios.get_by_user_id(user_id)
        if portfolio is None:
            raise RuntimeError("No portfolio found")
        symbol = Symbol(args["symbol"].upper())
        position = portfolio.get_position(symbol)
        if position is None:
            raise RuntimeError(f"No position found for {symbol}")
        return {
            "symbol": str(position.symbol),
            "quantity": position.quantity,
            "position_type": position.position_type.value,
            "avg_buy_price": float(position.average_buy_price.amount),
            "current_price": float(position.current_price.amount),
            "market_value": float(position.market_value.amount),
            "unrealized_pnl": float(position.unrealized_pnl_amount.amount),
            "total_cost": float(position.total_cost.amount),
        }

    # ------------------------------------------------------------------
    # Write helpers — audit-emitting
    # ------------------------------------------------------------------

    def _place_order(
        self, args: Mapping[str, Any], user_id: str
    ) -> Mapping[str, Any]:
        """Create a trade recommendation (NOT an execution).

        The UI surfaces this to the user for confirmation; actual execution
        happens only after they click through, and the audit event for that
        execution is emitted by the order use case (Phase 6).
        """
        recommendation = {
            "status": "pending_confirmation",
            "symbol": args["symbol"],
            "action": args["action"],
            "quantity": args["quantity"],
            "reasoning": args.get("reasoning", ""),
            "message": "Trade recommendation created. Waiting for user confirmation.",
        }
        self._emit_audit(
            action="TradeRecommendationCreated",
            actor=user_id,
            aggregate_id=f"{user_id}:{args['symbol']}",
            payload=recommendation,
            before=None,
            after=recommendation,
        )
        return recommendation

    def _cancel_order(
        self, args: Mapping[str, Any], user_id: str
    ) -> Mapping[str, Any]:
        if self._orders is None:
            raise RuntimeError("Order service not available")

        order = self._orders.get_by_id(args["order_id"])
        if order is None:
            raise RuntimeError("Order not found")
        if order.user_id != user_id:
            raise RuntimeError("Unauthorized")
        if order.status.value != "PENDING":
            raise RuntimeError(f"Cannot cancel order with status {order.status.value}")

        from src.domain.entities.trading import OrderStatus

        before_snapshot = {
            "id": order.id, "status": order.status.value, "symbol": str(order.symbol),
        }
        updated = self._orders.update_status(args["order_id"], OrderStatus.CANCELLED)
        if updated is None:
            raise RuntimeError("Failed to cancel order")

        after_snapshot = {
            "id": updated.id, "status": updated.status.value, "symbol": str(updated.symbol),
        }
        self._emit_audit(
            action="OrderCancelled",
            actor=user_id,
            aggregate_id=args["order_id"],
            payload={"order_id": args["order_id"], "symbol": str(updated.symbol)},
            before=before_snapshot,
            after=after_snapshot,
        )
        return {
            "status": "cancelled",
            "order_id": args["order_id"],
            "symbol": str(updated.symbol),
        }

    def _emit_audit(
        self,
        action: str,
        actor: str,
        aggregate_id: str,
        payload: Mapping[str, Any],
        before: Optional[Mapping[str, Any]],
        after: Optional[Mapping[str, Any]],
    ) -> None:
        try:
            from src.infrastructure.observability import get_correlation_id

            correlation_id = get_correlation_id() or None
        except Exception:
            correlation_id = None

        try:
            self._audit.append(
                AuditEvent(
                    id=str(uuid.uuid4()),
                    actor_user_id=actor,
                    action=action,
                    aggregate_type="order",
                    aggregate_id=aggregate_id,
                    before_hash=hash_state(before) if before is not None else None,
                    after_hash=hash_state(after) if after is not None else None,
                    payload_json=dict(payload),
                    occurred_at=datetime.utcnow(),
                    correlation_id=correlation_id,
                )
            )
        except Exception:  # noqa: BLE001 — never let audit break the tool call
            logger.exception("audit append failed action=%s aggregate=%s", action, aggregate_id)
