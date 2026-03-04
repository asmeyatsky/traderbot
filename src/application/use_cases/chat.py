"""
Chat Use Case

Architectural Intent:
- Orchestrates the AI chat flow: receives user messages, builds context,
  calls Claude with tools, processes tool results, and persists messages
- Tools map to existing backend services (market data, ML, portfolio, risk)
- The AI never auto-executes trades; it recommends + returns a TradeAction
  that requires explicit user confirmation via a separate endpoint
- One use case class following single responsibility principle
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any, AsyncIterator, Dict, List, Optional, TYPE_CHECKING

from src.domain.entities.conversation import (
    Conversation,
    Message,
    MessageRole,
    TradeAction,
    TradeActionType,
)
from src.domain.ports.ai_chat_port import (
    AIChatPort,
    ChatMessage,
    ChatStreamEvent,
    ToolCall,
    ToolDefinition,
    ToolResult,
    UserContext,
)
from src.domain.ports.conversation_repository_port import ConversationRepositoryPort
from src.domain.ports import (
    MarketDataPort,
    AIModelPort,
    NewsAnalysisPort,
    OrderRepositoryPort,
    PortfolioRepositoryPort,
    UserRepositoryPort,
)
from src.domain.value_objects import Symbol
from src.domain.services.technical_analysis import TechnicalAnalysisPort

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are TraderBot AI, an intelligent trading co-pilot. You help users analyze stocks, \
understand market conditions, manage their portfolio, and make informed trading decisions.

Key behaviors:
- Be concise and actionable. Lead with the answer, then explain.
- When recommending trades, ALWAYS use the place_order tool so the user sees a confirmation card.
- Never execute trades without the user explicitly confirming.
- Use real data from tools — never fabricate prices, predictions, or sentiment scores.
- If a tool call fails, acknowledge the error and suggest alternatives.
- Format numbers clearly: currency with $, percentages with %, large numbers with commas.
- When showing multiple stocks, use a structured comparison.
- Use get_technical_analysis for detailed indicator breakdowns (RSI, MACD, Bollinger Bands, etc.).
- Use screen_stocks to find stocks matching criteria or prebuilt screens like top_gainers, top_losers, most_active.
- Use get_market_status to check which global exchanges are currently open.
- Use run_backtest to test trading strategies against historical data.
"""

TOOL_DEFINITIONS: List[ToolDefinition] = [
    ToolDefinition(
        name="get_stock_price",
        description="Get the current real-time price for a stock symbol.",
        input_schema={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock ticker symbol (e.g., AAPL, MSFT, TSLA)",
                }
            },
            "required": ["symbol"],
        },
    ),
    ToolDefinition(
        name="get_ml_prediction",
        description="Get an AI/ML price movement prediction for a stock. Returns predicted direction and confidence.",
        input_schema={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                },
                "days": {
                    "type": "integer",
                    "description": "Number of days to predict ahead (default: 1)",
                    "default": 1,
                },
            },
            "required": ["symbol"],
        },
    ),
    ToolDefinition(
        name="get_trading_signal",
        description="Get a BUY/SELL/HOLD trading signal for a stock based on ensemble ML models.",
        input_schema={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                }
            },
            "required": ["symbol"],
        },
    ),
    ToolDefinition(
        name="get_news_sentiment",
        description="Get news sentiment analysis for a stock. Returns sentiment score and recent headlines.",
        input_schema={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                }
            },
            "required": ["symbol"],
        },
    ),
    ToolDefinition(
        name="get_portfolio",
        description="Get the user's current portfolio including positions, cash balance, and P&L.",
        input_schema={
            "type": "object",
            "properties": {},
        },
    ),
    ToolDefinition(
        name="place_order",
        description=(
            "Recommend a trade to the user. This creates a trade recommendation card "
            "that the user must explicitly confirm. Never call this without explaining "
            "the reasoning first."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Stock ticker symbol"},
                "action": {
                    "type": "string",
                    "enum": ["BUY", "SELL"],
                    "description": "Trade action",
                },
                "quantity": {
                    "type": "integer",
                    "description": "Number of shares",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation for the recommendation",
                },
            },
            "required": ["symbol", "action", "quantity", "reasoning"],
        },
    ),
    # Phase 2 — Real Technical Indicators
    ToolDefinition(
        name="get_technical_analysis",
        description="Get detailed technical analysis indicators for a stock: RSI, MACD, SMA, EMA, Bollinger Bands, ATR, Stochastic, ADX.",
        input_schema={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                }
            },
            "required": ["symbol"],
        },
    ),
    # Phase 3 — Chat Trading Improvements
    ToolDefinition(
        name="get_orders",
        description="Get the user's recent orders, optionally filtered by status (PENDING, EXECUTED, CANCELLED).",
        input_schema={
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["PENDING", "EXECUTED", "CANCELLED", "ALL"],
                    "description": "Filter by order status (default: ALL)",
                }
            },
        },
    ),
    ToolDefinition(
        name="cancel_order",
        description="Cancel a pending order by its order ID.",
        input_schema={
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The ID of the order to cancel",
                }
            },
            "required": ["order_id"],
        },
    ),
    ToolDefinition(
        name="get_position_details",
        description="Get detailed information about a specific position the user holds.",
        input_schema={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                }
            },
            "required": ["symbol"],
        },
    ),
    # Phase 1 — Stock Screening
    ToolDefinition(
        name="screen_stocks",
        description="Screen stocks by criteria or use prebuilt screens (top_gainers, top_losers, most_active, high_momentum, oversold_rsi).",
        input_schema={
            "type": "object",
            "properties": {
                "prebuilt_screen": {
                    "type": "string",
                    "enum": ["top_gainers", "top_losers", "most_active", "high_momentum", "oversold_rsi"],
                    "description": "Use a prebuilt screen",
                },
                "min_change_pct": {
                    "type": "number",
                    "description": "Minimum daily change percentage",
                },
                "max_change_pct": {
                    "type": "number",
                    "description": "Maximum daily change percentage",
                },
                "min_volume": {
                    "type": "integer",
                    "description": "Minimum trading volume",
                },
                "sectors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by sectors",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (default: 10)",
                },
            },
        },
    ),
    # Phase 4 — Multi-Market Coverage
    ToolDefinition(
        name="get_market_status",
        description="Check which global stock exchanges are currently open or closed, with next open/close times.",
        input_schema={
            "type": "object",
            "properties": {},
        },
    ),
    # Phase 6 — Backtesting
    ToolDefinition(
        name="run_backtest",
        description="Run a backtest of a trading strategy (sma_crossover, rsi_mean_reversion, momentum) on historical data.",
        input_schema={
            "type": "object",
            "properties": {
                "strategy": {
                    "type": "string",
                    "enum": ["sma_crossover", "rsi_mean_reversion", "momentum"],
                    "description": "Strategy to backtest",
                },
                "symbol": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                },
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
                },
            },
            "required": ["strategy", "symbol"],
        },
    ),
]


class ChatUseCase:
    """
    Use Case: Handle a chat message and stream an AI response.

    Orchestrates:
    1. Loading conversation history
    2. Building user context from existing services
    3. Calling Claude with tool definitions
    4. Executing tool calls against backend services
    5. Persisting user and assistant messages
    """

    def __init__(
        self,
        ai_chat_port: AIChatPort,
        conversation_repository: ConversationRepositoryPort,
        market_data_service: MarketDataPort,
        ai_model_service: AIModelPort,
        news_analysis_service: NewsAnalysisPort,
        portfolio_repository: PortfolioRepositoryPort,
        user_repository: UserRepositoryPort,
        order_repository: Optional[Any] = None,
        technical_analysis_port: Optional[TechnicalAnalysisPort] = None,
        stock_screener=None,
        exchange_registry=None,
        backtest_use_case=None,
        ensemble_predictor=None,
    ):
        self._ai = ai_chat_port
        self._conversations = conversation_repository
        self._market_data = market_data_service
        self._ai_model = ai_model_service
        self._news = news_analysis_service
        self._portfolios = portfolio_repository
        self._users = user_repository
        self._orders = order_repository
        self._technical_analysis = technical_analysis_port
        self._stock_screener = stock_screener
        self._exchange_registry = exchange_registry
        self._backtest_use_case = backtest_use_case
        self._ensemble_predictor = ensemble_predictor

    def create_conversation(self, user_id: str, title: str = "New conversation") -> Conversation:
        """Create a new conversation for a user."""
        conversation = Conversation(
            id=str(uuid.uuid4()),
            user_id=user_id,
            title=title,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        return self._conversations.save(conversation)

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID."""
        return self._conversations.get_by_id(conversation_id)

    def get_user_conversations(
        self, user_id: str, limit: int = 50, offset: int = 0
    ) -> List[Conversation]:
        """Get all conversations for a user."""
        return self._conversations.get_by_user_id(user_id, limit, offset)

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation."""
        return self._conversations.delete(conversation_id)

    async def send_message(
        self, conversation_id: str, user_id: str, content: str
    ) -> AsyncIterator[ChatStreamEvent]:
        """
        Process a user message and stream the AI response.

        Yields ChatStreamEvents for SSE streaming to the frontend.
        """
        # Load conversation
        conversation = self._conversations.get_by_id(conversation_id)
        if not conversation:
            yield ChatStreamEvent(type="error", content="Conversation not found")
            return
        if conversation.user_id != user_id:
            yield ChatStreamEvent(type="error", content="Unauthorized")
            return

        # Persist user message
        user_message = Message(
            id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            role=MessageRole.USER,
            content=content,
            created_at=datetime.utcnow(),
        )
        self._conversations.add_message(user_message)

        # Auto-generate title from first message
        if conversation.message_count == 0:
            title = content[:80] + ("..." if len(content) > 80 else "")
            updated = conversation.update_title(title)
            self._conversations.save(updated)

        # Build context
        user = self._users.get_by_id(user_id)
        user_context = UserContext(
            user_id=user_id,
            risk_tolerance=user.risk_tolerance.value if user else "moderate",
            investment_goal=user.investment_goal.value if user else "balanced_growth",
        )

        # Build message history for AI
        conversation = self._conversations.get_by_id(conversation_id)
        chat_messages = [
            ChatMessage(role=msg.role.value, content=msg.content)
            for msg in conversation.messages
        ]

        # Stream AI response with tool loop
        full_response = ""
        trade_actions: List[TradeAction] = []
        pending_tool_calls: List[ToolCall] = []

        async for event in self._ai.generate_response(
            messages=chat_messages,
            tools=TOOL_DEFINITIONS,
            user_context=user_context,
            system_prompt=SYSTEM_PROMPT,
        ):
            if event.type == "text_delta":
                full_response += event.content
                yield event

            elif event.type == "tool_call":
                pending_tool_calls.append(event.tool_call)
                # Execute tool and yield result
                tool_result = await self._execute_tool(
                    event.tool_call, user_id
                )
                yield ChatStreamEvent(
                    type="tool_result",
                    tool_result=tool_result,
                    metadata={"tool_name": event.tool_call.name},
                )

                # If it was a place_order tool, extract the trade action
                if event.tool_call.name == "place_order":
                    args = event.tool_call.arguments
                    try:
                        ta = TradeAction(
                            symbol=args["symbol"],
                            action=TradeActionType(args["action"]),
                            quantity=args["quantity"],
                            reasoning=args.get("reasoning", ""),
                            confidence=0.0,
                        )
                        trade_actions.append(ta)
                    except (KeyError, ValueError) as e:
                        logger.warning(f"Failed to parse trade action: {e}")

                # Continue the conversation with tool results
                # Build updated messages including tool call + result
                tool_messages = list(chat_messages)
                tool_messages.append(
                    ChatMessage(role="assistant", content=full_response)
                )
                tool_messages.append(
                    ChatMessage(
                        role="user",
                        content=f"[Tool result for {event.tool_call.name}]: {tool_result.content}",
                    )
                )

                # Get continuation from AI
                continuation = ""
                async for cont_event in self._ai.generate_response(
                    messages=tool_messages,
                    tools=TOOL_DEFINITIONS,
                    user_context=user_context,
                    system_prompt=SYSTEM_PROMPT,
                ):
                    if cont_event.type == "text_delta":
                        continuation += cont_event.content
                        yield cont_event
                    elif cont_event.type == "done":
                        break

                full_response += continuation

            elif event.type == "error":
                yield event
                return

            elif event.type == "done":
                pass

        # Persist assistant message
        assistant_message = Message(
            id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            role=MessageRole.ASSISTANT,
            content=full_response,
            created_at=datetime.utcnow(),
            trade_actions=trade_actions,
            metadata={
                "tool_calls": [
                    {"name": tc.name, "arguments": tc.arguments}
                    for tc in pending_tool_calls
                ]
            }
            if pending_tool_calls
            else {},
        )
        self._conversations.add_message(assistant_message)

        yield ChatStreamEvent(
            type="done",
            metadata={
                "message_id": assistant_message.id,
                "trade_actions": [
                    {
                        "symbol": ta.symbol,
                        "action": ta.action.value,
                        "quantity": ta.quantity,
                        "reasoning": ta.reasoning,
                    }
                    for ta in trade_actions
                ],
            },
        )

    async def _execute_tool(self, tool_call: ToolCall, user_id: str) -> ToolResult:
        """Execute a tool call against the appropriate backend service."""
        try:
            name = tool_call.name
            args = tool_call.arguments

            if name == "get_stock_price":
                return await self._tool_get_stock_price(args)
            elif name == "get_ml_prediction":
                return await self._tool_get_ml_prediction(args)
            elif name == "get_trading_signal":
                return await self._tool_get_trading_signal(args)
            elif name == "get_news_sentiment":
                return await self._tool_get_news_sentiment(args)
            elif name == "get_portfolio":
                return await self._tool_get_portfolio(user_id)
            elif name == "place_order":
                return await self._tool_place_order(args, user_id)
            elif name == "get_technical_analysis":
                return await self._tool_get_technical_analysis(args)
            elif name == "get_orders":
                return await self._tool_get_orders(args, user_id)
            elif name == "cancel_order":
                return await self._tool_cancel_order(args, user_id)
            elif name == "get_position_details":
                return await self._tool_get_position_details(args, user_id)
            elif name == "screen_stocks":
                return await self._tool_screen_stocks(args)
            elif name == "get_market_status":
                return await self._tool_get_market_status()
            elif name == "run_backtest":
                return await self._tool_run_backtest(args)
            else:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    content=f"Unknown tool: {name}",
                    is_error=True,
                )
        except Exception as e:
            logger.error(f"Tool execution error ({tool_call.name}): {e}")
            return ToolResult(
                tool_call_id=tool_call.id,
                content=f"Error executing {tool_call.name}: {str(e)}",
                is_error=True,
            )

    async def _tool_get_stock_price(self, args: Dict[str, Any]) -> ToolResult:
        if "symbol" not in args:
            return ToolResult(tool_call_id="", content="Missing required argument: symbol", is_error=True)
        symbol = Symbol(args["symbol"].upper())
        price = self._market_data.get_current_price(symbol)
        if price:
            return ToolResult(
                tool_call_id="",
                content=json.dumps({
                    "symbol": str(symbol),
                    "price": float(price.amount),
                    "currency": price.currency,
                }),
            )
        return ToolResult(
            tool_call_id="",
            content=f"Could not fetch price for {symbol}",
            is_error=True,
        )

    async def _tool_get_ml_prediction(self, args: Dict[str, Any]) -> ToolResult:
        if "symbol" not in args:
            return ToolResult(tool_call_id="", content="Missing required argument: symbol", is_error=True)
        symbol = Symbol(args["symbol"].upper())
        days = args.get("days", 1)

        # Use ensemble predictor when available for richer results
        if self._ensemble_predictor is not None:
            try:
                pred = self._ensemble_predictor.predict(symbol)
                return ToolResult(
                    tool_call_id="",
                    content=json.dumps({
                        "symbol": pred.symbol,
                        "direction": pred.direction,
                        "confidence": pred.confidence,
                        "predicted_change_pct": pred.predicted_change_pct,
                        "model_votes": pred.model_votes,
                        "top_features": pred.top_features,
                        "days_ahead": days,
                    }),
                )
            except Exception:
                pass

        prediction = self._ai_model.predict_price_movement(symbol, days)
        return ToolResult(
            tool_call_id="",
            content=json.dumps({
                "symbol": str(symbol),
                "predicted_movement": prediction,
                "days_ahead": days,
                "direction": "UP" if prediction > 0 else "DOWN" if prediction < 0 else "FLAT",
            }),
        )

    async def _tool_get_trading_signal(self, args: Dict[str, Any]) -> ToolResult:
        if "symbol" not in args:
            return ToolResult(tool_call_id="", content="Missing required argument: symbol", is_error=True)
        symbol = Symbol(args["symbol"].upper())

        # Ensemble predictor gives a richer signal with explanations
        if self._ensemble_predictor is not None:
            try:
                pred = self._ensemble_predictor.predict(symbol)
                # Map direction to trading signal
                if pred.direction == "UP" and pred.confidence >= 0.55:
                    signal = "BUY"
                elif pred.direction == "DOWN" and pred.confidence >= 0.55:
                    signal = "SELL"
                else:
                    signal = "HOLD"
                return ToolResult(
                    tool_call_id="",
                    content=json.dumps({
                        "symbol": pred.symbol,
                        "signal": signal,
                        "confidence": pred.confidence,
                        "direction": pred.direction,
                        "model_votes": pred.model_votes,
                        "top_features": pred.top_features,
                    }),
                )
            except Exception:
                pass

        signal = self._ai_model.get_trading_signal(symbol)
        return ToolResult(
            tool_call_id="",
            content=json.dumps({
                "symbol": str(symbol),
                "signal": signal,
            }),
        )

    async def _tool_get_news_sentiment(self, args: Dict[str, Any]) -> ToolResult:
        if "symbol" not in args:
            return ToolResult(tool_call_id="", content="Missing required argument: symbol", is_error=True)
        symbol = Symbol(args["symbol"].upper())
        news = self._market_data.get_market_news(symbol)
        if news:
            sentiments = self._news.batch_analyze_sentiment(news[:5])
            avg_score = (
                sum(s.score for s in sentiments) / len(sentiments)
                if sentiments
                else Decimal("0")
            )
            return ToolResult(
                tool_call_id="",
                content=json.dumps({
                    "symbol": str(symbol),
                    "average_sentiment": float(avg_score),
                    "sentiment_label": (
                        "positive" if avg_score > 5 else "negative" if avg_score < -5 else "neutral"
                    ),
                    "headlines": news[:5],
                    "num_articles": len(news),
                }),
            )
        return ToolResult(
            tool_call_id="",
            content=json.dumps({
                "symbol": str(symbol),
                "average_sentiment": 0,
                "sentiment_label": "no data",
                "headlines": [],
                "num_articles": 0,
            }),
        )

    async def _tool_get_portfolio(self, user_id: str) -> ToolResult:
        portfolio = self._portfolios.get_by_user_id(user_id)
        if portfolio:
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
            return ToolResult(
                tool_call_id="",
                content=json.dumps({
                    "total_value": float(portfolio.total_value.amount),
                    "cash_balance": float(portfolio.cash_balance.amount),
                    "positions_value": float(portfolio.positions_value.amount),
                    "num_positions": len(portfolio.positions),
                    "positions": positions_data,
                }),
            )
        return ToolResult(
            tool_call_id="",
            content=json.dumps({
                "total_value": 0,
                "cash_balance": 0,
                "positions_value": 0,
                "num_positions": 0,
                "positions": [],
            }),
        )

    async def _tool_place_order(
        self, args: Dict[str, Any], user_id: str
    ) -> ToolResult:
        """
        Create a trade recommendation (NOT an execution).
        The actual order is placed only when the user confirms via the chat UI.
        """
        return ToolResult(
            tool_call_id="",
            content=json.dumps({
                "status": "pending_confirmation",
                "symbol": args["symbol"],
                "action": args["action"],
                "quantity": args["quantity"],
                "reasoning": args.get("reasoning", ""),
                "message": "Trade recommendation created. Waiting for user confirmation.",
            }),
        )

    # ── Phase 2: Technical Analysis ──────────────────────────────────────

    async def _tool_get_technical_analysis(self, args: Dict[str, Any]) -> ToolResult:
        if "symbol" not in args:
            return ToolResult(tool_call_id="", content="Missing required argument: symbol", is_error=True)
        if self._technical_analysis is None:
            return ToolResult(tool_call_id="", content="Technical analysis service not available", is_error=True)

        symbol = Symbol(args["symbol"].upper())
        from src.domain.services.technical_analysis import generate_signal_summary
        indicators = self._technical_analysis.compute_indicators(symbol)
        signals = generate_signal_summary(indicators)

        data = {
            "symbol": indicators.symbol,
            "current_price": indicators.current_price,
            "indicators": {
                "RSI_14": indicators.rsi_14,
                "MACD_line": indicators.macd_line,
                "MACD_signal": indicators.macd_signal,
                "MACD_histogram": indicators.macd_histogram,
                "SMA_20": indicators.sma_20,
                "SMA_50": indicators.sma_50,
                "SMA_200": indicators.sma_200,
                "EMA_12": indicators.ema_12,
                "EMA_26": indicators.ema_26,
                "BB_upper": indicators.bb_upper,
                "BB_middle": indicators.bb_middle,
                "BB_lower": indicators.bb_lower,
                "ATR_14": indicators.atr_14,
                "Stochastic_K": indicators.stoch_k,
                "Stochastic_D": indicators.stoch_d,
                "ADX": indicators.adx,
            },
            "signals": signals,
        }
        return ToolResult(tool_call_id="", content=json.dumps(data))

    # ── Phase 3: Order Management ────────────────────────────────────────

    async def _tool_get_orders(self, args: Dict[str, Any], user_id: str) -> ToolResult:
        if self._orders is None:
            return ToolResult(tool_call_id="", content="Order service not available", is_error=True)

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
        return ToolResult(
            tool_call_id="",
            content=json.dumps({"orders": orders_data, "total": len(orders)}),
        )

    async def _tool_cancel_order(self, args: Dict[str, Any], user_id: str) -> ToolResult:
        if "order_id" not in args:
            return ToolResult(tool_call_id="", content="Missing required argument: order_id", is_error=True)
        if self._orders is None:
            return ToolResult(tool_call_id="", content="Order service not available", is_error=True)

        order = self._orders.get_by_id(args["order_id"])
        if not order:
            return ToolResult(tool_call_id="", content="Order not found", is_error=True)
        if order.user_id != user_id:
            return ToolResult(tool_call_id="", content="Unauthorized", is_error=True)
        if order.status.value != "PENDING":
            return ToolResult(
                tool_call_id="",
                content=json.dumps({"error": f"Cannot cancel order with status {order.status.value}"}),
                is_error=True,
            )

        from src.domain.entities.trading import OrderStatus
        updated = self._orders.update_status(args["order_id"], OrderStatus.CANCELLED)
        if updated:
            return ToolResult(
                tool_call_id="",
                content=json.dumps({
                    "status": "cancelled",
                    "order_id": args["order_id"],
                    "symbol": str(updated.symbol),
                }),
            )
        return ToolResult(tool_call_id="", content="Failed to cancel order", is_error=True)

    async def _tool_get_position_details(self, args: Dict[str, Any], user_id: str) -> ToolResult:
        if "symbol" not in args:
            return ToolResult(tool_call_id="", content="Missing required argument: symbol", is_error=True)

        portfolio = self._portfolios.get_by_user_id(user_id)
        if not portfolio:
            return ToolResult(tool_call_id="", content=json.dumps({"error": "No portfolio found"}), is_error=True)

        symbol = Symbol(args["symbol"].upper())
        position = portfolio.get_position(symbol)
        if not position:
            return ToolResult(
                tool_call_id="",
                content=json.dumps({"error": f"No position found for {symbol}"}),
                is_error=True,
            )

        return ToolResult(
            tool_call_id="",
            content=json.dumps({
                "symbol": str(position.symbol),
                "quantity": position.quantity,
                "position_type": position.position_type.value,
                "avg_buy_price": float(position.average_buy_price.amount),
                "current_price": float(position.current_price.amount),
                "market_value": float(position.market_value.amount),
                "unrealized_pnl": float(position.unrealized_pnl_amount.amount),
                "total_cost": float(position.total_cost.amount),
            }),
        )

    # ── Phase 1: Stock Screening ─────────────────────────────────────────

    async def _tool_screen_stocks(self, args: Dict[str, Any]) -> ToolResult:
        if self._stock_screener is None:
            return ToolResult(tool_call_id="", content="Stock screener not available", is_error=True)

        results = self._stock_screener.screen(args)
        return ToolResult(tool_call_id="", content=json.dumps(results))

    # ── Phase 4: Market Status ───────────────────────────────────────────

    async def _tool_get_market_status(self) -> ToolResult:
        if self._exchange_registry is None:
            return ToolResult(tool_call_id="", content="Exchange registry not available", is_error=True)

        statuses = self._exchange_registry.get_all_statuses()
        return ToolResult(tool_call_id="", content=json.dumps(statuses))

    # ── Phase 6: Backtesting ─────────────────────────────────────────────

    async def _tool_run_backtest(self, args: Dict[str, Any]) -> ToolResult:
        if "strategy" not in args or "symbol" not in args:
            return ToolResult(tool_call_id="", content="Missing required arguments: strategy, symbol", is_error=True)
        if self._backtest_use_case is None:
            return ToolResult(tool_call_id="", content="Backtesting not available", is_error=True)

        result = self._backtest_use_case.run(args)
        return ToolResult(tool_call_id="", content=json.dumps(result))
