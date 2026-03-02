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
from typing import Any, AsyncIterator, Dict, List, Optional

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
    PortfolioRepositoryPort,
    UserRepositoryPort,
)
from src.domain.value_objects import Symbol

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
    ):
        self._ai = ai_chat_port
        self._conversations = conversation_repository
        self._market_data = market_data_service
        self._ai_model = ai_model_service
        self._news = news_analysis_service
        self._portfolios = portfolio_repository
        self._users = user_repository

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
        symbol = Symbol(args["symbol"].upper())
        days = args.get("days", 1)
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
        symbol = Symbol(args["symbol"].upper())
        signal = self._ai_model.get_trading_signal(symbol)
        return ToolResult(
            tool_call_id="",
            content=json.dumps({
                "symbol": str(symbol),
                "signal": signal,
            }),
        )

    async def _tool_get_news_sentiment(self, args: Dict[str, Any]) -> ToolResult:
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
