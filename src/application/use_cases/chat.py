"""
Chat Use Case

Architectural Intent:
- Orchestrates the AI chat flow: receives user messages, builds context,
  calls Claude with tools, processes tool results, and persists messages.
- Tools are owned by MCP servers per bounded context (market_data / portfolio
  / research). This use case only talks to the `ToolRegistryPort` — it does
  not know what the tools are or where they come from.
- The AI never auto-executes trades; it recommends via `place_order` which
  returns a pending recommendation; user confirmation is a separate endpoint.
- Single responsibility — orchestrate streaming + tool loop + persistence.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import AsyncIterator, List, Optional

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
    ToolResult,
    UserContext,
)
from src.domain.ports.conversation_repository_port import ConversationRepositoryPort
from src.domain.ports import UserRepositoryPort
from src.domain.ports.tool_registry import ToolRegistryPort

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


class ChatUseCase:
    """Handle a chat message and stream an AI response.

    Collaborators:
    - `AIChatPort` — Claude streaming + tool calling
    - `ConversationRepositoryPort` — persist messages and conversations
    - `UserRepositoryPort` — read user for context
    - `ToolRegistryPort` — aggregated MCP tools across bounded contexts
    """

    def __init__(
        self,
        ai_chat_port: AIChatPort,
        conversation_repository: ConversationRepositoryPort,
        user_repository: UserRepositoryPort,
        tool_registry: ToolRegistryPort,
    ) -> None:
        self._ai = ai_chat_port
        self._conversations = conversation_repository
        self._users = user_repository
        self._tools = tool_registry

    # ------------------------------------------------------------------
    # Conversation CRUD
    # ------------------------------------------------------------------

    def create_conversation(
        self, user_id: str, title: str = "New conversation"
    ) -> Conversation:
        conversation = Conversation(
            id=str(uuid.uuid4()),
            user_id=user_id,
            title=title,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        return self._conversations.save(conversation)

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        return self._conversations.get_by_id(conversation_id)

    def get_user_conversations(
        self, user_id: str, limit: int = 50, offset: int = 0
    ) -> List[Conversation]:
        return self._conversations.get_by_user_id(user_id, limit, offset)

    def delete_conversation(self, conversation_id: str) -> bool:
        return self._conversations.delete(conversation_id)

    # ------------------------------------------------------------------
    # Streaming chat
    # ------------------------------------------------------------------

    async def send_message(
        self, conversation_id: str, user_id: str, content: str
    ) -> AsyncIterator[ChatStreamEvent]:
        """Process a user message and stream the AI response."""

        conversation = self._conversations.get_by_id(conversation_id)
        if not conversation:
            yield ChatStreamEvent(type="error", content="Conversation not found")
            return
        if conversation.user_id != user_id:
            yield ChatStreamEvent(type="error", content="Unauthorized")
            return

        # Persist the user message immediately so it appears in history even
        # if the AI call fails later.
        user_message = Message(
            id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            role=MessageRole.USER,
            content=content,
            created_at=datetime.utcnow(),
        )
        self._conversations.add_message(user_message)

        if conversation.message_count == 0:
            title = content[:80] + ("..." if len(content) > 80 else "")
            self._conversations.save(conversation.update_title(title))

        user = self._users.get_by_id(user_id)
        user_context = UserContext(
            user_id=user_id,
            risk_tolerance=user.risk_tolerance.value if user else "moderate",
            investment_goal=user.investment_goal.value if user else "balanced_growth",
        )

        # Rebuild the conversation so we include the message we just persisted.
        conversation = self._conversations.get_by_id(conversation_id)
        chat_messages = [
            ChatMessage(role=msg.role.value, content=msg.content)
            for msg in conversation.messages
        ]

        tools = self._tools.tool_definitions()

        full_response = ""
        trade_actions: List[TradeAction] = []
        tool_call_log: List[dict] = []

        async for event in self._ai.generate_response(
            messages=chat_messages,
            tools=tools,
            user_context=user_context,
            system_prompt=SYSTEM_PROMPT,
        ):
            if event.type == "text_delta":
                full_response += event.content
                yield event
                continue

            if event.type == "tool_call":
                tool_call = event.tool_call
                tool_call_log.append(
                    {"name": tool_call.name, "arguments": tool_call.arguments}
                )

                outcome = await self._tools.call_tool(
                    tool_name=tool_call.name,
                    args=tool_call.arguments,
                    actor_user_id=user_id,
                )

                # Build a ToolResult for the AI's next turn. Keep payloads as
                # JSON strings so the Claude SDK can embed them verbatim.
                tool_result = ToolResult(
                    tool_call_id=tool_call.id,
                    content=(
                        outcome.error_message
                        if outcome.is_error
                        else json.dumps(dict(outcome.payload))
                    ),
                    is_error=outcome.is_error,
                )
                yield ChatStreamEvent(
                    type="tool_result",
                    tool_result=tool_result,
                    metadata={"tool_name": tool_call.name},
                )

                if tool_call.name == "place_order" and not outcome.is_error:
                    try:
                        args = tool_call.arguments
                        trade_actions.append(
                            TradeAction(
                                symbol=args["symbol"],
                                action=TradeActionType(args["action"]),
                                quantity=args["quantity"],
                                reasoning=args.get("reasoning", ""),
                                confidence=0.0,
                            )
                        )
                    except (KeyError, ValueError) as exc:
                        logger.warning("failed to parse TradeAction: %s", exc)

                # Continue the conversation with the tool result inlined so
                # the AI can reason over it.
                continuation_messages = list(chat_messages)
                continuation_messages.append(
                    ChatMessage(role="assistant", content=full_response)
                )
                continuation_messages.append(
                    ChatMessage(
                        role="user",
                        content=f"[Tool result for {tool_call.name}]: {tool_result.content}",
                    )
                )

                async for cont_event in self._ai.generate_response(
                    messages=continuation_messages,
                    tools=tools,
                    user_context=user_context,
                    system_prompt=SYSTEM_PROMPT,
                ):
                    if cont_event.type == "text_delta":
                        full_response += cont_event.content
                        yield cont_event
                    elif cont_event.type == "done":
                        break

                continue

            if event.type == "error":
                yield event
                return

            # Ignore `done` here — the outer loop handles it after persistence.

        assistant_message = Message(
            id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            role=MessageRole.ASSISTANT,
            content=full_response,
            created_at=datetime.utcnow(),
            trade_actions=trade_actions,
            metadata={"tool_calls": tool_call_log} if tool_call_log else {},
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
