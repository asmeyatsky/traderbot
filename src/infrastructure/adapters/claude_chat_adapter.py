"""
Claude Chat Adapter

Architectural Intent:
- Implements AIChatPort using the Anthropic Claude API
- Handles streaming responses, tool calling, and prompt caching
- Infrastructure concern — domain layer is unaware of Claude specifics
"""
from __future__ import annotations

import json
import logging
from typing import AsyncIterator, Dict, List, Any

import anthropic

from src.domain.ports.ai_chat_port import (
    AIChatPort,
    ChatMessage,
    ChatStreamEvent,
    ToolCall,
    ToolDefinition,
    ToolResult,
    UserContext,
)
from src.infrastructure.config.settings import settings

logger = logging.getLogger(__name__)


class ClaudeChatAdapter(AIChatPort):
    """
    Adapter for the Anthropic Claude API.

    Implements streaming chat generation with tool calling support.
    Uses prompt caching on system prompts for cost reduction.
    """

    def __init__(self, api_key: str = None, model: str = None):
        self._api_key = api_key or settings.ANTHROPIC_API_KEY
        self._model = model or settings.CHAT_MODEL
        self._client = anthropic.AsyncAnthropic(api_key=self._api_key)

    def _build_tools(self, tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """Convert domain ToolDefinitions to Claude API tool format."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in tools
        ]

    def _build_messages(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """Convert domain ChatMessages to Claude API message format."""
        api_messages = []
        for msg in messages:
            if msg.role == "system":
                continue
            api_messages.append({"role": msg.role, "content": msg.content})
        return api_messages

    async def generate_response(
        self,
        messages: List[ChatMessage],
        tools: List[ToolDefinition],
        user_context: UserContext,
        system_prompt: str,
    ) -> AsyncIterator[ChatStreamEvent]:
        """
        Generate a streaming response from Claude with tool calling.

        Yields ChatStreamEvent instances as text deltas and tool calls arrive.
        """
        api_messages = self._build_messages(messages)
        api_tools = self._build_tools(tools) if tools else []

        system_with_context = (
            f"{system_prompt}\n\n"
            f"User context: risk_tolerance={user_context.risk_tolerance}, "
            f"investment_goal={user_context.investment_goal}"
        )
        if user_context.portfolio_summary:
            system_with_context += f"\nPortfolio: {user_context.portfolio_summary}"

        try:
            kwargs = {
                "model": self._model,
                "max_tokens": 4096,
                "system": [
                    {
                        "type": "text",
                        "text": system_with_context,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                "messages": api_messages,
            }
            if api_tools:
                kwargs["tools"] = api_tools

            async with self._client.messages.stream(**kwargs) as stream:
                async for event in stream:
                    if event.type == "content_block_start":
                        if hasattr(event.content_block, "type"):
                            if event.content_block.type == "tool_use":
                                # Tool call starting — we'll accumulate input
                                pass
                    elif event.type == "content_block_delta":
                        if hasattr(event.delta, "text"):
                            yield ChatStreamEvent(
                                type="text_delta",
                                content=event.delta.text,
                            )
                        elif hasattr(event.delta, "partial_json"):
                            # Tool input being streamed — skip, we get full result at end
                            pass
                    elif event.type == "message_stop":
                        pass

                # After stream completes, check for tool use in final message
                final_message = await stream.get_final_message()
                for block in final_message.content:
                    if block.type == "tool_use":
                        yield ChatStreamEvent(
                            type="tool_call",
                            tool_call=ToolCall(
                                id=block.id,
                                name=block.name,
                                arguments=block.input if isinstance(block.input, dict) else json.loads(block.input),
                            ),
                        )

            yield ChatStreamEvent(type="done")

        except anthropic.APIError as e:
            logger.error(f"Claude API error: {e}")
            yield ChatStreamEvent(
                type="error",
                content=f"AI service error: {str(e)}",
            )
        except Exception as e:
            logger.error(f"Unexpected error in Claude adapter: {e}")
            yield ChatStreamEvent(
                type="error",
                content="An unexpected error occurred while generating a response.",
            )
