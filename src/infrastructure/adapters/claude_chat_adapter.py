"""
Claude Chat Adapter

Architectural Intent:
- Implements AIChatPort using the Anthropic Claude API
- Handles streaming responses, tool calling, and prompt caching
- Infrastructure concern — domain layer is unaware of Claude specifics
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
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

# Per-million-token pricing (USD). Update when model or pricing changes.
# Source: Anthropic pricing, Haiku 4.5 as of 2026-04.
_MODEL_PRICING_USD_PER_MTOK: Dict[str, Dict[str, float]] = {
    "claude-haiku-4-5-20251001": {
        "input": 1.00,
        "output": 5.00,
        "cache_read": 0.08,
        "cache_write": 1.25,
    },
    "claude-sonnet-4-6": {
        "input": 3.00,
        "output": 15.00,
        "cache_read": 0.30,
        "cache_write": 3.75,
    },
    "claude-opus-4-7": {
        "input": 15.00,
        "output": 75.00,
        "cache_read": 1.50,
        "cache_write": 18.75,
    },
}


def _compute_cost_usd(model: str, usage: Any) -> float:
    """Compute USD cost for a single Claude call from Anthropic usage object."""
    pricing = _MODEL_PRICING_USD_PER_MTOK.get(model)
    if not pricing:
        return 0.0
    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    return (
        input_tokens * pricing["input"]
        + output_tokens * pricing["output"]
        + cache_read * pricing["cache_read"]
        + cache_write * pricing["cache_write"]
    ) / 1_000_000


class ClaudeChatAdapter(AIChatPort):
    """
    Adapter for the Anthropic Claude API.

    Implements streaming chat generation with tool calling support.
    Uses prompt caching on system prompts for cost reduction.
    """

    def __init__(self, api_key: str = None, model: str = None):
        self._api_key = api_key or settings.ANTHROPIC_API_KEY
        self._model = model or settings.CHAT_MODEL
        # Explicit timeout prevents hung streams from leaking workers.
        # The Anthropic SDK accepts seconds as a float; connect guard is tighter
        # than the overall cap so a dead server fails fast rather than slow.
        self._client = anthropic.AsyncAnthropic(
            api_key=self._api_key,
            timeout=anthropic.Timeout(60.0, connect=5.0),
            max_retries=0,  # we do our own retry with visibility — see generate_response
        )

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

        prompt_hash = hashlib.sha256(
            json.dumps(
                {"system": system_with_context, "messages": api_messages, "tools": api_tools},
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()[:16]

        max_retries = 3
        for attempt in range(max_retries):
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

                started_at = time.perf_counter()
                async with self._client.messages.stream(**kwargs) as stream:
                    async for event in stream:
                        if event.type == "content_block_start":
                            if hasattr(event.content_block, "type"):
                                if event.content_block.type == "tool_use":
                                    pass
                        elif event.type == "content_block_delta":
                            if hasattr(event.delta, "text"):
                                yield ChatStreamEvent(
                                    type="text_delta",
                                    content=event.delta.text,
                                )
                            elif hasattr(event.delta, "partial_json"):
                                pass
                        elif event.type == "message_stop":
                            pass

                    final_message = await stream.get_final_message()
                    latency_ms = int((time.perf_counter() - started_at) * 1000)
                    usage = getattr(final_message, "usage", None)
                    if usage is not None:
                        try:
                            from src.infrastructure.observability import get_correlation_id
                            correlation_id = get_correlation_id()
                        except Exception:
                            correlation_id = ""
                        logger.info(
                            "claude_api_call",
                            extra={
                                "event": "claude_api_call",
                                "model": self._model,
                                "prompt_hash": prompt_hash,
                                "input_tokens": getattr(usage, "input_tokens", 0),
                                "output_tokens": getattr(usage, "output_tokens", 0),
                                "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0),
                                "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0),
                                "latency_ms": latency_ms,
                                "cost_usd": round(_compute_cost_usd(self._model, usage), 6),
                                "user_id": user_context.user_id,
                                "correlation_id": correlation_id,
                                "stop_reason": getattr(final_message, "stop_reason", None),
                                "attempt": attempt + 1,
                            },
                        )

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
                return  # Success — exit retry loop

            except anthropic.RateLimitError as e:
                if attempt < max_retries - 1:
                    import asyncio
                    delay = 2 ** (attempt + 1)
                    logger.warning(f"Rate limited by Claude API, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                    continue
                logger.error(f"Claude API rate limit exceeded after {max_retries} attempts: {e}")
                yield ChatStreamEvent(type="error", content="AI service is temporarily overloaded. Please try again shortly.")

            except anthropic.InternalServerError as e:
                if attempt < max_retries - 1:
                    import asyncio
                    delay = 2 ** (attempt + 1)
                    logger.warning(f"Claude API server error, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                    continue
                logger.error(f"Claude API server error after {max_retries} attempts: {e}")
                yield ChatStreamEvent(type="error", content=f"AI service error: {str(e)}")

            except anthropic.APIError as e:
                logger.error(f"Claude API error: {e}")
                yield ChatStreamEvent(type="error", content=f"AI service error: {str(e)}")
                return

            except Exception as e:
                logger.error(f"Unexpected error in Claude adapter: {e}")
                yield ChatStreamEvent(type="error", content="An unexpected error occurred while generating a response.")
                return
