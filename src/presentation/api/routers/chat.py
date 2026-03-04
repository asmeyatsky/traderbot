"""
Chat API Router

Handles all chat-related endpoints including conversation management
and SSE-streamed AI responses.

Architectural Intent:
- Presentation layer only — delegates to ChatUseCase
- SSE streaming for real-time AI response delivery
- All endpoints require JWT authentication
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.infrastructure.security import get_current_user
from src.presentation.api.dependencies import get_chat_use_case
from src.application.use_cases.chat import ChatUseCase

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


# ============================================================================
# DTOs
# ============================================================================

class CreateConversationRequest(BaseModel):
    title: Optional[str] = Field(default="New conversation", max_length=500)


class SendMessageRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=10000)


class TradeActionResponse(BaseModel):
    symbol: str
    action: str
    quantity: int
    reasoning: str
    confidence: float
    executed: bool


class MessageResponse(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    created_at: datetime
    trade_actions: List[TradeActionResponse] = []


class ConversationResponse(BaseModel):
    id: str
    user_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0
    messages: List[MessageResponse] = []


class ConversationListResponse(BaseModel):
    conversations: List[ConversationResponse]
    total: int


# ============================================================================
# Helper functions
# ============================================================================

def _conversation_to_response(
    conv, include_messages: bool = False
) -> ConversationResponse:
    messages = []
    if include_messages:
        messages = [
            MessageResponse(
                id=msg.id,
                conversation_id=msg.conversation_id,
                role=msg.role.value,
                content=msg.content,
                created_at=msg.created_at,
                trade_actions=[
                    TradeActionResponse(
                        symbol=ta.symbol,
                        action=ta.action.value,
                        quantity=ta.quantity,
                        reasoning=ta.reasoning,
                        confidence=ta.confidence,
                        executed=ta.executed,
                    )
                    for ta in msg.trade_actions
                ],
            )
            for msg in conv.messages
        ]

    return ConversationResponse(
        id=conv.id,
        user_id=conv.user_id,
        title=conv.title,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        message_count=conv.message_count,
        messages=messages,
    )


# ============================================================================
# Endpoints
# ============================================================================

@router.post(
    "/conversations",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new conversation",
)
def create_conversation(
    request: CreateConversationRequest,
    current_user=Depends(get_current_user),
    chat_use_case: ChatUseCase = Depends(get_chat_use_case),
):
    conv = chat_use_case.create_conversation(
        user_id=current_user,
        title=request.title,
    )
    return _conversation_to_response(conv)


@router.get(
    "/conversations",
    response_model=ConversationListResponse,
    summary="List user's conversations",
)
def list_conversations(
    limit: int = 50,
    offset: int = 0,
    current_user=Depends(get_current_user),
    chat_use_case: ChatUseCase = Depends(get_chat_use_case),
):
    conversations = chat_use_case.get_user_conversations(
        user_id=current_user, limit=limit, offset=offset
    )
    return ConversationListResponse(
        conversations=[_conversation_to_response(c) for c in conversations],
        total=len(conversations),
    )


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationResponse,
    summary="Get a conversation with messages",
)
def get_conversation(
    conversation_id: str,
    current_user=Depends(get_current_user),
    chat_use_case: ChatUseCase = Depends(get_chat_use_case),
):
    conv = chat_use_case.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )
    if conv.user_id != current_user:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this conversation",
        )
    return _conversation_to_response(conv, include_messages=True)


@router.post(
    "/conversations/{conversation_id}/messages",
    summary="Send a message and receive SSE stream",
)
async def send_message(
    conversation_id: str,
    request: SendMessageRequest,
    current_user=Depends(get_current_user),
    chat_use_case: ChatUseCase = Depends(get_chat_use_case),
):
    """
    Send a message to a conversation and receive the AI response as an SSE stream.

    Event types:
    - text_delta: partial text content
    - tool_call: AI is calling a backend tool
    - tool_result: result of a tool execution
    - done: stream complete (includes message_id and trade_actions)
    - error: an error occurred
    """
    # Verify conversation exists and belongs to user
    conv = chat_use_case.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )
    if conv.user_id != current_user:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this conversation",
        )

    async def event_generator():
        async for event in chat_use_case.send_message(
            conversation_id=conversation_id,
            user_id=current_user,
            content=request.content,
        ):
            data = {"type": event.type}

            if event.content:
                data["content"] = event.content

            if event.tool_call:
                data["tool_call"] = {
                    "id": event.tool_call.id,
                    "name": event.tool_call.name,
                    "arguments": event.tool_call.arguments,
                }

            if event.tool_result:
                data["tool_result"] = {
                    "tool_call_id": event.tool_result.tool_call_id,
                    "content": event.tool_result.content,
                    "is_error": event.tool_result.is_error,
                }

            if event.metadata:
                data["metadata"] = event.metadata

            yield f"data: {json.dumps(data)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.delete(
    "/conversations/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a conversation",
)
def delete_conversation(
    conversation_id: str,
    current_user=Depends(get_current_user),
    chat_use_case: ChatUseCase = Depends(get_chat_use_case),
):
    conv = chat_use_case.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )
    if conv.user_id != current_user:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this conversation",
        )
    chat_use_case.delete_conversation(conversation_id)
