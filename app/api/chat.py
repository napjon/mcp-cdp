"""Chat REST + SSE. Include this router from platform `create_app()`."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.mcp.config import redact
from app.services.chat import (
    ChatClientIdConflict,
    create_conversation,
    get_conversation,
    list_messages,
    resolve_user_message,
    stream_user_message,
)

router = APIRouter(prefix="/api", tags=["chat"])


class CreateConversationIn(BaseModel):
    project_id: str | None = None


class PostMessageIn(BaseModel):
    content: str = Field(min_length=1)
    client_id: str = Field(min_length=1)


@router.post("/conversations")
def post_conversation(body: CreateConversationIn) -> dict:
    try:
        return create_conversation(body.project_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=redact(str(exc))) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=redact(str(exc))) from exc


@router.get("/conversations/{conversation_id}/messages")
def get_messages(conversation_id: str) -> dict:
    conv = get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="conversation not found")
    return {"messages": list_messages(conversation_id)}


@router.post("/conversations/{conversation_id}/messages")
async def post_message(
    conversation_id: str,
    body: PostMessageIn,
    request: Request,
) -> StreamingResponse:
    conv = get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="conversation not found")
    try:
        resolve_user_message(conversation_id, body.content, body.client_id)
    except ChatClientIdConflict as exc:
        raise HTTPException(status_code=409, detail=redact(str(exc))) from exc

    async def events() -> AsyncIterator[str]:
        async for event in stream_user_message(
            conversation_id,
            body.content,
            body.client_id,
            disconnected=request.is_disconnected,
        ):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
