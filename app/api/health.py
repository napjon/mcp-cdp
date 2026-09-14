"""Health and status endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from app.settings import get_settings

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
def health() -> dict:
    return {"ok": True}


@router.get("/status")
def status() -> dict:
    settings = get_settings()
    return {
        "llm_configured": bool(settings.openai_api_key or settings.anthropic_api_key),
        "provider": settings.llm_provider,
        "mcp_token_set": bool(settings.mcp_token),
    }
