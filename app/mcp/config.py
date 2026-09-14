"""Env/settings accessors for MCP and chat. Never return API keys to callers."""

from __future__ import annotations

import os
import re
from pathlib import Path

_KEY_RE = re.compile(
    r"(?:sk-[A-Za-z0-9_\-]{8,}|sk-ant-[A-Za-z0-9_\-]{8,}|sk-proj-[A-Za-z0-9_\-]{8,})"
)


def _settings():
    try:
        from app.settings import get_settings

        return get_settings()
    except ImportError:
        return None


def _attr(name: str, default=None):
    s = _settings()
    if s is None:
        return default
    return getattr(s, name, default)


def mcp_token() -> str | None:
    value = _attr("mcp_token")
    if value:
        return str(value)
    env = os.environ.get("MCP_TOKEN")
    return env or None


def data_dir() -> Path:
    value = _attr("data_dir")
    if value:
        return Path(value)
    raw = os.environ.get("MCP_CDP_DATA_DIR") or os.environ.get("DATA_DIR") or "data"
    return Path(raw)


def sqlite_path() -> Path:
    value = _attr("db_path")
    if value:
        return Path(value)
    value = _attr("sqlite_path")
    if value:
        return Path(value)
    return data_dir() / "mcp_cdp.sqlite"


def app_base_url() -> str:
    host = _attr("host") or "127.0.0.1"
    port = _attr("port") or 8765
    return f"http://{host}:{port}"


def mcp_http_url() -> str:
    value = _attr("mcp_http_url")
    if value:
        return str(value)
    return os.environ.get("MCP_HTTP_URL", "http://127.0.0.1:8765/mcp")


def max_upload_bytes() -> int:
    value = _attr("max_upload_bytes")
    try:
        if value is not None:
            return int(value)
    except (TypeError, ValueError):
        pass
    raw = os.environ.get("MAX_UPLOAD_BYTES")
    if raw:
        try:
            return int(raw)
        except ValueError:
            pass
    return 50_000_000


def import_roots() -> list[Path]:
    value = _attr("mcp_import_roots")
    parts: list[str] = []
    if isinstance(value, (list, tuple)):
        parts = [str(p) for p in value if p]
    elif value:
        parts = _split_roots(str(value))
    if not parts:
        parts = _split_roots(os.environ.get("MCP_IMPORT_ROOTS", ""))
    return [Path(p).expanduser() for p in parts if p]


def _split_roots(raw: str) -> list[str]:
    raw = (raw or "").strip()
    if not raw:
        return []
    if os.pathsep in raw:
        return [p.strip() for p in raw.split(os.pathsep) if p.strip()]
    return [p.strip() for p in raw.split(",") if p.strip()]


def openai_api_key() -> str | None:
    value = _attr("openai_api_key")
    if value:
        return str(value)
    return os.environ.get("OPENAI_API_KEY") or None


def anthropic_api_key() -> str | None:
    value = _attr("anthropic_api_key")
    if value:
        return str(value)
    return os.environ.get("ANTHROPIC_API_KEY") or None


def llm_provider_name() -> str | None:
    value = _attr("llm_provider")
    if value:
        return str(value).strip().lower()
    env = (os.environ.get("LLM_PROVIDER") or "").strip().lower()
    if env:
        return env
    if openai_api_key():
        return "openai"
    if anthropic_api_key():
        return "anthropic"
    return None


def openai_model() -> str:
    value = _attr("openai_model")
    if value:
        return str(value)
    return os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"


def anthropic_model() -> str:
    value = _attr("anthropic_model")
    if value:
        return str(value)
    return os.environ.get("ANTHROPIC_MODEL") or "claude-sonnet-4-20250514"


def known_secrets() -> list[str]:
    secrets = [
        openai_api_key(),
        anthropic_api_key(),
        mcp_token(),
    ]
    return [s for s in secrets if s]


def redact(text: str | None) -> str:
    if not text:
        return ""
    out = str(text)
    try:
        from app.services.security import redact_secrets

        out = redact_secrets(out, _settings())
    except ImportError:
        pass
    for secret in known_secrets():
        if secret:
            out = out.replace(secret, "[redacted]")
    return _KEY_RE.sub("[redacted]", out)
