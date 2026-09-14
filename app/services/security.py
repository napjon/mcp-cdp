"""Loopback bind, CSRF origin/host checks, secret redaction, SSRF host allowlist."""

from __future__ import annotations

import re
from urllib.parse import urlparse

from starlette.requests import Request

from app.models import AppError
from app.settings import Settings

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
SHEET_EXACT_HOSTS = frozenset({"docs.google.com", "spreadsheets.google.com"})
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

_KEY_RE = re.compile(
    r"(?i)(sk-[a-z0-9_-]{8,}|sk-ant-[a-z0-9_-]{8,}|Bearer\s+\S+)"
)


def is_loopback_host(host: str | None) -> bool:
    if not host:
        return False
    return host.strip("[]").lower() in LOOPBACK_HOSTS


def hostname_of_host_header(host_header: str) -> str:
    value = (host_header or "").strip()
    if not value:
        return ""
    if value.startswith("["):
        end = value.find("]")
        if end != -1:
            return value[1:end].lower()
    if value.count(":") == 1:
        return value.split(":", 1)[0].lower()
    return value.lower()


def host_allowed(host_header: str) -> bool:
    return is_loopback_host(hostname_of_host_header(host_header))


def origin_allowed(origin: str, settings: Settings) -> bool:
    try:
        parsed = urlparse(origin)
    except ValueError:
        return False
    if parsed.scheme not in {"http", "https"}:
        return False
    host = (parsed.hostname or "").lower()
    if not is_loopback_host(host):
        return False
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return port in {5173, settings.port}


def is_allowed_sheet_host(host: str | None) -> bool:
    if not host:
        return False
    h = host.lower().rstrip(".")
    if h in SHEET_EXACT_HOSTS:
        return True
    return h.endswith(".googleusercontent.com") and h.count(".") >= 2


def check_mutating_request(request: Request, settings: Settings) -> None:
    if request.method in SAFE_METHODS:
        return
    path = request.url.path
    if path == "/mcp" or path.startswith("/mcp/"):
        return
    host = request.headers.get("host", "")
    if not host_allowed(host):
        raise AppError("invalid host", status_code=403)
    origin = request.headers.get("origin")
    requested_with = request.headers.get("x-requested-with", "")
    if origin:
        if not origin_allowed(origin, settings):
            raise AppError("invalid origin", status_code=403)
        return
    if requested_with != "mcp-cdp":
        raise AppError("invalid origin", status_code=403)


def redact_secrets(message: str, settings: Settings | None = None) -> str:
    text = _KEY_RE.sub("[redacted]", message or "")
    if settings:
        for secret in (settings.mcp_token, settings.openai_api_key, settings.anthropic_api_key):
            if secret:
                text = text.replace(secret, "[redacted]")
    return text[:2000]
