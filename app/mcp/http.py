"""Mount the MCP Streamable HTTP app on FastAPI.

Platform `create_app()` should:

1. Create `lifespan_hooks: list = []`
2. Build a host lifespan that enters every hook:
       async with AsyncExitStack() as stack:
           for hook in lifespan_hooks:
               await stack.enter_async_context(hook)
           yield
3. Construct FastAPI(lifespan=...), include routers, then call
   `mount_mcp(app, lifespan_hooks)`.

`streamable_http_app()` must run before `session_manager` is used.
"""

from __future__ import annotations

import hmac
from collections.abc import Awaitable, Callable
from typing import Any

from app.mcp.config import mcp_token
from app.mcp.server import mcp

Send = Callable[[dict[str, Any]], Awaitable[None]]


class RewriteMCPSlash:
    """POST /mcp (no trailing slash) would 405 under FastAPI slash handling; map it to /mcp/."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Send) -> None:
        if scope.get("type") == "http" and scope.get("path") == "/mcp":
            scope = dict(scope)
            scope["path"] = "/mcp/"
            if isinstance(scope.get("raw_path"), (bytes, bytearray)):
                scope["raw_path"] = b"/mcp/"
        await self.app(scope, receive, send)


class BearerAuthASGI:
    """Require `Authorization: Bearer $MCP_TOKEN`. Fail closed if the token is unset."""

    def __init__(self, app: Any, token: str | None) -> None:
        self.app = app
        self._token = token or ""
        self._expected = f"Bearer {self._token}".encode() if self._token else b""

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        if (scope.get("method") or "").upper() == "OPTIONS":
            await self.app(scope, receive, send)
            return
        if not self._token:
            await _send_401(send)
            return
        headers = {k.decode("latin1").lower(): v for k, v in (scope.get("headers") or [])}
        provided = headers.get("authorization", b"")
        if not _bearer_ok(provided, self._expected):
            await _send_401(send)
            return
        await self.app(scope, receive, send)


async def _send_401(send: Send) -> None:
    body = b'{"error":"unauthorized"}'
    await send(
        {
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                (b"www-authenticate", b"Bearer"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


def _bearer_ok(provided: bytes, expected: bytes) -> bool:
    if len(provided) != len(expected):
        return False
    return hmac.compare_digest(provided, expected)


def mount_mcp(app: Any, lifespan_hooks: list) -> None:
    """Mount MCP at `/mcp` and append `session_manager.run()` to lifespan_hooks.

    Call this after the FastAPI app exists. Do not start a worker here.
    """
    asgi = mcp.streamable_http_app(streamable_http_path="/")
    asgi = BearerAuthASGI(asgi, mcp_token())
    app.mount("/mcp", asgi)
    app.add_middleware(RewriteMCPSlash)
    lifespan_hooks.append(mcp.session_manager.run())
