"""Stdio MCP client <-> HTTP MCP at http://127.0.0.1:8765/mcp.

Does not start a worker. Logs to stderr only. Hosts import_local_file locally
so HTTP MCP never exposes filesystem import.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from mcp.server.lowlevel.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool, ToolAnnotations

from app.mcp.config import app_base_url, mcp_http_url, mcp_token, redact
from app.mcp.domain import confirm_or_error, read_import_bytes, resolve_import_path
from app.mcp.server import INSTRUCTIONS

logger = logging.getLogger("mcp_cdp.stdio")

IMPORT_TOOL = Tool(
    name="import_local_file",
    description=(
        "Import a local CSV (or similar) from a path under MCP_IMPORT_ROOTS. "
        "stdio only. confirm must be true. Rejects path traversal and symlink escape."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "project_id": {"type": "string"},
            "path": {"type": "string", "description": "Path relative to MCP_IMPORT_ROOTS or absolute under a root"},
            "confirm": {"type": "boolean"},
            "name": {"type": "string", "description": "Optional dataset name"},
        },
        "required": ["project_id", "path", "confirm"],
        "additionalProperties": False,
    },
    annotations=ToolAnnotations(
        read_only_hint=False,
        destructive_hint=False,
        open_world_hint=False,
    ),
)


def _configure_logging() -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def _tool_result(payload: dict[str, Any], *, is_error: bool = False) -> CallToolResult:
    text = json.dumps(payload, default=str)
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        structured_content=payload,
        is_error=is_error or payload.get("ok") is False,
    )


async def _import_via_http(
    httpx_mod: Any,
    project_id: str,
    path: str,
    confirm: Any,
    name: str | None,
) -> dict[str, Any]:
    err = confirm_or_error(confirm, "import a local file")
    if err:
        return err
    try:
        resolved = resolve_import_path(path)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    if resolved.exists() and (resolved.is_dir() or resolved.is_symlink()):
        return {"ok": False, "error": "path is outside MCP_IMPORT_ROOTS"}
    if not resolved.is_file():
        return {"ok": False, "error": "path is not a file"}
    data, oversize = read_import_bytes(resolved)
    if oversize:
        return {"ok": False, "error": "file too large", "code": oversize}
    url = f"{app_base_url()}/api/projects/{project_id}/datasets/upload"
    timeout = httpx_mod.Timeout(30.0, read=120.0)
    filename = name or resolved.name
    async with httpx_mod.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            url,
            files={"file": (filename, data, "text/csv")},
            headers={"X-Requested-With": "mcp-cdp"},
        )
    if response.status_code >= 400:
        detail: Any
        try:
            detail = response.json()
        except ValueError:
            detail = {"error": redact(response.text)}
        if isinstance(detail, dict):
            return {
                "ok": False,
                "error": redact(str(detail.get("error") or detail.get("detail") or response.status_code)),
            }
        return {"ok": False, "error": redact(str(detail))}
    try:
        body = response.json()
    except ValueError:
        return {"ok": True, "status_code": response.status_code}
    if isinstance(body, dict):
        body.setdefault("ok", True)
        return body
    return {"ok": True, "result": body}


async def _run() -> None:
    try:
        import httpx2 as httpx
    except ImportError:  # pragma: no cover
        import httpx  # type: ignore

    from mcp import Client
    from mcp.client.streamable_http import streamable_http_client

    url = mcp_http_url()
    headers: dict[str, str] = {}
    token = mcp_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    timeout = httpx.Timeout(30.0, read=300.0)
    async with httpx.AsyncClient(headers=headers, timeout=timeout) as http_client:
        transport = streamable_http_client(url, http_client=http_client)
        async with Client(transport) as remote:

            async def on_list_tools(_ctx: Any, _params: Any) -> ListToolsResult:
                listed = await remote.list_tools()
                tools = [t for t in listed.tools if t.name != "import_local_file"]
                tools.append(IMPORT_TOOL)
                return ListToolsResult(tools=tools)

            async def on_call_tool(_ctx: Any, params: Any) -> CallToolResult:
                name = params.name
                arguments = dict(params.arguments or {})
                if name == "import_local_file":
                    try:
                        payload = await _import_via_http(
                            httpx,
                            str(arguments.get("project_id") or ""),
                            str(arguments.get("path") or ""),
                            arguments.get("confirm", False),
                            arguments.get("name"),
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.error("import_local_file failed: %s", redact(str(exc)))
                        payload = {"ok": False, "error": redact(str(exc))}
                    return _tool_result(payload, is_error=payload.get("ok") is False)
                return await remote.call_tool(name, arguments)

            server = Server(
                "mcp-cdp",
                instructions=INSTRUCTIONS,
                on_list_tools=on_list_tools,
                on_call_tool=on_call_tool,
            )
            logger.info("stdio bridge forwarding to %s", url)
            async with stdio_server() as (read_stream, write_stream):
                await server.run(
                    read_stream,
                    write_stream,
                    server.create_initialization_options(),
                )


def main() -> None:
    _configure_logging()
    try:
        import anyio

        anyio.run(_run)
    except KeyboardInterrupt:
        print("mcp-cdp stdio bridge: interrupted", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:  # noqa: BLE001
        print(f"mcp-cdp stdio bridge: {redact(str(exc))}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
