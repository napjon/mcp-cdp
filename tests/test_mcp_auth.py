"""MCP HTTP fails closed without a valid bearer token."""

from __future__ import annotations


def test_mcp_http_unauthorized_without_token(client) -> None:
    for path in ("/mcp", "/mcp/"):
        response = client.post(path, json={"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        assert response.status_code == 401, path
        assert "sk-" not in response.text
        body = response.json()
        assert body.get("error") == "unauthorized"


def test_mcp_http_unauthorized_wrong_token(client, monkeypatch) -> None:
    from app.settings import reset_settings

    monkeypatch.setenv("MCP_TOKEN", "expected-token")
    reset_settings()
    from app.main import create_app

    application = create_app(worker_enabled=False)
    from fastapi.testclient import TestClient

    with TestClient(application, base_url="http://127.0.0.1:8765") as c:
        denied = c.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "initialize"},
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert denied.status_code == 401
        assert "expected-token" not in denied.text
