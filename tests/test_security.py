from __future__ import annotations

POST_HEADERS = {
    "Origin": "http://127.0.0.1:5173",
    "X-Requested-With": "mcp-cdp",
}


def test_health_ok(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_status_never_includes_keys(data_dir, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-should-not-leak")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret-should-not-leak")
    monkeypatch.setenv("MCP_TOKEN", "mcp-secret-token")
    from fastapi.testclient import TestClient

    from app.main import create_app
    from app.settings import reset_settings

    reset_settings()
    application = create_app(worker_enabled=False)
    with TestClient(application, base_url="http://127.0.0.1:8765") as c:
        response = c.get("/api/status")
        assert response.status_code == 200
        body = response.json()
        assert body["llm_configured"] is True
        assert body["provider"] == "openai"
        assert body["mcp_token_set"] is True
        raw = response.text
        assert "sk-secret-should-not-leak" not in raw
        assert "sk-ant-secret-should-not-leak" not in raw
        assert "mcp-secret-token" not in raw
        assert "sk-" not in raw


def test_origin_reject(client):
    response = client.post(
        "/api/projects",
        json={"name": "Nope"},
        headers={"Origin": "http://evil.example", "X-Requested-With": "mcp-cdp"},
    )
    assert response.status_code == 403
    assert response.json()["error"] == "invalid origin"
    assert "sk-" not in response.text


def test_origin_allow_local_ui(client):
    response = client.post(
        "/api/projects",
        json={"name": "Workspace two"},
        headers=POST_HEADERS,
    )
    assert response.status_code == 200, response.text
    assert response.json()["name"] == "Workspace two"


def test_loopback_post_without_origin_allowed(client):
    response = client.post("/api/projects", json={"name": "cli"})
    assert response.status_code == 200, response.text


def test_default_project_exists(client):
    response = client.get("/api/projects")
    names = [p["name"] for p in response.json()]
    assert "Local workspace" in names


def test_settings_repr_hides_secrets(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-hidden")
    monkeypatch.setenv("MCP_TOKEN", "tok-hidden")
    from app.settings import get_settings, reset_settings

    reset_settings()
    text = repr(get_settings())
    assert "sk-hidden" not in text
    assert "tok-hidden" not in text
    assert "openai_set=True" in text
