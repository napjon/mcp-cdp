"""API keys must never appear in JSON/SSE errors. No-key chat stays honest."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.chat import router
from app.mcp.config import sqlite_path
from app.services.chat import NO_KEY_MESSAGE

SCHEMA = """
CREATE TABLE projects (id TEXT PRIMARY KEY, name TEXT, created_at TEXT);
CREATE TABLE conversations (id TEXT PRIMARY KEY, project_id TEXT, created_at TEXT);
CREATE TABLE messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT,
    role TEXT,
    content TEXT,
    client_id TEXT UNIQUE,
    status TEXT,
    provider TEXT,
    model TEXT,
    usage_json TEXT,
    created_at TEXT
);
"""

SECRET = "sk-secret-test-key-12345ABCDEFG"


def _parse_sse(body: str) -> list[dict]:
    events = []
    for block in body.split("\n\n"):
        line = block.strip()
        if line.startswith("data:"):
            events.append(json.loads(line[5:].strip()))
    return events


@pytest.fixture
def chat_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("MCP_CDP_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    db = sqlite_path()
    conn = sqlite3.connect(db)
    conn.executescript(SCHEMA)
    conn.execute(
        "INSERT INTO projects (id, name, created_at) VALUES ('p1', 'Local workspace', '2020-01-01T00:00:00+00:00')"
    )
    conn.commit()
    conn.close()
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_no_api_key_in_json_error(chat_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)

    async def boom(project_id: str, conversation_id: str, user_text: str, **_: object) -> AsyncIterator[str]:
        raise RuntimeError(f"AuthenticationError: invalid api key {SECRET}")
        yield "unreachable"

    monkeypatch.setattr("app.services.chat.iter_assistant_tokens", boom)

    conv_id = chat_client.post("/api/conversations", json={"project_id": "p1"}).json()["id"]
    resp = chat_client.post(
        f"/api/conversations/{conv_id}/messages",
        json={"content": "hello", "client_id": "cid-err"},
    )
    assert resp.status_code == 200
    assert SECRET not in resp.text
    events = _parse_sse(resp.text)
    err = next(e for e in events if e.get("type") == "error")
    blob = json.dumps(err)
    assert SECRET not in blob
    assert err.get("status") == "failed"


def test_no_key_explains_cards_still_work(chat_client: TestClient) -> None:
    conv_id = chat_client.post("/api/conversations", json={"project_id": "p1"}).json()["id"]
    resp = chat_client.post(
        f"/api/conversations/{conv_id}/messages",
        json={"content": "how accurate is the model?", "client_id": "cid-nokey"},
    )
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    text = "".join(e.get("content") or "" for e in events if e.get("type") == "token")
    assert text == NO_KEY_MESSAGE
    assert "cards" in text.lower()
    assert events[-1].get("type") == "done"
    assert events[-1].get("status") == "complete"
    assert "0.99" not in text
    assert "accuracy" not in text.lower() or "invent" in NO_KEY_MESSAGE.lower() or "cards" in text.lower()
