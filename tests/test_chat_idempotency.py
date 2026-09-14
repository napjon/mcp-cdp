"""Chat client_id idempotency: a second POST must not call the LLM again."""

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


def test_client_id_idempotency_does_not_call_llm_twice(chat_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    async def fake_tokens(project_id: str, conversation_id: str, user_text: str, **_: object) -> AsyncIterator[str]:
        calls["n"] += 1
        yield "Hello"
        yield " world"

    monkeypatch.setattr("app.services.chat.iter_assistant_tokens", fake_tokens)

    created = chat_client.post("/api/conversations", json={"project_id": "p1"})
    assert created.status_code == 200
    conv_id = created.json()["id"]
    url = f"/api/conversations/{conv_id}/messages"
    payload = {"content": "Hi there", "client_id": "cid-1"}

    first = chat_client.post(url, json=payload)
    assert first.status_code == 200
    first_events = _parse_sse(first.text)
    assert any(e.get("type") == "token" for e in first_events)
    assert first_events[-1].get("type") == "done"
    assert first_events[-1].get("status") == "complete"
    assert calls["n"] == 1

    second = chat_client.post(url, json=payload)
    assert second.status_code == 200
    second_events = _parse_sse(second.text)
    assert calls["n"] == 1
    assert second_events[-1].get("type") == "done"
    text = "".join(e.get("content") or "" for e in second_events if e.get("type") == "token")
    assert "Hello" in text


def test_same_client_id_does_not_replay_other_conversation(
    chat_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_tokens(
        project_id: str, conversation_id: str, user_text: str, **_: object
    ) -> AsyncIterator[str]:
        yield f"assistant-for-{conversation_id}"

    monkeypatch.setattr("app.services.chat.iter_assistant_tokens", fake_tokens)

    conv_a = chat_client.post("/api/conversations", json={"project_id": "p1"}).json()["id"]
    conv_b = chat_client.post("/api/conversations", json={"project_id": "p1"}).json()["id"]
    first = chat_client.post(
        f"/api/conversations/{conv_a}/messages",
        json={"content": "hello-a", "client_id": "shared-cid"},
    )
    assert first.status_code == 200
    first_text = "".join(
        e.get("content") or "" for e in _parse_sse(first.text) if e.get("type") == "token"
    )
    assert first_text == f"assistant-for-{conv_a}"

    second = chat_client.post(
        f"/api/conversations/{conv_b}/messages",
        json={"content": "hello-b", "client_id": "shared-cid"},
    )
    assert f"assistant-for-{conv_a}" not in (second.text or "")
    assert second.status_code in {200, 409}
    if second.status_code == 409:
        listed = chat_client.get(f"/api/conversations/{conv_b}/messages").json()["messages"]
        assert listed == []
    else:
        second_text = "".join(
            e.get("content") or "" for e in _parse_sse(second.text) if e.get("type") == "token"
        )
        assert f"assistant-for-{conv_a}" not in second_text
        listed = chat_client.get(f"/api/conversations/{conv_b}/messages").json()["messages"]
        contents = [m.get("content") or "" for m in listed]
        assert all(f"assistant-for-{conv_a}" not in c for c in contents)


def test_same_client_id_is_new_send_when_unique_is_per_conversation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MCP_CDP_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    db = sqlite_path()
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE projects (id TEXT PRIMARY KEY, name TEXT, created_at TEXT);
        CREATE TABLE conversations (id TEXT PRIMARY KEY, project_id TEXT, created_at TEXT);
        CREATE TABLE messages (
            id TEXT PRIMARY KEY,
            conversation_id TEXT,
            role TEXT,
            content TEXT,
            client_id TEXT,
            status TEXT,
            provider TEXT,
            model TEXT,
            usage_json TEXT,
            created_at TEXT,
            UNIQUE(conversation_id, client_id)
        );
        """
    )
    conn.execute(
        "INSERT INTO projects (id, name, created_at) VALUES ('p1', 'Local workspace', '2020-01-01T00:00:00+00:00')"
    )
    conn.commit()
    conn.close()
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    async def fake_tokens(
        project_id: str, conversation_id: str, user_text: str, **_: object
    ) -> AsyncIterator[str]:
        yield f"assistant-for-{conversation_id}:{user_text}"

    monkeypatch.setattr("app.services.chat.iter_assistant_tokens", fake_tokens)

    conv_a = client.post("/api/conversations", json={"project_id": "p1"}).json()["id"]
    conv_b = client.post("/api/conversations", json={"project_id": "p1"}).json()["id"]
    first = client.post(
        f"/api/conversations/{conv_a}/messages",
        json={"content": "hello-a", "client_id": "shared-cid"},
    )
    second = client.post(
        f"/api/conversations/{conv_b}/messages",
        json={"content": "hello-b", "client_id": "shared-cid"},
    )
    assert first.status_code == 200
    assert second.status_code == 200
    first_text = "".join(
        e.get("content") or "" for e in _parse_sse(first.text) if e.get("type") == "token"
    )
    second_text = "".join(
        e.get("content") or "" for e in _parse_sse(second.text) if e.get("type") == "token"
    )
    assert first_text == f"assistant-for-{conv_a}:hello-a"
    assert second_text == f"assistant-for-{conv_b}:hello-b"
    assert first_text not in second_text
    listed_b = client.get(f"/api/conversations/{conv_b}/messages").json()["messages"]
    assert any(m.get("role") == "user" and m.get("content") == "hello-b" for m in listed_b)
    assert all("hello-a" not in (m.get("content") or "") for m in listed_b)
