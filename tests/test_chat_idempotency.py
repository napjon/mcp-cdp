"""Chat client_id idempotency: a second POST must not call the LLM again."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from app.api.chat import router
from app.mcp.config import sqlite_path
from app.services.chat import create_conversation, list_messages, stream_user_message

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


def _parent_id(message: dict) -> str | None:
    usage = message.get("usage_json")
    if isinstance(usage, str) and usage:
        usage = json.loads(usage)
    if isinstance(usage, dict):
        parent = usage.get("parent_id")
        return str(parent) if parent else None
    return None


@pytest.mark.asyncio
async def test_overlapping_client_ids_do_not_cross_attach(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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

    calls = {"n": 0}
    entered = asyncio.Event()
    gate = asyncio.Event()
    snapshots: list[tuple[str, list[str]]] = []

    async def fake_tokens(
        project_id: str, conversation_id: str, user_text: str, **_: object
    ) -> AsyncIterator[str]:
        calls["n"] += 1
        snapshots.append(
            (
                user_text,
                [
                    m.get("content") or ""
                    for m in list_messages(conversation_id)
                    if m.get("role") == "user"
                ],
            )
        )
        entered.set()
        await gate.wait()
        yield f"reply:{user_text}"

    monkeypatch.setattr("app.services.chat.iter_assistant_tokens", fake_tokens)

    app = FastAPI()
    app.include_router(router)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        conv_id = (await ac.post("/api/conversations", json={"project_id": "p1"})).json()["id"]
        url = f"/api/conversations/{conv_id}/messages"
        first = asyncio.create_task(
            ac.post(url, json={"content": "hello-a", "client_id": "cid-a"}, timeout=10.0)
        )
        await asyncio.wait_for(entered.wait(), 2)
        second = asyncio.create_task(
            ac.post(url, json={"content": "hello-b", "client_id": "cid-b"}, timeout=10.0)
        )
        await asyncio.sleep(0.1)
        assert calls["n"] == 1
        gate.set()
        r1, r2 = await asyncio.gather(first, second)
        assert r1.status_code == 200
        assert r2.status_code == 200
        text_a = "".join(
            e.get("content") or "" for e in _parse_sse(r1.text) if e.get("type") == "token"
        )
        text_b = "".join(
            e.get("content") or "" for e in _parse_sse(r2.text) if e.get("type") == "token"
        )
        assert text_a == "reply:hello-a"
        assert text_b == "reply:hello-b"
        assert calls["n"] == 2
        assert snapshots[0][0] == "hello-a"
        assert "hello-b" not in snapshots[0][1]

        listed = (await ac.get(url)).json()["messages"]
        users = [m for m in listed if m.get("role") == "user"]
        assistants = [m for m in listed if m.get("role") == "assistant"]
        assert len(users) == 2
        assert len(assistants) == 2
        by_parent = {_parent_id(a): a for a in assistants}
        assert set(by_parent) == {u["id"] for u in users}
        for user in users:
            assistant = by_parent[user["id"]]
            assert assistant.get("content") == f"reply:{user.get('content')}"

        retry = await ac.post(url, json={"content": "hello-a", "client_id": "cid-a"}, timeout=10.0)
        assert retry.status_code == 200
        assert calls["n"] == 2
        retry_text = "".join(
            e.get("content") or "" for e in _parse_sse(retry.text) if e.get("type") == "token"
        )
        assert retry_text == "reply:hello-a"


@pytest.mark.asyncio
async def test_cancelled_stream_releases_conversation_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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

    entered = asyncio.Event()
    hold = asyncio.Event()

    async def fake_tokens(
        project_id: str, conversation_id: str, user_text: str, **_: object
    ) -> AsyncIterator[str]:
        if user_text == "hello-a":
            entered.set()
            await hold.wait()
        yield f"reply:{user_text}"

    monkeypatch.setattr("app.services.chat.iter_assistant_tokens", fake_tokens)
    conv_id = create_conversation("p1")["id"]

    async def consume_first() -> None:
        async for _event in stream_user_message(conv_id, "hello-a", "cid-a"):
            pass

    first = asyncio.create_task(consume_first())
    await asyncio.wait_for(entered.wait(), 2)
    first.cancel()
    await asyncio.gather(first, return_exceptions=True)

    events: list[dict] = []

    async def collect() -> None:
        async for event in stream_user_message(conv_id, "hello-b", "cid-b"):
            events.append(event)

    await asyncio.wait_for(collect(), 2)
    text = "".join(e.get("content") or "" for e in events if e.get("type") == "token")
    assert text == "reply:hello-b"
    assert events[-1].get("type") == "done"


@pytest.mark.asyncio
async def test_http_disconnect_allows_next_send(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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

    entered = asyncio.Event()
    hold = asyncio.Event()

    async def fake_tokens(
        project_id: str, conversation_id: str, user_text: str, **_: object
    ) -> AsyncIterator[str]:
        if user_text == "hello-a":
            yield "partial:hello-a"
            entered.set()
            await hold.wait()
            yield "more"
            return
        yield f"reply:{user_text}"

    monkeypatch.setattr("app.services.chat.iter_assistant_tokens", fake_tokens)
    app = FastAPI()
    app.include_router(router)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        conv_id = (await ac.post("/api/conversations", json={"project_id": "p1"})).json()["id"]
        url = f"/api/conversations/{conv_id}/messages"
        first = asyncio.create_task(
            ac.post(url, json={"content": "hello-a", "client_id": "cid-a"}, timeout=10.0)
        )
        await asyncio.wait_for(entered.wait(), 2)
        first.cancel()
        await asyncio.gather(first, return_exceptions=True)
        second = await asyncio.wait_for(
            ac.post(url, json={"content": "hello-b", "client_id": "cid-b"}, timeout=10.0),
            2,
        )
        assert second.status_code == 200
        text_b = "".join(
            e.get("content") or "" for e in _parse_sse(second.text) if e.get("type") == "token"
        )
        assert text_b == "reply:hello-b"
        listed = (await ac.get(url)).json()["messages"]
        assistants = [m for m in listed if m.get("role") == "assistant"]
        users = [m for m in listed if m.get("role") == "user"]
        by_parent = {_parent_id(a): a for a in assistants}
        user_b = next(u for u in users if u.get("content") == "hello-b")
        assert by_parent[user_b["id"]].get("content") == "reply:hello-b"
        user_a = next((u for u in users if u.get("content") == "hello-a"), None)
        if user_a is not None and user_a["id"] in by_parent:
            assert by_parent[user_a["id"]].get("status") in {"interrupted", "failed", "complete"}


@pytest.mark.asyncio
async def test_aclose_after_token_then_retry_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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

    calls = {"n": 0}
    hold = asyncio.Event()

    async def fake_tokens(
        project_id: str, conversation_id: str, user_text: str, **_: object
    ) -> AsyncIterator[str]:
        calls["n"] += 1
        if calls["n"] == 1:
            yield "partial"
            await hold.wait()
            yield "more"
            return
        yield "final"

    monkeypatch.setattr("app.services.chat.iter_assistant_tokens", fake_tokens)
    conv_id = create_conversation("p1")["id"]
    agen = stream_user_message(conv_id, "hello", "cid-1")
    first = await anext(agen)
    assert first.get("type") == "token"
    assert first.get("content") == "partial"
    await agen.aclose()

    listed = list_messages(conv_id)
    users = [m for m in listed if m.get("role") == "user"]
    assistants = [m for m in listed if m.get("role") == "assistant"]
    assert len(users) == 1
    assert len(assistants) == 1
    assert assistants[0].get("status") == "interrupted"
    assert assistants[0].get("content") == "partial"
    assert _parent_id(assistants[0]) == users[0]["id"]

    retry_events: list[dict] = []
    async for event in stream_user_message(conv_id, "hello", "cid-1"):
        retry_events.append(event)
    retry_text = "".join(
        e.get("content") or "" for e in retry_events if e.get("type") == "token"
    )
    assert retry_text == "final"
    assert calls["n"] == 2
    assert retry_events[-1].get("type") == "done"

    replay_events: list[dict] = []
    async for event in stream_user_message(conv_id, "hello", "cid-1"):
        replay_events.append(event)
    replay_text = "".join(
        e.get("content") or "" for e in replay_events if e.get("type") == "token"
    )
    assert replay_text == "final"
    assert calls["n"] == 2
    listed = list_messages(conv_id)
    assistants = [m for m in listed if m.get("role") == "assistant"]
    assert len(assistants) == 1
    assert assistants[0].get("status") == "complete"
    assert assistants[0].get("content") == "final"
