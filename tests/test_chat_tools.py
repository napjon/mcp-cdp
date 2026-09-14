"""In-app chat tools: required args, call cap, no submit_training, persist usage."""

from __future__ import annotations

import json
import sqlite3
import sys
import types
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.chat import router
from app.mcp.config import sqlite_path
from app.mcp.domain import chat_tool_specs, dispatch_chat_tool
from app.services.chat import (
    MAX_CONTEXT_CHARS,
    MAX_MESSAGE_CHARS,
    MAX_OUTPUT_TOKENS,
    MAX_TOOL_CALLS,
    _record_usage,
    build_project_context,
    iter_assistant_tokens,
    run_chat_tool_calls,
)

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

POST_HEADERS = {
    "Origin": "http://127.0.0.1:5173",
    "X-Requested-With": "mcp-cdp",
}


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


def test_chat_tool_list_excludes_submit_training() -> None:
    names = [spec["name"] for spec in chat_tool_specs()]
    assert "submit_training" not in names
    assert "configure_experiment" in names


@pytest.mark.asyncio
async def test_submit_training_not_dispatched_from_chat() -> None:
    result = await dispatch_chat_tool(
        "p1", "submit_training", {"experiment_id": "e1", "confirm": True}
    )
    assert result.get("ok") is False
    assert "unknown" in (result.get("error") or "").lower()


@pytest.mark.asyncio
async def test_missing_required_chat_arg_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"n": 0}

    async def boom(*_a, **_k):
        called["n"] += 1
        return {"ok": True}

    monkeypatch.setattr("app.mcp.domain.inspect_dataset", boom)
    result = await dispatch_chat_tool("p1", "inspect_dataset", {})
    assert result.get("ok") is False
    assert "dataset_id" in (result.get("error") or "")
    assert called["n"] == 0

    empty = await dispatch_chat_tool("p1", "inspect_dataset", {"dataset_id": "  "})
    assert empty.get("ok") is False
    assert called["n"] == 0


@pytest.mark.asyncio
async def test_seventh_chat_tool_call_not_dispatched(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    async def fake_dispatch(project_id: str, name: str, arguments: dict) -> dict:
        seen.append(name)
        return {"ok": True, "name": name}

    monkeypatch.setattr("app.services.chat.dispatch_chat_tool", fake_dispatch)
    calls = [{"name": "list_datasets", "arguments": {}} for _ in range(7)]
    used, results = await run_chat_tool_calls("p1", calls, used=0)
    assert MAX_TOOL_CALLS == 6
    assert used == 6
    assert seen == ["list_datasets"] * 6
    assert len(results) == 6


def test_record_usage_keeps_provider_tokens_without_inventing_cost() -> None:
    out: dict = {}
    _record_usage(out, {"input_tokens": 3, "output_tokens": 7})
    _record_usage(out, {"input_tokens": 1, "output_tokens": 2})
    assert out == {"input_tokens": 4, "output_tokens": 9}
    assert "cost" not in out


def test_openai_stream_usage_persisted(chat_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.services.chat.openai_api_key", lambda: "sk-test-not-real")
    monkeypatch.setattr("app.services.chat.anthropic_api_key", lambda: None)
    monkeypatch.setattr("app.services.chat.llm_provider_name", lambda: "openai")
    monkeypatch.setattr("app.services.chat.openai_model", lambda: "gpt-4o-mini")

    class FakeUsage:
        prompt_tokens = 11
        completion_tokens = 5
        total_tokens = 16

        def model_dump(self, exclude_none: bool = True) -> dict:
            return {
                "prompt_tokens": 11,
                "completion_tokens": 5,
                "total_tokens": 16,
            }

    class FakeDelta:
        def __init__(self, content: str | None = None) -> None:
            self.content = content
            self.tool_calls = None

    class FakeChoice:
        def __init__(self, content: str | None) -> None:
            self.delta = FakeDelta(content)

    class FakeChunk:
        def __init__(self, content: str | None = None, usage: FakeUsage | None = None) -> None:
            self.choices = [FakeChoice(content)] if content is not None else []
            self.usage = usage

    class FakeStream:
        def __init__(self) -> None:
            self._chunks = [
                FakeChunk(content="Hello from mock"),
                FakeChunk(usage=FakeUsage()),
            ]

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self._chunks:
                raise StopAsyncIteration
            return self._chunks.pop(0)

    class FakeCompletions:
        async def create(self, **kwargs):
            assert kwargs.get("stream") is True
            assert kwargs.get("max_tokens") == MAX_OUTPUT_TOKENS
            return FakeStream()

    class FakeClient:
        def __init__(self, api_key: str | None = None) -> None:
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    fake_openai = types.ModuleType("openai")
    fake_openai.AsyncOpenAI = FakeClient
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    conv_id = chat_client.post("/api/conversations", json={"project_id": "p1"}).json()["id"]
    resp = chat_client.post(
        f"/api/conversations/{conv_id}/messages",
        json={"content": "hi", "client_id": "cid-usage"},
        headers=POST_HEADERS,
    )
    assert resp.status_code == 200, resp.text
    listed = chat_client.get(f"/api/conversations/{conv_id}/messages")
    messages = listed.json()["messages"]
    assistant = next(m for m in messages if m["role"] == "assistant")
    usage = assistant.get("usage_json")
    if isinstance(usage, str):
        usage = json.loads(usage)
    assert usage is not None
    assert usage.get("prompt_tokens") == 11
    assert usage.get("completion_tokens") == 5
    assert usage.get("total_tokens") == 16
    assert "cost" not in usage
    assert "cost_usd" not in usage
    assert "Hello from mock" in (assistant.get("content") or "")


def test_build_project_context_caps_huge_fake_context(monkeypatch: pytest.MonkeyPatch) -> None:
    datasets = [
        {"id": f"d{i}", "name": f"Dataset {i}", "source_type": "upload"} for i in range(80)
    ]
    columns = [
        {
            "name": f"col_{j}_" + ("n" * 120),
            "inferred_role": "feature",
            "user_role": None,
            "n_missing": 0,
            "n_unique": 4,
            "is_constant": 0,
            "sample_preview": "preview" * 40,
        }
        for j in range(120)
    ]

    def fake_fetchall(sql: str, params: tuple = ()) -> list[dict]:
        if "FROM datasets" in sql:
            return datasets
        if "FROM columns" in sql:
            return columns
        if "FROM experiments" in sql:
            return [{"id": f"e{i}", "dataset_id": "d0", "created_at": "t"} for i in range(40)]
        if "FROM jobs" in sql:
            return [
                {"id": f"j{i}", "type": "train", "status": "complete", "error": None, "finished_at": "t"}
                for i in range(8)
            ]
        return []

    def fake_fetchone(sql: str, params: tuple = ()) -> dict | None:
        if "FROM projects" in sql:
            return {"id": "p1", "name": "Local workspace"}
        if "FROM dataset_versions" in sql:
            return {"id": "v1", "version": 1, "n_rows": 10_000, "n_cols": 120}
        if "FROM experiment_revisions" in sql:
            return {"revision": 1, "config_json": "{" + ("k" * 2000) + "}"}
        if "FROM reports" in sql:
            return {"report_json": "{" + ("r" * 2000) + "}"}
        if "FROM models" in sql:
            return {"task": "classification", "metrics_json": "{" + ("m" * 2000) + "}"}
        return None

    monkeypatch.setattr("app.services.chat.fetchall", fake_fetchall)
    monkeypatch.setattr("app.services.chat.fetchone", fake_fetchone)
    monkeypatch.setattr("app.services.chat.samples_allowed", lambda *_a, **_k: False)

    truncation: dict = {}
    text = build_project_context("p1", truncation=truncation)
    assert len(text) <= MAX_CONTEXT_CHARS
    assert "[truncated:" in text
    assert truncation.get("context") is True


@pytest.mark.asyncio
async def test_openai_kwargs_include_max_tokens_with_truncated_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    class FakeStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    class FakeCompletions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return FakeStream()

    class FakeClient:
        def __init__(self, api_key: str | None = None) -> None:
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    fake_openai = types.ModuleType("openai")
    fake_openai.AsyncOpenAI = FakeClient
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setattr("app.services.chat.openai_api_key", lambda: "sk-test-not-real")
    monkeypatch.setattr("app.services.chat.anthropic_api_key", lambda: None)
    monkeypatch.setattr("app.services.chat.llm_provider_name", lambda: "openai")
    monkeypatch.setattr("app.services.chat.openai_model", lambda: "gpt-4o-mini")
    monkeypatch.setattr("app.services.chat.llm_configured", lambda: True)

    def fake_context(_project_id: str, truncation: dict | None = None) -> str:
        if truncation is not None:
            truncation["context"] = True
        return "x" * 100

    monkeypatch.setattr("app.services.chat.build_project_context", fake_context)
    monkeypatch.setattr("app.services.chat._history_messages", lambda *_a, **_k: [])

    usage: dict = {}
    async for _token in iter_assistant_tokens("p1", "c1", "hi" * 5000, usage_out=usage):
        pass
    assert captured.get("max_tokens") == MAX_OUTPUT_TOKENS
    assert captured.get("stream") is True
    assert usage.get("truncated") is True
    assert usage.get("truncation", {}).get("context") is True
    assert usage.get("truncation", {}).get("user") is True
    for msg in captured.get("messages") or []:
        content = msg.get("content")
        if isinstance(content, str) and msg.get("role") != "system":
            assert len(content) <= MAX_MESSAGE_CHARS
