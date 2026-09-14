"""LLM chat: persist first, stream tokens, never leak API keys."""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from datetime import UTC, datetime
from typing import Any

from app.mcp.config import (
    anthropic_api_key,
    anthropic_model,
    llm_provider_name,
    openai_api_key,
    openai_model,
    redact,
)
from app.mcp.db import db_conn, fetchall, fetchone
from app.mcp.domain import (
    chat_tool_specs,
    dispatch_chat_tool,
    get_sample_disclosure,
    preview_sample_disclosure,
    samples_allowed,
    set_sample_disclosure,
)

# Data card / in-process UI can import disclosure helpers from this module.
__all__ = [
    "NO_KEY_MESSAGE",
    "ChatClientIdConflict",
    "build_project_context",
    "create_conversation",
    "get_conversation",
    "get_sample_disclosure",
    "iter_assistant_tokens",
    "list_messages",
    "llm_configured",
    "preview_sample_disclosure",
    "resolve_user_message",
    "set_sample_disclosure",
    "stream_user_message",
]

logger = logging.getLogger("mcp_cdp.chat")

NO_KEY_MESSAGE = (
    "No LLM API key is configured. Training and prediction still work from the cards. "
    "Chat is optional and does not invent metrics."
)
MAX_TOOL_CALLS = 6
MAX_CONTEXT_CHARS = 20_000
MAX_DATASETS_IN_CONTEXT = 20
MAX_COLUMNS_PER_DATASET = 40
MAX_MESSAGE_CHARS = 8_000
MAX_HISTORY_CHARS = 24_000
MAX_TOOL_RESULT_CHARS = 4_000
MAX_OUTPUT_TOKENS = 2048
SYSTEM_PROMPT = (
    "You are the assistant for a local AutoML app. Help the user understand their data, "
    "experiment settings, and computed metrics. You only see schema, aggregates, and metrics "
    "unless the user enabled sample sharing. Never invent metrics or scores. If a number is "
    "not in the context or a tool result, say you do not have it. Never run SQL, shell, or code. "
    "You may save experiment config; you cannot start training — the user does that on Confirm & train. "
    "Prediction returns a job_id; poll get_job. Stay inside this project. "
    "Set confirm=true on write tools only when the user clearly asked to do that action."
)

Disconnected = Callable[[], Awaitable[bool]]
_locks: dict[str, asyncio.Lock] = {}
_locks_guard = asyncio.Lock()


class ChatClientIdConflict(Exception):
    """client_id is already stored on a different conversation."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _new_id() -> str:
    return str(uuid.uuid4())


def llm_configured() -> bool:
    return bool(openai_api_key() or anthropic_api_key())


def create_conversation(project_id: str | None = None) -> dict[str, Any]:
    pid = project_id or _default_project_id()
    if not pid:
        raise ValueError("project_id is required")
    conv_id = _new_id()
    created = _now()
    try:
        with db_conn() as conn:
            conn.execute(
                "INSERT INTO conversations (id, project_id, created_at) VALUES (?, ?, ?)",
                (conv_id, pid, created),
            )
            conn.commit()
    except sqlite3.IntegrityError as exc:
        raise ValueError("project not found") from exc
    return {"id": conv_id, "project_id": pid, "created_at": created}


def _default_project_id() -> str | None:
    row = fetchone("SELECT id FROM projects ORDER BY created_at LIMIT 1")
    return str(row["id"]) if row else None


def get_conversation(conversation_id: str) -> dict[str, Any] | None:
    return fetchone(
        "SELECT id, project_id, created_at FROM conversations WHERE id = ?",
        (conversation_id,),
    )


def list_messages(conversation_id: str) -> list[dict[str, Any]]:
    rows = fetchall(
        """
        SELECT id, conversation_id, role, content, client_id, status, provider, model, usage_json, created_at
        FROM messages WHERE conversation_id = ? ORDER BY created_at, id
        """,
        (conversation_id,),
    )
    for row in rows:
        usage = row.get("usage_json")
        if isinstance(usage, str) and usage:
            try:
                row["usage_json"] = json.loads(usage)
            except json.JSONDecodeError:
                pass
    return rows


def _message_by_client_id(client_id: str, conversation_id: str) -> dict[str, Any] | None:
    return fetchone(
        """
        SELECT id, conversation_id, role, content, client_id, status, provider, model, created_at
        FROM messages WHERE client_id = ? AND conversation_id = ?
        """,
        (client_id, conversation_id),
    )


def resolve_user_message(conversation_id: str, content: str, client_id: str) -> dict[str, Any]:
    """Return the user row for this conversation, inserting if needed.

    Same-conversation retries are idempotent. A leftover global UNIQUE on
    client_id raises ChatClientIdConflict instead of attaching another
    conversation's messages. After UNIQUE(conversation_id, client_id), a
    reused client_id is a new send in this conversation.
    """
    existing = _message_by_client_id(client_id, conversation_id)
    if existing:
        return existing
    try:
        return _insert_message(
            conversation_id=conversation_id,
            role="user",
            content=content,
            client_id=client_id,
            status="complete",
        )
    except sqlite3.IntegrityError as exc:
        existing = _message_by_client_id(client_id, conversation_id)
        if existing:
            return existing
        raise ChatClientIdConflict("client_id already used in another conversation") from exc


def _assistant_after(conversation_id: str, user_created_at: str, user_id: str) -> dict[str, Any] | None:
    return fetchone(
        """
        SELECT id, conversation_id, role, content, client_id, status, provider, model, created_at
        FROM messages
        WHERE conversation_id = ? AND role = 'assistant' AND (created_at > ? OR (created_at = ? AND id > ?))
        ORDER BY created_at, id LIMIT 1
        """,
        (conversation_id, user_created_at, user_created_at, user_id),
    )


def _insert_message(
    *,
    conversation_id: str,
    role: str,
    content: str,
    client_id: str | None,
    status: str,
    provider: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    msg_id = _new_id()
    created = _now()
    with db_conn() as conn:
        conn.execute(
            """
            INSERT INTO messages (
                id, conversation_id, role, content, client_id, status, provider, model, usage_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)
            """,
            (msg_id, conversation_id, role, content, client_id, status, provider, model, created),
        )
        conn.commit()
    return {
        "id": msg_id,
        "conversation_id": conversation_id,
        "role": role,
        "content": content,
        "client_id": client_id,
        "status": status,
        "provider": provider,
        "model": model,
        "created_at": created,
    }


def _update_message(
    msg_id: str,
    *,
    content: str | None = None,
    status: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    usage_json: str | None = None,
) -> None:
    sets: list[str] = []
    params: list[Any] = []
    if content is not None:
        sets.append("content = ?")
        params.append(content)
    if status is not None:
        sets.append("status = ?")
        params.append(status)
    if provider is not None:
        sets.append("provider = ?")
        params.append(provider)
    if model is not None:
        sets.append("model = ?")
        params.append(model)
    if usage_json is not None:
        sets.append("usage_json = ?")
        params.append(usage_json)
    if not sets:
        return
    params.append(msg_id)
    with db_conn() as conn:
        conn.execute(f"UPDATE messages SET {', '.join(sets)} WHERE id = ?", tuple(params))
        conn.commit()


async def _lock_for(conversation_id: str, client_id: str) -> asyncio.Lock:
    key = f"{conversation_id}:{client_id}"
    async with _locks_guard:
        lock = _locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            _locks[key] = lock
        return lock


def _sse(event: dict[str, Any]) -> dict[str, Any]:
    return event


async def stream_user_message(
    conversation_id: str,
    content: str,
    client_id: str,
    disconnected: Disconnected | None = None,
) -> AsyncIterator[dict[str, Any]]:
    conv = get_conversation(conversation_id)
    if not conv:
        yield _sse({"type": "error", "message": "conversation not found"})
        return

    lock = await _lock_for(conversation_id, client_id)
    async with lock:
        async for event in _stream_locked(conv, content, client_id, disconnected):
            yield event


async def _stream_locked(
    conv: dict[str, Any],
    content: str,
    client_id: str,
    disconnected: Disconnected | None,
) -> AsyncIterator[dict[str, Any]]:
    conversation_id = conv["id"]
    project_id = conv["project_id"]
    try:
        user_msg = resolve_user_message(conversation_id, content, client_id)
    except ChatClientIdConflict as exc:
        yield _sse({"type": "error", "message": str(exc), "status": "conflict"})
        return
    assistant = _assistant_after(conversation_id, user_msg["created_at"], user_msg["id"])
    if assistant and assistant.get("status") == "complete":
        text = assistant.get("content") or ""
        if text:
            yield _sse({"type": "token", "content": text})
        yield _sse({"type": "done", "status": "complete"})
        return
    if assistant and assistant.get("status") == "streaming":
        yield _sse({"type": "error", "message": "generation already in progress"})
        return

    assistant = _insert_message(
        conversation_id=conversation_id,
        role="assistant",
        content="",
        client_id=None,
        status="streaming",
        provider=llm_provider_name() if llm_configured() else None,
        model=_active_model() if llm_configured() else None,
    )
    chunks: list[str] = []
    status = "complete"
    usage_out: dict[str, Any] = {}
    try:
        async for token in iter_assistant_tokens(
            project_id,
            conversation_id,
            user_msg.get("content") or content,
            usage_out=usage_out,
        ):
            if disconnected is not None and await disconnected():
                status = "interrupted"
                break
            chunks.append(token)
            yield _sse({"type": "token", "content": token})
        if status != "interrupted" and disconnected is not None and await disconnected():
            status = "interrupted"
        text = "".join(chunks)
        _update_message(
            assistant["id"],
            content=text,
            status=status,
            usage_json=_usage_json(usage_out),
        )
        if status == "interrupted":
            yield _sse({"type": "error", "message": "interrupted", "status": "interrupted"})
            return
        yield _sse({"type": "done", "status": "complete"})
    except asyncio.CancelledError:
        _update_message(
            assistant["id"],
            content="".join(chunks),
            status="interrupted",
            usage_json=_usage_json(usage_out),
        )
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("assistant generation failed: %s", redact(str(exc)))
        _update_message(
            assistant["id"],
            content="".join(chunks),
            status="failed",
            usage_json=_usage_json(usage_out),
        )
        yield _sse({"type": "error", "message": redact(str(exc)), "status": "failed"})


def _active_model() -> str | None:
    name = llm_provider_name()
    if name == "anthropic":
        return anthropic_model()
    if name == "openai":
        return openai_model()
    if anthropic_api_key():
        return anthropic_model()
    if openai_api_key():
        return openai_model()
    return None


def _bound_text(text: str, limit: int) -> tuple[str, bool]:
    if limit <= 0:
        return "", bool(text)
    if len(text) <= limit:
        return text, False
    note = "…[truncated]"
    keep = max(0, limit - len(note))
    return text[:keep] + note, True


def _bound_tool_result(result: Any, usage_out: dict[str, Any] | None = None) -> str:
    raw = json.dumps(result, default=str)
    text, truncated = _bound_text(raw, MAX_TOOL_RESULT_CHARS)
    if truncated:
        _mark_truncated(usage_out, tool_result=True)
    return text


def _mark_truncated(usage_out: dict[str, Any] | None, **flags: Any) -> None:
    if usage_out is None:
        return
    useful = {key: val for key, val in flags.items() if val}
    if not useful:
        return
    usage_out["truncated"] = True
    prev = usage_out.get("truncation")
    merged: dict[str, Any] = dict(prev) if isinstance(prev, dict) else {}
    merged.update(useful)
    usage_out["truncation"] = merged


def _clip_context(text: str, datasets_omitted: int, columns_omitted: int) -> tuple[str, bool]:
    notes: list[str] = []
    if datasets_omitted:
        notes.append(f"{datasets_omitted} datasets omitted")
    if columns_omitted:
        notes.append(f"{columns_omitted} columns omitted")
    over_chars = len(text) > MAX_CONTEXT_CHARS
    if over_chars:
        notes.append(f"project context limited to {MAX_CONTEXT_CHARS} characters")
    if not notes:
        return text, False
    note = "[truncated: " + "; ".join(notes) + "]"
    if len(text) + 1 + len(note) <= MAX_CONTEXT_CHARS:
        return text + "\n" + note, True
    keep = MAX_CONTEXT_CHARS - len(note) - 1
    if keep < 0:
        return note[:MAX_CONTEXT_CHARS], True
    return text[:keep] + "\n" + note, True


def build_project_context(
    project_id: str,
    truncation: dict[str, Any] | None = None,
) -> str:
    lines = [f"Project id: {project_id}"]
    project = fetchone("SELECT id, name FROM projects WHERE id = ?", (project_id,))
    if project:
        lines.append(f"Project name: {project.get('name')}")
    datasets = fetchall(
        "SELECT id, name, source_type FROM datasets WHERE project_id = ? ORDER BY created_at",
        (project_id,),
    )
    datasets_omitted = 0
    columns_omitted = 0
    if not datasets:
        lines.append("No datasets yet.")
    else:
        if len(datasets) > MAX_DATASETS_IN_CONTEXT:
            datasets_omitted = len(datasets) - MAX_DATASETS_IN_CONTEXT
            datasets = datasets[:MAX_DATASETS_IN_CONTEXT]
        for ds in datasets:
            lines.append(f"Dataset {ds.get('name')} id={ds.get('id')} source={ds.get('source_type')}")
            version = fetchone(
                """
                SELECT id, version, n_rows, n_cols FROM dataset_versions
                WHERE dataset_id = ? ORDER BY version DESC LIMIT 1
                """,
                (ds["id"],),
            )
            if version:
                lines.append(
                    f"  version {version.get('version')} rows={version.get('n_rows')} cols={version.get('n_cols')}"
                )
                cols = fetchall(
                    """
                    SELECT name, inferred_role, user_role, n_missing, n_unique, is_constant, sample_preview
                    FROM columns WHERE version_id = ? ORDER BY name
                    """,
                    (version["id"],),
                )
                if len(cols) > MAX_COLUMNS_PER_DATASET:
                    columns_omitted += len(cols) - MAX_COLUMNS_PER_DATASET
                    cols = cols[:MAX_COLUMNS_PER_DATASET]
                disclosed = samples_allowed(str(ds["id"]))
                for col in cols:
                    role = col.get("user_role") or col.get("inferred_role")
                    piece = (
                        f"  - {col.get('name')} role={role} missing={col.get('n_missing')} "
                        f"unique={col.get('n_unique')} constant={col.get('is_constant')}"
                    )
                    if disclosed and col.get("sample_preview"):
                        piece += f" preview={col.get('sample_preview')}"
                    lines.append(piece)
                if disclosed:
                    lines.append("  sample sharing is enabled for this dataset")
    experiments = fetchall(
        "SELECT id, dataset_id, created_at FROM experiments WHERE project_id = ? ORDER BY created_at",
        (project_id,),
    )
    for exp in experiments:
        rev = fetchone(
            """
            SELECT revision, config_json FROM experiment_revisions
            WHERE experiment_id = ? ORDER BY revision DESC LIMIT 1
            """,
            (exp["id"],),
        )
        cfg = ""
        if rev and rev.get("config_json"):
            cfg = str(rev["config_json"])
            if len(cfg) > 1500:
                cfg = cfg[:1500] + "…"
        lines.append(f"Experiment {exp['id']} dataset={exp.get('dataset_id')} config={cfg}")
    jobs = fetchall(
        """
        SELECT id, type, status, error, finished_at FROM jobs
        WHERE project_id = ? ORDER BY created_at DESC LIMIT 8
        """,
        (project_id,),
    )
    for job in jobs:
        lines.append(f"Job {job['id']} type={job.get('type')} status={job.get('status')}")
        report = fetchone("SELECT report_json FROM reports WHERE job_id = ?", (job["id"],))
        if report and report.get("report_json"):
            blob = str(report["report_json"])
            if len(blob) > 1500:
                blob = blob[:1500] + "…"
            lines.append(f"  report={blob}")
        model = fetchone(
            "SELECT task, metrics_json FROM models WHERE job_id = ? LIMIT 1",
            (job["id"],),
        )
        if model and model.get("metrics_json"):
            metrics = str(model["metrics_json"])
            if len(metrics) > 1500:
                metrics = metrics[:1500] + "…"
            lines.append(f"  metrics={metrics} task={model.get('task')}")
    text, truncated = _clip_context("\n".join(lines), datasets_omitted, columns_omitted)
    if truncation is not None and truncated:
        truncation["context"] = True
        if datasets_omitted:
            truncation["datasets_omitted"] = datasets_omitted
        if columns_omitted:
            truncation["columns_omitted"] = columns_omitted
    return text


def _history_messages(
    conversation_id: str,
    truncation: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    rows = list_messages(conversation_id)
    out: list[dict[str, str]] = []
    truncated = False
    for row in rows:
        if row.get("role") not in {"user", "assistant"}:
            continue
        if row.get("role") == "assistant" and row.get("status") == "streaming":
            continue
        text = row.get("content") or ""
        if not text:
            continue
        text, was = _bound_text(text, MAX_MESSAGE_CHARS)
        truncated = truncated or was
        out.append({"role": row["role"], "content": text})
    out = out[-20:]
    total = sum(len(item["content"]) for item in out)
    while len(out) > 1 and total > MAX_HISTORY_CHARS:
        dropped = out.pop(0)
        total -= len(dropped["content"])
        truncated = True
    if truncation is not None and truncated:
        truncation["history"] = True
    return out


def _usage_json(usage_out: dict[str, Any] | None) -> str | None:
    if not usage_out:
        return None
    return json.dumps(usage_out)


def _usage_payload(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    data: dict[str, Any]
    if hasattr(usage, "model_dump"):
        dumped = usage.model_dump(exclude_none=True)
        data = dumped if isinstance(dumped, dict) else {}
    elif isinstance(usage, dict):
        data = {k: v for k, v in usage.items() if v is not None}
    else:
        data = {}
        for key in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        ):
            val = getattr(usage, key, None)
            if val is not None:
                data[key] = val
    return data or None


def _record_usage(usage_out: dict[str, Any] | None, usage: Any) -> None:
    payload = _usage_payload(usage)
    if not payload or usage_out is None:
        return
    for key, val in payload.items():
        existing = usage_out.get(key)
        if isinstance(val, (int, float)) and isinstance(existing, (int, float)):
            usage_out[key] = existing + val
        elif key not in usage_out:
            usage_out[key] = val


async def run_chat_tool_calls(
    project_id: str,
    calls: list[dict[str, Any]],
    used: int = 0,
) -> tuple[int, list[dict[str, Any]]]:
    """Dispatch at most MAX_TOOL_CALLS including `used`. The remainder is not dispatched."""
    results: list[dict[str, Any]] = []
    for call in calls:
        if used >= MAX_TOOL_CALLS:
            break
        used += 1
        name = str(call.get("name") or "")
        raw_args = call.get("arguments")
        args = raw_args if isinstance(raw_args, dict) else {}
        results.append(await dispatch_chat_tool(project_id, name, args))
    return used, results


async def iter_assistant_tokens(
    project_id: str,
    conversation_id: str,
    user_text: str,
    usage_out: dict[str, Any] | None = None,
) -> AsyncIterator[str]:
    """Yield assistant text tokens. Patch this in tests to avoid calling providers."""
    if not llm_configured():
        yield NO_KEY_MESSAGE
        return
    trunc_info: dict[str, Any] = {}
    context = build_project_context(project_id, truncation=trunc_info)
    history = _history_messages(conversation_id, truncation=trunc_info)
    user_text, user_trunc = _bound_text(user_text, MAX_MESSAGE_CHARS)
    if user_trunc:
        trunc_info["user"] = True
    _mark_truncated(usage_out, **trunc_info)
    provider = llm_provider_name()
    if provider == "anthropic" and anthropic_api_key():
        async for token in _stream_anthropic(
            project_id, context, history, user_text, usage_out=usage_out
        ):
            yield token
        return
    if openai_api_key():
        async for token in _stream_openai(
            project_id, context, history, user_text, usage_out=usage_out
        ):
            yield token
        return
    if anthropic_api_key():
        async for token in _stream_anthropic(
            project_id, context, history, user_text, usage_out=usage_out
        ):
            yield token
        return
    yield NO_KEY_MESSAGE


def _openai_tools() -> list[dict[str, Any]]:
    tools = []
    for spec in chat_tool_specs():
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": spec["name"],
                    "description": spec["description"],
                    "parameters": spec["parameters"],
                },
            }
        )
    return tools


def _anthropic_tools() -> list[dict[str, Any]]:
    tools = []
    for spec in chat_tool_specs():
        tools.append(
            {
                "name": spec["name"],
                "description": spec["description"],
                "input_schema": spec["parameters"],
            }
        )
    return tools


def _parse_tool_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {}
    return {}


async def _stream_openai(
    project_id: str,
    context: str,
    history: list[dict[str, str]],
    user_text: str,
    usage_out: dict[str, Any] | None = None,
) -> AsyncIterator[str]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=openai_api_key())
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\nProject context:\n" + context},
    ]
    for item in history:
        messages.append({"role": item["role"], "content": item["content"]})
    if not history or history[-1].get("content") != user_text:
        messages.append({"role": "user", "content": user_text})

    tools = _openai_tools()
    tool_calls_used = 0
    model = openai_model()
    while True:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": MAX_OUTPUT_TOKENS,
        }
        if tool_calls_used < MAX_TOOL_CALLS:
            kwargs["tools"] = tools
        stream = await client.chat.completions.create(**kwargs)
        role_content: list[str] = []
        acc: dict[int, dict[str, str]] = {}
        async for chunk in stream:
            _record_usage(usage_out, getattr(chunk, "usage", None))
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                role_content.append(delta.content)
                yield delta.content
            if delta and delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index if tc.index is not None else 0
                    slot = acc.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function and tc.function.name:
                        slot["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        slot["arguments"] += tc.function.arguments
        if not acc:
            return
        remaining = MAX_TOOL_CALLS - tool_calls_used
        if remaining <= 0:
            return
        openai_tool_calls = []
        for idx in sorted(acc)[:remaining]:
            slot = acc[idx]
            openai_tool_calls.append(
                {
                    "id": slot["id"] or f"call_{idx}",
                    "type": "function",
                    "function": {"name": slot["name"], "arguments": slot["arguments"] or "{}"},
                }
            )
        if not openai_tool_calls:
            return
        messages.append(
            {
                "role": "assistant",
                "content": "".join(role_content) or None,
                "tool_calls": openai_tool_calls,
            }
        )
        calls = [
            {
                "name": call["function"]["name"],
                "arguments": _parse_tool_args(call["function"]["arguments"]),
            }
            for call in openai_tool_calls
        ]
        tool_calls_used, results = await run_chat_tool_calls(
            project_id, calls, used=tool_calls_used
        )
        for call, result in zip(openai_tool_calls, results, strict=True):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": _bound_tool_result(result, usage_out),
                }
            )


async def _stream_anthropic(
    project_id: str,
    context: str,
    history: list[dict[str, str]],
    user_text: str,
    usage_out: dict[str, Any] | None = None,
) -> AsyncIterator[str]:
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(api_key=anthropic_api_key())
    messages: list[dict[str, Any]] = []
    for item in history:
        messages.append({"role": item["role"], "content": item["content"]})
    if not history or history[-1].get("content") != user_text:
        messages.append({"role": "user", "content": user_text})
    if not messages:
        messages.append({"role": "user", "content": user_text})

    tools = _anthropic_tools()
    tool_calls_used = 0
    model = anthropic_model()
    system = SYSTEM_PROMPT + "\n\nProject context:\n" + context
    while True:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "system": system,
        }
        if tool_calls_used < MAX_TOOL_CALLS:
            kwargs["tools"] = tools
        async with client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text
            final = await stream.get_final_message()
        _record_usage(usage_out, getattr(final, "usage", None))
        tool_uses = [b for b in final.content if getattr(b, "type", None) == "tool_use"]
        remaining = MAX_TOOL_CALLS - tool_calls_used
        if not tool_uses or remaining <= 0:
            return
        tool_uses = tool_uses[:remaining]
        assistant_content = []
        tool_use_ids = {getattr(b, "id", None) for b in tool_uses}
        for block in final.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                assistant_content.append({"type": "text", "text": block.text})
            elif btype == "tool_use" and block.id in tool_use_ids:
                assistant_content.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input if isinstance(block.input, Mapping) else {},
                    }
                )
        messages.append({"role": "assistant", "content": assistant_content})
        calls = [
            {
                "name": block.name,
                "arguments": block.input if isinstance(block.input, dict) else {},
            }
            for block in tool_uses
        ]
        tool_calls_used, results = await run_chat_tool_calls(
            project_id, calls, used=tool_calls_used
        )
        tool_results = [
            {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": _bound_tool_result(result, usage_out),
            }
            for block, result in zip(tool_uses, results, strict=True)
        ]
        messages.append({"role": "user", "content": tool_results})
