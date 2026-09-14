"""Chat routes through the real app factory, no invented metrics without a key."""

from __future__ import annotations

import json

POST_HEADERS = {
    "Origin": "http://127.0.0.1:5173",
    "X-Requested-With": "mcp-cdp",
}


def _parse_sse(body: str) -> list[dict]:
    events = []
    for block in body.split("\n\n"):
        line = block.strip()
        if line.startswith("data:"):
            events.append(json.loads(line[5:].strip()))
    return events


def test_chat_no_key_through_app(client) -> None:
    created = client.post("/api/conversations", json={"project_id": "local"}, headers=POST_HEADERS)
    assert created.status_code == 200
    conv_id = created.json()["id"]
    resp = client.post(
        f"/api/conversations/{conv_id}/messages",
        json={"content": "How good is the model?", "client_id": "app-cid-1"},
        headers=POST_HEADERS,
    )
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    text = "".join(e.get("content") or "" for e in events if e.get("type") == "token")
    assert "cards" in text.lower()
    assert events[-1].get("type") == "done"
    assert events[-1].get("status") == "complete"
    listed = client.get(f"/api/conversations/{conv_id}/messages")
    assert listed.status_code == 200
    messages = listed.json()["messages"]
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["status"] == "complete"
