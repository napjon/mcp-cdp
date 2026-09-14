"""MCP reads go through datasets/jobs services when present and stay project-scoped."""

from __future__ import annotations

import json

import pytest

from app.mcp.domain import ServiceUnavailable, get_project, get_report, list_projects


@pytest.mark.asyncio
async def test_list_projects_prefers_datasets_service(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_call(module, names, **kwargs):
        assert module == "app.services.datasets"
        assert "list_projects" in names
        return [{"id": "from-service", "name": "Svc"}]

    monkeypatch.setattr("app.mcp.domain.call_service", fake_call)
    out = await list_projects()
    assert out.get("projects")[0]["id"] == "from-service"


@pytest.mark.asyncio
async def test_get_project_prefers_datasets_service(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_call(module, names, **kwargs):
        assert module == "app.services.datasets"
        assert "get_project" in names
        assert kwargs.get("project_id") == "p9"
        return {"id": "p9", "name": "Svc project"}

    monkeypatch.setattr("app.mcp.domain.call_service", fake_call)
    out = await get_project("p9")
    assert out.get("id") == "p9"
    assert out.get("name") == "Svc project"


@pytest.mark.asyncio
async def test_get_report_sql_fallback_does_not_leak_other_project(
    db, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.db import get_db, utcnow

    async def unavailable(*_a, **_k):
        raise ServiceUnavailable("app.services.jobs")

    monkeypatch.setattr("app.mcp.domain.call_service", unavailable)
    now = utcnow()
    secret = {"metric": "only-for-local"}
    with get_db() as conn:
        conn.execute(
            "INSERT INTO projects (id, name, created_at) VALUES (?, ?, ?)",
            ("p2", "Other", now),
        )
        conn.execute(
            """
            INSERT INTO jobs (
              id, project_id, type, status, attempt_id, progress_json, created_at
            ) VALUES (?, ?, 'train', 'succeeded', 'a1', '{}', ?)
            """,
            ("job-a", "local", now),
        )
        conn.execute(
            "INSERT INTO reports (id, job_id, report_json, plot_dir) VALUES (?, ?, ?, NULL)",
            ("rep-a", "job-a", json.dumps(secret)),
        )

    leaked = await get_report("p2", "job-a")
    assert leaked.get("ok") is False
    assert "only-for-local" not in json.dumps(leaked)

    own = await get_report("local", "job-a")
    assert own.get("ok") is True
    assert (own.get("report") or {}).get("metric") == "only-for-local"


@pytest.mark.asyncio
async def test_get_report_service_path_stays_in_project(db) -> None:
    from app.db import get_db, utcnow

    now = utcnow()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO projects (id, name, created_at) VALUES (?, ?, ?)",
            ("p2", "Other", now),
        )
        conn.execute(
            """
            INSERT INTO jobs (
              id, project_id, type, status, attempt_id, progress_json, created_at
            ) VALUES (?, ?, 'train', 'succeeded', 'a1', '{}', ?)
            """,
            ("job-b", "local", now),
        )
        conn.execute(
            "INSERT INTO reports (id, job_id, report_json, plot_dir) VALUES (?, ?, ?, NULL)",
            ("rep-b", "job-b", json.dumps({"metric": "secret-b"})),
        )

    leaked = await get_report("p2", "job-b")
    blob = json.dumps(leaked)
    assert "secret-b" not in blob
    assert leaked.get("ok") is False
