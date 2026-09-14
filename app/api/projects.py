"""Project list and create."""

from __future__ import annotations

from fastapi import APIRouter

from app.db import get_db, new_id, require_project, row_dict, utcnow
from app.models import ProjectCreate

router = APIRouter(prefix="/api/projects", tags=["projects"])


@router.get("")
def list_projects() -> list[dict]:
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM projects ORDER BY created_at ASC").fetchall()
        return [row_dict(r) for r in rows]


@router.post("")
def create_project(body: ProjectCreate) -> dict:
    with get_db() as conn:
        pid = new_id()
        now = utcnow()
        conn.execute(
            "INSERT INTO projects (id, name, created_at) VALUES (?, ?, ?)",
            (pid, body.name.strip(), now),
        )
        return {"id": pid, "name": body.name.strip(), "created_at": now}


@router.get("/{project_id}")
def get_project(project_id: str) -> dict:
    with get_db() as conn:
        return require_project(conn, project_id)
