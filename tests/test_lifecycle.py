from __future__ import annotations

import sqlite3

import pytest

POST_HEADERS = {
    "Origin": "http://127.0.0.1:5173",
    "X-Requested-With": "mcp-cdp",
}

OLD_SCHEMA = """
CREATE TABLE projects (id TEXT PRIMARY KEY, name TEXT, created_at TEXT);
CREATE TABLE datasets (
  id TEXT PRIMARY KEY,
  project_id TEXT,
  name TEXT,
  source_type TEXT,
  connection_id TEXT NULL,
  created_at TEXT
);
CREATE TABLE conversations (
  id TEXT PRIMARY KEY,
  project_id TEXT,
  created_at TEXT,
  FOREIGN KEY(project_id) REFERENCES projects(id)
);
CREATE TABLE messages (
  id TEXT PRIMARY KEY,
  conversation_id TEXT,
  role TEXT,
  content TEXT,
  client_id TEXT UNIQUE,
  status TEXT,
  provider TEXT NULL,
  model TEXT NULL,
  usage_json TEXT NULL,
  created_at TEXT,
  FOREIGN KEY(conversation_id) REFERENCES conversations(id)
);
"""


def test_migrate_needs_review_and_messages_unique(data_dir):
    from app.db import connect, migrate, utcnow

    conn = connect()
    conn.executescript(OLD_SCHEMA)
    now = utcnow()
    conn.execute(
        "INSERT INTO projects (id, name, created_at) VALUES ('p1', 'P', ?)", (now,)
    )
    conn.execute(
        """
        INSERT INTO datasets (id, project_id, name, source_type, created_at)
        VALUES ('d1', 'p1', 't', 'csv', ?)
        """,
        (now,),
    )
    conn.execute(
        "INSERT INTO conversations (id, project_id, created_at) VALUES ('c1', 'p1', ?)",
        (now,),
    )
    conn.execute(
        "INSERT INTO conversations (id, project_id, created_at) VALUES ('c2', 'p1', ?)",
        (now,),
    )
    conn.execute(
        """
        INSERT INTO messages (
          id, conversation_id, role, content, client_id, status, created_at
        ) VALUES ('m1', 'c1', 'user', 'keep-me', 'cid-shared', 'complete', ?)
        """,
        (now,),
    )
    conn.commit()
    cols_before = {row[1] for row in conn.execute("PRAGMA table_info(datasets)")}
    assert "needs_review" not in cols_before

    migrate(conn)
    conn.commit()

    cols = {row["name"] for row in conn.execute("PRAGMA table_info(datasets)")}
    assert "needs_review" in cols
    kept = conn.execute("SELECT content FROM messages WHERE id = 'm1'").fetchone()
    assert kept["content"] == "keep-me"
    conn.execute(
        """
        INSERT INTO messages (
          id, conversation_id, role, content, client_id, status, created_at
        ) VALUES ('m2', 'c2', 'user', 'other', 'cid-shared', 'complete', ?)
        """,
        (now,),
    )
    conn.commit()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            INSERT INTO messages (
              id, conversation_id, role, content, client_id, status, created_at
            ) VALUES ('m3', 'c1', 'user', 'dup', 'cid-shared', 'complete', ?)
            """,
            (now,),
        )
    conn.rollback()
    versions = {
        row["version"]
        for row in conn.execute("SELECT version FROM schema_migrations").fetchall()
    }
    assert 1 in versions
    assert 2 in versions
    conn.close()


def test_init_db_allows_same_client_id_across_conversations(db):
    from app.db import get_db, utcnow

    now = utcnow()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO conversations (id, project_id, created_at) VALUES ('c1', 'local', ?)",
            (now,),
        )
        conn.execute(
            "INSERT INTO conversations (id, project_id, created_at) VALUES ('c2', 'local', ?)",
            (now,),
        )
        conn.execute(
            """
            INSERT INTO messages (
              id, conversation_id, role, content, client_id, status, created_at
            ) VALUES ('m1', 'c1', 'user', 'a', 'same-cid', 'complete', ?)
            """,
            (now,),
        )
        conn.execute(
            """
            INSERT INTO messages (
              id, conversation_id, role, content, client_id, status, created_at
            ) VALUES ('m2', 'c2', 'user', 'b', 'same-cid', 'complete', ?)
            """,
            (now,),
        )
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(datasets)")}
        assert "needs_review" in cols


def test_delete_project_removes_rows_and_files(client):
    from pathlib import Path

    from app.db import get_db
    from app.lifecycle import delete_project

    created = client.post(
        "/api/projects", json={"name": "Temp gone"}, headers=POST_HEADERS
    )
    assert created.status_code == 200, created.text
    project_id = created.json()["id"]
    uploaded = client.post(
        f"/api/projects/{project_id}/datasets/upload",
        files={"file": ("t.csv", b"a,b\n1,2\n3,4\n", "text/csv")},
        headers=POST_HEADERS,
    )
    assert uploaded.status_code == 200, uploaded.text
    dataset_id = uploaded.json()["id"]
    folder = Path(uploaded.json()["current_version"]["normalized_path"]).parent
    assert folder.is_dir()

    result = delete_project(project_id)
    assert result["ok"] is True
    assert not folder.exists()
    with get_db() as conn:
        assert conn.execute(
            "SELECT 1 FROM projects WHERE id = ?", (project_id,)
        ).fetchone() is None
        assert conn.execute(
            "SELECT 1 FROM datasets WHERE id = ?", (dataset_id,)
        ).fetchone() is None
        assert conn.execute(
            "SELECT 1 FROM dataset_versions WHERE dataset_id = ?", (dataset_id,)
        ).fetchone() is None
        assert conn.execute("SELECT 1 FROM projects WHERE id = 'local'").fetchone()


def test_backup_restore_roundtrip_includes_datasets_and_artifacts(db):
    from pathlib import Path

    from app.lifecycle import backup, restore
    from app.settings import get_settings

    settings = get_settings()
    dataset_file = settings.datasets_dir / "ds-keep" / "table.csv"
    artifact_file = settings.artifacts_dir / "job-keep" / "model.joblib"
    dataset_file.parent.mkdir(parents=True, exist_ok=True)
    artifact_file.parent.mkdir(parents=True, exist_ok=True)
    dataset_file.write_text("a,b\n1,2\n", encoding="utf-8")
    artifact_file.write_bytes(b"artifact")

    result = backup()
    archive = Path(result["path"])
    assert archive.is_file()
    assert archive.name.endswith(".tar.gz")
    assert result["manifest"]["db"]
    assert result["manifest"]["datasets"] == "datasets"
    assert result["manifest"]["artifacts"] == "artifacts"
    assert result["manifest"]["created"]

    dataset_file.unlink()
    artifact_file.unlink()
    assert not dataset_file.exists()
    assert not artifact_file.exists()

    restored = restore(archive)
    assert restored["ok"] is True
    assert dataset_file.is_file()
    assert dataset_file.read_text(encoding="utf-8") == "a,b\n1,2\n"
    assert artifact_file.is_file()
    assert artifact_file.read_bytes() == b"artifact"


def test_maybe_run_retention_startup_hook_and_purge(db):
    import inspect

    from app.db import get_db
    from app.lifecycle import maybe_run_retention, purge
    from app.main import create_app

    assert callable(maybe_run_retention)
    assert callable(purge)
    assert "maybe_run_retention" in inspect.getsource(create_app)

    old = "2020-01-01T00:00:00Z"
    with get_db() as conn:
        conn.execute(
            "INSERT INTO conversations (id, project_id, created_at) VALUES ('c-old', 'local', ?)",
            (old,),
        )
        conn.execute(
            """
            INSERT INTO messages (
              id, conversation_id, role, content, client_id, status, created_at
            ) VALUES ('m-old', 'c-old', 'user', 'stale', 'cid-old', 'complete', ?)
            """,
            (old,),
        )

    first = purge(days=30)
    assert first["ok"] is True
    assert first["conversations_deleted"] == 1
    assert first["messages_deleted"] >= 1
    again = maybe_run_retention()
    assert again["ok"] is True
    with get_db() as conn:
        assert conn.execute("SELECT 1 FROM conversations WHERE id = 'c-old'").fetchone() is None
        assert conn.execute("SELECT 1 FROM messages WHERE id = 'm-old'").fetchone() is None
