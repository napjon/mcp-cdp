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
