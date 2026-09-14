"""SQLite access: WAL, foreign keys, contract schema."""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from app.settings import get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
  id TEXT PRIMARY KEY,
  name TEXT,
  created_at TEXT
);
CREATE TABLE IF NOT EXISTS datasets (
  id TEXT PRIMARY KEY,
  project_id TEXT,
  name TEXT,
  source_type TEXT,
  connection_id TEXT NULL,
  created_at TEXT,
  FOREIGN KEY(project_id) REFERENCES projects(id)
);
CREATE TABLE IF NOT EXISTS dataset_versions (
  id TEXT PRIMARY KEY,
  dataset_id TEXT,
  version INTEGER,
  content_hash TEXT,
  original_path TEXT,
  normalized_path TEXT,
  n_rows INTEGER,
  n_cols INTEGER,
  header_row INTEGER,
  encoding TEXT,
  delimiter TEXT,
  import_meta_json TEXT,
  created_at TEXT,
  FOREIGN KEY(dataset_id) REFERENCES datasets(id)
);
CREATE TABLE IF NOT EXISTS sheet_connections (
  id TEXT PRIMARY KEY,
  project_id TEXT,
  spreadsheet_id TEXT,
  gid TEXT NULL,
  sheet_name TEXT NULL,
  public_url TEXT,
  last_status TEXT,
  last_checked_at TEXT,
  last_error TEXT NULL,
  last_version_id TEXT NULL,
  FOREIGN KEY(project_id) REFERENCES projects(id)
);
CREATE TABLE IF NOT EXISTS columns (
  id TEXT PRIMARY KEY,
  version_id TEXT,
  name TEXT,
  inferred_role TEXT,
  user_role TEXT NULL,
  n_missing INTEGER,
  n_unique INTEGER,
  is_constant INTEGER,
  sample_preview TEXT,
  FOREIGN KEY(version_id) REFERENCES dataset_versions(id)
);
CREATE TABLE IF NOT EXISTS experiments (
  id TEXT PRIMARY KEY,
  project_id TEXT,
  dataset_id TEXT,
  created_at TEXT,
  FOREIGN KEY(project_id) REFERENCES projects(id),
  FOREIGN KEY(dataset_id) REFERENCES datasets(id)
);
CREATE TABLE IF NOT EXISTS experiment_revisions (
  id TEXT PRIMARY KEY,
  experiment_id TEXT,
  revision INTEGER,
  config_json TEXT,
  created_at TEXT,
  FOREIGN KEY(experiment_id) REFERENCES experiments(id)
);
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY,
  project_id TEXT,
  experiment_revision_id TEXT NULL,
  model_id TEXT NULL,
  type TEXT,
  status TEXT,
  attempt_id TEXT,
  lease_until TEXT NULL,
  heartbeat_at TEXT NULL,
  progress_json TEXT,
  error TEXT NULL,
  created_at TEXT,
  started_at TEXT,
  finished_at TEXT,
  FOREIGN KEY(project_id) REFERENCES projects(id)
);
CREATE TABLE IF NOT EXISTS models (
  id TEXT PRIMARY KEY,
  project_id TEXT,
  job_id TEXT,
  dataset_version_id TEXT,
  task TEXT,
  artifact_path TEXT,
  metrics_json TEXT,
  created_at TEXT,
  FOREIGN KEY(project_id) REFERENCES projects(id),
  FOREIGN KEY(job_id) REFERENCES jobs(id),
  FOREIGN KEY(dataset_version_id) REFERENCES dataset_versions(id)
);
CREATE TABLE IF NOT EXISTS reports (
  id TEXT PRIMARY KEY,
  job_id TEXT,
  report_json TEXT,
  plot_dir TEXT,
  FOREIGN KEY(job_id) REFERENCES jobs(id)
);
CREATE TABLE IF NOT EXISTS predictions (
  id TEXT PRIMARY KEY,
  job_id TEXT,
  model_id TEXT,
  input_version_id TEXT,
  output_path TEXT,
  created_at TEXT,
  FOREIGN KEY(job_id) REFERENCES jobs(id),
  FOREIGN KEY(model_id) REFERENCES models(id),
  FOREIGN KEY(input_version_id) REFERENCES dataset_versions(id)
);
CREATE TABLE IF NOT EXISTS conversations (
  id TEXT PRIMARY KEY,
  project_id TEXT,
  created_at TEXT,
  FOREIGN KEY(project_id) REFERENCES projects(id)
);
CREATE TABLE IF NOT EXISTS messages (
  id TEXT PRIMARY KEY,
  conversation_id TEXT,
  role TEXT,
  content TEXT,
  client_id TEXT,
  status TEXT,
  provider TEXT NULL,
  model TEXT NULL,
  usage_json TEXT NULL,
  created_at TEXT,
  FOREIGN KEY(conversation_id) REFERENCES conversations(id)
);
CREATE TABLE IF NOT EXISTS sample_disclosure (
  dataset_id TEXT PRIMARY KEY,
  enabled INTEGER,
  previewed_at TEXT NULL,
  FOREIGN KEY(dataset_id) REFERENCES datasets(id)
);
CREATE TABLE IF NOT EXISTS schema_migrations (
  version INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  applied_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_dataset_versions_uniq
  ON dataset_versions(dataset_id, version);
CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_conversation_client
  ON messages(conversation_id, client_id);
CREATE INDEX IF NOT EXISTS idx_datasets_project ON datasets(project_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, created_at);
CREATE INDEX IF NOT EXISTS idx_columns_version ON columns(version_id);
CREATE INDEX IF NOT EXISTS idx_experiments_project ON experiments(project_id);
"""


def utcnow() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def new_id() -> str:
    return uuid.uuid4().hex


def connect(path: Path | None = None) -> sqlite3.Connection:
    db_path = Path(path) if path else get_settings().db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


@contextmanager
def get_db() -> Iterator[sqlite3.Connection]:
    conn = connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


get_connection = get_db


def row_dict(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    return {k: row[k] for k in row.keys()}  # noqa: SIM118 — sqlite3.Row iterates values


def init_db() -> None:
    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.datasets_dir.mkdir(parents=True, exist_ok=True)
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    with get_db() as conn:
        conn.executescript(SCHEMA)
        migrate(conn)
        _ensure_default_project(conn)


def _ensure_default_project(conn: sqlite3.Connection) -> None:
    existing = conn.execute(
        "SELECT id FROM projects WHERE name = ?", ("Local workspace",)
    ).fetchone()
    if existing:
        return
    pid = "local"
    if conn.execute("SELECT 1 FROM projects WHERE id = ?", (pid,)).fetchone():
        pid = new_id()
    conn.execute(
        "INSERT INTO projects (id, name, created_at) VALUES (?, ?, ?)",
        (pid, "Local workspace", utcnow()),
    )


def require_project(conn: sqlite3.Connection, project_id: str) -> dict:
    from app.models import AppError

    row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    if not row:
        raise AppError("project not found", status_code=404)
    return row_dict(row)  # type: ignore[return-value]


def migrate(conn: sqlite3.Connection | None = None) -> None:
    """Apply additive SQLite migrations. Safe on fresh and existing databases."""
    if conn is None:
        with get_db() as owned:
            _migrate(owned)
        return
    _migrate(conn)


def _migrate(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
          version INTEGER PRIMARY KEY,
          name TEXT NOT NULL,
          applied_at TEXT NOT NULL
        )
        """
    )
    applied = {
        int(row["version"])
        for row in conn.execute("SELECT version FROM schema_migrations").fetchall()
    }
    _migrate_needs_review(conn)
    _record_migration(conn, applied, 1, "datasets_needs_review")
    _migrate_messages_unique(conn)
    _record_migration(conn, applied, 2, "messages_client_id_per_conversation")


def _record_migration(
    conn: sqlite3.Connection, applied: set[int], version: int, name: str
) -> None:
    if version in applied:
        return
    conn.execute(
        "INSERT INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?)",
        (version, name, utcnow()),
    )
    applied.add(version)


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _migrate_needs_review(conn: sqlite3.Connection) -> None:
    if "datasets" not in _table_names(conn):
        return
    if "needs_review" not in _table_columns(conn, "datasets"):
        conn.execute(
            "ALTER TABLE datasets ADD COLUMN needs_review INTEGER NOT NULL DEFAULT 0"
        )


def _table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    return {row["name"] for row in rows}


def _migrate_messages_unique(conn: sqlite3.Connection) -> None:
    if "messages" not in _table_names(conn):
        return
    if _messages_has_global_client_id_unique(conn):
        conn.execute("PRAGMA foreign_keys=OFF")
        try:
            conn.executescript(
                """
                CREATE TABLE messages_new (
                  id TEXT PRIMARY KEY,
                  conversation_id TEXT,
                  role TEXT,
                  content TEXT,
                  client_id TEXT,
                  status TEXT,
                  provider TEXT NULL,
                  model TEXT NULL,
                  usage_json TEXT NULL,
                  created_at TEXT,
                  FOREIGN KEY(conversation_id) REFERENCES conversations(id)
                );
                INSERT INTO messages_new (
                  id, conversation_id, role, content, client_id, status,
                  provider, model, usage_json, created_at
                )
                SELECT
                  id, conversation_id, role, content, client_id, status,
                  provider, model, usage_json, created_at
                FROM messages;
                DROP TABLE messages;
                ALTER TABLE messages_new RENAME TO messages;
                """
            )
        finally:
            conn.execute("PRAGMA foreign_keys=ON")
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_conversation_client
        ON messages(conversation_id, client_id)
        """
    )


def _messages_has_global_client_id_unique(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='messages'"
    ).fetchone()
    sql = (row["sql"] if row else "") or ""
    compact = " ".join(sql.split())
    if "client_id TEXT UNIQUE" in compact:
        return True
    for idx in conn.execute("PRAGMA index_list(messages)").fetchall():
        if not idx["unique"]:
            continue
        cols = [
            info["name"]
            for info in conn.execute(
                f"PRAGMA index_info('{idx['name']}')"
            ).fetchall()
        ]
        if cols == ["client_id"]:
            return True
    return False
