"""Backup, restore, project deletion, and thirty-day chat retention."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.db import get_db, init_db, migrate, require_project
from app.models import AppError
from app.settings import get_settings


def backup(dest: Path | None = None) -> dict:
    settings = get_settings()
    backups = settings.data_dir / "backups"
    backups.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    dest = Path(dest) if dest is not None else backups / f"mcp_cdp-{stamp}.sqlite"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with get_db() as conn:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    shutil.copy2(settings.db_path, dest)
    return {"ok": True, "path": str(dest)}


def restore(path: str | Path | None = None) -> dict:
    settings = get_settings()
    src = Path(path) if path is not None else _latest_backup()
    if src is None or not src.is_file():
        raise AppError("backup not found", status_code=404)
    dest = settings.db_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        with get_db() as conn:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    shutil.copy2(src, dest)
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(dest) + suffix)
        if sidecar.exists():
            sidecar.unlink()
    return {"ok": True, "path": str(dest)}


def _latest_backup() -> Path | None:
    backups = get_settings().data_dir / "backups"
    if not backups.is_dir():
        return None
    files = [p for p in backups.glob("*.sqlite") if p.is_file()]
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def delete_project(project_id: str) -> dict:
    settings = get_settings()
    with get_db() as conn:
        require_project(conn, project_id)
        dataset_ids = _ids(conn, "SELECT id FROM datasets WHERE project_id = ?", (project_id,))
        job_ids = _ids(conn, "SELECT id FROM jobs WHERE project_id = ?", (project_id,))
        conv_ids = _ids(
            conn, "SELECT id FROM conversations WHERE project_id = ?", (project_id,)
        )
        experiment_ids = _ids(
            conn, "SELECT id FROM experiments WHERE project_id = ?", (project_id,)
        )
        version_ids: list[str] = []
        if dataset_ids:
            q, params = _in_clause(dataset_ids)
            version_ids = _ids(
                conn, f"SELECT id FROM dataset_versions WHERE dataset_id IN ({q})", params
            )
        predict_dirs: list[Path] = []
        for row in conn.execute(
            "SELECT progress_json FROM jobs WHERE project_id = ?", (project_id,)
        ).fetchall():
            raw = row["progress_json"]
            if not raw:
                continue
            try:
                progress = json.loads(raw)
            except json.JSONDecodeError:
                continue
            path = progress.get("predict_path")
            if path:
                predict_dirs.append(Path(path).parent)

        if conv_ids:
            q, params = _in_clause(conv_ids)
            conn.execute(f"DELETE FROM messages WHERE conversation_id IN ({q})", params)
        conn.execute("DELETE FROM conversations WHERE project_id = ?", (project_id,))
        if dataset_ids:
            q, params = _in_clause(dataset_ids)
            conn.execute(f"DELETE FROM sample_disclosure WHERE dataset_id IN ({q})", params)
        if job_ids:
            q, params = _in_clause(job_ids)
            conn.execute(f"DELETE FROM reports WHERE job_id IN ({q})", params)
            conn.execute(f"DELETE FROM predictions WHERE job_id IN ({q})", params)
        conn.execute("DELETE FROM models WHERE project_id = ?", (project_id,))
        if version_ids:
            q, params = _in_clause(version_ids)
            conn.execute(f"DELETE FROM columns WHERE version_id IN ({q})", params)
        conn.execute("DELETE FROM jobs WHERE project_id = ?", (project_id,))
        if experiment_ids:
            q, params = _in_clause(experiment_ids)
            conn.execute(
                f"DELETE FROM experiment_revisions WHERE experiment_id IN ({q})", params
            )
        conn.execute("DELETE FROM experiments WHERE project_id = ?", (project_id,))
        if dataset_ids:
            q, params = _in_clause(dataset_ids)
            conn.execute(f"DELETE FROM dataset_versions WHERE dataset_id IN ({q})", params)
        conn.execute("DELETE FROM datasets WHERE project_id = ?", (project_id,))
        conn.execute("DELETE FROM sheet_connections WHERE project_id = ?", (project_id,))
        conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))

    for did in dataset_ids:
        _rm_under(settings.datasets_dir / did, settings.data_dir)
    for jid in job_ids:
        _rm_under(settings.artifacts_dir / jid, settings.data_dir)
    for folder in predict_dirs:
        _rm_under(folder, settings.data_dir)
    return {"ok": True, "id": project_id}


def apply_retention(days: int = 30) -> dict:
    cutoff = datetime.now(UTC) - timedelta(days=days)
    with get_db() as conn:
        old_messages = [
            row["id"]
            for row in conn.execute("SELECT id, created_at FROM messages").fetchall()
            if _is_older(row["created_at"], cutoff)
        ]
        old_conversations = [
            row["id"]
            for row in conn.execute("SELECT id, created_at FROM conversations").fetchall()
            if _is_older(row["created_at"], cutoff)
        ]
        deleted_messages = 0
        if old_messages:
            q, params = _in_clause(old_messages)
            cur = conn.execute(f"DELETE FROM messages WHERE id IN ({q})", params)
            deleted_messages += cur.rowcount
        if old_conversations:
            q, params = _in_clause(old_conversations)
            cur = conn.execute(
                f"DELETE FROM messages WHERE conversation_id IN ({q})", params
            )
            deleted_messages += cur.rowcount
            cur = conn.execute(f"DELETE FROM conversations WHERE id IN ({q})", params)
            deleted_conversations = cur.rowcount
        else:
            deleted_conversations = 0
    return {
        "ok": True,
        "cutoff": cutoff.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "messages_deleted": deleted_messages,
        "conversations_deleted": deleted_conversations,
    }


def _ids(conn, sql: str, params: tuple | list) -> list[str]:
    return [row["id"] for row in conn.execute(sql, params).fetchall()]


def _in_clause(ids: list[str]) -> tuple[str, list[str]]:
    return ",".join("?" * len(ids)), list(ids)


def _rm_under(path: Path, root: Path) -> None:
    try:
        resolved = path.resolve()
        root_r = root.resolve()
    except OSError:
        return
    if resolved == root_r or root_r not in resolved.parents:
        return
    if resolved.is_dir():
        shutil.rmtree(resolved, ignore_errors=True)
    elif resolved.is_file():
        resolved.unlink(missing_ok=True)


def _is_older(value: str | None, cutoff: datetime) -> bool:
    ts = _parse_ts(value)
    if ts is None:
        return False
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts < cutoff


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.lifecycle")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("migrate")
    p_backup = sub.add_parser("backup")
    p_backup.add_argument("path", nargs="?")
    p_restore = sub.add_parser("restore")
    p_restore.add_argument("path", nargs="?")
    p_del = sub.add_parser("delete-project")
    p_del.add_argument("project_id")
    p_ret = sub.add_parser("retention")
    p_ret.add_argument("--days", type=int, default=30)
    args = parser.parse_args(argv)
    if args.cmd == "migrate":
        init_db()
        migrate()
        print("ok")
        return 0
    if args.cmd == "backup":
        print(backup(Path(args.path) if args.path else None)["path"])
        return 0
    if args.cmd == "restore":
        print(restore(args.path)["path"])
        return 0
    if args.cmd == "delete-project":
        print(delete_project(args.project_id)["id"])
        return 0
    if args.cmd == "retention":
        result = apply_retention(days=args.days)
        print(
            f"{result['conversations_deleted']} conversations, "
            f"{result['messages_deleted']} messages"
        )
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
