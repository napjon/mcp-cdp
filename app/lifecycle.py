"""Backup, restore, project deletion, and thirty-day chat retention."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sqlite3
import sys
import tarfile
import tempfile
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.db import get_db, init_db, migrate, require_project
from app.models import AppError
from app.services.jobs import terminate_job_process
from app.settings import get_settings

log = logging.getLogger("mcp_cdp.lifecycle")

RETENTION_INTERVAL_S = 6 * 60 * 60
_retention_run_lock = threading.Lock()


def backup(dest: Path | None = None) -> dict:
    settings = get_settings()
    backups = settings.data_dir / "backups"
    backups.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    dest = Path(dest) if dest is not None else backups / f"mcp_cdp-{stamp}.tar.gz"
    dest = dest.resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    created = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    db_name = settings.db_path.name
    manifest = {
        "created": created,
        "db": db_name,
        "datasets": "datasets",
        "artifacts": "artifacts",
    }
    with tempfile.TemporaryDirectory(prefix=".backup-staging-", dir=str(dest.parent)) as tmp:
        staging = Path(tmp)
        with get_db() as conn:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            _snapshot_sqlite(conn, staging / db_name)
        _copy_tree(settings.datasets_dir, staging / "datasets")
        _copy_tree(settings.artifacts_dir, staging / "artifacts")
        (staging / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        _atomic_write_archive(staging, dest)
    return {"ok": True, "path": str(dest), "manifest": manifest}


def restore(path: str | Path | None = None) -> dict:
    settings = get_settings()
    src = Path(path) if path is not None else _latest_backup()
    if src is None or not src.exists():
        raise AppError("backup not found", status_code=404)
    src = src.resolve()
    dest_db = settings.db_path
    dest_db.parent.mkdir(parents=True, exist_ok=True)
    if dest_db.exists():
        with get_db() as conn:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    with tempfile.TemporaryDirectory(prefix=".restore-staging-", dir=str(settings.data_dir)) as tmp:
        root = _unpack_backup(src, Path(tmp))
        manifest = _read_manifest(root)
        db_src = root / str(manifest.get("db") or settings.db_path.name)
        if not db_src.is_file():
            raise AppError("backup not found", status_code=404)
        datasets_src = root / str(manifest.get("datasets") or "datasets")
        artifacts_src = root / str(manifest.get("artifacts") or "artifacts")
        _replace_sqlite(db_src, dest_db)
        _replace_tree(datasets_src if datasets_src.exists() else None, settings.datasets_dir)
        _replace_tree(artifacts_src if artifacts_src.exists() else None, settings.artifacts_dir)
    return {"ok": True, "path": str(dest_db), "manifest": manifest}


def _latest_backup() -> Path | None:
    backups = get_settings().data_dir / "backups"
    if not backups.is_dir():
        return None
    cands: list[Path] = [
        p
        for p in backups.iterdir()
        if (p.is_file() and _is_tar_path(p))
        or (p.is_dir() and (p / "manifest.json").is_file())
    ]
    if not cands:
        cands = [p for p in backups.glob("*.sqlite") if p.is_file()]
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def _is_tar_path(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith((".tar.gz", ".tgz")) or tarfile.is_tarfile(path)


def _snapshot_sqlite(conn: sqlite3.Connection, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    snap = sqlite3.connect(str(dest))
    try:
        conn.backup(snap)
        snap.commit()
    finally:
        snap.close()


def _replace_sqlite(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.restore-tmp")
    if tmp.exists():
        tmp.unlink()
    shutil.copy2(src, tmp)
    os.replace(tmp, dest)
    for suffix in ("-wal", "-shm"):
        Path(str(dest) + suffix).unlink(missing_ok=True)


def _copy_tree(src: Path, dest: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dest, dirs_exist_ok=True)
    else:
        dest.mkdir(parents=True, exist_ok=True)


def _replace_tree(src: Path | None, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.restore-tmp")
    old = dest.with_name(f".{dest.name}.restore-old")
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    if src is not None and src.is_dir():
        shutil.copytree(src, tmp)
    else:
        tmp.mkdir(parents=True)
    if old.exists():
        shutil.rmtree(old, ignore_errors=True)
    if dest.exists():
        dest.rename(old)
    tmp.rename(dest)
    if old.exists():
        shutil.rmtree(old, ignore_errors=True)


def _atomic_write_archive(staging: Path, dest: Path) -> None:
    name = dest.name.lower()
    if name.endswith((".tar.gz", ".tgz")):
        tmp = dest.parent / f".{dest.name}.{os.getpid()}.partial"
        try:
            with tarfile.open(tmp, "w:gz") as tar:
                for item in sorted(staging.iterdir(), key=lambda p: p.name):
                    tar.add(item, arcname=item.name)
            os.replace(tmp, dest)
        finally:
            tmp.unlink(missing_ok=True)
        return
    _atomic_replace_dir(staging, dest)


def _atomic_replace_dir(staging: Path, dest: Path) -> None:
    new = dest.parent / f".{dest.name}.{os.getpid()}.new"
    old = dest.parent / f".{dest.name}.{os.getpid()}.old"
    if new.exists():
        shutil.rmtree(new, ignore_errors=True)
    shutil.copytree(staging, new)
    if old.exists():
        shutil.rmtree(old, ignore_errors=True)
    if dest.exists():
        dest.rename(old)
    new.rename(dest)
    if old.exists():
        shutil.rmtree(old, ignore_errors=True)


def _unpack_backup(src: Path, tmp: Path) -> Path:
    if src.is_dir():
        if (src / "manifest.json").is_file():
            return src
        raise AppError("backup not found", status_code=404)
    if tarfile.is_tarfile(src):
        with tarfile.open(src, "r:*") as tar:
            _safe_extract(tar, tmp)
        return _archive_root(tmp)
    raise AppError("backup not found", status_code=404)


def _safe_extract(tar: tarfile.TarFile, dest: Path) -> None:
    dest_r = dest.resolve()
    members = []
    for member in tar.getmembers():
        name = member.name.replace("\\", "/")
        if name.startswith(("/", "..")) or "/../" in f"/{name}/":
            raise AppError("invalid backup archive")
        target = (dest / name).resolve()
        if target != dest_r and dest_r not in target.parents:
            raise AppError("invalid backup archive")
        members.append(member)
    tar.extractall(dest, members=members)


def _archive_root(extracted: Path) -> Path:
    if (extracted / "manifest.json").is_file():
        return extracted
    kids = [p for p in extracted.iterdir() if p.name != "__MACOSX"]
    if len(kids) == 1 and kids[0].is_dir() and (kids[0] / "manifest.json").is_file():
        return kids[0]
    found = list(extracted.rglob("manifest.json"))
    if len(found) == 1:
        return found[0].parent
    raise AppError("backup not found", status_code=404)


def _read_manifest(root: Path) -> dict:
    path = root / "manifest.json"
    if not path.is_file():
        return {"created": None, "db": "mcp_cdp.sqlite", "datasets": "datasets", "artifacts": "artifacts"}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AppError("invalid backup archive") from exc
    if not isinstance(data, dict):
        raise AppError("invalid backup archive")
    return data


def delete_project(project_id: str) -> dict:
    settings = get_settings()
    with get_db() as conn:
        require_project(conn, project_id)
        dataset_ids = _ids(conn, "SELECT id FROM datasets WHERE project_id = ?", (project_id,))
        job_ids = _ids(conn, "SELECT id FROM jobs WHERE project_id = ?", (project_id,))
        # Stop active work before removing the rows and files it may still use.
        for job_id in job_ids:
            terminate_job_process(job_id)
        if job_ids:
            conn.execute(
                "UPDATE jobs SET status = 'canceled', finished_at = ?, error = NULL "
                "WHERE project_id = ? AND status IN ('queued', 'running')",
                (datetime.now(UTC).isoformat().replace('+00:00', 'Z'), project_id),
            )
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


def purge(days: int = 30) -> dict:
    """Delete chat traces older than `days`. Safe to call repeatedly."""
    return apply_retention(days=days)


def maybe_run_retention(days: int = 30) -> dict:
    """Idempotent retention entrypoint for startup and the background timer."""
    with _retention_run_lock:
        return purge(days=days)


def start_retention_loop(
    interval_s: float = RETENTION_INTERVAL_S,
) -> tuple[threading.Event, threading.Thread]:
    stop = threading.Event()

    def _loop() -> None:
        while not stop.wait(interval_s):
            try:
                maybe_run_retention()
            except Exception:
                log.exception("retention failed")

    thread = threading.Thread(target=_loop, name="mcp-cdp-retention", daemon=True)
    thread.start()
    return stop, thread


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
        result = purge(days=args.days)
        print(
            f"{result['conversations_deleted']} conversations, "
            f"{result['messages_deleted']} messages"
        )
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
