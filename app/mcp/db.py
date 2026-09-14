"""SQLite access via platform helper when present, else the contracted file."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from app.mcp.config import sqlite_path


def _row_dict(row: Any) -> dict[str, Any]:
    if row is None:
        return {}
    if isinstance(row, dict):
        return dict(row)
    try:
        from app.db import row_dict

        converted = row_dict(row)
        if converted is not None:
            return converted
    except ImportError:
        pass
    if hasattr(row, "keys"):
        return {k: row[k] for k in row.keys()}  # noqa: SIM118 — sqlite3.Row iterates values
    return dict(row)


@contextmanager
def db_conn() -> Iterator[sqlite3.Connection]:
    try:
        from app import db as app_db  # type: ignore
    except ImportError:
        app_db = None

    if app_db is not None:
        for name in ("get_db", "get_connection", "connect", "connection"):
            fn = getattr(app_db, name, None)
            if not callable(fn):
                continue
            conn = fn()
            if hasattr(conn, "__enter__"):
                with conn as inner:
                    yield inner
                return
            try:
                if hasattr(conn, "row_factory") and conn.row_factory is None:
                    conn.row_factory = sqlite3.Row
                yield conn
                if hasattr(conn, "commit"):
                    conn.commit()
            finally:
                closer = getattr(app_db, "put_connection", None)
                if callable(closer):
                    closer(conn)
            return

    path = sqlite_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def fetchall(sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    with db_conn() as conn:
        try:
            cur = conn.execute(sql, params)
        except sqlite3.OperationalError:
            return []
        return [_row_dict(r) for r in cur.fetchall()]


def fetchone(sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
    rows = fetchall(sql, params)
    return rows[0] if rows else None


def execute(sql: str, params: tuple[Any, ...] = ()) -> None:
    with db_conn() as conn:
        conn.execute(sql, params)
        conn.commit()
