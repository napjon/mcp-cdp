"""Public Google Sheets ingest. Never fetches the user-supplied URL."""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from urllib.parse import parse_qs, urljoin, urlparse

import httpx

from app.db import get_db, new_id, require_project, row_dict, utcnow
from app.models import AppError
from app.services.datasets import (
    IngestError,
    _decode_bytes,
    _set_needs_review,
    _sniff_delimiter,
    add_dataset_version,
    detect_encoding,
    parse_tabular,
)
from app.services.security import is_allowed_sheet_host
from app.settings import get_settings

_SHEET_ID_RE = re.compile(r"/spreadsheets/d/([a-zA-Z0-9_-]+)")
_GID_RE = re.compile(r"gid=(\d+)")
_LOGIN_HOSTS = frozenset(
    {
        "accounts.google.com",
        "account.google.com",
        "www.google.com",
        "google.com",
    }
)


@dataclass
class SheetRef:
    spreadsheet_id: str
    gid: str | None
    public_url: str


class SheetFetchError(AppError):
    pass


def parse_sheets_url(url: str) -> SheetRef:
    if not url or not str(url).strip():
        raise SheetFetchError("not a Google Sheet URL", code="not_a_sheet")
    raw = str(url).strip()
    parsed = urlparse(raw)
    host = (parsed.hostname or "").lower()
    if parsed.scheme and parsed.scheme not in {"http", "https"}:
        raise SheetFetchError("not a Google Sheet URL", code="not_a_sheet")
    if parsed.scheme and host not in {"docs.google.com", "spreadsheets.google.com"}:
        raise SheetFetchError("not a Google Sheet URL", code="not_a_sheet")
    match = _SHEET_ID_RE.search(parsed.path or "")
    if not match:
        raise SheetFetchError("not a Google Sheet URL", code="not_a_sheet")
    spreadsheet_id = match.group(1)
    gid = None
    query = parse_qs(parsed.query)
    if query.get("gid"):
        gid = str(query["gid"][0])
    if parsed.fragment:
        frag = _GID_RE.search(parsed.fragment)
        if frag:
            gid = frag.group(1)
    return SheetRef(spreadsheet_id=spreadsheet_id, gid=gid, public_url=raw)


def _require_gid(gid: str | int | None) -> str:
    if gid is None or str(gid).strip() == "":
        raise SheetFetchError(
            "This link does not identify a tab. Open the sheet tab and copy the URL that includes #gid=.",
            code="gid_unspecified",
        )
    gid_s = str(gid).strip()
    if not re.fullmatch(r"\d+", gid_s):
        raise SheetFetchError("invalid gid", code="not_a_sheet")
    return gid_s


def export_url(spreadsheet_id: str, gid: str | int | None) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", spreadsheet_id or ""):
        raise SheetFetchError("not a Google Sheet URL", code="not_a_sheet")
    gid_s = _require_gid(gid)
    return (
        f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?format=csv&gid={gid_s}"
    )


def download_export_csv(
    spreadsheet_id: str,
    gid: str | int | None,
    *,
    client: httpx.Client | None = None,
) -> bytes:
    """GET the canonical export URL only. Follows allowlisted redirects."""
    settings = get_settings()
    url = export_url(spreadsheet_id, gid)
    owns_client = client is None
    if owns_client:
        client = httpx.Client(follow_redirects=False, timeout=settings.sheets_timeout_s)
    try:
        return _download_with_redirects(client, url, settings.max_upload_bytes)
    except httpx.TimeoutException as exc:
        raise SheetFetchError("sheet request timed out", code="timeout") from exc
    except httpx.RequestError as exc:
        raise SheetFetchError("sheet request failed", code="timeout") from exc
    finally:
        if owns_client and client is not None:
            client.close()


def _download_with_redirects(client: httpx.Client, url: str, max_bytes: int) -> bytes:
    current = url
    for _ in range(8):
        _assert_request_url(current)
        response = client.get(current)
        if response.status_code in {301, 302, 303, 307, 308}:
            location = response.headers.get("location")
            if not location:
                raise SheetFetchError("sheet is private", code="private")
            current = urljoin(current, location)
            host = (urlparse(current).hostname or "").lower()
            if host in _LOGIN_HOSTS or not is_allowed_sheet_host(host):
                raise SheetFetchError("sheet is private", code="private")
            continue
        return _classify_body(response, max_bytes)
    raise SheetFetchError("sheet is private", code="private")


def _assert_request_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise SheetFetchError("sheet is private", code="private")
    if not is_allowed_sheet_host(parsed.hostname):
        raise SheetFetchError("sheet is private", code="private")


def _classify_body(response: httpx.Response, max_bytes: int) -> bytes:
    status = response.status_code
    if status in {401, 403}:
        raise SheetFetchError("sheet is private", code="private")
    if status == 404:
        raise SheetFetchError("sheet was revoked", code="revoked")
    if status >= 400:
        raise SheetFetchError("sheet is private", code="private")
    length = response.headers.get("content-length")
    if length and length.isdigit() and int(length) > max_bytes:
        raise SheetFetchError("sheet is too large", code="too_large")
    content = response.content
    if len(content) > max_bytes:
        raise SheetFetchError("sheet is too large", code="too_large")
    ctype = (response.headers.get("content-type") or "").lower()
    stripped = content.lstrip()
    looks_html = (
        "text/html" in ctype
        or stripped[:15].lower().startswith(b"<!doctype")
        or stripped[:6].lower().startswith(b"<html")
    )
    if looks_html:
        raise SheetFetchError("sheet is private", code="private")
    return content


def suggest_header_row(data: bytes, *, max_scan: int = 10) -> int:
    if not data or not data.strip():
        return 0
    try:
        text = _decode_bytes(data, detect_encoding(data))
    except (IngestError, UnicodeDecodeError):
        return 0
    stream = io.StringIO(text)
    sample = stream.read(16384)
    stream.seek(0)
    delimiter = _sniff_delimiter(sample)
    rows: list[list[str]] = []
    try:
        for i, row in enumerate(csv.reader(stream, delimiter=delimiter)):
            if i >= max_scan:
                break
            rows.append(row)
    except csv.Error:
        return 0
    if not rows:
        return 0
    best_i = 0
    best_score = float("-inf")
    for i, row in enumerate(rows):
        nonempty = [(c or "").strip() for c in row if (c or "").strip()]
        if len(nonempty) < 2:
            score = -100.0
        else:
            unique_ratio = len(set(nonempty)) / len(nonempty)
            numeric = 0
            for cell in nonempty:
                try:
                    float(cell.replace(",", ""))
                    numeric += 1
                except ValueError:
                    pass
            score = unique_ratio - (numeric / len(nonempty))
        if score > best_score:
            best_score = score
            best_i = i
    return best_i


def preview_sheet(
    project_id: str,
    *,
    url: str,
    gid: str | int | None = None,
    header_row: int | None = None,
    name: str | None = None,
) -> dict:
    with get_db() as conn:
        require_project(conn, project_id)
    ref = parse_sheets_url(url)
    use_gid = str(gid) if gid is not None and str(gid) != "" else ref.gid
    use_gid = _require_gid(use_gid)
    data = download_export_csv(ref.spreadsheet_id, use_gid)
    suggested = suggest_header_row(data)
    used = suggested if header_row is None else int(header_row)
    if used < 0:
        raise IngestError("invalid header_row")
    table = parse_tabular(data, header_row=used, filename=f"{name or 'sheet'}.csv")
    sample = table.rows[:20]
    return {
        "headers": table.headers,
        "columns": table.headers,
        "rows": sample,
        "suggested_header_row": suggested,
        "header_row": used,
        "n": len(sample),
        "n_rows": len(table.rows),
        "n_cols": len(table.headers),
        "spreadsheet_id": ref.spreadsheet_id,
        "gid": use_gid,
    }


def connect_sheet(
    project_id: str,
    *,
    url: str,
    gid: str | int | None = None,
    header_row: int | None = 0,
    name: str | None = None,
) -> dict:
    if header_row is None:
        header_row = 0
    ref = parse_sheets_url(url)
    use_gid = str(gid) if gid is not None and str(gid) != "" else ref.gid
    use_gid = _require_gid(use_gid)
    with get_db() as conn:
        require_project(conn, project_id)
        connection_id = new_id()
        now = utcnow()
        conn.execute(
            """
            INSERT INTO sheet_connections (
              id, project_id, spreadsheet_id, gid, sheet_name, public_url,
              last_status, last_checked_at, last_error, last_version_id
            ) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, NULL, NULL)
            """,
            (
                connection_id,
                project_id,
                ref.spreadsheet_id,
                use_gid,
                name,
                ref.public_url,
                now,
            ),
        )

    try:
        data = download_export_csv(ref.spreadsheet_id, use_gid)
        table = parse_tabular(
            data,
            header_row=header_row,
            filename=f"{name or 'sheet'}.csv",
        )
    except (SheetFetchError, IngestError) as exc:
        _mark_connection(connection_id, status=exc.code or "error", error=exc.message)
        raise

    dataset_name = name or "Google Sheet"
    # create_csv_dataset would mark source csv; insert sheets dataset here.
    from app.services.datasets import persist_version

    with get_db() as conn:
        dataset_id = new_id()
        now = utcnow()
        conn.execute(
            """
            INSERT INTO datasets (id, project_id, name, source_type, connection_id, created_at)
            VALUES (?, ?, ?, 'sheets', ?, ?)
            """,
            (dataset_id, project_id, dataset_name, connection_id, now),
        )
        version = persist_version(conn, dataset_id=dataset_id, table=table, version=1)
        conn.execute(
            """
            UPDATE sheet_connections
            SET last_status = 'ok', last_checked_at = ?, last_error = NULL,
                last_version_id = ?, gid = COALESCE(gid, ?)
            WHERE id = ?
            """,
            (utcnow(), version["id"], use_gid, connection_id),
        )
        from app.services.datasets import get_dataset

        return get_dataset(project_id, dataset_id, conn=conn)


def refresh_sheet(project_id: str, connection_id: str, *, header_row: int | None = None) -> dict:
    with get_db() as conn:
        require_project(conn, project_id)
        row = conn.execute(
            "SELECT * FROM sheet_connections WHERE id = ? AND project_id = ?",
            (connection_id, project_id),
        ).fetchone()
        if not row:
            raise AppError("sheet connection not found", status_code=404)
        connection = row_dict(row)
        dataset_row = row_dict(
            conn.execute(
                "SELECT * FROM datasets WHERE connection_id = ? AND project_id = ?",
                (connection_id, project_id),
            ).fetchone()
        )
        previous_version_id = connection.get("last_version_id")
        prev_cols: list[dict] | None = None
        if dataset_row is not None:
            latest = conn.execute(
                """
                SELECT id FROM dataset_versions
                WHERE dataset_id = ? ORDER BY version DESC LIMIT 1
                """,
                (dataset_row["id"],),
            ).fetchone()
            if latest:
                prev_cols = [
                    row_dict(r)
                    for r in conn.execute(
                        """
                        SELECT name, inferred_role, user_role
                        FROM columns WHERE version_id = ? ORDER BY rowid
                        """,
                        (latest["id"],),
                    ).fetchall()
                ]
        if header_row is None:
            header_row = 0
            if previous_version_id:
                ver = conn.execute(
                    "SELECT header_row FROM dataset_versions WHERE id = ?",
                    (previous_version_id,),
                ).fetchone()
                if ver is not None and ver["header_row"] is not None:
                    header_row = int(ver["header_row"])

    try:
        data = download_export_csv(connection["spreadsheet_id"], connection["gid"])
        table = parse_tabular(
            data,
            header_row=header_row,
            filename=f"{connection.get('sheet_name') or 'sheet'}.csv",
        )
    except (SheetFetchError, IngestError) as exc:
        _mark_connection(
            connection_id,
            status=exc.code or "error",
            error=exc.message,
            last_version_id=previous_version_id,
        )
        raise

    with get_db() as conn:
        if dataset_row is None:
            dataset_id = new_id()
            conn.execute(
                """
                INSERT INTO datasets (id, project_id, name, source_type, connection_id, created_at)
                VALUES (?, ?, ?, 'sheets', ?, ?)
                """,
                (
                    dataset_id,
                    project_id,
                    connection.get("sheet_name") or "Google Sheet",
                    connection_id,
                    utcnow(),
                ),
            )
        else:
            dataset_id = dataset_row["id"]
        changed = add_dataset_version(
            conn, project_id=project_id, dataset_id=dataset_id, table=table
        )
        latest = conn.execute(
            """
            SELECT id FROM dataset_versions WHERE dataset_id = ?
            ORDER BY version DESC LIMIT 1
            """,
            (dataset_id,),
        ).fetchone()
        if changed is not None and prev_cols is not None and latest is not None:
            _apply_refresh_roles(
                conn,
                dataset_id=dataset_id,
                new_version_id=latest["id"],
                prev_cols=prev_cols,
                new_stats=table.column_stats,
            )
        conn.execute(
            """
            UPDATE sheet_connections
            SET last_status = 'ok', last_checked_at = ?, last_error = NULL, last_version_id = ?
            WHERE id = ?
            """,
            (utcnow(), latest["id"] if latest else previous_version_id, connection_id),
        )
        from app.services.datasets import get_dataset

        payload = get_dataset(project_id, dataset_id, conn=conn)
        payload["unchanged"] = changed is None
        return payload


def _type_family(role: str | None) -> str:
    r = (role or "").strip().lower()
    # Keep categorical/text/identifier/constant distinct: each changes how
    # feature selection and preprocessing interpret a refreshed column.
    if r in {"numeric", "date", "categorical", "text", "identifier", "constant"}:
        return r
    return "text"


def _apply_refresh_roles(
    conn,
    *,
    dataset_id: str,
    new_version_id: str,
    prev_cols: list[dict],
    new_stats: list[dict],
) -> None:
    prev_by_name = {c["name"]: c for c in prev_cols}
    prev_names = [c["name"] for c in prev_cols]
    new_names = [c["name"] for c in new_stats]
    needs_review = list(new_names) != list(prev_names)
    for col in new_stats:
        name = col.get("name")
        prev = prev_by_name.get(name)
        if prev is None:
            needs_review = True
            continue
        new_inferred = col.get("inferred_role")
        old_inferred = prev.get("inferred_role")
        if _type_family(new_inferred) != _type_family(old_inferred):
            needs_review = True
        user_role = prev.get("user_role")
        if user_role:
            conn.execute(
                "UPDATE columns SET user_role = ? WHERE version_id = ? AND name = ?",
                (user_role, new_version_id, name),
            )
    if needs_review:
        _set_needs_review(conn, dataset_id, 1)


def _mark_connection(
    connection_id: str,
    *,
    status: str,
    error: str,
    last_version_id: str | None = None,
) -> None:
    with get_db() as conn:
        if last_version_id is not None:
            conn.execute(
                """
                UPDATE sheet_connections
                SET last_status = ?, last_checked_at = ?, last_error = ?, last_version_id = ?
                WHERE id = ?
                """,
                (status, utcnow(), error, last_version_id, connection_id),
            )
        else:
            conn.execute(
                """
                UPDATE sheet_connections
                SET last_status = ?, last_checked_at = ?, last_error = ?
                WHERE id = ?
                """,
                (status, utcnow(), error, connection_id),
            )


connect_google_sheet = connect_sheet
connect = connect_sheet
refresh_google_sheet = refresh_sheet
refresh = refresh_sheet


def get_connection(project_id: str, connection_id: str) -> dict:
    with get_db() as conn:
        require_project(conn, project_id)
        row = conn.execute(
            "SELECT * FROM sheet_connections WHERE id = ? AND project_id = ?",
            (connection_id, project_id),
        ).fetchone()
        if not row:
            raise AppError("sheet connection not found", status_code=404)
        return row_dict(row)
