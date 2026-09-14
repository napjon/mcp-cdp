"""CSV ingest, profiling, dataset versions."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from app.db import get_db, new_id, require_project, row_dict, utcnow
from app.models import VALID_ROLES, AppError
from app.settings import get_settings

MISSING_TOKENS = frozenset({"", "na", "n/a", "nan", "null", "none", "#n/a"})
_ID_NAME_RE = re.compile(r"(^id$|_id$|^sku$|uuid|identifier)", re.IGNORECASE)
_DATE_NAME_RE = re.compile(r"date|time|timestamp|dt$", re.IGNORECASE)
_LEADING_ZERO_RE = re.compile(r"^0\d+$")
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
)


class IngestError(AppError):
    pass


@dataclass
class ParsedTable:
    encoding: str
    delimiter: str
    raw_headers: list[str]
    headers: list[str]
    rows: list[list[str]]
    import_meta: dict
    content_hash: str
    original_bytes: bytes
    header_row: int
    profile: dict = field(default_factory=dict)
    column_stats: list[dict] = field(default_factory=list)


def parse_tabular(
    data: bytes,
    *,
    header_row: int = 0,
    filename: str = "upload.csv",
    max_bytes: int | None = None,
    max_rows: int | None = None,
) -> ParsedTable:
    settings = get_settings()
    max_bytes = settings.max_upload_bytes if max_bytes is None else max_bytes
    max_rows = settings.max_rows if max_rows is None else max_rows
    if not data or not data.strip():
        raise IngestError("empty file")
    if len(data) > max_bytes:
        raise IngestError("file too large", code="too_large")
    if header_row < 0:
        raise IngestError("invalid header_row")

    encoding = detect_encoding(data)
    try:
        text = _decode_bytes(data, encoding)
    except UnicodeDecodeError as exc:
        raise IngestError("unable to decode file") from exc
    if "\x00" in text:
        raise IngestError("malformed CSV")
    if not text.strip():
        raise IngestError("empty file")

    stream = io.StringIO(text)
    for _ in range(header_row):
        line = stream.readline()
        if line == "":
            raise IngestError("header row out of range")

    sample_pos = stream.tell()
    sample = stream.read(16384)
    stream.seek(sample_pos)
    delimiter = _sniff_delimiter(sample)

    reader = csv.reader(stream, delimiter=delimiter, skipinitialspace=False)
    try:
        raw_headers = next(reader)
    except csv.Error as exc:
        raise IngestError("malformed CSV") from exc
    except StopIteration:
        raise IngestError("empty file") from None
    if not raw_headers or all(not (h or "").strip() for h in raw_headers):
        raise IngestError("empty file")

    headers, header_meta = resolve_headers(raw_headers)
    rows: list[list[str]] = []
    n_cols = len(raw_headers)
    try:
        for raw in reader:
            if not raw or all(c == "" for c in raw):
                continue
            if len(raw) != n_cols:
                raise IngestError("inconsistent column counts")
            rows.append(raw)
            if len(rows) > max_rows:
                raise IngestError(f"too many rows (max {max_rows})", code="too_many_rows")
    except csv.Error as exc:
        raise IngestError("malformed CSV") from exc

    if not rows:
        raise IngestError("empty file")

    table = ParsedTable(
        encoding=encoding,
        delimiter=delimiter,
        raw_headers=list(raw_headers),
        headers=headers,
        rows=rows,
        import_meta={
            "filename": filename,
            **header_meta,
            "header_row": header_row,
        },
        content_hash=hashlib.sha256(data).hexdigest(),
        original_bytes=data,
        header_row=header_row,
    )
    _profile(table)
    return table


def detect_encoding(data: bytes) -> str:
    if data.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if data.startswith(b"\xff\xfe"):
        return "utf-16-le"
    if data.startswith(b"\xfe\xff"):
        return "utf-16-be"
    try:
        data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise IngestError("unable to decode file") from exc
    return "utf-8"


def _decode_bytes(data: bytes, encoding: str) -> str:
    if encoding in {"utf-16-le", "utf-16-be"}:
        return data.decode("utf-16")
    return data.decode(encoding)


def _sniff_delimiter(sample: str) -> str:
    if not sample.strip():
        return ","
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        if dialect.delimiter:
            return dialect.delimiter
    except csv.Error:
        pass
    return ","


def resolve_headers(raw: list[str]) -> tuple[list[str], dict]:
    seen: dict[str, int] = {}
    resolved: list[str] = []
    renames: list[dict] = []
    for i, original in enumerate(raw):
        name = (original or "").strip()
        if not name:
            name = f"unnamed_{i + 1}"
        base = name
        if base in seen:
            seen[base] += 1
            name = f"{base}_{seen[base]}"
        else:
            seen[base] = 1
        while name in resolved:
            seen[base] += 1
            name = f"{base}_{seen[base]}"
        resolved.append(name)
        if name != original:
            renames.append({"index": i, "original": original, "resolved": name})
    return resolved, {
        "original_headers": list(raw),
        "resolved_headers": resolved,
        "renames": renames,
    }


def _is_missing(value: str) -> bool:
    return value.strip().lower() in MISSING_TOKENS if value is not None else True


def _profile(table: ParsedTable) -> None:
    frame = pd.DataFrame(table.rows, columns=table.headers, dtype="string")
    n_rows = len(frame)
    n_duplicate_rows = int(frame.duplicated().sum())
    column_stats: list[dict] = []
    n_constant = 0
    for name in table.headers:
        series = frame[name]
        missing_mask = series.isna() | series.map(lambda v: _is_missing(str(v)) if pd.notna(v) else True)
        n_missing = int(missing_mask.sum())
        present = series.loc[~missing_mask].astype(str)
        n_unique = int(present.nunique())
        is_constant = n_unique <= 1
        if is_constant:
            n_constant += 1
        role = infer_role(name, present, n_rows=n_rows, n_unique=n_unique, is_constant=is_constant)
        samples = [str(v)[:80] for v in list(present.unique())[:5]]
        column_stats.append(
            {
                "name": name,
                "inferred_role": role,
                "n_missing": n_missing,
                "n_unique": n_unique,
                "is_constant": int(is_constant),
                "sample_preview": ", ".join(samples),
            }
        )
    table.column_stats = column_stats
    table.import_meta["n_duplicate_rows"] = n_duplicate_rows
    table.import_meta["n_constant_columns"] = n_constant
    table.profile = {
        "n_rows": n_rows,
        "n_cols": len(table.headers),
        "n_duplicate_rows": n_duplicate_rows,
        "n_constant_columns": n_constant,
        "columns": column_stats,
    }


def infer_role(
    name: str,
    present: pd.Series,
    *,
    n_rows: int,
    n_unique: int,
    is_constant: bool,
) -> str:
    if is_constant:
        return "constant"
    values = present.astype(str).tolist()
    n = len(values)
    if n == 0:
        return "text"
    leading_zeros = sum(1 for v in values if _LEADING_ZERO_RE.match(v.strip()))
    uuidish = sum(1 for v in values if _UUID_RE.match(v.strip()))
    if leading_zeros / n >= 0.5 or uuidish / n >= 0.8:
        return "identifier"
    if _ID_NAME_RE.search(name) and n_unique == n_rows:
        return "identifier"

    date_hits = _datetime_hits(values)
    numeric_hits = _numeric_hits(values)
    date_named = bool(_DATE_NAME_RE.search(name))
    if date_named and date_hits / n >= 0.6:
        return "date"
    if date_hits / n >= 0.9 and numeric_hits / n < 0.9:
        return "date"
    if numeric_hits / n >= 0.9 and leading_zeros / n < 0.5:
        return "numeric"
    if date_hits / n >= 0.9:
        return "date"
    if n_unique <= 50 and (n_unique / max(n_rows, 1)) <= 0.2:
        return "categorical"
    return "text"


def _numeric_hits(values: list[str]) -> int:
    hits = 0
    for raw in values:
        s = raw.strip().replace(",", "")
        if s in {"", ".", "-", "+", "inf", "+inf", "-inf"}:
            continue
        try:
            float(s)
            hits += 1
        except ValueError:
            continue
    return hits


def _datetime_hits(values: list[str]) -> int:
    try:
        parsed = pd.to_datetime(pd.Series(values), errors="coerce", format="mixed")
        return int(parsed.notna().sum())
    except (ValueError, TypeError):
        return 0


def write_normalized_csv(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(headers)
        writer.writerows(rows)


def persist_version(
    conn,
    *,
    dataset_id: str,
    table: ParsedTable,
    version: int,
) -> dict:
    settings = get_settings()
    version_id = new_id()
    folder = settings.datasets_dir / dataset_id
    folder.mkdir(parents=True, exist_ok=True)
    original_path = folder / f"v{version}-original.csv"
    normalized_path = folder / f"v{version}-normalized.csv"
    original_path.write_bytes(table.original_bytes)
    write_normalized_csv(normalized_path, table.headers, table.rows)
    now = utcnow()
    conn.execute(
        """
        INSERT INTO dataset_versions (
          id, dataset_id, version, content_hash, original_path, normalized_path,
          n_rows, n_cols, header_row, encoding, delimiter, import_meta_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            version_id,
            dataset_id,
            version,
            table.content_hash,
            str(original_path),
            str(normalized_path),
            len(table.rows),
            len(table.headers),
            table.header_row,
            table.encoding,
            table.delimiter,
            json.dumps(table.import_meta),
            now,
        ),
    )
    for col in table.column_stats:
        conn.execute(
            """
            INSERT INTO columns (
              id, version_id, name, inferred_role, user_role, n_missing, n_unique,
              is_constant, sample_preview
            ) VALUES (?, ?, ?, ?, NULL, ?, ?, ?, ?)
            """,
            (
                new_id(),
                version_id,
                col["name"],
                col["inferred_role"],
                col["n_missing"],
                col["n_unique"],
                col["is_constant"],
                col["sample_preview"],
            ),
        )
    return {"id": version_id, "version": version}


def create_csv_dataset(
    project_id: str,
    data: bytes,
    *,
    filename: str,
    name: str | None = None,
    header_row: int = 0,
) -> dict:
    table = parse_tabular(data, header_row=header_row, filename=filename)
    dataset_name = name or _stem(filename)
    with get_db() as conn:
        require_project(conn, project_id)
        dataset_id = new_id()
        now = utcnow()
        conn.execute(
            """
            INSERT INTO datasets (id, project_id, name, source_type, connection_id, created_at)
            VALUES (?, ?, ?, 'csv', NULL, ?)
            """,
            (dataset_id, project_id, dataset_name, now),
        )
        persist_version(conn, dataset_id=dataset_id, table=table, version=1)
    return get_dataset(project_id, dataset_id)


def add_dataset_version(
    conn,
    *,
    project_id: str,
    dataset_id: str,
    table: ParsedTable,
) -> dict | None:
    """Append a version unless content_hash matches the latest. Returns None if unchanged."""
    latest = conn.execute(
        """
        SELECT id, version, content_hash FROM dataset_versions
        WHERE dataset_id = ? ORDER BY version DESC LIMIT 1
        """,
        (dataset_id,),
    ).fetchone()
    if latest and latest["content_hash"] == table.content_hash:
        return None
    next_version = (latest["version"] + 1) if latest else 1
    persist_version(conn, dataset_id=dataset_id, table=table, version=next_version)
    return get_dataset(project_id, dataset_id, conn=conn)


def _stem(filename: str) -> str:
    name = Path(filename or "dataset").name
    stem = Path(name).stem.strip() or "dataset"
    return stem[:200]


def latest_version_row(conn, dataset_id: str):
    return conn.execute(
        """
        SELECT * FROM dataset_versions
        WHERE dataset_id = ? ORDER BY version DESC LIMIT 1
        """,
        (dataset_id,),
    ).fetchone()


def list_projects() -> dict:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM projects ORDER BY created_at ASC"
        ).fetchall()
        return {"projects": [row_dict(r) for r in rows]}


def get_project(project_id: str) -> dict:
    with get_db() as conn:
        row = require_project(conn, project_id)
        datasets = conn.execute(
            """
            SELECT id, name, source_type, created_at
            FROM datasets WHERE project_id = ?
            ORDER BY created_at ASC
            """,
            (project_id,),
        ).fetchall()
        row["datasets"] = [row_dict(r) for r in datasets]
        row["ok"] = True
        return row


def list_datasets(project_id: str) -> list[dict]:
    with get_db() as conn:
        require_project(conn, project_id)
        rows = conn.execute(
            """
            SELECT d.*, v.version, v.n_rows, v.n_cols, v.content_hash, v.id AS version_id
            FROM datasets d
            LEFT JOIN dataset_versions v
              ON v.id = (
                SELECT id FROM dataset_versions
                WHERE dataset_id = d.id ORDER BY version DESC LIMIT 1
              )
            WHERE d.project_id = ?
            ORDER BY d.created_at DESC
            """,
            (project_id,),
        ).fetchall()
        return [row_dict(r) for r in rows]


def get_dataset(project_id: str, dataset_id: str, conn=None) -> dict:
    def _load(c) -> dict:
        require_project(c, project_id)
        ds = c.execute(
            "SELECT * FROM datasets WHERE id = ? AND project_id = ?",
            (dataset_id, project_id),
        ).fetchone()
        if not ds:
            raise AppError("dataset not found", status_code=404)
        out = row_dict(ds)
        version = latest_version_row(c, dataset_id)
        if version:
            payload = _version_payload(c, version)
            out["current_version"] = payload
            out["version"] = payload
            profile = payload.get("profile") or {}
            out["profile"] = profile
            out["columns"] = profile.get("columns") or []
        else:
            out["current_version"] = None
            out["version"] = None
            out["profile"] = None
            out["columns"] = []
        if ds["connection_id"]:
            conn_row = c.execute(
                "SELECT * FROM sheet_connections WHERE id = ?", (ds["connection_id"],)
            ).fetchone()
            out["connection"] = row_dict(conn_row)
        return out

    if conn is not None:
        return _load(conn)
    with get_db() as c:
        return _load(c)


def _version_payload(conn, version_row) -> dict:
    payload = row_dict(version_row)
    meta = {}
    if version_row["import_meta_json"]:
        try:
            meta = json.loads(version_row["import_meta_json"])
        except json.JSONDecodeError:
            meta = {}
    cols = conn.execute(
        "SELECT name, inferred_role, user_role, n_missing, n_unique, is_constant, sample_preview "
        "FROM columns WHERE version_id = ? ORDER BY rowid",
        (version_row["id"],),
    ).fetchall()
    columns = [row_dict(c) for c in cols]
    payload["import_meta"] = meta
    payload.pop("import_meta_json", None)
    payload["profile"] = {
        "n_rows": version_row["n_rows"],
        "n_cols": version_row["n_cols"],
        "n_duplicate_rows": meta.get("n_duplicate_rows", 0),
        "n_constant_columns": meta.get("n_constant_columns", 0),
        "columns": columns,
    }
    return payload


def preview_dataset(project_id: str, dataset_id: str, n: int = 20) -> dict:
    n = max(1, min(int(n), 100))
    with get_db() as conn:
        ds = get_dataset(project_id, dataset_id, conn=conn)
        version = ds.get("current_version")
        if not version:
            raise AppError("dataset has no versions", status_code=404)
        path = Path(version["normalized_path"])
        if not path.is_file():
            raise AppError("normalized file missing", status_code=404)
        n_rows = int(version.get("n_rows") or 0)
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            headers = next(reader, [])
            rows = []
            for i, row in enumerate(reader):
                if i >= n:
                    break
                rows.append(row)
        return {
            "dataset_id": dataset_id,
            "version_id": version["id"],
            "columns": headers,
            "rows": rows,
            "n": len(rows),
            "total": n_rows,
            "n_rows": n_rows,
            "sampled": len(rows) < n_rows,
        }


def update_column_roles(project_id: str, dataset_id: str, updates: list[dict]) -> dict:
    with get_db() as conn:
        ds = get_dataset(project_id, dataset_id, conn=conn)
        version = ds.get("current_version")
        if not version:
            raise AppError("dataset has no versions", status_code=404)
        known = {
            r["name"]
            for r in conn.execute(
                "SELECT name FROM columns WHERE version_id = ?", (version["id"],)
            ).fetchall()
        }
        for item in updates:
            name = item["name"]
            role = item["role"]
            if role not in VALID_ROLES:
                raise AppError(f"invalid role: {role}")
            if name not in known:
                raise AppError(f"unknown column: {name}", status_code=404)
            conn.execute(
                "UPDATE columns SET user_role = ? WHERE version_id = ? AND name = ?",
                (role, version["id"], name),
            )
        _set_needs_review(conn, dataset_id, 0)
    return get_dataset(project_id, dataset_id)


def clear_needs_review(project_id: str, dataset_id: str) -> dict:
    with get_db() as conn:
        get_dataset(project_id, dataset_id, conn=conn)
        _set_needs_review(conn, dataset_id, 0)
    return get_dataset(project_id, dataset_id)


def _set_needs_review(conn, dataset_id: str, value: int) -> None:
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(datasets)").fetchall()}
    if "needs_review" not in cols:
        return
    conn.execute(
        "UPDATE datasets SET needs_review = ? WHERE id = ?",
        (int(value), dataset_id),
    )


def get_profile(project_id: str, dataset_id: str) -> dict:
    ds = get_dataset(project_id, dataset_id)
    return {
        "ok": True,
        "dataset_id": dataset_id,
        "profile": ds.get("profile"),
        "columns": ds.get("columns") or [],
    }


def import_local_file(project_id: str, path: str, name: str | None = None) -> dict:
    file_path = Path(path)
    if not file_path.is_file():
        raise AppError("file not found", status_code=404)
    data = file_path.read_bytes()
    return create_csv_dataset(
        project_id, data, filename=name or file_path.name, name=name or file_path.stem
    )


def load_normalized_table(path: str | Path) -> tuple[list[str], list[list[str]]]:
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        headers = next(reader)
        rows = [row for row in reader]
    return headers, rows


DISCLOSURE_SAMPLE_ROWS = 5


def preview_disclosure(project_id: str, dataset_id: str) -> dict:
    payload = preview_dataset(project_id, dataset_id, n=DISCLOSURE_SAMPLE_ROWS)
    now = utcnow()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO sample_disclosure (dataset_id, enabled, previewed_at)
            VALUES (?, 0, ?)
            ON CONFLICT(dataset_id) DO UPDATE SET previewed_at = excluded.previewed_at
            """,
            (dataset_id, now),
        )
    payload["previewed_at"] = now
    payload["ok"] = True
    payload["sample_rows"] = len(payload.get("rows") or [])
    return payload


def set_disclosure(project_id: str, dataset_id: str, enabled: bool) -> dict:
    with get_db() as conn:
        get_dataset(project_id, dataset_id, conn=conn)
        row = conn.execute(
            "SELECT enabled, previewed_at FROM sample_disclosure WHERE dataset_id = ?",
            (dataset_id,),
        ).fetchone()
        previewed_at = row["previewed_at"] if row else None
        if enabled and not previewed_at:
            raise AppError("preview sample sharing before enabling")
        flag = 1 if enabled else 0
        conn.execute(
            """
            INSERT INTO sample_disclosure (dataset_id, enabled, previewed_at)
            VALUES (?, ?, ?)
            ON CONFLICT(dataset_id) DO UPDATE SET enabled = excluded.enabled
            """,
            (dataset_id, flag, previewed_at),
        )
        out = conn.execute(
            "SELECT dataset_id, enabled, previewed_at FROM sample_disclosure WHERE dataset_id = ?",
            (dataset_id,),
        ).fetchone()
        result = row_dict(out)
        result["ok"] = True
        return result
