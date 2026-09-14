"""Predict jobs: column alignment and formula-injection-safe downloads."""

from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path

from app.db import get_db, new_id, require_project, row_dict
from app.models import AppError
from app.services.datasets import (
    MISSING_TOKENS,
    create_csv_dataset,
    get_dataset,
    load_normalized_table,
    parse_tabular,
    write_normalized_csv,
)
from app.services.jobs import enqueue_job
from app.settings import get_settings

_DANGEROUS_PREFIXES = ("=", "+", "-", "@", "\t", "\r")
_ID_NAME_RE = re.compile(r"(^id$|_id$|^sku$|uuid|identifier)", re.IGNORECASE)
_LEADING_ZERO_RE = re.compile(r"^0\d+$")


def required_columns(config: dict, train_headers: list[str]) -> list[str]:
    excluded = set(config.get("excluded") or [])
    target = config.get("target")
    features = config.get("features")
    if features:
        required = list(features)
    else:
        required = [c for c in train_headers if c != target and c not in excluded]
    extra: list[str] = []
    if config.get("task") == "forecast":
        extra.extend(config.get("group_columns") or [])
        if config.get("date_column"):
            extra.append(config["date_column"])
    seen = set(required)
    for col in extra:
        if col not in seen:
            required.append(col)
            seen.add(col)
    return required


def align_columns(
    headers: list[str],
    rows: list[list[str]],
    required: list[str],
    *,
    roles: dict[str, str] | None = None,
    target: str | None = None,
) -> tuple[list[str], list[list[str]]]:
    roles = roles or {}
    present = set(headers)
    missing = [c for c in required if c not in present]
    if missing:
        raise AppError(f"missing required columns: {', '.join(missing)}")
    idx = {h: i for i, h in enumerate(headers)}
    keep = list(required)
    seen = set(keep)
    for name in headers:
        if name in seen:
            continue
        cells = _column_cells(rows, idx[name])
        if name == target or _identifier_like(name, roles.get(name), cells):
            keep.append(name)
            seen.add(name)
    for name in required:
        if roles.get(name) != "numeric":
            continue
        _require_numeric_cells(name, _column_cells(rows, idx[name]))
    aligned = []
    for row in rows:
        aligned.append([row[idx[c]] if idx[c] < len(row) else "" for c in keep])
    return keep, aligned


def _column_cells(rows: list[list[str]], index: int) -> list[str]:
    return [row[index] if index < len(row) else "" for row in rows]


def _is_missing_cell(value: str) -> bool:
    return (value or "").strip().lower() in MISSING_TOKENS


def _identifier_like(name: str, role: str | None, values: list[str]) -> bool:
    if role == "identifier":
        return True
    if role == "numeric":
        return False
    if _ID_NAME_RE.search(name):
        return True
    present = [v for v in values if not _is_missing_cell(v)]
    if not present:
        return False
    hits = sum(1 for v in present if _LEADING_ZERO_RE.match(v.strip()))
    return hits / len(present) >= 0.5


def _require_numeric_cells(name: str, values: list[str]) -> None:
    for i, raw in enumerate(values):
        if _is_missing_cell(raw):
            continue
        text = raw.strip().replace(",", "")
        try:
            float(text)
        except ValueError as exc:
            raise AppError(
                f"conversion failed for numeric feature {name} at row {i + 1}"
            ) from exc


def escape_formula_cell(value: str) -> str:
    if value and value[0] in _DANGEROUS_PREFIXES:
        return "'" + value
    return value


def escape_csv_text(text: str) -> str:
    reader = csv.reader(io.StringIO(text))
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    first = True
    for row in reader:
        if first:
            writer.writerow(row)
            first = False
            continue
        writer.writerow([escape_formula_cell(cell) for cell in row])
    return out.getvalue()


def submit_predict(
    project_id: str,
    *,
    model_id: str,
    dataset_id: str | None = None,
    upload_bytes: bytes | None = None,
    filename: str = "predict.csv",
) -> dict:
    with get_db() as conn:
        require_project(conn, project_id)
        model = conn.execute(
            "SELECT * FROM models WHERE id = ? AND project_id = ?",
            (model_id, project_id),
        ).fetchone()
        if not model:
            raise AppError("model not found", status_code=404)
        train_job = conn.execute(
            "SELECT * FROM jobs WHERE id = ?", (model["job_id"],)
        ).fetchone()
        config: dict = {}
        revision_id = None
        if train_job and train_job["experiment_revision_id"]:
            revision_id = train_job["experiment_revision_id"]
            rev = conn.execute(
                "SELECT * FROM experiment_revisions WHERE id = ?",
                (revision_id,),
            ).fetchone()
            if rev:
                config = json.loads(rev["config_json"])
        train_version = conn.execute(
            "SELECT * FROM dataset_versions WHERE id = ?",
            (model["dataset_version_id"],),
        ).fetchone()
        train_headers: list[str] = []
        roles: dict[str, str] = {}
        if train_version:
            cols = conn.execute(
                "SELECT name, inferred_role, user_role FROM columns WHERE version_id = ? ORDER BY rowid",
                (train_version["id"],),
            ).fetchall()
            train_headers = [c["name"] for c in cols]
            for col in cols:
                role = col["user_role"] or col["inferred_role"]
                if role:
                    roles[col["name"]] = role
        for name, role in (config.get("roles") or {}).items():
            if role:
                roles[name] = role
        required = required_columns(config, train_headers)
        target = config.get("target")

    predict_path: str
    input_version_id: str
    if upload_bytes is not None:
        table = parse_tabular(upload_bytes, filename=filename)
        headers, rows = align_columns(
            table.headers, table.rows, required, roles=roles, target=target
        )
        ds = create_csv_dataset(
            project_id, upload_bytes, filename=filename, name=f"predict-{filename}"
        )
        input_version_id = ds["current_version"]["id"]
        predict_path = _write_aligned_csv(headers, rows)
        dataset_id = ds["id"]
    elif dataset_id:
        ds = get_dataset(project_id, dataset_id)
        version = ds.get("current_version")
        if not version:
            raise AppError("dataset has no versions", status_code=404)
        headers, rows = load_normalized_table(version["normalized_path"])
        headers, rows = align_columns(
            headers, rows, required, roles=roles, target=target
        )
        input_version_id = version["id"]
        predict_path = _write_aligned_csv(headers, rows)
    else:
        raise AppError("dataset_id or file is required")

    progress = {
        "model_id": model_id,
        "input_version_id": input_version_id,
        "dataset_id": dataset_id,
        "predict_path": predict_path,
    }
    job = enqueue_job(
        project_id,
        job_type="predict",
        experiment_revision_id=revision_id,
        model_id=model_id,
        progress=progress,
    )
    return {"job_id": job["id"], "status": "queued"}


def _write_aligned_csv(headers: list[str], rows: list[list[str]]) -> str:
    folder = get_settings().artifacts_dir / new_id()
    path = folder / "aligned.csv"
    write_normalized_csv(path, headers, rows)
    return str(path)


def get_prediction(project_id: str, prediction_id: str) -> dict:
    with get_db() as conn:
        require_project(conn, project_id)
        row = conn.execute(
            "SELECT * FROM predictions WHERE id = ?", (prediction_id,)
        ).fetchone()
        if not row:
            row = conn.execute(
                "SELECT * FROM predictions WHERE job_id = ?", (prediction_id,)
            ).fetchone()
        if not row:
            raise AppError("prediction not found", status_code=404)
        job = conn.execute(
            "SELECT project_id FROM jobs WHERE id = ?", (row["job_id"],)
        ).fetchone()
        if not job or job["project_id"] != project_id:
            raise AppError("prediction not found", status_code=404)
        return row_dict(row)


def download_prediction_csv(project_id: str, prediction_id: str) -> tuple[bytes, str]:
    pred = get_prediction(project_id, prediction_id)
    path = Path(pred["output_path"])
    if not path.is_file():
        raise AppError("prediction file missing", status_code=404)
    raw = path.read_text(encoding="utf-8")
    escaped = escape_csv_text(raw)
    name = f"predictions-{pred['id'][:8]}.csv"
    return escaped.encode("utf-8"), name
