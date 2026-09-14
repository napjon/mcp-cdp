"""Domain adapters shared by MCP tools and chat. Import platform services lazily."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

from app.mcp.config import import_roots, max_upload_bytes, redact
from app.mcp.db import db_conn, fetchall, fetchone

MAX_SAMPLE_ROWS = 5
SAMPLE_KEYS = {
    "sample",
    "samples",
    "rows",
    "preview",
    "head",
    "data",
    "records",
    "sample_rows",
    "sample_preview",
}


class ServiceUnavailable(RuntimeError):
    pass


def confirmed(confirm: Any) -> bool:
    if confirm is True:
        return True
    if isinstance(confirm, (int, float)) and confirm == 1:
        return True
    return isinstance(confirm, str) and confirm.strip().lower() in {"true", "1", "yes"}


def confirm_or_error(confirm: Any, action: str) -> dict[str, Any] | None:
    if confirmed(confirm):
        return None
    return {"ok": False, "error": f"confirm=true is required to {action}"}


def clamp_sample_rows(sample_rows: Any) -> int:
    try:
        n = int(sample_rows or 0)
    except (TypeError, ValueError):
        n = 0
    return max(0, min(n, MAX_SAMPLE_ROWS))


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _symlink_escapes(path: Path, roots: list[Path]) -> bool:
    current = path
    seen: set[Path] = set()
    while current not in seen:
        seen.add(current)
        if current.exists() and current.is_symlink():
            try:
                target = current.readlink()
                dest = target if target.is_absolute() else current.parent / target
                resolved = dest.resolve()
            except OSError:
                return True
            if not any(_is_under(resolved, root) for root in roots):
                return True
        if current.parent == current:
            break
        current = current.parent
    return False


def read_import_bytes(path: Path) -> tuple[bytes, str | None]:
    """Read at most settings.max_upload_bytes+1 so oversize files are not fully slurped."""
    limit = max_upload_bytes()
    with path.open("rb") as fh:
        data = fh.read(limit + 1)
    if len(data) > limit:
        return data, "too_large"
    return data, None


def resolve_import_path(user_path: str) -> Path:
    if not user_path or not str(user_path).strip():
        raise ValueError("path is required")
    if "\x00" in str(user_path):
        raise ValueError("invalid path")
    roots = [p.resolve() for p in import_roots()]
    if not roots:
        raise ValueError("MCP_IMPORT_ROOTS is not configured")
    raw = Path(user_path)
    candidates = [raw] if raw.is_absolute() else [root / raw for root in roots]
    for cand in candidates:
        try:
            resolved = cand.resolve()
        except OSError as exc:
            raise ValueError("invalid path") from exc
        if not any(_is_under(resolved, root) for root in roots):
            continue
        if _symlink_escapes(cand, roots):
            continue
        return resolved
    raise ValueError("path is outside MCP_IMPORT_ROOTS")


def samples_allowed(dataset_id: str | None) -> bool:
    if not dataset_id:
        return False
    row = fetchone(
        "SELECT enabled FROM sample_disclosure WHERE dataset_id = ?",
        (dataset_id,),
    )
    if not row:
        return False
    return bool(row.get("enabled"))


def _looks_like_rows(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    return isinstance(value[0], (dict, list))


def sanitize_inspect(
    payload: Any,
    sample_rows: int,
    *,
    dataset_id: str | None = None,
    allow_samples: bool | None = None,
) -> dict[str, Any]:
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        payload = {"result": payload}
    n = clamp_sample_rows(sample_rows)
    if allow_samples is None:
        allow_samples = samples_allowed(dataset_id) if dataset_id else False
    show = allow_samples and n > 0
    out = _sanitize_obj(payload, n, show)
    if not isinstance(out, dict):
        out = {"result": out}
    out.setdefault("ok", True)
    out["sample_rows"] = n if show else 0
    if not show:
        out["samples_included"] = False
        if n > 0 and not allow_samples:
            out["sample_note"] = "Raw rows are omitted until sample sharing is enabled."
    return out


def _sanitize_obj(value: Any, n: int, show_samples: bool) -> Any:
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            key_l = str(key).lower()
            if key_l in SAMPLE_KEYS or key_l.endswith("_preview"):
                if not show_samples:
                    continue
                if _looks_like_rows(item) or isinstance(item, list):
                    cleaned[key] = item[:n]
                elif isinstance(item, str) and len(item) > 2000:
                    cleaned[key] = item[:2000]
                else:
                    cleaned[key] = item
                continue
            cleaned[key] = _sanitize_obj(item, n, show_samples)
        return cleaned
    if _looks_like_rows(value):
        return value[:n] if show_samples else []
    if isinstance(value, list) and len(value) > 200:
        return [_sanitize_obj(v, n, show_samples) for v in value[:200]]
    if isinstance(value, list):
        return [_sanitize_obj(v, n, show_samples) for v in value]
    if isinstance(value, str) and len(value) > 8000:
        return value[:8000] + "…"
    return value


def _filter_kwargs(fn: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return kwargs
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def _app_error_type() -> type[Exception] | tuple:
    try:
        from app.models import AppError

        return AppError
    except ImportError:
        return ()


async def call_service(module: str, names: tuple[str, ...], **kwargs: Any) -> Any:
    import importlib

    try:
        mod = importlib.import_module(module)
    except ImportError as exc:
        raise ServiceUnavailable(module) from exc
    fn = None
    for name in names:
        cand = getattr(mod, name, None)
        if callable(cand):
            fn = cand
            break
    if fn is None:
        raise ServiceUnavailable(f"{module} has none of {names}")
    try:
        result = fn(**_filter_kwargs(fn, kwargs))
        if inspect.isawaitable(result):
            result = await result
        return result
    except _app_error_type() as exc:  # type: ignore[misc]
        payload: dict[str, Any] = {"ok": False, "error": redact(getattr(exc, "message", str(exc)))}
        code = getattr(exc, "code", None)
        if code:
            payload["code"] = code
        return payload


def _as_dict(result: Any, *, list_key: str | None = None) -> dict[str, Any]:
    if result is None:
        return {"ok": True}
    if isinstance(result, dict):
        return result
    if isinstance(result, list):
        return {"ok": True, list_key or "items": result}
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if hasattr(result, "dict"):
        return result.dict()
    return {"result": result}


async def _service_or_none(module: str, names: tuple[str, ...], **kwargs: Any) -> Any:
    try:
        return await call_service(module, names, **kwargs)
    except ServiceUnavailable:
        return None


# --- reads ---


async def list_projects() -> dict[str, Any]:
    result = await _service_or_none("app.services.datasets", ("list_projects",))
    if result is not None:
        return _as_dict(result, list_key="projects")
    rows = fetchall("SELECT id, name, created_at FROM projects ORDER BY created_at")
    return {"projects": rows}


async def get_project(project_id: str) -> dict[str, Any]:
    result = await _service_or_none(
        "app.services.datasets", ("get_project",), project_id=project_id
    )
    if result is not None:
        return _as_dict(result)
    row = fetchone(
        "SELECT id, name, created_at FROM projects WHERE id = ?",
        (project_id,),
    )
    if not row:
        return {"ok": False, "error": "project not found"}
    datasets = fetchall(
        "SELECT id, name, source_type, created_at FROM datasets WHERE project_id = ?",
        (project_id,),
    )
    row["datasets"] = datasets
    row["ok"] = True
    return row


async def list_datasets(project_id: str) -> dict[str, Any]:
    result = await _service_or_none(
        "app.services.datasets",
        ("list_datasets", "list_for_project"),
        project_id=project_id,
    )
    if result is not None:
        return _as_dict(result, list_key="datasets")
    rows = fetchall(
        "SELECT id, name, source_type, connection_id, created_at FROM datasets WHERE project_id = ?",
        (project_id,),
    )
    return {"datasets": rows, "project_id": project_id}


async def _fetch_inspect_payload(project_id: str, dataset_id: str, sample_rows: int) -> dict[str, Any]:
    result = await _service_or_none(
        "app.services.datasets",
        ("get_dataset", "inspect_dataset", "inspect"),
        project_id=project_id,
        dataset_id=dataset_id,
    )
    payload: dict[str, Any]
    if result is not None:
        payload = _as_dict(result)
        if payload.get("ok") is False:
            return payload
        version = payload.get("current_version") or {}
        if isinstance(version, dict):
            payload.setdefault("n_rows", version.get("n_rows"))
            payload.setdefault("n_cols", version.get("n_cols"))
            profile = version.get("profile") or {}
            if isinstance(profile, dict) and profile.get("columns"):
                payload.setdefault("columns", profile.get("columns"))
    else:
        ds = fetchone(
            "SELECT id, name, source_type, created_at FROM datasets WHERE id = ? AND project_id = ?",
            (dataset_id, project_id),
        )
        if not ds:
            return {"ok": False, "error": "dataset not found"}
        version = fetchone(
            """
            SELECT id, version, n_rows, n_cols, encoding, delimiter, header_row, created_at
            FROM dataset_versions WHERE dataset_id = ? ORDER BY version DESC LIMIT 1
            """,
            (dataset_id,),
        )
        columns = []
        if version:
            columns = fetchall(
                """
                SELECT name, inferred_role, user_role, n_missing, n_unique, is_constant, sample_preview
                FROM columns WHERE version_id = ? ORDER BY name
                """,
                (version["id"],),
            )
        payload = {
            "ok": True,
            "dataset": ds,
            "version": version,
            "n_rows": version.get("n_rows") if version else None,
            "n_cols": version.get("n_cols") if version else None,
            "columns": columns,
        }
    n = clamp_sample_rows(sample_rows)
    if n > 0 and samples_allowed(dataset_id):
        preview = await _service_or_none(
            "app.services.datasets",
            ("preview_dataset",),
            project_id=project_id,
            dataset_id=dataset_id,
            n=n,
        )
        if isinstance(preview, dict) and preview.get("rows") is not None:
            payload["rows"] = preview["rows"][:n]
    return payload


async def inspect_dataset(
    project_id: str,
    dataset_id: str,
    sample_rows: int = 0,
) -> dict[str, Any]:
    n = clamp_sample_rows(sample_rows)
    payload = await _fetch_inspect_payload(project_id, dataset_id, n)
    return sanitize_inspect(payload, n, dataset_id=dataset_id)


async def get_profile(project_id: str, dataset_id: str) -> dict[str, Any]:
    result = await _service_or_none(
        "app.services.datasets",
        ("get_dataset", "get_profile", "profile"),
        project_id=project_id,
        dataset_id=dataset_id,
    )
    if result is None:
        inspected = await inspect_dataset(project_id, dataset_id, sample_rows=0)
        return {"ok": True, "profile": inspected, "dataset_id": dataset_id}
    return sanitize_inspect(_as_dict(result), 0, dataset_id=dataset_id)


async def get_report(project_id: str, job_id: str) -> dict[str, Any]:
    result = await _service_or_none(
        "app.services.jobs",
        ("get_report", "report"),
        project_id=project_id,
        job_id=job_id,
    )
    if result is not None:
        return sanitize_inspect(_as_dict(result), 0, allow_samples=False)
    row = fetchone(
        """
        SELECT r.job_id AS job_id, r.report_json AS report_json, r.plot_dir AS plot_dir
        FROM reports r
        JOIN jobs j ON j.id = r.job_id
        WHERE j.project_id = ? AND r.job_id = ?
        """,
        (project_id, job_id),
    )
    if not row:
        return {"ok": False, "error": "report not found"}
    report = row.get("report_json")
    if isinstance(report, str):
        try:
            report = json.loads(report)
        except json.JSONDecodeError:
            pass
    return {"ok": True, "job_id": job_id, "report": report, "plot_dir": row.get("plot_dir")}


async def get_job(project_id: str, job_id: str) -> dict[str, Any]:
    result = await _service_or_none(
        "app.services.jobs",
        ("get_job", "get"),
        project_id=project_id,
        job_id=job_id,
    )
    if result is not None:
        return _as_dict(result)
    row = fetchone(
        """
        SELECT id, project_id, type, status, progress_json, error, created_at, started_at, finished_at
        FROM jobs WHERE id = ? AND project_id = ?
        """,
        (job_id, project_id),
    )
    if not row:
        return {"ok": False, "error": "job not found"}
    progress = row.get("progress_json")
    if isinstance(progress, str):
        try:
            row["progress_json"] = json.loads(progress)
        except json.JSONDecodeError:
            pass
    row["ok"] = True
    return row


# --- writes ---


async def connect_google_sheet(
    project_id: str,
    url: str,
    confirm: Any = False,
    gid: str | None = None,
    header_row: int | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    err = confirm_or_error(confirm, "connect a Google Sheet")
    if err:
        return err
    kwargs: dict[str, Any] = {
        "project_id": project_id,
        "url": url,
        "header_row": header_row,
        "name": name,
    }
    if gid is not None and str(gid) != "":
        kwargs["gid"] = gid
    result = await _service_or_none(
        "app.services.sheets",
        ("connect_sheet", "connect_google_sheet", "connect", "create_connection"),
        **kwargs,
    )
    if result is None:
        return {"ok": False, "error": "sheets service is not available"}
    return _as_dict(result)


async def refresh_google_sheet(
    project_id: str,
    connection_id: str,
    confirm: Any = False,
) -> dict[str, Any]:
    err = confirm_or_error(confirm, "refresh a Google Sheet")
    if err:
        return err
    result = await _service_or_none(
        "app.services.sheets",
        ("refresh_sheet", "refresh_google_sheet", "refresh", "refresh_connection"),
        project_id=project_id,
        connection_id=connection_id,
        id=connection_id,
    )
    if result is None:
        return {"ok": False, "error": "sheets service is not available"}
    return _as_dict(result)


def _parse_config(config: Any) -> Any:
    if isinstance(config, str):
        try:
            return json.loads(config)
        except json.JSONDecodeError:
            return config
    return config


async def configure_experiment(
    project_id: str,
    dataset_id: str,
    config: Any,
    confirm: Any = False,
) -> dict[str, Any]:
    err = confirm_or_error(confirm, "configure an experiment")
    if err:
        return err
    parsed = _parse_config(config)
    result = await _service_or_none(
        "app.services.experiments",
        ("create_experiment", "configure_experiment", "create"),
        project_id=project_id,
        dataset_id=dataset_id,
        config=parsed,
    )
    if result is None:
        return {"ok": False, "error": "experiments service is not available"}
    return _as_dict(result)


async def submit_training(
    project_id: str,
    experiment_id: str,
    confirm: Any = False,
) -> dict[str, Any]:
    err = confirm_or_error(confirm, "submit training")
    if err:
        return err
    result = await _service_or_none(
        "app.services.jobs",
        ("submit_training", "enqueue_train", "enqueue"),
        project_id=project_id,
        experiment_id=experiment_id,
    )
    if result is None:
        result = await _service_or_none(
            "app.services.experiments",
            ("submit_training", "submit"),
            project_id=project_id,
            experiment_id=experiment_id,
        )
    if result is None:
        return {"ok": False, "error": "training service is not available"}
    data = _as_dict(result)
    job_id = data.get("job_id") or data.get("id")
    if job_id:
        data["job_id"] = job_id
    return data


async def cancel_job(
    project_id: str,
    job_id: str,
    confirm: Any = False,
) -> dict[str, Any]:
    err = confirm_or_error(confirm, "cancel a job")
    if err:
        return err
    result = await _service_or_none(
        "app.services.jobs",
        ("cancel_job", "cancel"),
        project_id=project_id,
        job_id=job_id,
    )
    if result is None:
        return {"ok": False, "error": "jobs service is not available"}
    return _as_dict(result)


async def submit_predict(
    project_id: str,
    confirm: Any = False,
    model_id: str | None = None,
    dataset_id: str | None = None,
) -> dict[str, Any]:
    err = confirm_or_error(confirm, "submit a prediction job")
    if err:
        return err
    if not model_id:
        return {"ok": False, "error": "model_id is required"}
    result = await _service_or_none(
        "app.services.predict",
        ("submit_predict", "submit", "enqueue"),
        project_id=project_id,
        model_id=model_id,
        dataset_id=dataset_id,
    )
    if result is None:
        return {"ok": False, "error": "predict service is not available"}
    data = _as_dict(result)
    job_id = data.get("job_id") or data.get("id")
    if job_id:
        data["job_id"] = job_id
    return data


async def import_local_file(
    project_id: str,
    path: str,
    confirm: Any = False,
    name: str | None = None,
) -> dict[str, Any]:
    err = confirm_or_error(confirm, "import a local file")
    if err:
        return err
    try:
        resolved = resolve_import_path(path)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    if resolved.exists() and resolved.is_dir():
        return {"ok": False, "error": "path is a directory"}
    if resolved.exists() and resolved.is_symlink():
        return {"ok": False, "error": "path is outside MCP_IMPORT_ROOTS"}
    if not resolved.is_file():
        return {"ok": False, "error": "path is not a file"}
    data, oversize = read_import_bytes(resolved)
    if oversize:
        return {"ok": False, "error": "file too large", "code": oversize}
    result = await _service_or_none(
        "app.services.datasets",
        ("create_csv_dataset", "import_local_file", "import_file", "ingest_path", "ingest_local"),
        project_id=project_id,
        data=data,
        filename=resolved.name,
        name=name or resolved.stem,
        path=str(resolved),
    )
    if result is None:
        return {
            "ok": False,
            "error": "datasets service is not available",
            "path": str(resolved),
        }
    return _as_dict(result)


def get_sample_disclosure(dataset_id: str) -> dict[str, Any]:
    row = fetchone(
        "SELECT dataset_id, enabled, previewed_at FROM sample_disclosure WHERE dataset_id = ?",
        (dataset_id,),
    )
    if not row:
        return {"dataset_id": dataset_id, "enabled": 0, "previewed_at": None}
    return row


async def preview_sample_disclosure(project_id: str, dataset_id: str) -> dict[str, Any]:
    """Record that the operator previewed the bounded sample that chat would send."""
    preview = await _service_or_none(
        "app.services.datasets",
        ("preview_dataset",),
        project_id=project_id,
        dataset_id=dataset_id,
        n=MAX_SAMPLE_ROWS,
    )
    rows = []
    columns: list[Any] = []
    if isinstance(preview, dict):
        rows = (preview.get("rows") or [])[:MAX_SAMPLE_ROWS]
        columns = preview.get("columns") or []
    try:
        from app.db import utcnow

        now = utcnow()
    except ImportError:
        from datetime import UTC, datetime

        now = datetime.now(UTC).isoformat()
    with db_conn() as conn:
        conn.execute(
            """
            INSERT INTO sample_disclosure (dataset_id, enabled, previewed_at)
            VALUES (?, 0, ?)
            ON CONFLICT(dataset_id) DO UPDATE SET previewed_at = excluded.previewed_at
            """,
            (dataset_id, now),
        )
        conn.commit()
    return {
        "ok": True,
        "dataset_id": dataset_id,
        "previewed_at": now,
        "columns": columns,
        "rows": rows,
        "sample_rows": len(rows),
    }


def set_sample_disclosure(dataset_id: str, enabled: bool) -> dict[str, Any]:
    current = get_sample_disclosure(dataset_id)
    if enabled and not current.get("previewed_at"):
        return {"ok": False, "error": "preview sample sharing before enabling"}
    flag = 1 if enabled else 0
    previewed_at = current.get("previewed_at")
    with db_conn() as conn:
        conn.execute(
            """
            INSERT INTO sample_disclosure (dataset_id, enabled, previewed_at)
            VALUES (?, ?, ?)
            ON CONFLICT(dataset_id) DO UPDATE SET enabled = excluded.enabled
            """,
            (dataset_id, flag, previewed_at),
        )
        conn.commit()
    return get_sample_disclosure(dataset_id) | {"ok": True}


CHAT_TOOL_REQUIRED: dict[str, tuple[str, ...]] = {
    "list_datasets": (),
    "inspect_dataset": ("dataset_id",),
    "get_profile": ("dataset_id",),
    "get_job": ("job_id",),
    "get_report": ("job_id",),
    "configure_experiment": ("dataset_id", "config", "confirm"),
    "submit_predict": ("confirm",),
    "cancel_job": ("job_id", "confirm"),
}


def validate_chat_tool_args(name: str, arguments: dict[str, Any] | None) -> dict[str, Any] | None:
    required = CHAT_TOOL_REQUIRED.get(name)
    if required is None:
        return {"ok": False, "error": f"unknown tool {name}"}
    args = arguments or {}
    for key in required:
        if key not in args or args[key] is None:
            return {"ok": False, "error": f"missing required argument: {key}"}
        value = args[key]
        if isinstance(value, str) and not value.strip():
            return {"ok": False, "error": f"missing required argument: {key}"}
    return None


def chat_tool_specs() -> list[dict[str, Any]]:
    """JSON-schema tool specs for the LLM, project-scoped at call time."""
    return [
        {
            "name": "list_datasets",
            "description": "List datasets in this project.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        {
            "name": "inspect_dataset",
            "description": "Inspect dataset schema and aggregates. Raw rows only if sample sharing is enabled.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string"},
                    "sample_rows": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": MAX_SAMPLE_ROWS,
                        "default": 0,
                    },
                },
                "required": ["dataset_id"],
                "additionalProperties": False,
            },
        },
        {
            "name": "get_profile",
            "description": "Get the column profile for a dataset (no raw table dump).",
            "parameters": {
                "type": "object",
                "properties": {"dataset_id": {"type": "string"}},
                "required": ["dataset_id"],
                "additionalProperties": False,
            },
        },
        {
            "name": "get_job",
            "description": "Get a training or predict job by id.",
            "parameters": {
                "type": "object",
                "properties": {"job_id": {"type": "string"}},
                "required": ["job_id"],
                "additionalProperties": False,
            },
        },
        {
            "name": "get_report",
            "description": "Get computed metrics and plot paths for a finished training job.",
            "parameters": {
                "type": "object",
                "properties": {"job_id": {"type": "string"}},
                "required": ["job_id"],
                "additionalProperties": False,
            },
        },
        {
            "name": "configure_experiment",
            "description": "Create an experiment config. Set confirm true only if the user asked to save it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string"},
                    "config": {"type": "object"},
                    "confirm": {"type": "boolean"},
                },
                "required": ["dataset_id", "config", "confirm"],
                "additionalProperties": False,
            },
        },
        {
            "name": "submit_predict",
            "description": "Queue a prediction job. Returns job_id immediately. confirm must be true.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_id": {"type": "string"},
                    "dataset_id": {"type": "string"},
                    "confirm": {"type": "boolean"},
                },
                "required": ["confirm"],
                "additionalProperties": False,
            },
        },
        {
            "name": "cancel_job",
            "description": "Cancel a queued or running job. confirm must be true.",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "string"},
                    "confirm": {"type": "boolean"},
                },
                "required": ["job_id", "confirm"],
                "additionalProperties": False,
            },
        },
    ]


async def dispatch_chat_tool(project_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    args = dict(arguments or {})
    args.pop("project_id", None)
    invalid = validate_chat_tool_args(name, args)
    if invalid:
        return invalid
    if name == "list_datasets":
        return await list_datasets(project_id)
    if name == "inspect_dataset":
        return await inspect_dataset(
            project_id,
            str(args.get("dataset_id") or ""),
            clamp_sample_rows(args.get("sample_rows", 0)),
        )
    if name == "get_profile":
        return await get_profile(project_id, str(args.get("dataset_id") or ""))
    if name == "get_job":
        return await get_job(project_id, str(args.get("job_id") or ""))
    if name == "get_report":
        return await get_report(project_id, str(args.get("job_id") or ""))
    if name == "configure_experiment":
        return await configure_experiment(
            project_id,
            str(args.get("dataset_id") or ""),
            args.get("config"),
            args.get("confirm", False),
        )
    if name == "submit_predict":
        return await submit_predict(
            project_id,
            args.get("confirm", False),
            model_id=args.get("model_id"),
            dataset_id=args.get("dataset_id"),
        )
    if name == "cancel_job":
        return await cancel_job(
            project_id,
            str(args.get("job_id") or ""),
            args.get("confirm", False),
        )
    return {"ok": False, "error": f"unknown tool {name}"}
