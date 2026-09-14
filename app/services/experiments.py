"""Immutable experiment revisions and config checks."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pandas as pd

from app.db import get_db, new_id, require_project, row_dict, utcnow
from app.models import VALID_BUDGETS, VALID_FREQUENCIES, VALID_SPLITS, VALID_TASKS, AppError
from app.services.datasets import MISSING_TOKENS, _numeric_hits, get_dataset, load_normalized_table
from app.settings import get_settings

_ISO_DATE = re.compile(
    r"^\d{4}-\d{1,2}-\d{1,2}(?:[ T]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?$"
)
_YMD_DATE = re.compile(r"^(\d{4})[/-](\d{1,2})[/-](\d{1,2})(?:$|[ T])")
_SLASH_DATE = re.compile(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})(?:$|[ T])")

MIN_CLASS_SUPPORT = 2


def validate_config(config: dict, dataset: dict | None = None) -> dict:
    if not isinstance(config, dict):
        raise AppError("config must be an object")
    task = config.get("task")
    if task not in VALID_TASKS:
        raise AppError("invalid task")
    if task in {"classification", "regression", "forecast"}:
        target = config.get("target")
        if target is None or not str(target).strip():
            raise AppError("target is required")
        config = {**config, "target": str(target).strip()}
    horizon = config.get("horizon")
    if horizon is not None:
        try:
            horizon_i = int(horizon)
        except (TypeError, ValueError) as exc:
            raise AppError("horizon must be an integer") from exc
        if horizon_i < 1 or horizon_i > 90:
            raise AppError("horizon must be between 1 and 90")
        config = {**config, "horizon": horizon_i}
    if task == "forecast":
        if not config.get("date_column"):
            raise AppError("date_column is required for forecast")
        if config.get("horizon") is None:
            raise AppError("horizon is required for forecast")
    budget = config.get("budget", "quick")
    if budget not in VALID_BUDGETS:
        raise AppError("invalid budget")
    split = config.get("split")
    if split is not None and split not in VALID_SPLITS:
        raise AppError("invalid split")
    freq = config.get("frequency")
    if task == "forecast":
        if freq not in VALID_FREQUENCIES:
            if freq is None or (isinstance(freq, str) and not str(freq).strip()):
                raise AppError("frequency is required for forecast")
            raise AppError("invalid frequency")
    elif freq is not None and freq not in VALID_FREQUENCIES:
        raise AppError("invalid frequency")
    if dataset:
        if int(dataset.get("needs_review") or 0):
            raise AppError(
                "dataset schema changed; review columns before training",
                code="needs_review",
            )
        _validate_against_dataset(config, dataset)
    return config


def _validate_against_dataset(config: dict, dataset: dict) -> None:
    version = dataset.get("current_version") or {}
    profile = version.get("profile") or {}
    names = {c["name"] for c in profile.get("columns") or []}
    if not names:
        raise AppError("dataset has no columns")
    target = config.get("target")
    if target and target not in names:
        raise AppError(f"unknown target column: {target}")
    for key in ("features", "excluded", "group_columns"):
        vals = config.get(key) or []
        if not isinstance(vals, list):
            raise AppError(f"{key} must be a list")
        missing = [c for c in vals if c not in names]
        if missing:
            raise AppError(f"unknown columns in {key}: {', '.join(missing)}")
    for key in ("date_column", "entity_column"):
        val = config.get(key)
        if val and val not in names:
            raise AppError(f"unknown {key}: {val}")
    if config.get("task") == "forecast":
        _check_forecast_series(config, version)
    if config.get("task") == "classification":
        _check_classification_target(config, version)


def _check_forecast_series(config: dict, version: dict) -> None:
    settings = get_settings()
    groups = config.get("group_columns") or []
    path = version.get("normalized_path")
    if path and Path(path).is_file():
        headers, rows = load_normalized_table(path)
        idx = {h: i for i, h in enumerate(headers)}
        target = config.get("target")
        date_column = config.get("date_column")
        if target and target in idx:
            cells = [row[idx[target]] if idx[target] < len(row) else "" for row in rows]
            if not _numeric_capable(cells):
                raise AppError(f"forecast target {target} must be numeric")
        if date_column and date_column in idx:
            cells = [
                row[idx[date_column]] if idx[date_column] < len(row) else ""
                for row in rows
            ]
            _require_parseable_dates(date_column, cells)
        if groups:
            missing = [g for g in groups if g not in idx]
            if missing:
                raise AppError(f"unknown columns in group_columns: {', '.join(missing)}")
            keys = set()
            for row in rows:
                keys.add(tuple(row[idx[g]] if idx[g] < len(row) else "" for g in groups))
                if len(keys) > settings.max_forecast_series:
                    raise AppError(
                        f"too many forecast series (max {settings.max_forecast_series})",
                        code="too_many_series",
                    )


def _numeric_capable(values: list[str]) -> bool:
    present = [
        str(v)
        for v in values
        if v is not None and str(v).strip().lower() not in MISSING_TOKENS
    ]
    if not present:
        return False
    return _numeric_hits(present) / len(present) >= 0.9


def _require_parseable_dates(name: str, values: list[str]) -> None:
    present = [
        str(v).strip()
        for v in values
        if v is not None and str(v).strip() and str(v).strip().lower() not in MISSING_TOKENS
    ]
    if not present:
        raise AppError(f"date_column {name} has no parseable dates")
    families: set[str] = set()
    for raw in present:
        family, ok = _classify_date(raw)
        if family == "ambiguous":
            raise AppError(f"ambiguous date in {name}: {raw!r}")
        if not ok:
            raise AppError(f"unparseable date in {name}: {raw!r}")
        families.add(family)
    if len(families) > 1:
        raise AppError(f"mixed date formats in {name}")


def _classify_date(value: str) -> tuple[str, bool]:
    s = value.strip()
    if _ISO_DATE.match(s) or _YMD_DATE.match(s):
        parsed = pd.to_datetime(s, errors="coerce")
        return "ymd", bool(pd.notna(parsed))
    match = _SLASH_DATE.match(s)
    if match:
        a, b = int(match.group(1)), int(match.group(2))
        if a <= 12 and b <= 12 and a != b:
            return "ambiguous", False
        parsed = pd.to_datetime(s, errors="coerce", dayfirst=a > 12)
        return ("dmy" if a > 12 else "mdy"), bool(pd.notna(parsed))
    parsed = pd.to_datetime(s, errors="coerce")
    return "other", bool(pd.notna(parsed))


def _check_classification_target(config: dict, version: dict) -> None:
    target = config.get("target")
    profile = version.get("profile") or {}
    col = next((c for c in (profile.get("columns") or []) if c.get("name") == target), None)
    if col is not None and col.get("n_unique") is not None and int(col["n_unique"]) < 2:
        raise AppError(
            "classification needs at least two classes in the target",
            code="single_class",
        )
    path = version.get("normalized_path")
    if not path or not Path(path).is_file():
        return
    headers, rows = load_normalized_table(path)
    if target not in headers:
        return
    idx = headers.index(target)
    counts: dict[str, int] = {}
    for row in rows:
        raw = row[idx] if idx < len(row) else ""
        if raw is None or str(raw).strip().lower() in MISSING_TOKENS:
            continue
        key = str(raw)
        counts[key] = counts.get(key, 0) + 1
    if len(counts) < 2:
        raise AppError(
            "classification needs at least two classes in the target",
            code="single_class",
        )
    weak = [f"{name} ({n})" for name, n in sorted(counts.items()) if n < MIN_CLASS_SUPPORT]
    if weak:
        raise AppError(
            "each class needs at least 2 rows; insufficient support: " + ", ".join(weak),
            code="insufficient_class_support",
        )
    split = str(config.get("split") or "random").strip().lower()
    if split in {"random", "stratified"}:
        try:
            test_size = float(config.get("test_size", 0.2))
        except (TypeError, ValueError):
            test_size = 0.2
        if not classification_split_feasible(counts, test_size=test_size):
            raise AppError(
                "this classification split cannot keep at least one row of each class in the test set",
                code="infeasible_split",
            )


def classification_split_feasible(counts: dict[str, int], test_size: float = 0.2) -> bool:
    n = sum(int(c) for c in counts.values())
    n_classes = len(counts)
    if n_classes < 2 or n <= 0:
        return False
    try:
        test_size_f = float(test_size)
    except (TypeError, ValueError):
        return False
    helper = None
    try:
        from app.ml.preprocess import classification_split_feasible as helper
    except ImportError:
        helper = None
    if callable(helper):
        try:
            if not helper(n, n_classes, test_size_f, MIN_CLASS_SUPPORT):
                return False
        except TypeError:
            pass
    if not 0.0 < test_size_f < 1.0:
        return False
    n_test = min(n - 1, max(1, math.ceil(test_size_f * n - 1e-12)))
    for count in counts.values():
        # A class with 2 rows in n=10 at test_size=0.2 gets ~0.4 test rows.
        if (n_test * int(count)) / n < 0.5:
            return False
    return True


def create_experiment(project_id: str, dataset_id: str, config: dict) -> dict:
    with get_db() as conn:
        require_project(conn, project_id)
        dataset = get_dataset(project_id, dataset_id, conn=conn)
        config = validate_config(config, dataset)
        experiment_id = new_id()
        revision_id = new_id()
        now = utcnow()
        conn.execute(
            "INSERT INTO experiments (id, project_id, dataset_id, created_at) VALUES (?, ?, ?, ?)",
            (experiment_id, project_id, dataset_id, now),
        )
        conn.execute(
            """
            INSERT INTO experiment_revisions (id, experiment_id, revision, config_json, created_at)
            VALUES (?, ?, 1, ?, ?)
            """,
            (revision_id, experiment_id, json.dumps(config), now),
        )
        return get_experiment(project_id, experiment_id, conn=conn)


configure_experiment = create_experiment


def submit_training(project_id: str, experiment_id: str) -> dict:
    from app.services.jobs import submit_training as enqueue_training

    return enqueue_training(project_id, experiment_id)


def add_revision(project_id: str, experiment_id: str, config: dict) -> dict:
    with get_db() as conn:
        exp = _require_experiment(conn, project_id, experiment_id)
        dataset = get_dataset(project_id, exp["dataset_id"], conn=conn)
        config = validate_config(config, dataset)
        latest = conn.execute(
            """
            SELECT MAX(revision) AS r FROM experiment_revisions WHERE experiment_id = ?
            """,
            (experiment_id,),
        ).fetchone()
        nxt = int(latest["r"] or 0) + 1
        conn.execute(
            """
            INSERT INTO experiment_revisions (id, experiment_id, revision, config_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (new_id(), experiment_id, nxt, json.dumps(config), utcnow()),
        )
        return get_experiment(project_id, experiment_id, conn=conn)


def get_experiment(project_id: str, experiment_id: str, conn=None) -> dict:
    def _load(c) -> dict:
        row = _require_experiment(c, project_id, experiment_id)
        revs = c.execute(
            """
            SELECT * FROM experiment_revisions
            WHERE experiment_id = ? ORDER BY revision ASC
            """,
            (experiment_id,),
        ).fetchall()
        revisions = []
        for r in revs:
            item = row_dict(r)
            item["config"] = json.loads(r["config_json"])
            item.pop("config_json", None)
            revisions.append(item)
        row["revisions"] = revisions
        row["current_revision"] = revisions[-1] if revisions else None
        return row

    if conn is not None:
        return _load(conn)
    with get_db() as c:
        return _load(c)


def list_experiments(project_id: str) -> list[dict]:
    with get_db() as conn:
        require_project(conn, project_id)
        rows = conn.execute(
            """
            SELECT e.*, r.id AS revision_id, r.revision, r.config_json
            FROM experiments e
            LEFT JOIN experiment_revisions r
              ON r.id = (
                SELECT id FROM experiment_revisions
                WHERE experiment_id = e.id ORDER BY revision DESC LIMIT 1
              )
            WHERE e.project_id = ?
            ORDER BY e.created_at DESC
            """,
            (project_id,),
        ).fetchall()
        out = []
        for r in rows:
            item = row_dict(r)
            item["config"] = json.loads(r["config_json"]) if r["config_json"] else None
            item.pop("config_json", None)
            out.append(item)
        return out


def latest_revision(conn, experiment_id: str):
    return conn.execute(
        """
        SELECT * FROM experiment_revisions
        WHERE experiment_id = ? ORDER BY revision DESC LIMIT 1
        """,
        (experiment_id,),
    ).fetchone()


def _require_experiment(conn, project_id: str, experiment_id: str) -> dict:
    require_project(conn, project_id)
    row = conn.execute(
        "SELECT * FROM experiments WHERE id = ? AND project_id = ?",
        (experiment_id, project_id),
    ).fetchone()
    if not row:
        raise AppError("experiment not found", status_code=404)
    return row_dict(row)
