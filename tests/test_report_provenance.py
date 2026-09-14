from __future__ import annotations

import json
from pathlib import Path

from app.db import get_db, new_id, utcnow
from app.worker import _store_predict_result, _store_train_result


def _seed_train_job(config: dict, progress: dict | None = None) -> dict:
    now = utcnow()
    ids = {
        "dataset_id": new_id(),
        "version_id": new_id(),
        "experiment_id": new_id(),
        "revision_id": new_id(),
        "job_id": new_id(),
    }
    progress = {"dataset_version_id": ids["version_id"], **(progress or {})}
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO datasets (id, project_id, name, source_type, connection_id, created_at)
            VALUES (?, 'local', 't', 'csv', NULL, ?)
            """,
            (ids["dataset_id"], now),
        )
        conn.execute(
            """
            INSERT INTO dataset_versions (
              id, dataset_id, version, content_hash, original_path, normalized_path,
              n_rows, n_cols, header_row, encoding, delimiter, import_meta_json, created_at
            ) VALUES (?, ?, 1, 'h', 'o.csv', 'n.csv', 5, 2, 0, 'utf-8', ',', '{}', ?)
            """,
            (ids["version_id"], ids["dataset_id"], now),
        )
        conn.execute(
            "INSERT INTO experiments (id, project_id, dataset_id, created_at) VALUES (?, 'local', ?, ?)",
            (ids["experiment_id"], ids["dataset_id"], now),
        )
        conn.execute(
            """
            INSERT INTO experiment_revisions (id, experiment_id, revision, config_json, created_at)
            VALUES (?, ?, 1, ?, ?)
            """,
            (ids["revision_id"], ids["experiment_id"], json.dumps(config), now),
        )
        conn.execute(
            """
            INSERT INTO jobs (
              id, project_id, experiment_revision_id, model_id, type, status,
              attempt_id, progress_json, created_at
            ) VALUES (?, 'local', ?, NULL, 'train', 'running', ?, ?, ?)
            """,
            (ids["job_id"], ids["revision_id"], new_id(), json.dumps(progress), now),
        )
    ids["progress"] = progress
    return ids


def _job_dict(ids: dict, *, job_type: str = "train", model_id: str | None = None) -> dict:
    return {
        "id": ids["job_id"],
        "project_id": "local",
        "type": job_type,
        "experiment_revision_id": ids["revision_id"],
        "model_id": model_id,
        "progress_json": json.dumps(ids["progress"]),
    }


def _read_report(job_id: str) -> dict:
    with get_db() as conn:
        row = conn.execute("SELECT report_json FROM reports WHERE job_id = ?", (job_id,)).fetchone()
    assert row is not None
    return json.loads(row["report_json"])


def test_train_report_includes_provenance_and_dummy_baseline(db, tmp_path: Path):
    ids = _seed_train_job(
        {
            "task": "classification",
            "target": "y",
            "features": ["x"],
            "split": "group",
            "seed": 42,
        }
    )
    result = {
        "metrics": {
            "task": "classification",
            "test": {"macro_f1": 0.8, "balanced_accuracy": 0.75},
            "validation": {"score": 0.7},
            "candidates": {
                "dummy": {"val_score": 0.5, "status": "ok", "name": "dummy"},
                "linear": {"val_score": 0.7, "status": "ok"},
            },
            "split_indices": {"train": [0, 1, 2], "val": [3], "test": [4, 5, 6]},
            "y_true": ["0", "1", "0"],
        },
        "selected_candidate": "linear",
        "warnings": [],
        "plot_files": [],
        "model_path": str(tmp_path / "model.joblib"),
    }
    _store_train_result(_job_dict(ids), {"artifact_dir": str(tmp_path / "art")}, result)
    report = _read_report(ids["job_id"])
    assert report["dataset_version_id"] == ids["version_id"]
    assert report["split"] == "group"
    assert report["n_test"] == 3
    assert report["task"] == "classification"
    assert report["units"] == "none"
    assert report["baseline_metrics"] == {"val_score": 0.5}
    assert "test" not in report["baseline_metrics"]


def test_train_report_preserves_canonical_baseline_bags(db, tmp_path: Path):
    ids = _seed_train_job(
        {"task": "regression", "target": "y", "features": ["x"], "split": "random"}
    )
    baseline = {
        "family": "dummy",
        "validation": {"mae": 2.0, "rmse": 2.2},
        "test": {"mae": 2.5, "rmse": 2.7},
    }
    result = {
        "metrics": {
            "task": "regression",
            "test": {"mae": 1.0, "rmse": 1.1},
            "validation": {"mae": 1.2, "rmse": 1.3},
            "baseline": baseline,
            "y_true": [1.0, 2.0],
        },
        "selected_candidate": "linear",
        "warnings": [],
        "plot_files": [],
    }
    _store_train_result(_job_dict(ids), {"artifact_dir": str(tmp_path / "canonical")}, result)
    report = _read_report(ids["job_id"])
    assert report["baseline_metrics"] == baseline


def test_regression_report_units_from_config_or_default(db, tmp_path: Path):
    labeled = _seed_train_job(
        {
            "task": "regression",
            "target": "y",
            "features": ["x"],
            "split": "random",
            "units": "kg",
        }
    )
    metrics = {
        "task": "regression",
        "test": {"mae": 1.2},
        "candidates": {"dummy": {"val_score": -2.0, "status": "ok"}},
        "y_true": [1.0, 2.0],
    }
    _store_train_result(
        _job_dict(labeled),
        {"artifact_dir": str(tmp_path / "kg")},
        {"metrics": metrics, "selected_candidate": "dummy", "warnings": [], "plot_files": []},
    )
    assert _read_report(labeled["job_id"])["units"] == "kg"

    unlabeled = _seed_train_job(
        {"task": "regression", "target": "y", "features": ["x"], "split": "time"}
    )
    _store_train_result(
        _job_dict(unlabeled),
        {"artifact_dir": str(tmp_path / "plain")},
        {"metrics": metrics, "selected_candidate": "linear", "warnings": [], "plot_files": []},
    )
    report = _read_report(unlabeled["job_id"])
    assert report["units"] == "target units"
    assert report["split"] == "time"


def test_forecast_baseline_from_last_value_candidate(db, tmp_path: Path):
    ids = _seed_train_job(
        {
            "task": "forecast",
            "target": "y",
            "date_column": "d",
            "split": "time",
            "horizon": 7,
        }
    )
    result = {
        "metrics": {
            "task": "forecast",
            "test": {"mae": 4.0},
            "candidates": {
                "last_value": {"val_mae": 5.0, "status": "ok"},
                "xgboost": {"val_mae": 3.0, "status": "ok"},
            },
            "y_true": [1, 2, 3, 4],
        },
        "selected_candidate": "xgboost",
        "warnings": [],
        "plot_files": [],
    }
    _store_train_result(_job_dict(ids), {"artifact_dir": str(tmp_path / "fc")}, result)
    report = _read_report(ids["job_id"])
    assert report["baseline_metrics"] == {"val_mae": 5.0}
    assert report["n_test"] == 4
    assert report["units"] == "target units"


def test_predict_job_stores_metrics_on_report_without_refit(db, tmp_path: Path):
    train_ids = _seed_train_job(
        {"task": "classification", "target": "y", "features": ["x"], "split": "random"}
    )
    now = utcnow()
    model_id = new_id()
    predict_job_id = new_id()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO models (
              id, project_id, job_id, dataset_version_id, task, artifact_path,
              metrics_json, created_at
            ) VALUES (?, 'local', ?, ?, 'classification', 'model.joblib', '{}', ?)
            """,
            (model_id, train_ids["job_id"], train_ids["version_id"], now),
        )
        conn.execute(
            """
            INSERT INTO jobs (
              id, project_id, experiment_revision_id, model_id, type, status,
              attempt_id, progress_json, created_at
            ) VALUES (?, 'local', ?, ?, 'predict', 'running', ?, ?, ?)
            """,
            (
                predict_job_id,
                train_ids["revision_id"],
                model_id,
                new_id(),
                json.dumps(
                    {
                        "input_version_id": train_ids["version_id"],
                        "model_id": model_id,
                    }
                ),
                now,
            ),
        )
    job = {
        "id": predict_job_id,
        "project_id": "local",
        "model_id": model_id,
        "progress_json": json.dumps(
            {"input_version_id": train_ids["version_id"], "model_id": model_id}
        ),
    }
    metrics = {"task": "classification", "test": {"macro_f1": 0.91}}
    _store_predict_result(
        job,
        {"artifact_dir": str(tmp_path / "pred")},
        {
            "output_path": str(tmp_path / "pred" / "predictions.csv"),
            "metrics": metrics,
            "warnings": ["Ignored extra columns: ['z']"],
            "selected_candidate": "linear",
            "plot_files": [],
        },
    )
    report = _read_report(predict_job_id)
    assert report["metrics"] == metrics
    assert report["task"] == "classification"
    with get_db() as conn:
        model = conn.execute("SELECT artifact_path FROM models WHERE id = ?", (model_id,)).fetchone()
        pred = conn.execute(
            "SELECT id FROM predictions WHERE job_id = ?", (predict_job_id,)
        ).fetchone()
    assert model["artifact_path"] == "model.joblib"
    assert pred is not None


def test_predict_without_metrics_does_not_invent_report(db, tmp_path: Path):
    train_ids = _seed_train_job(
        {"task": "classification", "target": "y", "features": ["x"], "split": "random"}
    )
    now = utcnow()
    model_id = new_id()
    predict_job_id = new_id()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO models (
              id, project_id, job_id, dataset_version_id, task, artifact_path,
              metrics_json, created_at
            ) VALUES (?, 'local', ?, ?, 'classification', 'model.joblib', '{}', ?)
            """,
            (model_id, train_ids["job_id"], train_ids["version_id"], now),
        )
        conn.execute(
            """
            INSERT INTO jobs (
              id, project_id, experiment_revision_id, model_id, type, status,
              attempt_id, progress_json, created_at
            ) VALUES (?, 'local', ?, ?, 'predict', 'running', ?, ?, ?)
            """,
            (
                predict_job_id,
                train_ids["revision_id"],
                model_id,
                new_id(),
                json.dumps({"input_version_id": train_ids["version_id"]}),
                now,
            ),
        )
    _store_predict_result(
        {
            "id": predict_job_id,
            "project_id": "local",
            "model_id": model_id,
            "progress_json": json.dumps({"input_version_id": train_ids["version_id"]}),
        },
        {"artifact_dir": str(tmp_path / "pred2")},
        {"output_path": str(tmp_path / "pred2" / "predictions.csv"), "metrics": None},
    )
    with get_db() as conn:
        row = conn.execute("SELECT id FROM reports WHERE job_id = ?", (predict_job_id,)).fetchone()
    assert row is None
