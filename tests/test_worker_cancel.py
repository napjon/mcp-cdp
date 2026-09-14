from __future__ import annotations

import multiprocessing
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi.testclient import TestClient

POST_HEADERS = {
    "Origin": "http://127.0.0.1:5173",
    "X-Requested-With": "mcp-cdp",
}

CSV = (
    b"id,y,x\n"
    b"00123,1,0.2\n00124,0,0.8\n00125,1,0.4\n00126,0,0.6\n"
    b"00127,1,0.3\n00128,0,0.7\n00129,1,0.5\n00130,0,0.9\n"
    b"00131,1,0.1\n00132,0,0.55\n"
)


def slow_ml_job(job_dict, paths):
    time.sleep(60)
    artifact = Path(paths["artifact_dir"])
    artifact.mkdir(parents=True, exist_ok=True)
    model = artifact / "model.joblib"
    model.write_bytes(b"stub")
    return {
        "metrics": {"macro_f1": 1.0},
        "model_path": str(model),
        "plot_files": [],
        "warnings": [],
        "selected_candidate": "dummy",
        "task": "classification",
    }


def write_marker_after_sleep(path: str, delay: float) -> None:
    time.sleep(delay)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("stale", encoding="utf-8")


def bulky_ml_job(job_dict, paths):
    artifact = Path(paths["artifact_dir"])
    artifact.mkdir(parents=True, exist_ok=True)
    model = artifact / "model.joblib"
    model.write_bytes(b"stub")
    return {
        "metrics": {"blob": "x" * (4 * 1024 * 1024), "macro_f1": 1.0},
        "model_path": str(model),
        "plot_files": [],
        "warnings": [],
        "selected_candidate": "dummy",
        "task": "classification",
    }


def _dataset_and_experiment(client) -> tuple[str, str]:
    uploaded = client.post(
        "/api/projects/local/datasets/upload",
        files={"file": ("train.csv", CSV, "text/csv")},
        headers=POST_HEADERS,
    )
    assert uploaded.status_code == 200, uploaded.text
    dataset_id = uploaded.json()["id"]
    exp = client.post(
        "/api/projects/local/experiments",
        json={
            "dataset_id": dataset_id,
            "config": {
                "task": "classification",
                "target": "y",
                "features": ["x"],
                "budget": "quick",
                "seed": 42,
            },
        },
        headers=POST_HEADERS,
    )
    assert exp.status_code == 200, exp.text
    return dataset_id, exp.json()["id"]


def test_cancel_queued_job(client):
    _, experiment_id = _dataset_and_experiment(client)
    submitted = client.post(
        f"/api/projects/local/experiments/{experiment_id}/submit",
        headers=POST_HEADERS,
    )
    job_id = submitted.json()["job_id"]
    canceled = client.post(
        f"/api/projects/local/jobs/{job_id}/cancel",
        headers=POST_HEADERS,
    )
    assert canceled.status_code == 200, canceled.text
    assert canceled.json()["status"] == "canceled"
    again = client.get(f"/api/projects/local/jobs/{job_id}")
    assert again.json()["status"] == "canceled"


def test_cancel_running_terminates_process(db):
    from app.db import get_db, new_id, utcnow
    from app.services.jobs import attach_job_process, cancel_job, get_job

    job_id = new_id()
    now = utcnow()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO jobs (
              id, project_id, experiment_revision_id, model_id, type, status,
              attempt_id, lease_until, heartbeat_at, progress_json, error,
              created_at, started_at, finished_at
            ) VALUES (?, 'local', NULL, NULL, 'train', 'running', ?, ?, ?, '{}', NULL, ?, ?, NULL)
            """,
            (job_id, new_id(), now, now, now, now),
        )
    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(target=time.sleep, args=(60,), daemon=True)
    proc.start()
    attach_job_process(job_id, proc)
    try:
        out = cancel_job("local", job_id)
        assert out["status"] == "canceled"
        proc.join(timeout=5)
        assert not proc.is_alive()
        assert get_job("local", job_id)["status"] == "canceled"
    finally:
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=2)
        attach_job_process(job_id, None)


def test_cancel_running_job_does_not_succeed(data_dir, monkeypatch):
    monkeypatch.setattr("app.worker.run_ml_job", slow_ml_job)
    from app.main import create_app

    application = create_app(worker_enabled=True)
    with TestClient(application, base_url="http://127.0.0.1:8765") as client:
        _, experiment_id = _dataset_and_experiment(client)
        submitted = client.post(
            f"/api/projects/local/experiments/{experiment_id}/submit",
            headers=POST_HEADERS,
        )
        assert submitted.status_code == 200, submitted.text
        job_id = submitted.json()["job_id"]
        body = None
        for _ in range(100):
            body = client.get(f"/api/projects/local/jobs/{job_id}").json()
            if body["status"] == "running":
                break
            time.sleep(0.1)
        assert body is not None
        assert body["status"] == "running", body
        canceled = client.post(
            f"/api/projects/local/jobs/{job_id}/cancel",
            headers=POST_HEADERS,
        )
        assert canceled.status_code == 200, canceled.text
        assert canceled.json()["status"] == "canceled"
        for _ in range(30):
            status = client.get(f"/api/projects/local/jobs/{job_id}").json()["status"]
            assert status != "succeeded"
            if status == "canceled":
                break
            time.sleep(0.1)
        time.sleep(0.5)
        final = client.get(f"/api/projects/local/jobs/{job_id}").json()
        assert final["status"] == "canceled"


def test_ml_child_large_result_does_not_deadlock(data_dir, monkeypatch):
    monkeypatch.setattr("app.worker.run_ml_job", bulky_ml_job)
    from app.main import create_app

    application = create_app(worker_enabled=True)
    with TestClient(application, base_url="http://127.0.0.1:8765") as client:
        _, experiment_id = _dataset_and_experiment(client)
        submitted = client.post(
            f"/api/projects/local/experiments/{experiment_id}/submit",
            headers=POST_HEADERS,
        )
        assert submitted.status_code == 200, submitted.text
        job_id = submitted.json()["job_id"]
        body = None
        for _ in range(100):
            body = client.get(f"/api/projects/local/jobs/{job_id}").json()
            if body["status"] in {"succeeded", "failed", "canceled"}:
                break
            time.sleep(0.1)
        assert body is not None
        assert body["status"] == "succeeded", body


def _iso_offset(hours: int) -> str:
    return (
        (datetime.now(UTC) + timedelta(hours=hours))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _insert_running_job(*, lease_until: str | None, attempt_id: str | None = None) -> tuple[str, str]:
    from app.db import get_db, new_id, utcnow

    job_id = new_id()
    attempt = attempt_id or new_id()
    now = utcnow()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO jobs (
              id, project_id, experiment_revision_id, model_id, type, status,
              attempt_id, lease_until, heartbeat_at, progress_json, error,
              created_at, started_at, finished_at
            ) VALUES (?, 'local', NULL, NULL, 'train', 'running', ?, ?, ?, '{}', NULL, ?, ?, NULL)
            """,
            (job_id, attempt, lease_until, now, now, now),
        )
    return job_id, attempt


def _attempt_job(job_id: str, attempt_id: str) -> dict:
    return {
        "id": job_id,
        "type": "train",
        "attempt_id": attempt_id,
        "progress_json": "{}",
        "experiment_revision_id": None,
        "model_id": None,
    }


def test_recover_does_not_write_previous_attempt_dir(db):
    from app.db import get_db
    from app.services.jobs import attach_job_process, recover_stale_jobs
    from app.worker import _build_call

    job_id, attempt = _insert_running_job(lease_until=_iso_offset(-1))
    _, paths = _build_call(_attempt_job(job_id, attempt))
    old_dir = Path(paths["artifact_dir"])
    marker = old_dir / "late.txt"
    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(target=write_marker_after_sleep, args=(str(marker), 3.0), daemon=True)
    proc.start()
    attach_job_process(job_id, proc, attempt_id=attempt)
    try:
        recover_stale_jobs()
        proc.join(timeout=5)
        assert not proc.is_alive()
        assert not marker.exists()
        with get_db() as conn:
            row = conn.execute(
                "SELECT status, attempt_id FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        assert row["status"] == "queued"
        assert row["attempt_id"] != attempt
        _, new_paths = _build_call(_attempt_job(job_id, row["attempt_id"]))
        assert Path(new_paths["artifact_dir"]) != old_dir
        assert not marker.exists()
        assert not (old_dir / "late.txt").exists()
    finally:
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=2)
        attach_job_process(job_id, None)


def test_cancel_does_not_write_previous_attempt_dir(db):
    from app.services.jobs import attach_job_process, cancel_job
    from app.worker import _build_call

    job_id, attempt = _insert_running_job(lease_until=_iso_offset(1))
    _, paths = _build_call(_attempt_job(job_id, attempt))
    old_dir = Path(paths["artifact_dir"])
    marker = old_dir / "late.txt"
    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(target=write_marker_after_sleep, args=(str(marker), 3.0), daemon=True)
    proc.start()
    attach_job_process(job_id, proc, attempt_id=attempt)
    try:
        out = cancel_job("local", job_id)
        assert out["status"] == "canceled"
        proc.join(timeout=5)
        assert not proc.is_alive()
        assert not marker.exists()
        assert not (old_dir / "late.txt").exists()
    finally:
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=2)
        attach_job_process(job_id, None)


def test_recover_does_not_terminate_replacement_attempt(db):
    from app.db import new_id
    from app.services.jobs import attach_job_process, terminate_job_process

    job_id, old_attempt = _insert_running_job(lease_until=_iso_offset(-1))
    replacement_attempt = new_id()
    ctx = multiprocessing.get_context("spawn")
    old_proc = ctx.Process(target=time.sleep, args=(60,), daemon=True)
    new_proc = ctx.Process(target=time.sleep, args=(60,), daemon=True)
    old_proc.start()
    new_proc.start()
    attach_job_process(job_id, old_proc, attempt_id=old_attempt)
    attach_job_process(job_id, new_proc, attempt_id=replacement_attempt)
    try:
        terminate_job_process(job_id, attempt_id=old_attempt)
        time.sleep(0.2)
        assert new_proc.is_alive()
        assert old_proc.is_alive()
        terminate_job_process(job_id, attempt_id=replacement_attempt)
        new_proc.join(timeout=5)
        assert not new_proc.is_alive()
    finally:
        for proc in (old_proc, new_proc):
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=2)
        attach_job_process(job_id, None)
