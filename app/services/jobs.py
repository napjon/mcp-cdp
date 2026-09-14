"""Job queue: max 10 queued, one running, lease/heartbeat, cancel."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from datetime import UTC

from app.db import get_db, new_id, require_project, row_dict, utcnow
from app.models import TERMINAL_JOB_STATUSES, AppError
from app.services.datasets import get_dataset
from app.services.experiments import get_experiment, latest_revision, validate_config
from app.settings import get_settings

LEASE_SECONDS = 60

_job_processes: dict[str, object] = {}
_job_processes_lock = threading.Lock()


def attach_job_process(job_id: str, proc: object | None) -> None:
    with _job_processes_lock:
        if proc is None:
            _job_processes.pop(job_id, None)
        else:
            _job_processes[job_id] = proc


def terminate_job_process(job_id: str) -> None:
    with _job_processes_lock:
        proc = _job_processes.get(job_id)
    _terminate_proc(proc)


def terminate_all_job_processes() -> None:
    with _job_processes_lock:
        procs = list(_job_processes.values())
    for proc in procs:
        _terminate_proc(proc)


def _terminate_proc(proc: object | None) -> None:
    if proc is None:
        return
    try:
        is_alive = getattr(proc, "is_alive", None)
        if callable(is_alive) and not is_alive():
            return
        terminate = getattr(proc, "terminate", None)
        if callable(terminate):
            terminate()
    except OSError:
        pass


def enqueue_job(
    project_id: str,
    *,
    job_type: str,
    experiment_revision_id: str | None = None,
    model_id: str | None = None,
    progress: dict | None = None,
) -> dict:
    if job_type not in {"train", "predict"}:
        raise AppError("invalid job type")
    with get_db() as conn:
        require_project(conn, project_id)
        queued = conn.execute(
            "SELECT COUNT(*) AS c FROM jobs WHERE status = 'queued'"
        ).fetchone()["c"]
        if queued >= get_settings().max_queue:
            raise AppError("job queue is full", status_code=429, code="queue_full")
        job_id = new_id()
        now = utcnow()
        conn.execute(
            """
            INSERT INTO jobs (
              id, project_id, experiment_revision_id, model_id, type, status,
              attempt_id, lease_until, heartbeat_at, progress_json, error,
              created_at, started_at, finished_at
            ) VALUES (?, ?, ?, ?, ?, 'queued', ?, NULL, NULL, ?, NULL, ?, NULL, NULL)
            """,
            (
                job_id,
                project_id,
                experiment_revision_id,
                model_id,
                job_type,
                new_id(),
                json.dumps(progress or {}),
                now,
            ),
        )
        return get_job(project_id, job_id, conn=conn)


def submit_training(project_id: str, experiment_id: str) -> dict:
    with get_db() as conn:
        exp = get_experiment(project_id, experiment_id, conn=conn)
        rev = latest_revision(conn, experiment_id)
        if not rev:
            raise AppError("experiment has no revision")
        dataset = get_dataset(project_id, exp["dataset_id"], conn=conn)
        raw_config = rev["config_json"]
        config = json.loads(raw_config) if raw_config else {}
        validate_config(config, dataset)
        version = conn.execute(
            """
            SELECT id FROM dataset_versions
            WHERE dataset_id = ? ORDER BY version DESC LIMIT 1
            """,
            (exp["dataset_id"],),
        ).fetchone()
        progress = {
            "dataset_id": exp["dataset_id"],
            "dataset_version_id": version["id"] if version else None,
            "experiment_id": experiment_id,
            "revision": rev["revision"],
        }
        revision_id = rev["id"]
    job = enqueue_job(
        project_id,
        job_type="train",
        experiment_revision_id=revision_id,
        progress=progress,
    )
    return {"job_id": job["id"], "status": "queued"}


def get_job(project_id: str, job_id: str, conn=None) -> dict:
    def _load(c) -> dict:
        require_project(c, project_id)
        row = c.execute(
            "SELECT * FROM jobs WHERE id = ? AND project_id = ?",
            (job_id, project_id),
        ).fetchone()
        if not row:
            raise AppError("job not found", status_code=404)
        out = row_dict(row)
        raw_progress = out.get("progress_json")
        if raw_progress:
            try:
                out["progress"] = json.loads(raw_progress)
            except json.JSONDecodeError:
                out["progress"] = {}
        else:
            out["progress"] = {}
        pred = c.execute(
            "SELECT id FROM predictions WHERE job_id = ? ORDER BY created_at DESC LIMIT 1",
            (job_id,),
        ).fetchone()
        if pred:
            out["prediction_id"] = pred["id"]
            out["result"] = {
                "prediction_id": pred["id"],
                "model_id": out.get("model_id"),
            }
        return out

    if conn is not None:
        return _load(conn)
    with get_db() as c:
        return _load(c)


def cancel_job(project_id: str, job_id: str) -> dict:
    with get_db() as conn:
        job = get_job(project_id, job_id, conn=conn)
        if job["status"] in TERMINAL_JOB_STATUSES:
            raise AppError("job already finished", status_code=409)
        conn.execute(
            """
            UPDATE jobs
            SET status = 'canceled', finished_at = ?, error = NULL
            WHERE id = ? AND status IN ('queued', 'running')
            """,
            (utcnow(), job_id),
        )
        out = get_job(project_id, job_id, conn=conn)
    terminate_job_process(job_id)
    return out


def claim_next_job() -> dict | None:
    attempt = new_id()
    now = utcnow()
    lease = _lease_deadline()
    with get_db() as conn:
        # isolation_level None so Python does not emit a second BEGIN around DML.
        conn.isolation_level = None
        conn.execute("BEGIN IMMEDIATE")
        queued = conn.execute(
            "SELECT id FROM jobs WHERE status = 'queued' ORDER BY created_at ASC LIMIT 1"
        ).fetchone()
        if not queued:
            return None
        cur = conn.execute(
            """
            UPDATE jobs
            SET status = 'running', attempt_id = ?, lease_until = ?, heartbeat_at = ?,
                started_at = COALESCE(started_at, ?)
            WHERE id = ? AND status = 'queued'
              AND NOT EXISTS (SELECT 1 FROM jobs WHERE status = 'running')
            """,
            (attempt, lease, now, now, queued["id"]),
        )
        if cur.rowcount != 1:
            return None
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (queued["id"],)).fetchone()
        return row_dict(row)


def heartbeat(job_id: str, attempt_id: str | None = None) -> None:
    with get_db() as conn:
        sql = """
            UPDATE jobs SET heartbeat_at = ?, lease_until = ?
            WHERE id = ? AND status = 'running'
        """
        params: list = [utcnow(), _lease_deadline(), job_id]
        if attempt_id is not None:
            sql += " AND attempt_id = ?"
            params.append(attempt_id)
        conn.execute(sql, params)


def job_status(job_id: str) -> str | None:
    with get_db() as conn:
        row = conn.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return row["status"] if row else None


def attempt_is_current(job_id: str, attempt_id: str | None, conn=None) -> bool:
    def _check(c) -> bool:
        row = c.execute(
            "SELECT status, attempt_id FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        if not row or row["status"] != "running":
            return False
        return attempt_id is None or row["attempt_id"] == attempt_id

    if conn is not None:
        return _check(conn)
    with get_db() as c:
        return _check(c)


def finish_job(
    job_id: str,
    *,
    status: str,
    error: str | None = None,
    progress: dict | None = None,
    attempt_id: str | None = None,
    conn=None,
) -> bool:
    def _run(c) -> bool:
        current = c.execute(
            "SELECT status, attempt_id FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        if not current:
            return False
        if current["status"] == "canceled":
            return False
        if attempt_id is not None and current["attempt_id"] != attempt_id:
            return False
        fields = ["status = ?", "finished_at = ?", "error = ?"]
        params: list = [status, utcnow(), error]
        if progress is not None:
            fields.append("progress_json = ?")
            params.append(json.dumps(progress))
        where = "id = ? AND status = 'running'"
        params.append(job_id)
        if attempt_id is not None:
            where += " AND attempt_id = ?"
            params.append(attempt_id)
        cur = c.execute(
            f"UPDATE jobs SET {', '.join(fields)} WHERE {where}",
            params,
        )
        return cur.rowcount == 1

    if conn is not None:
        return _run(conn)
    with get_db() as c:
        return _run(c)


def update_progress(job_id: str, progress: dict) -> None:
    with get_db() as conn:
        conn.execute(
            "UPDATE jobs SET progress_json = ? WHERE id = ? AND status = 'running'",
            (json.dumps(progress), job_id),
        )


def recover_stale_jobs() -> None:
    now = utcnow()
    with get_db() as conn:
        conn.execute(
            """
            UPDATE jobs SET status = 'queued', lease_until = NULL, attempt_id = ?
            WHERE status = 'running'
              AND (lease_until IS NULL OR lease_until < ?)
            """,
            (new_id(), now),
        )


def get_report(project_id: str, job_id: str) -> dict:
    with get_db() as conn:
        get_job(project_id, job_id, conn=conn)
        row = conn.execute("SELECT * FROM reports WHERE job_id = ?", (job_id,)).fetchone()
        if not row:
            raise AppError("report not found", status_code=404)
        out = row_dict(row)
        try:
            out["report"] = json.loads(row["report_json"]) if row["report_json"] else {}
        except json.JSONDecodeError:
            out["report"] = {}
        out.pop("report_json", None)
        return out


def iter_job_events(project_id: str, job_id: str) -> Iterator[str]:
    last = None
    while True:
        job = get_job(project_id, job_id)
        snapshot = {
            "type": "status",
            "job_id": job["id"],
            "status": job["status"],
            "progress": job.get("progress") or {},
            "error": job.get("error"),
        }
        encoded = json.dumps(snapshot)
        if encoded != last:
            yield f"data: {encoded}\n\n"
            last = encoded
        if job["status"] in TERMINAL_JOB_STATUSES:
            done = json.dumps({"type": "done", "job_id": job["id"], "status": job["status"]})
            yield f"data: {done}\n\n"
            return
        time.sleep(0.4)


def _lease_deadline() -> str:
    from datetime import datetime, timedelta

    return (
        (datetime.now(UTC) + timedelta(seconds=LEASE_SECONDS))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
