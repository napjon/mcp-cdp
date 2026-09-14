from __future__ import annotations

import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

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


def test_enqueue_returns_queued_job_immediately(client):
    _, experiment_id = _dataset_and_experiment(client)
    submitted = client.post(
        f"/api/projects/local/experiments/{experiment_id}/submit",
        headers=POST_HEADERS,
    )
    assert submitted.status_code == 200, submitted.text
    body = submitted.json()
    assert "job_id" in body
    assert body["status"] == "queued"
    job = client.get(f"/api/projects/local/jobs/{body['job_id']}")
    assert job.status_code == 200
    assert job.json()["status"] in {"queued", "running", "failed"}
    assert job.json()["type"] == "train"


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


def test_queue_max_10(client):
    _, experiment_id = _dataset_and_experiment(client)
    for _ in range(10):
        response = client.post(
            f"/api/projects/local/experiments/{experiment_id}/submit",
            headers=POST_HEADERS,
        )
        assert response.status_code == 200, response.text
    full = client.post(
        f"/api/projects/local/experiments/{experiment_id}/submit",
        headers=POST_HEADERS,
    )
    assert full.status_code == 429
    assert full.json()["code"] == "queue_full"


def test_worker_fails_when_ml_missing(worker_client, monkeypatch):
    def _missing():
        raise ImportError("ml engine not loaded")

    monkeypatch.setattr("app.worker.load_ml", _missing)
    _, experiment_id = _dataset_and_experiment(worker_client)
    submitted = worker_client.post(
        f"/api/projects/local/experiments/{experiment_id}/submit",
        headers=POST_HEADERS,
    )
    job_id = submitted.json()["job_id"]
    body = None
    for _ in range(50):
        body = worker_client.get(f"/api/projects/local/jobs/{job_id}").json()
        if body["status"] in {"failed", "succeeded", "canceled"}:
            break
        time.sleep(0.1)
    assert body is not None
    assert body["status"] == "failed"
    assert body["error"] == "ml engine not loaded"


def _job_count() -> int:
    from app.db import get_db

    with get_db() as conn:
        return conn.execute("SELECT COUNT(*) AS c FROM jobs").fetchone()["c"]


def test_submit_rejects_single_class_without_queueing(client):
    from app.db import get_db, new_id, utcnow

    uploaded = client.post(
        "/api/projects/local/datasets/upload",
        files={"file": ("oneclass.csv", b"id,y,x\n1,yes,0.1\n2,yes,0.2\n3,yes,0.3\n", "text/csv")},
        headers=POST_HEADERS,
    )
    assert uploaded.status_code == 200, uploaded.text
    dataset_id = uploaded.json()["id"]
    experiment_id = new_id()
    revision_id = new_id()
    now = utcnow()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO experiments (id, project_id, dataset_id, created_at) VALUES (?, 'local', ?, ?)",
            (experiment_id, dataset_id, now),
        )
        conn.execute(
            """
            INSERT INTO experiment_revisions (id, experiment_id, revision, config_json, created_at)
            VALUES (?, ?, 1, ?, ?)
            """,
            (
                revision_id,
                experiment_id,
                '{"task":"classification","target":"y","features":["x"],"budget":"quick"}',
                now,
            ),
        )
    before = _job_count()
    submitted = client.post(
        f"/api/projects/local/experiments/{experiment_id}/submit",
        headers=POST_HEADERS,
    )
    assert submitted.status_code == 400
    assert submitted.json()["code"] == "single_class"
    assert _job_count() == before


def test_formula_injection_escape():
    from app.services.predict import escape_csv_text, escape_formula_cell

    assert escape_formula_cell("=1+1") == "'=1+1"
    assert escape_formula_cell("+cmd") == "'+cmd"
    assert escape_formula_cell("ok") == "ok"
    escaped = escape_csv_text("name,val\n=cmd,1\nhello,2\n")
    lines = escaped.strip().splitlines()
    assert lines[0] == "name,val"
    assert lines[1].startswith("'=") or lines[1].startswith("\"'=")
    assert "hello" in lines[2]


def test_predict_missing_required_columns(client):
    from app.db import get_db, new_id, utcnow

    dataset_id, _ = _dataset_and_experiment(client)
    ds = client.get(f"/api/projects/local/datasets/{dataset_id}").json()
    version_id = ds["current_version"]["id"]
    job_id = new_id()
    model_id = new_id()
    now = utcnow()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO jobs (
              id, project_id, experiment_revision_id, model_id, type, status,
              attempt_id, progress_json, created_at
            ) VALUES (?, 'local', NULL, NULL, 'train', 'succeeded', ?, '{}', ?)
            """,
            (job_id, new_id(), now),
        )
        conn.execute(
            """
            INSERT INTO models (
              id, project_id, job_id, dataset_version_id, task, artifact_path,
              metrics_json, created_at
            ) VALUES (?, 'local', ?, ?, 'classification', 'x', '{}', ?)
            """,
            (model_id, job_id, version_id, now),
        )
    response = client.post(
        "/api/projects/local/predict",
        files={"file": ("p.csv", b"id,z\n1,2\n", "text/csv")},
        data={"model_id": model_id},
        headers=POST_HEADERS,
    )
    assert response.status_code == 400
    assert "missing required columns" in response.json()["error"]


def test_prediction_download_escapes_formulas(client):
    from pathlib import Path

    from app.db import get_db, new_id, utcnow
    from app.settings import get_settings

    dataset_id, _ = _dataset_and_experiment(client)
    ds = client.get(f"/api/projects/local/datasets/{dataset_id}").json()
    version_id = ds["current_version"]["id"]
    job_id = new_id()
    model_id = new_id()
    pred_id = new_id()
    now = utcnow()
    out = get_settings().artifacts_dir / job_id
    out.mkdir(parents=True, exist_ok=True)
    path = out / "predictions.csv"
    path.write_text("pred,note\n1,=1+1\n", encoding="utf-8")
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO jobs (
              id, project_id, experiment_revision_id, model_id, type, status,
              attempt_id, progress_json, created_at, finished_at
            ) VALUES (?, 'local', NULL, ?, 'predict', 'succeeded', ?, '{}', ?, ?)
            """,
            (job_id, model_id, new_id(), now, now),
        )
        conn.execute(
            """
            INSERT INTO models (
              id, project_id, job_id, dataset_version_id, task, artifact_path,
              metrics_json, created_at
            ) VALUES (?, 'local', ?, ?, 'classification', 'x', '{}', ?)
            """,
            (model_id, job_id, version_id, now),
        )
        conn.execute(
            """
            INSERT INTO predictions (id, job_id, model_id, input_version_id, output_path, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (pred_id, job_id, model_id, version_id, str(path), now),
        )
    response = client.get(f"/api/projects/local/predictions/{pred_id}/download")
    assert response.status_code == 200
    text = response.content.decode()
    assert "'=1+1" in text
    assert Path(path).read_text(encoding="utf-8").splitlines()[1].startswith("1,=1+1")


def test_predict_align_preserves_identifiers_drops_extras_keeps_order():
    from app.services.predict import align_columns

    headers = ["id", "x", "extra", "y"]
    rows = [
        ["00123", "0.2", "noise", "1"],
        ["00124", "0.8", "noise", "0"],
        ["00125", "", "noise", "1"],
    ]
    out_h, out_r = align_columns(
        headers,
        rows,
        ["x"],
        roles={"x": "numeric", "id": "identifier", "y": "categorical"},
        target="y",
    )
    assert "extra" not in out_h
    assert out_h == ["x", "id", "y"]
    assert [row[out_h.index("id")] for row in out_r] == ["00123", "00124", "00125"]
    assert [row[out_h.index("x")] for row in out_r] == ["0.2", "0.8", ""]


def test_predict_numeric_conversion_failure_errors():
    from app.models import AppError
    from app.services.predict import align_columns

    with pytest.raises(AppError, match="conversion failed for numeric feature x"):
        align_columns(
            ["id", "x"],
            [["00123", "not-a-number"]],
            ["x"],
            roles={"x": "numeric", "id": "identifier"},
        )


def test_predict_submit_conversion_and_identifier_file(client):
    from app.db import get_db, new_id, utcnow

    dataset_id, experiment_id = _dataset_and_experiment(client)
    ds = client.get(f"/api/projects/local/datasets/{dataset_id}").json()
    version_id = ds["current_version"]["id"]
    exp = client.get(f"/api/projects/local/experiments/{experiment_id}").json()
    revision_id = exp["current_revision"]["id"]
    job_id = new_id()
    model_id = new_id()
    now = utcnow()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO jobs (
              id, project_id, experiment_revision_id, model_id, type, status,
              attempt_id, progress_json, created_at
            ) VALUES (?, 'local', ?, NULL, 'train', 'succeeded', ?, '{}', ?)
            """,
            (job_id, revision_id, new_id(), now),
        )
        conn.execute(
            """
            INSERT INTO models (
              id, project_id, job_id, dataset_version_id, task, artifact_path,
              metrics_json, created_at
            ) VALUES (?, 'local', ?, ?, 'classification', 'x', '{}', ?)
            """,
            (model_id, job_id, version_id, now),
        )
    bad = client.post(
        "/api/projects/local/predict",
        files={"file": ("p.csv", b"id,x,extra\n00123,abc,drop-me\n", "text/csv")},
        data={"model_id": model_id},
        headers=POST_HEADERS,
    )
    assert bad.status_code == 400
    assert "conversion failed" in bad.json()["error"]

    ok = client.post(
        "/api/projects/local/predict",
        files={
            "file": (
                "p.csv",
                b"id,x,extra\n00123,0.2,drop-me\n00124,0.8,drop-me\n",
                "text/csv",
            )
        },
        data={"model_id": model_id},
        headers=POST_HEADERS,
    )
    assert ok.status_code == 200, ok.text
    queued = client.get(f"/api/projects/local/jobs/{ok.json()['job_id']}").json()
    aligned = Path(queued["progress"]["predict_path"]).read_text(encoding="utf-8")
    assert "00123" in aligned
    assert "00124" in aligned
    assert "drop-me" not in aligned
    lines = [line for line in aligned.splitlines() if line.strip()]
    assert lines[0].split(",")[0] == "x" or "id" in lines[0]
    body_rows = lines[1:]
    assert body_rows[0].split(",")[0] in {"00123", "0.2"}
    assert "00123" in body_rows[0]
    assert "00124" in body_rows[1]


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


def test_finish_job_ignores_mismatched_attempt(db):
    from app.db import get_db
    from app.services.jobs import finish_job

    job_id, attempt = _insert_running_job(lease_until=_iso_offset(1))
    finish_job(job_id, status="succeeded", attempt_id="not-the-attempt")
    with get_db() as conn:
        row = conn.execute(
            "SELECT status, finished_at FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
    assert row["status"] == "running"
    assert row["finished_at"] is None

    finish_job(job_id, status="succeeded", attempt_id=attempt)
    with get_db() as conn:
        row = conn.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
    assert row["status"] == "succeeded"


def test_recover_stale_jobs_only_expired_or_null_leases(db):
    from app.db import get_db
    from app.services.jobs import recover_stale_jobs

    live_id, live_attempt = _insert_running_job(lease_until=_iso_offset(1))
    stale_id, stale_attempt = _insert_running_job(lease_until=_iso_offset(-1))
    null_id, null_attempt = _insert_running_job(lease_until=None)

    recover_stale_jobs()
    with get_db() as conn:
        live = conn.execute(
            "SELECT status, attempt_id, lease_until FROM jobs WHERE id = ?", (live_id,)
        ).fetchone()
        stale = conn.execute(
            "SELECT status, attempt_id, lease_until FROM jobs WHERE id = ?", (stale_id,)
        ).fetchone()
        missing = conn.execute(
            "SELECT status, attempt_id, lease_until FROM jobs WHERE id = ?", (null_id,)
        ).fetchone()
    assert live["status"] == "running"
    assert live["attempt_id"] == live_attempt
    assert live["lease_until"] is not None
    assert stale["status"] == "queued"
    assert stale["attempt_id"] != stale_attempt
    assert stale["lease_until"] is None
    assert missing["status"] == "queued"
    assert missing["attempt_id"] != null_attempt
    assert missing["lease_until"] is None


def _insert_queued_job() -> str:
    from app.db import get_db, new_id, utcnow

    job_id = new_id()
    now = utcnow()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO jobs (
              id, project_id, experiment_revision_id, model_id, type, status,
              attempt_id, lease_until, heartbeat_at, progress_json, error,
              created_at, started_at, finished_at
            ) VALUES (?, 'local', NULL, NULL, 'train', 'queued', ?, NULL, NULL, '{}', NULL, ?, NULL, NULL)
            """,
            (job_id, new_id(), now),
        )
    return job_id


def test_claim_next_job_atomic_one_running(db):
    from app.db import get_db
    from app.services.jobs import claim_next_job

    for _ in range(5):
        _insert_queued_job()

    n_threads = 16
    barrier = threading.Barrier(n_threads, timeout=10)
    results: list = [None] * n_threads
    errors: list = []

    def _claim(index: int) -> None:
        try:
            barrier.wait()
            results[index] = claim_next_job()
        except Exception as exc:  # noqa: BLE001 — surface thread failures in the parent
            errors.append(exc)

    threads = [threading.Thread(target=_claim, args=(i,)) for i in range(n_threads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
        assert not thread.is_alive()

    assert errors == []
    claimed = [row for row in results if row is not None]
    assert len(claimed) == 1
    with get_db() as conn:
        running = conn.execute(
            "SELECT COUNT(*) AS c FROM jobs WHERE status = 'running'"
        ).fetchone()["c"]
        queued = conn.execute(
            "SELECT COUNT(*) AS c FROM jobs WHERE status = 'queued'"
        ).fetchone()["c"]
    assert running == 1
    assert queued == 4
    assert claimed[0]["status"] == "running"


def test_two_attempts_do_not_share_artifact_dir(db):
    from app.db import new_id
    from app.worker import _build_call

    job_id = new_id()
    first, second = new_id(), new_id()
    base = {
        "id": job_id,
        "type": "train",
        "progress_json": "{}",
        "experiment_revision_id": None,
        "model_id": None,
    }
    _, paths_a = _build_call({**base, "attempt_id": first})
    _, paths_b = _build_call({**base, "attempt_id": second})
    dir_a = Path(paths_a["artifact_dir"])
    dir_b = Path(paths_b["artifact_dir"])
    assert dir_a != dir_b
    assert dir_a.name == first
    assert dir_b.name == second
    assert dir_a.parent == dir_b.parent
    assert dir_a.parent.name == job_id
    assert dir_a.parent.parent.name == "artifacts"


def test_enqueue_job_concurrent_respects_max_queue(db):
    from app.db import get_db
    from app.models import AppError
    from app.services.jobs import enqueue_job
    from app.settings import get_settings

    max_queue = get_settings().max_queue
    n_threads = max_queue * 3
    barrier = threading.Barrier(n_threads, timeout=10)
    accepted: list = []
    rejected: list = []
    errors: list = []

    def _enqueue() -> None:
        try:
            barrier.wait()
            accepted.append(enqueue_job("local", job_type="train"))
        except AppError as exc:
            if exc.code == "queue_full" and exc.status_code == 429:
                rejected.append(exc)
            else:
                errors.append(exc)
        except Exception as exc:  # noqa: BLE001 — surface thread failures in the parent
            errors.append(exc)

    threads = [threading.Thread(target=_enqueue) for _ in range(n_threads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
        assert not thread.is_alive()

    assert errors == []
    assert len(accepted) == max_queue
    assert len(rejected) == n_threads - max_queue
    with get_db() as conn:
        queued = conn.execute(
            "SELECT COUNT(*) AS c FROM jobs WHERE status = 'queued'"
        ).fetchone()["c"]
        total = conn.execute("SELECT COUNT(*) AS c FROM jobs").fetchone()["c"]
    assert queued == max_queue
    assert total == max_queue
