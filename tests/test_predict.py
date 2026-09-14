from __future__ import annotations

import inspect
from pathlib import Path

from app.api.predict import predict_submit

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


def _dataset_and_model(client) -> tuple[str, str, str]:
    from app.db import get_db, new_id, utcnow

    uploaded = client.post(
        "/api/projects/local/datasets/upload",
        files={"file": ("train.csv", CSV, "text/csv")},
        headers=POST_HEADERS,
    )
    assert uploaded.status_code == 200, uploaded.text
    dataset_id = uploaded.json()["id"]
    version_id = uploaded.json()["current_version"]["id"]
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
    revision_id = exp.json()["current_revision"]["id"]
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
    return dataset_id, version_id, model_id


def _version_count(dataset_id: str | None = None) -> int:
    from app.db import get_db

    with get_db() as conn:
        if dataset_id:
            return conn.execute(
                "SELECT COUNT(*) AS c FROM dataset_versions WHERE dataset_id = ?",
                (dataset_id,),
            ).fetchone()["c"]
        return conn.execute("SELECT COUNT(*) AS c FROM dataset_versions").fetchone()["c"]


def test_predict_existing_dataset_does_not_insert_version(client):
    from app.settings import get_settings

    dataset_id, version_id, model_id = _dataset_and_model(client)
    before = _version_count(dataset_id)
    response = client.post(
        "/api/projects/local/predict",
        json={"model_id": model_id, "dataset_id": dataset_id},
        headers=POST_HEADERS,
    )
    assert response.status_code == 200, response.text
    assert _version_count(dataset_id) == before
    job = client.get(f"/api/projects/local/jobs/{response.json()['job_id']}").json()
    progress = job["progress"]
    assert progress["input_version_id"] == version_id
    predict_path = Path(progress["predict_path"])
    assert predict_path.is_file()
    assert get_settings().artifacts_dir.resolve() in predict_path.resolve().parents
    text = predict_path.read_text(encoding="utf-8")
    assert "00123" in text
    assert "x" in text.splitlines()[0]


def test_predict_upload_creates_separate_dataset_not_train_version(client):
    dataset_id, version_id, model_id = _dataset_and_model(client)
    before_train = _version_count(dataset_id)
    before_all = _version_count()
    response = client.post(
        "/api/projects/local/predict",
        files={"file": ("p.csv", b"id,x\n00123,0.2\n", "text/csv")},
        data={"model_id": model_id},
        headers=POST_HEADERS,
    )
    assert response.status_code == 200, response.text
    assert _version_count(dataset_id) == before_train
    assert _version_count() == before_all + 1
    job = client.get(f"/api/projects/local/jobs/{response.json()['job_id']}").json()
    assert job["progress"]["input_version_id"] != version_id
    assert Path(job["progress"]["predict_path"]).is_file()


def test_predict_upload_read_is_bounded(client, monkeypatch):
    from app.settings import get_settings

    _, _, model_id = _dataset_and_model(client)
    monkeypatch.setattr(get_settings(), "max_upload_bytes", 32)
    src = inspect.getsource(predict_submit)
    assert "max_upload_bytes + 1" in src
    response = client.post(
        "/api/projects/local/predict",
        files={"file": ("big.csv", b"id,x\n" + b"1,2\n" * 40, "text/csv")},
        data={"model_id": model_id},
        headers=POST_HEADERS,
    )
    assert response.status_code == 400
    assert response.json()["code"] == "too_large"
