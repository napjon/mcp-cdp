from __future__ import annotations

import pytest

POST_HEADERS = {
    "Origin": "http://127.0.0.1:5173",
    "X-Requested-With": "mcp-cdp",
}

TWO_CLASS_CSV = b"id,y,x\n1,yes,0.1\n2,no,0.2\n3,yes,0.3\n4,no,0.4\n"
BALANCED_CLASS_CSV = (
    b"id,y,x\n"
    b"1,yes,0.1\n2,yes,0.2\n3,yes,0.3\n4,yes,0.4\n5,yes,0.5\n"
    b"6,no,0.6\n7,no,0.7\n8,no,0.8\n9,no,0.9\n10,no,1.0\n"
)
TINY_STRATIFY_CSV = (
    b"id,y,x\n"
    b"1,yes,0.1\n2,yes,0.2\n3,yes,0.3\n4,yes,0.4\n"
    b"5,yes,0.5\n6,yes,0.6\n7,yes,0.7\n8,yes,0.8\n"
    b"9,no,0.9\n10,no,1.0\n"
)
SINGLE_CLASS_CSV = b"id,y,x\n1,yes,0.1\n2,yes,0.2\n3,yes,0.3\n"
WEAK_CLASS_CSV = b"id,y,x\n1,yes,0.1\n2,yes,0.2\n3,no,0.3\n"


def _upload(client, name: str, data: bytes) -> str:
    response = client.post(
        "/api/projects/local/datasets/upload",
        files={"file": (name, data, "text/csv")},
        headers=POST_HEADERS,
    )
    assert response.status_code == 200, response.text
    return response.json()["id"]


def _job_count() -> int:
    from app.db import get_db

    with get_db() as conn:
        return conn.execute("SELECT COUNT(*) AS c FROM jobs").fetchone()["c"]


def test_validate_config_requires_nonempty_target(client):
    from app.models import AppError
    from app.services.datasets import get_dataset
    from app.services.experiments import validate_config

    dataset_id = _upload(client, "ok.csv", TWO_CLASS_CSV)
    dataset = get_dataset("local", dataset_id)
    with pytest.raises(AppError, match="target is required"):
        validate_config({"task": "classification", "features": ["x"]}, dataset)
    with pytest.raises(AppError, match="target is required"):
        validate_config({"task": "regression", "target": "  ", "features": ["x"]}, dataset)
    create = client.post(
        "/api/projects/local/experiments",
        json={"dataset_id": dataset_id, "config": {"task": "classification", "features": ["x"]}},
        headers=POST_HEADERS,
    )
    assert create.status_code == 400
    assert _job_count() == 0


def test_validate_config_rejects_single_class_and_weak_support(client):
    from app.models import AppError
    from app.services.datasets import get_dataset
    from app.services.experiments import validate_config

    single_id = _upload(client, "single.csv", SINGLE_CLASS_CSV)
    single = get_dataset("local", single_id)
    with pytest.raises(AppError) as single_exc:
        validate_config(
            {"task": "classification", "target": "y", "features": ["x"]},
            single,
        )
    assert single_exc.value.code == "single_class"
    blocked = client.post(
        "/api/projects/local/experiments",
        json={
            "dataset_id": single_id,
            "config": {"task": "classification", "target": "y", "features": ["x"]},
        },
        headers=POST_HEADERS,
    )
    assert blocked.status_code == 400
    assert blocked.json()["code"] == "single_class"

    weak_id = _upload(client, "weak.csv", WEAK_CLASS_CSV)
    weak = get_dataset("local", weak_id)
    with pytest.raises(AppError) as weak_exc:
        validate_config(
            {"task": "classification", "target": "y", "features": ["x"]},
            weak,
        )
    assert weak_exc.value.code == "insufficient_class_support"
    weak_create = client.post(
        "/api/projects/local/experiments",
        json={
            "dataset_id": weak_id,
            "config": {"task": "classification", "target": "y", "features": ["x"]},
        },
        headers=POST_HEADERS,
    )
    assert weak_create.status_code == 400
    assert weak_create.json()["code"] == "insufficient_class_support"
    assert _job_count() == 0


def test_validate_config_rejects_unknown_split(client):
    from app.models import AppError
    from app.services.datasets import get_dataset
    from app.services.experiments import validate_config

    dataset_id = _upload(client, "split.csv", BALANCED_CLASS_CSV)
    dataset = get_dataset("local", dataset_id)
    with pytest.raises(AppError, match="invalid split"):
        validate_config(
            {
                "task": "classification",
                "target": "y",
                "features": ["x"],
                "split": "stratified",
            },
            dataset,
        )
    response = client.post(
        "/api/projects/local/experiments",
        json={
            "dataset_id": dataset_id,
            "config": {
                "task": "classification",
                "target": "y",
                "features": ["x"],
                "split": "holdout",
            },
        },
        headers=POST_HEADERS,
    )
    assert response.status_code == 400
    assert "split" in response.json()["error"]
    ok = validate_config(
        {
            "task": "classification",
            "target": "y",
            "features": ["x"],
            "split": "random",
        },
        dataset,
    )
    assert ok["split"] == "random"


def test_validate_config_rejects_infeasible_classification_split(client):
    from app.models import AppError
    from app.services.datasets import get_dataset
    from app.services.experiments import validate_config

    tiny_id = _upload(client, "sklearn-tiny.csv", TWO_CLASS_CSV)
    tiny = get_dataset("local", tiny_id)
    with pytest.raises(AppError) as tiny_exc:
        validate_config(
            {"task": "classification", "target": "y", "features": ["x"]},
            tiny,
        )
    assert tiny_exc.value.code == "infeasible_split"

    dataset_id = _upload(client, "tiny-split.csv", TINY_STRATIFY_CSV)
    dataset = get_dataset("local", dataset_id)
    before = _job_count()
    with pytest.raises(AppError) as exc:
        validate_config(
            {"task": "classification", "target": "y", "features": ["x"]},
            dataset,
        )
    assert exc.value.code == "infeasible_split"
    blocked = client.post(
        "/api/projects/local/experiments",
        json={
            "dataset_id": dataset_id,
            "config": {"task": "classification", "target": "y", "features": ["x"]},
        },
        headers=POST_HEADERS,
    )
    assert blocked.status_code == 400
    assert blocked.json()["code"] == "infeasible_split"
    assert _job_count() == before


FORECAST_CSV = b"date,sales,sku\n2024-01-01,10,A\n2024-01-02,11,A\n2024-01-03,12,A\n"
FORECAST_TEXT_TARGET = b"date,label,sku\n2024-01-01,foo,A\n2024-01-02,bar,A\n"
FORECAST_AMBIGUOUS = b"date,sales\n01/02/2024,10\n03/04/2024,11\n"


def _experiment_count() -> int:
    from app.db import get_db

    with get_db() as conn:
        return conn.execute("SELECT COUNT(*) AS c FROM experiments").fetchone()["c"]


def test_forecast_validate_config_requires_target_and_frequency(client):
    from app.models import AppError
    from app.services.datasets import get_dataset
    from app.services.experiments import validate_config

    dataset_id = _upload(client, "fc.csv", FORECAST_CSV)
    dataset = get_dataset("local", dataset_id)
    before = _experiment_count()
    with pytest.raises(AppError, match="target is required"):
        validate_config(
            {"task": "forecast", "date_column": "date", "frequency": "D", "horizon": 2},
            dataset,
        )
    with pytest.raises(AppError, match="frequency is required"):
        validate_config(
            {
                "task": "forecast",
                "target": "sales",
                "date_column": "date",
                "horizon": 2,
            },
            dataset,
        )
    with pytest.raises(AppError, match="invalid frequency"):
        validate_config(
            {
                "task": "forecast",
                "target": "sales",
                "date_column": "date",
                "frequency": "daily",
                "horizon": 2,
            },
            dataset,
        )
    missing_freq = client.post(
        "/api/projects/local/experiments",
        json={
            "dataset_id": dataset_id,
            "config": {
                "task": "forecast",
                "target": "sales",
                "date_column": "date",
                "horizon": 2,
            },
        },
        headers=POST_HEADERS,
    )
    assert missing_freq.status_code == 400
    assert _experiment_count() == before
    ok = validate_config(
        {
            "task": "forecast",
            "target": "sales",
            "date_column": "date",
            "frequency": "D",
            "horizon": 2,
            "group_columns": ["sku"],
        },
        dataset,
    )
    assert ok["frequency"] == "D"
    created = client.post(
        "/api/projects/local/experiments",
        json={
            "dataset_id": dataset_id,
            "config": {
                "task": "forecast",
                "target": "sales",
                "date_column": "date",
                "frequency": "D",
                "horizon": 2,
            },
        },
        headers=POST_HEADERS,
    )
    assert created.status_code == 200, created.text
    assert _experiment_count() == before + 1


def test_forecast_validate_config_rejects_non_numeric_target_and_ambiguous_dates(client):
    from app.models import AppError
    from app.services.datasets import get_dataset
    from app.services.experiments import validate_config

    text_id = _upload(client, "text.csv", FORECAST_TEXT_TARGET)
    text_ds = get_dataset("local", text_id)
    before = _experiment_count()
    with pytest.raises(AppError, match="must be numeric"):
        validate_config(
            {
                "task": "forecast",
                "target": "label",
                "date_column": "date",
                "frequency": "D",
                "horizon": 2,
            },
            text_ds,
        )
    blocked = client.post(
        "/api/projects/local/experiments",
        json={
            "dataset_id": text_id,
            "config": {
                "task": "forecast",
                "target": "label",
                "date_column": "date",
                "frequency": "D",
                "horizon": 2,
            },
        },
        headers=POST_HEADERS,
    )
    assert blocked.status_code == 400
    assert _experiment_count() == before

    amb_id = _upload(client, "amb.csv", FORECAST_AMBIGUOUS)
    amb = get_dataset("local", amb_id)
    with pytest.raises(AppError, match="ambiguous date"):
        validate_config(
            {
                "task": "forecast",
                "target": "sales",
                "date_column": "date",
                "frequency": "D",
                "horizon": 2,
            },
            amb,
        )
    amb_create = client.post(
        "/api/projects/local/experiments",
        json={
            "dataset_id": amb_id,
            "config": {
                "task": "forecast",
                "target": "sales",
                "date_column": "date",
                "frequency": "D",
                "horizon": 2,
            },
        },
        headers=POST_HEADERS,
    )
    assert amb_create.status_code == 400
    assert _experiment_count() == before
