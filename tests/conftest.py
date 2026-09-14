from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

POST_HEADERS = {
    "Origin": "http://127.0.0.1:5173",
    "X-Requested-With": "mcp-cdp",
}


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from app.settings import reset_settings

    reset_settings()
    yield tmp_path
    reset_settings()


@pytest.fixture
def db(data_dir):
    from app.db import init_db

    init_db()
    return data_dir


@pytest.fixture
def client(data_dir):
    from app.main import create_app

    application = create_app(worker_enabled=False)
    with TestClient(application, base_url="http://127.0.0.1:8765") as c:
        yield c


@pytest.fixture
def worker_client(data_dir):
    from app.main import create_app

    application = create_app(worker_enabled=True)
    with TestClient(application, base_url="http://127.0.0.1:8765") as c:
        yield c
