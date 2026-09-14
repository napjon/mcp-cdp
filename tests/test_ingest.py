from __future__ import annotations

import io

import pytest

POST_HEADERS = {
    "Origin": "http://127.0.0.1:5173",
    "X-Requested-With": "mcp-cdp",
}

CSV_VALID = b'id,name,city\n00123,"Doe, Jane",Paris\n00124,Smith,Lyon\n'
CSV_DUP_HEADERS = b"a,a,\n1,2,3\n4,5,6\n"


def test_parse_valid_quoted_commas_and_leading_zeros():
    from app.services.datasets import parse_tabular

    table = parse_tabular(CSV_VALID, filename="ids.csv")
    assert table.headers == ["id", "name", "city"]
    assert table.rows[0][0] == "00123"
    assert table.rows[0][1] == "Doe, Jane"
    assert table.rows[1][0] == "00124"
    id_col = next(c for c in table.column_stats if c["name"] == "id")
    assert "00123" in id_col["sample_preview"]
    assert id_col["inferred_role"] == "identifier"


def test_rejects_100001_rows():
    from app.models import AppError
    from app.services.datasets import parse_tabular

    buf = io.StringIO()
    buf.write("a,b\n")
    for i in range(100001):
        buf.write(f"{i},x\n")
    with pytest.raises(AppError) as exc:
        parse_tabular(buf.getvalue().encode())
    assert exc.value.code == "too_many_rows"


def test_rejects_empty_malformed_inconsistent_and_oversize():
    from app.models import AppError
    from app.services.datasets import parse_tabular

    with pytest.raises(AppError, match="empty"):
        parse_tabular(b"")
    with pytest.raises(AppError, match="empty"):
        parse_tabular(b"a,b\n")
    with pytest.raises(AppError, match="inconsistent"):
        parse_tabular(b"a,b\n1,2,3\n")
    with pytest.raises(AppError) as exc:
        parse_tabular(b"a,b\n1,2\n" + b"x" * 50_000_001)
    assert exc.value.code == "too_large"
    with pytest.raises(AppError, match="malformed"):
        parse_tabular(b"a,b\n1,\x00\n")


def test_duplicate_and_blank_headers_resolved_in_metadata():
    from app.services.datasets import parse_tabular

    table = parse_tabular(CSV_DUP_HEADERS)
    assert table.headers == ["a", "a_2", "unnamed_3"]
    assert table.import_meta["original_headers"] == ["a", "a", ""]
    assert table.import_meta["renames"]


def test_upload_api_preserves_00123(client):
    response = client.post(
        "/api/projects/local/datasets/upload",
        files={"file": ("ids.csv", CSV_VALID, "text/csv")},
        headers=POST_HEADERS,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["current_version"]["n_rows"] == 2
    preview = client.get(f"/api/projects/local/datasets/{body['id']}/preview")
    assert preview.status_code == 200
    rows = preview.json()["rows"]
    assert rows[0][0] == "00123"
    assert rows[0][1] == "Doe, Jane"


def test_upload_rejects_100001_rows(client):
    buf = io.BytesIO()
    buf.write(b"a,b\n")
    for i in range(100001):
        buf.write(f"{i},x\n".encode())
    response = client.post(
        "/api/projects/local/datasets/upload",
        files={"file": ("big.csv", buf.getvalue(), "text/csv")},
        headers=POST_HEADERS,
    )
    assert response.status_code == 400
    assert response.json()["code"] == "too_many_rows"


def _version_count() -> int:
    from app.db import get_db

    with get_db() as conn:
        return conn.execute("SELECT COUNT(*) AS c FROM dataset_versions").fetchone()["c"]


def test_upload_rejects_empty_and_malformed_without_versions(client):
    before = _version_count()
    empty = client.post(
        "/api/projects/local/datasets/upload",
        files={"file": ("empty.csv", b"", "text/csv")},
        headers=POST_HEADERS,
    )
    assert empty.status_code == 400
    malformed = client.post(
        "/api/projects/local/datasets/upload",
        files={"file": ("bad.csv", b"a,b\n1,\x00\n", "text/csv")},
        headers=POST_HEADERS,
    )
    assert malformed.status_code == 400
    inconsistent = client.post(
        "/api/projects/local/datasets/upload",
        files={"file": ("wide.csv", b"a,b\n1,2,3\n", "text/csv")},
        headers=POST_HEADERS,
    )
    assert inconsistent.status_code == 400
    assert _version_count() == before


def test_utf16_decodes_before_nul_check_and_rejects_non_utf8():
    from app.models import AppError
    from app.services.datasets import parse_tabular

    table = parse_tabular("id,name\n00123,Ada\n".encode("utf-16"), filename="wide.csv")
    assert table.encoding in {"utf-16-le", "utf-16-be", "utf-16"}
    assert table.headers == ["id", "name"]
    assert table.rows[0][0] == "00123"

    quoted = parse_tabular(b'a,b\n"x\ny",1\n"q""q",2\n')
    assert quoted.rows[0][0] == "x\ny"
    assert quoted.rows[1][0] == 'q"q'

    with pytest.raises(AppError, match="decode"):
        parse_tabular(b"a,b\n1,\xff\n")


def test_preview_reports_version_n_rows_and_sampled_flag(client):
    buf = io.StringIO()
    buf.write("id,val\n")
    for i in range(50):
        buf.write(f"{i:05d},{i}\n")
    uploaded = client.post(
        "/api/projects/local/datasets/upload",
        files={"file": ("many.csv", buf.getvalue().encode(), "text/csv")},
        headers=POST_HEADERS,
    )
    assert uploaded.status_code == 200, uploaded.text
    body = uploaded.json()
    n_rows = body["current_version"]["n_rows"]
    assert n_rows == 50
    preview = client.get(f"/api/projects/local/datasets/{body['id']}/preview?n=20")
    assert preview.status_code == 200
    payload = preview.json()
    assert payload["total"] == n_rows
    assert payload["n_rows"] == n_rows
    assert payload["total"] != len(payload["rows"])
    assert len(payload["rows"]) == 20
    assert payload["sampled"] is True

    small = client.get(f"/api/projects/local/datasets/{body['id']}/preview?n=100")
    full = small.json()
    assert full["total"] == n_rows
    assert full["sampled"] is False
    assert len(full["rows"]) == n_rows


def test_list_projects_and_get_project(client):
    from app.models import AppError
    from app.services.datasets import get_project, list_projects

    listed = list_projects()
    projects = listed["projects"]
    assert any(p["id"] == "local" for p in projects)
    project = get_project("local")
    assert project["id"] == "local"
    assert project["ok"] is True
    assert "datasets" in project
    with pytest.raises(AppError) as exc:
        get_project("missing-project")
    assert exc.value.status_code == 404


def test_column_role_update(client):
    uploaded = client.post(
        "/api/projects/local/datasets/upload",
        files={"file": ("ids.csv", CSV_VALID, "text/csv")},
        headers=POST_HEADERS,
    ).json()
    response = client.put(
        f"/api/projects/local/datasets/{uploaded['id']}/columns",
        json={"columns": [{"name": "name", "role": "text"}]},
        headers=POST_HEADERS,
    )
    assert response.status_code == 200
    cols = response.json()["current_version"]["profile"]["columns"]
    name_col = next(c for c in cols if c["name"] == "name")
    assert name_col["user_role"] == "text"


def test_disclosure_preview_required_before_enable(client):
    uploaded = client.post(
        "/api/projects/local/datasets/upload",
        files={"file": ("ids.csv", CSV_VALID, "text/csv")},
        headers=POST_HEADERS,
    )
    assert uploaded.status_code == 200, uploaded.text
    dataset_id = uploaded.json()["id"]
    enable = client.put(
        f"/api/projects/local/datasets/{dataset_id}/disclosure",
        json={"enabled": True},
        headers=POST_HEADERS,
    )
    assert enable.status_code == 400
    preview = client.post(
        f"/api/projects/local/datasets/{dataset_id}/disclosure/preview",
        headers=POST_HEADERS,
    )
    assert preview.status_code == 200, preview.text
    body = preview.json()
    assert len(body["rows"]) <= 5
    assert body["previewed_at"]
    assert body["enabled"] == 0
    enabled = client.put(
        f"/api/projects/local/datasets/{dataset_id}/disclosure",
        json={"enabled": True},
        headers=POST_HEADERS,
    )
    assert enabled.status_code == 200, enabled.text
    assert enabled.json()["enabled"] == 1
    preview_again = client.post(
        f"/api/projects/local/datasets/{dataset_id}/disclosure/preview",
        headers=POST_HEADERS,
    )
    assert preview_again.status_code == 200
    assert preview_again.json()["enabled"] == 1
    off = client.put(
        f"/api/projects/local/datasets/{dataset_id}/disclosure",
        json={"enabled": False},
        headers=POST_HEADERS,
    )
    assert off.status_code == 200
    assert off.json()["enabled"] == 0
