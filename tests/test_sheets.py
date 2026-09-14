from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest

POST_HEADERS = {
    "Origin": "http://127.0.0.1:5173",
    "X-Requested-With": "mcp-cdp",
}

EXPORT_PREFIX = "https://docs.google.com/spreadsheets/d/"


class FakeResponse:
    def __init__(self, status_code=200, content=b"a,b\n1,2\n", headers=None):
        self.status_code = status_code
        self.content = content
        self.headers = headers or {"content-type": "text/csv"}


def test_parse_sheet_urls():
    from app.services.sheets import parse_sheets_url

    ref = parse_sheets_url(
        "https://docs.google.com/spreadsheets/d/ABC_123-id/edit#gid=42"
    )
    assert ref.spreadsheet_id == "ABC_123-id"
    assert ref.gid == "42"

    ref = parse_sheets_url(
        "https://docs.google.com/spreadsheets/d/ABC_123-id/edit?gid=99#gid=99"
    )
    assert ref.gid == "99"

    ref = parse_sheets_url("https://docs.google.com/spreadsheets/d/ABC_123-id/edit")
    assert ref.gid is None


def test_parse_rejects_non_sheet_urls():
    from app.services.sheets import parse_sheets_url

    with pytest.raises(Exception) as exc:
        parse_sheets_url("https://evil.example/spreadsheets/d/ABC123/edit")
    assert exc.value.code == "not_a_sheet"
    with pytest.raises(Exception) as exc:
        parse_sheets_url("https://docs.google.com/document/d/ABC123/edit")
    assert exc.value.code == "not_a_sheet"


def test_export_url_is_canonical():
    from app.services.sheets import export_url

    assert export_url("AbC-1", 7).startswith(EXPORT_PREFIX)
    assert export_url("AbC-1", 7).endswith("/export?format=csv&gid=7")
    assert export_url("AbC-1", 0).endswith("/export?format=csv&gid=0")


def test_export_url_requires_gid():
    from app.services.sheets import export_url

    with pytest.raises(Exception) as exc:
        export_url("AbC-1", None)
    assert exc.value.code == "gid_unspecified"
    assert "#gid=" in exc.value.message
    with pytest.raises(Exception) as exc:
        export_url("AbC-1", "")
    assert exc.value.code == "gid_unspecified"


def test_connect_sheet_requires_gid(client):
    response = client.post(
        "/api/projects/local/datasets/sheets",
        json={"url": "https://docs.google.com/spreadsheets/d/NOGIDTABID/edit"},
        headers=POST_HEADERS,
    )
    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "gid_unspecified"
    assert "#gid=" in body["error"]


@patch("app.services.sheets.httpx.Client")
def test_never_fetches_user_url(mock_client):
    from app.services.sheets import download_export_csv

    inst = mock_client.return_value
    inst.get.return_value = FakeResponse()
    download_export_csv("SheetId1", 0)
    inst.get.assert_called()
    url = inst.get.call_args[0][0]
    assert url == f"{EXPORT_PREFIX}SheetId1/export?format=csv&gid=0"
    assert "evil" not in url


@patch("app.services.sheets.httpx.Client")
def test_mocked_private_sheet(mock_client, client):
    inst = mock_client.return_value
    inst.get.return_value = FakeResponse(
        status_code=403, content=b"<html>login</html>", headers={"content-type": "text/html"}
    )
    response = client.post(
        "/api/projects/local/datasets/sheets",
        json={"url": "https://docs.google.com/spreadsheets/d/PRIVATESHEETID/edit#gid=0"},
        headers=POST_HEADERS,
    )
    assert response.status_code == 400
    assert response.json()["code"] == "private"
    assert "sk-" not in response.text


@patch("app.services.sheets.httpx.Client")
def test_mocked_revoked_and_timeout(mock_client):
    from app.services.sheets import download_export_csv

    inst = mock_client.return_value
    inst.get.return_value = FakeResponse(status_code=404, content=b"gone")
    with pytest.raises(Exception) as exc:
        download_export_csv("GONEID", 0)
    assert exc.value.code == "revoked"

    inst.get.side_effect = httpx.TimeoutException("timeout")
    with pytest.raises(Exception) as exc:
        download_export_csv("SLOW", 0)
    assert exc.value.code == "timeout"


@patch("app.services.sheets.httpx.Client")
def test_login_html_is_private(mock_client):
    from app.services.sheets import download_export_csv

    inst = mock_client.return_value
    inst.get.return_value = FakeResponse(
        status_code=200,
        content=b"<!DOCTYPE html><html>sign in</html>",
        headers={"content-type": "text/html; charset=utf-8"},
    )
    with pytest.raises(Exception) as exc:
        download_export_csv("LOGINWALL", 0)
    assert exc.value.code == "private"


@patch("app.services.sheets.httpx.Client")
def test_same_hash_refresh_does_not_add_version(mock_client, client):
    csv_bytes = b"sku,store,sales\n001,A,10\n002,B,20\n"
    inst = mock_client.return_value
    inst.get.return_value = FakeResponse(content=csv_bytes)

    created = client.post(
        "/api/projects/local/datasets/sheets",
        json={
            "url": "https://docs.google.com/spreadsheets/d/SHEETREFRESH1/edit#gid=0",
            "name": "Sales",
        },
        headers=POST_HEADERS,
    )
    assert created.status_code == 200, created.text
    body = created.json()
    version_1 = body["current_version"]["version"]
    connection_id = body["connection_id"]

    refreshed = client.post(
        f"/api/projects/local/sheet-connections/{connection_id}/refresh",
        headers=POST_HEADERS,
    )
    assert refreshed.status_code == 200, refreshed.text
    assert refreshed.json()["unchanged"] is True
    assert refreshed.json()["current_version"]["version"] == version_1


@patch("app.services.sheets.httpx.Client")
def test_failed_refresh_keeps_previous_snapshot(mock_client, client):
    csv_bytes = b"a,b\n1,2\n"
    inst = mock_client.return_value
    inst.get.return_value = FakeResponse(content=csv_bytes)
    created = client.post(
        "/api/projects/local/datasets/sheets",
        json={"url": "https://docs.google.com/spreadsheets/d/KEEPVER1/edit#gid=0"},
        headers=POST_HEADERS,
    )
    assert created.status_code == 200, created.text
    dataset_id = created.json()["id"]
    connection_id = created.json()["connection_id"]
    previous = created.json()["current_version"]["id"]

    inst.get.return_value = FakeResponse(status_code=403, content=b"no")
    failed = client.post(
        f"/api/projects/local/sheet-connections/{connection_id}/refresh",
        headers=POST_HEADERS,
    )
    assert failed.status_code == 400
    assert failed.json()["code"] == "private"

    still = client.get(f"/api/projects/local/datasets/{dataset_id}")
    assert still.json()["current_version"]["id"] == previous
    assert still.json()["connection"]["last_version_id"] == previous
    assert still.json()["connection"]["last_status"] == "private"


def test_suggest_header_row_skips_title_line():
    from app.services.sheets import suggest_header_row

    data = b"Monthly report,\ndate,sales\n2024-01-01,10\n2024-01-02,11\n"
    assert suggest_header_row(data) == 1


def _table_counts():
    from app.db import get_db

    with get_db() as conn:
        return {
            "datasets": conn.execute("SELECT COUNT(*) AS c FROM datasets").fetchone()["c"],
            "connections": conn.execute(
                "SELECT COUNT(*) AS c FROM sheet_connections"
            ).fetchone()["c"],
            "versions": conn.execute(
                "SELECT COUNT(*) AS c FROM dataset_versions"
            ).fetchone()["c"],
        }


@patch("app.services.sheets.httpx.Client")
def test_sheets_preview_does_not_persist(mock_client, client):
    csv_bytes = b"title row,\ndate,sales\n2024-01-01,10\n2024-01-02,12\n"
    inst = mock_client.return_value
    inst.get.return_value = FakeResponse(content=csv_bytes)
    before = _table_counts()
    response = client.post(
        "/api/projects/local/datasets/sheets/preview",
        json={"url": "https://docs.google.com/spreadsheets/d/PREVIEW1/edit#gid=0"},
        headers=POST_HEADERS,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert "headers" in body
    assert "rows" in body
    assert "suggested_header_row" in body
    assert body["suggested_header_row"] == 1
    assert _table_counts() == before
    url = inst.get.call_args[0][0]
    assert url.endswith("/export?format=csv&gid=0")


@patch("app.services.sheets.httpx.Client")
def test_schema_change_sets_needs_review_same_schema_does_not(mock_client, client):
    inst = mock_client.return_value
    original = (
        b"id,y,x\n"
        b"1,yes,0.1\n2,no,0.2\n3,yes,0.3\n4,no,0.4\n5,yes,0.5\n"
        b"6,no,0.6\n7,yes,0.7\n8,no,0.8\n9,yes,0.9\n10,no,1.0\n"
    )
    inst.get.return_value = FakeResponse(content=original)
    created = client.post(
        "/api/projects/local/datasets/sheets",
        json={"url": "https://docs.google.com/spreadsheets/d/SCHEMA1/edit#gid=0"},
        headers=POST_HEADERS,
    )
    assert created.status_code == 200, created.text
    body = created.json()
    dataset_id = body["id"]
    connection_id = body["connection_id"]
    assert int(body.get("needs_review") or 0) == 0
    exp = client.post(
        "/api/projects/local/experiments",
        json={
            "dataset_id": dataset_id,
            "config": {"task": "classification", "target": "y", "features": ["x"]},
        },
        headers=POST_HEADERS,
    )
    assert exp.status_code == 200, exp.text
    experiment_id = exp.json()["id"]

    same_schema = (
        b"id,y,x\n"
        b"11,yes,1.1\n12,no,1.2\n13,yes,1.3\n14,no,1.4\n15,yes,1.5\n"
        b"16,no,1.6\n17,yes,1.7\n18,no,1.8\n19,yes,1.9\n20,no,2.0\n"
    )
    inst.get.return_value = FakeResponse(content=same_schema)
    refreshed = client.post(
        f"/api/projects/local/sheet-connections/{connection_id}/refresh",
        headers=POST_HEADERS,
    )
    assert refreshed.status_code == 200, refreshed.text
    assert refreshed.json()["unchanged"] is False
    assert int(refreshed.json().get("needs_review") or 0) == 0

    changed = (
        b"id,y,x,extra\n"
        b"1,yes,0.1,z\n2,no,0.2,z\n3,yes,0.3,z\n4,no,0.4,z\n5,yes,0.5,z\n"
        b"6,no,0.6,z\n7,yes,0.7,z\n8,no,0.8,z\n9,yes,0.9,z\n10,no,1.0,z\n"
    )
    inst.get.return_value = FakeResponse(content=changed)
    schema = client.post(
        f"/api/projects/local/sheet-connections/{connection_id}/refresh",
        headers=POST_HEADERS,
    )
    assert schema.status_code == 200, schema.text
    assert int(schema.json().get("needs_review") or 0) == 1

    blocked_submit = client.post(
        f"/api/projects/local/experiments/{experiment_id}/submit",
        headers=POST_HEADERS,
    )
    assert blocked_submit.status_code == 400
    assert blocked_submit.json()["code"] == "needs_review"

    blocked = client.post(
        "/api/projects/local/experiments",
        json={
            "dataset_id": dataset_id,
            "config": {"task": "classification", "target": "y", "features": ["x"]},
        },
        headers=POST_HEADERS,
    )
    assert blocked.status_code == 400
    assert blocked.json()["code"] == "needs_review"

    reviewed = client.put(
        f"/api/projects/local/datasets/{dataset_id}/columns",
        json={"columns": [{"name": "extra", "role": "text"}]},
        headers=POST_HEADERS,
    )
    assert reviewed.status_code == 200, reviewed.text
    assert int(reviewed.json().get("needs_review") or 0) == 0

    inst.get.return_value = FakeResponse(
        content=(
            b"id,y,x,other\n"
            b"1,yes,0.1,z\n2,no,0.2,z\n3,yes,0.3,z\n4,no,0.4,z\n5,yes,0.5,z\n"
            b"6,no,0.6,z\n7,yes,0.7,z\n8,no,0.8,z\n9,yes,0.9,z\n10,no,1.0,z\n"
        )
    )
    again = client.post(
        f"/api/projects/local/sheet-connections/{connection_id}/refresh",
        headers=POST_HEADERS,
    )
    assert again.status_code == 200
    assert int(again.json().get("needs_review") or 0) == 1
    cleared = client.post(
        f"/api/projects/local/datasets/{dataset_id}/review",
        headers=POST_HEADERS,
    )
    assert cleared.status_code == 200
    assert int(cleared.json().get("needs_review") or 0) == 0
    ok = client.post(
        "/api/projects/local/experiments",
        json={
            "dataset_id": dataset_id,
            "config": {"task": "classification", "target": "y", "features": ["x"]},
        },
        headers=POST_HEADERS,
    )
    assert ok.status_code == 200, ok.text


@patch("app.services.sheets.httpx.Client")
def test_refresh_type_change_sets_needs_review_and_preserves_user_role(mock_client, client):
    inst = mock_client.return_value
    numeric = b"id,amount\n1,10\n2,20\n3,30\n4,40\n"
    inst.get.return_value = FakeResponse(content=numeric)
    created = client.post(
        "/api/projects/local/datasets/sheets",
        json={"url": "https://docs.google.com/spreadsheets/d/TYPECHANGE1/edit#gid=0"},
        headers=POST_HEADERS,
    )
    assert created.status_code == 200, created.text
    dataset_id = created.json()["id"]
    connection_id = created.json()["connection_id"]
    amount = next(c for c in created.json()["columns"] if c["name"] == "amount")
    assert amount["inferred_role"] == "numeric"

    reviewed = client.put(
        f"/api/projects/local/datasets/{dataset_id}/columns",
        json={"columns": [{"name": "amount", "role": "numeric"}]},
        headers=POST_HEADERS,
    )
    assert reviewed.status_code == 200, reviewed.text
    assert (
        next(c for c in reviewed.json()["columns"] if c["name"] == "amount")["user_role"]
        == "numeric"
    )

    same_types = b"id,amount\n5,50\n6,60\n7,70\n8,80\n"
    inst.get.return_value = FakeResponse(content=same_types)
    copied = client.post(
        f"/api/projects/local/sheet-connections/{connection_id}/refresh",
        headers=POST_HEADERS,
    )
    assert copied.status_code == 200, copied.text
    assert int(copied.json().get("needs_review") or 0) == 0
    copied_amount = next(c for c in copied.json()["columns"] if c["name"] == "amount")
    assert copied_amount["user_role"] == "numeric"
    assert copied_amount["inferred_role"] == "numeric"

    as_text = b"id,amount\n1,ten\n2,twenty\n3,thirty\n4,forty\n"
    inst.get.return_value = FakeResponse(content=as_text)
    drifted = client.post(
        f"/api/projects/local/sheet-connections/{connection_id}/refresh",
        headers=POST_HEADERS,
    )
    assert drifted.status_code == 200, drifted.text
    assert int(drifted.json().get("needs_review") or 0) == 1
    drifted_amount = next(c for c in drifted.json()["columns"] if c["name"] == "amount")
    assert drifted_amount["inferred_role"] == "text"


@patch("app.services.sheets.httpx.Client")
def test_refresh_semantic_role_change_sets_needs_review(mock_client, client):
    inst = mock_client.return_value
    categorical = (
        b"id,status\n"
        b"1,new\n2,new\n3,new\n4,new\n5,new\n"
        b"6,done\n7,done\n8,done\n9,done\n10,done\n"
    )
    inst.get.return_value = FakeResponse(content=categorical)
    created = client.post(
        "/api/projects/local/datasets/sheets",
        json={"url": "https://docs.google.com/spreadsheets/d/SEMANTIC1/edit#gid=0"},
        headers=POST_HEADERS,
    )
    assert created.status_code == 200, created.text
    connection_id = created.json()["connection_id"]
    status = next(c for c in created.json()["columns"] if c["name"] == "status")
    assert status["inferred_role"] == "categorical"

    text = (
        b"id,status\n"
        b"1,awaiting approval\n2,shipped to customer\n3,returned\n4,backordered\n"
        b"5,delayed at port\n6,ready for pickup\n7,customs hold\n8,damaged in transit\n"
        b"9,delivered\n10,cancelled by buyer\n"
    )
    inst.get.return_value = FakeResponse(content=text)
    refreshed = client.post(
        f"/api/projects/local/sheet-connections/{connection_id}/refresh",
        headers=POST_HEADERS,
    )
    assert refreshed.status_code == 200, refreshed.text
    assert int(refreshed.json().get("needs_review") or 0) == 1
    refreshed_status = next(c for c in refreshed.json()["columns"] if c["name"] == "status")
    assert refreshed_status["inferred_role"] == "text"
