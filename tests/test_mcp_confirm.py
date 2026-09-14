"""Write tools require confirm=true."""

from __future__ import annotations

import pytest

from app.mcp.domain import (
    cancel_job,
    configure_experiment,
    connect_google_sheet,
    import_local_file,
    refresh_google_sheet,
    submit_predict,
    submit_training,
)
from app.mcp.server import http_tool_names


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "factory",
    [
        lambda: submit_training("p", "e", confirm=False),
        lambda: submit_predict("p", confirm=False, model_id="m"),
        lambda: cancel_job("p", "j", confirm=False),
        lambda: configure_experiment("p", "d", {"task": "classification"}, confirm=False),
        lambda: connect_google_sheet("p", "https://docs.google.com/spreadsheets/d/x/edit", confirm=False),
        lambda: refresh_google_sheet("p", "c1", confirm=False),
        lambda: import_local_file("p", "/tmp/x.csv", confirm=False),
    ],
)
async def test_writes_require_confirm(factory) -> None:
    result = await factory()
    assert result.get("ok") is False
    assert "confirm" in (result.get("error") or "").lower()


@pytest.mark.asyncio
async def test_submit_training_omitted_confirm() -> None:
    result = await submit_training("p", "e")
    assert result.get("ok") is False
    assert "confirm=true" in (result.get("error") or "")


@pytest.mark.asyncio
async def test_connect_google_sheet_without_gid_does_not_default_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict = {}

    async def fake_call(module, names, **kwargs):
        seen.update(kwargs)
        return {"ok": False, "error": "copy the tab URL including gid", "code": "gid_unspecified"}

    monkeypatch.setattr("app.mcp.domain.call_service", fake_call)
    result = await connect_google_sheet(
        "p",
        "https://docs.google.com/spreadsheets/d/ABC123id/edit",
        confirm=True,
    )
    assert "gid" not in seen
    assert seen.get("gid") not in {0, "0"}
    assert result.get("code") == "gid_unspecified"
    assert result.get("ok") is False


def test_import_local_file_not_on_http_server() -> None:
    names = http_tool_names()
    assert "import_local_file" not in names
    for required in (
        "list_projects",
        "get_project",
        "list_datasets",
        "inspect_dataset",
        "get_profile",
        "get_report",
        "connect_google_sheet",
        "refresh_google_sheet",
        "configure_experiment",
        "submit_training",
        "get_job",
        "cancel_job",
        "submit_predict",
    ):
        assert required in names
