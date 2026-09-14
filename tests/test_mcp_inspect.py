"""inspect_dataset must not dump a full table."""

from __future__ import annotations

from typing import Any

import pytest

from app.mcp.domain import MAX_SAMPLE_ROWS, inspect_dataset


def _huge_payload(**_: Any) -> dict[str, Any]:
    return {
        "ok": True,
        "n_rows": 10_000,
        "n_cols": 3,
        "columns": [
            {"name": "a", "inferred_role": "numeric", "sample_preview": "1,2,3," + "4," * 500},
            {"name": "b", "inferred_role": "categorical", "sample_preview": "x" * 4000},
        ],
        "rows": [{"a": i, "b": "row"} for i in range(10_000)],
        "sample": [{"a": i} for i in range(10_000)],
        "preview": [{"a": i} for i in range(10_000)],
    }


@pytest.mark.asyncio
async def test_inspect_dataset_does_not_dump_full_table(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_fetch(project_id: str, dataset_id: str, sample_rows: int) -> dict[str, Any]:
        return _huge_payload()

    monkeypatch.setattr("app.mcp.domain._fetch_inspect_payload", fake_fetch)
    monkeypatch.setattr("app.mcp.domain.samples_allowed", lambda dataset_id: True)

    out = await inspect_dataset("p1", "d1", sample_rows=5)
    for key in ("rows", "sample", "preview"):
        assert key not in out or len(out[key]) <= MAX_SAMPLE_ROWS
        if key in out:
            assert len(out[key]) <= 5
    assert out.get("n_rows") == 10_000
    dumped = str(out)
    assert dumped.count("'a':") <= 20 or len(out.get("rows") or []) <= 5


@pytest.mark.asyncio
async def test_inspect_dataset_default_has_no_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_fetch(project_id: str, dataset_id: str, sample_rows: int) -> dict[str, Any]:
        return _huge_payload()

    monkeypatch.setattr("app.mcp.domain._fetch_inspect_payload", fake_fetch)
    monkeypatch.setattr("app.mcp.domain.samples_allowed", lambda dataset_id: True)

    out = await inspect_dataset("p1", "d1")
    assert out.get("sample_rows") == 0
    assert not out.get("rows")
    assert not out.get("sample")
    assert not out.get("preview")
    assert out.get("n_rows") == 10_000


@pytest.mark.asyncio
async def test_inspect_dataset_caps_requested_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_fetch(project_id: str, dataset_id: str, sample_rows: int) -> dict[str, Any]:
        assert sample_rows <= MAX_SAMPLE_ROWS
        return _huge_payload()

    monkeypatch.setattr("app.mcp.domain._fetch_inspect_payload", fake_fetch)
    monkeypatch.setattr("app.mcp.domain.samples_allowed", lambda dataset_id: True)

    out = await inspect_dataset("p1", "d1", sample_rows=100)
    assert (out.get("sample_rows") or 0) <= MAX_SAMPLE_ROWS
    assert len(out.get("rows") or []) <= MAX_SAMPLE_ROWS


@pytest.mark.asyncio
async def test_inspect_dataset_hides_rows_without_disclosure(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_fetch(project_id: str, dataset_id: str, sample_rows: int) -> dict[str, Any]:
        return _huge_payload()

    monkeypatch.setattr("app.mcp.domain._fetch_inspect_payload", fake_fetch)
    monkeypatch.setattr("app.mcp.domain.samples_allowed", lambda dataset_id: False)

    out = await inspect_dataset("p1", "d1", sample_rows=5)
    assert not out.get("rows")
    assert not out.get("sample")
    assert "sample_note" in out
    assert out.get("n_rows") == 10_000
