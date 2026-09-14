"""import_local_file rejects traversal and symlink escape."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.mcp.domain import import_local_file, read_import_bytes, resolve_import_path


@pytest.fixture
def import_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "root"
    root.mkdir()
    (root / "ok.csv").write_text("a,b\n1,2\n")
    monkeypatch.setenv("MCP_IMPORT_ROOTS", str(root))
    return root


@pytest.mark.asyncio
async def test_import_local_file_rejects_absolute_outside(import_root: Path, tmp_path: Path) -> None:
    secret = tmp_path / "secret.csv"
    secret.write_text("x\n")
    result = await import_local_file("p1", str(secret), confirm=True)
    assert result.get("ok") is False
    assert "outside" in (result.get("error") or "").lower()


@pytest.mark.asyncio
async def test_import_local_file_rejects_dotdot(import_root: Path, tmp_path: Path) -> None:
    secret = tmp_path / "secret.csv"
    secret.write_text("x\n")
    result = await import_local_file("p1", "../secret.csv", confirm=True)
    assert result.get("ok") is False
    assert "outside" in (result.get("error") or "").lower()


@pytest.mark.asyncio
async def test_import_local_file_rejects_symlink_escape(import_root: Path, tmp_path: Path) -> None:
    secret = tmp_path / "secret.csv"
    secret.write_text("x\n")
    link = import_root / "link.csv"
    link.symlink_to(secret)
    result = await import_local_file("p1", str(link), confirm=True)
    assert result.get("ok") is False
    assert "outside" in (result.get("error") or "").lower()
    with pytest.raises(ValueError, match="outside"):
        resolve_import_path(str(link))


def test_resolve_import_path_accepts_file_under_root(import_root: Path) -> None:
    resolved = resolve_import_path("ok.csv")
    assert resolved == (import_root / "ok.csv").resolve()


def test_read_import_bytes_does_not_exceed_max_plus_one(
    import_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("app.mcp.domain.max_upload_bytes", lambda: 32)
    big = import_root / "big.csv"
    big.write_bytes(b"a,b\n" + b"1,2\n" * 500)
    data, code = read_import_bytes(big)
    assert code == "too_large"
    assert len(data) == 33


@pytest.mark.asyncio
async def test_import_local_file_rejects_oversize(
    import_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("app.mcp.domain.max_upload_bytes", lambda: 32)
    big = import_root / "big.csv"
    big.write_bytes(b"a,b\n" + b"1,2\n" * 500)

    def fail_if_called(**kwargs: object) -> None:
        raise AssertionError("oversize import must not call the datasets service")

    monkeypatch.setattr("app.services.datasets.create_csv_dataset", fail_if_called)
    monkeypatch.setattr("app.services.datasets.import_local_file", fail_if_called)
    result = await import_local_file("p1", str(big), confirm=True)
    assert result.get("ok") is False
    assert result.get("code") == "too_large"
    assert "large" in (result.get("error") or "").lower()
