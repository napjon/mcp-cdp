"""Runtime settings. API keys and MCP_TOKEN are never logged."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    host: str = "127.0.0.1"
    port: int = 8765
    data_dir: Path = Path("data")
    max_upload_bytes: int = 50_000_000
    max_rows: int = 100_000
    max_queue: int = 10
    max_forecast_series: int = 100
    sheets_timeout_s: float = 30.0
    mcp_token: str | None = None
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    # Comma- or path-separator-delimited roots allowed to the stdio MCP importer.
    mcp_import_roots: str | None = None

    @field_validator("host")
    @classmethod
    def _loopback_only(cls, v: str) -> str:
        host = (v or "").strip().lower()
        if host not in _LOOPBACK:
            return "127.0.0.1"
        return "127.0.0.1" if host == "localhost" else host

    @field_validator("port")
    @classmethod
    def _not_reserved(cls, v: int) -> int:
        if v in {3000, 8000}:
            return 8765
        return v

    @property
    def db_path(self) -> Path:
        return self.data_dir / "mcp_cdp.sqlite"

    @property
    def sqlite_path(self) -> Path:
        return self.db_path

    @property
    def datasets_dir(self) -> Path:
        return self.data_dir / "datasets"

    @property
    def artifacts_dir(self) -> Path:
        return self.data_dir / "artifacts"

    @property
    def llm_provider(self) -> str | None:
        if self.openai_api_key:
            return "openai"
        if self.anthropic_api_key:
            return "anthropic"
        return None

    def __repr__(self) -> str:
        return (
            f"Settings(host={self.host!r}, port={self.port}, data_dir={str(self.data_dir)!r}, "
            f"mcp_token_set={bool(self.mcp_token)}, "
            f"openai_set={bool(self.openai_api_key)}, "
            f"anthropic_set={bool(self.anthropic_api_key)})"
        )

    __str__ = __repr__


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings() -> None:
    get_settings.cache_clear()


class _SettingsProxy:
    """Live view of get_settings() for `from app.settings import settings`."""

    def __getattr__(self, name: str):
        return getattr(get_settings(), name)

    def __repr__(self) -> str:
        return repr(get_settings())


settings = _SettingsProxy()
