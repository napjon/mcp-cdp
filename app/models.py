"""API models and error type."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AppError(Exception):
    def __init__(self, message: str, status_code: int = 400, code: str | None = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class ProjectOut(BaseModel):
    id: str
    name: str
    created_at: str


class SheetsConnect(BaseModel):
    url: str
    gid: str | int | None = None
    header_row: int = 0
    name: str | None = None


class ColumnRoleUpdate(BaseModel):
    name: str
    role: str


class ColumnsUpdate(BaseModel):
    columns: list[ColumnRoleUpdate]


class ExperimentCreate(BaseModel):
    dataset_id: str
    config: dict[str, Any]


class PredictJson(BaseModel):
    model_id: str
    dataset_id: str | None = None


VALID_ROLES = frozenset(
    {"numeric", "categorical", "text", "date", "identifier", "constant", "excluded"}
)
VALID_TASKS = frozenset({"classification", "regression", "forecast"})
VALID_BUDGETS = frozenset({"quick", "thorough"})
VALID_SPLITS = frozenset({"random", "group", "time"})
VALID_FREQUENCIES = frozenset({"D", "W", "M"})
TERMINAL_JOB_STATUSES = frozenset({"succeeded", "failed", "canceled"})
