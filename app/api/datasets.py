"""Dataset upload, sheets, preview, column roles."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Query, UploadFile
from pydantic import BaseModel

from app.models import ColumnsUpdate, SheetsConnect
from app.services.datasets import (
    clear_needs_review,
    create_csv_dataset,
    get_dataset,
    list_datasets,
    preview_dataset,
    preview_disclosure,
    set_disclosure,
    update_column_roles,
)
from app.services.sheets import connect_sheet, preview_sheet, refresh_sheet
from app.settings import get_settings


class DisclosureUpdate(BaseModel):
    enabled: bool


router = APIRouter(prefix="/api/projects", tags=["datasets"])


@router.get("/{project_id}/datasets")
def datasets_list(project_id: str) -> list[dict]:
    return list_datasets(project_id)


@router.get("/{project_id}/datasets/{dataset_id}")
def dataset_get(project_id: str, dataset_id: str) -> dict:
    return get_dataset(project_id, dataset_id)


@router.get("/{project_id}/datasets/{dataset_id}/preview")
def dataset_preview(
    project_id: str, dataset_id: str, n: int = Query(default=20, ge=1, le=100)
) -> dict:
    return preview_dataset(project_id, dataset_id, n=n)


@router.post("/{project_id}/datasets/upload")
async def dataset_upload(project_id: str, file: Annotated[UploadFile, File()]) -> dict:
    settings = get_settings()
    data = await file.read(settings.max_upload_bytes + 1)
    filename = file.filename or "upload.csv"
    return create_csv_dataset(project_id, data, filename=filename)


@router.put("/{project_id}/datasets/{dataset_id}/columns")
def dataset_columns(project_id: str, dataset_id: str, body: ColumnsUpdate) -> dict:
    return update_column_roles(
        project_id, dataset_id, [c.model_dump() for c in body.columns]
    )


@router.post("/{project_id}/datasets/sheets/preview")
def dataset_sheets_preview(project_id: str, body: SheetsConnect) -> dict:
    return preview_sheet(
        project_id,
        url=body.url,
        gid=body.gid,
        header_row=body.header_row,
        name=body.name,
    )


@router.post("/{project_id}/datasets/sheets")
def dataset_sheets(project_id: str, body: SheetsConnect) -> dict:
    return connect_sheet(
        project_id,
        url=body.url,
        gid=body.gid,
        header_row=body.header_row,
        name=body.name,
    )


@router.post("/{project_id}/sheet-connections/{connection_id}/refresh")
def dataset_sheet_refresh(project_id: str, connection_id: str) -> dict:
    return refresh_sheet(project_id, connection_id)


@router.post("/{project_id}/datasets/{dataset_id}/review")
def dataset_review(project_id: str, dataset_id: str) -> dict:
    return clear_needs_review(project_id, dataset_id)


@router.post("/{project_id}/datasets/{dataset_id}/disclosure/preview")
def dataset_disclosure_preview(project_id: str, dataset_id: str) -> dict:
    return preview_disclosure(project_id, dataset_id)


@router.put("/{project_id}/datasets/{dataset_id}/disclosure")
def dataset_disclosure_set(project_id: str, dataset_id: str, body: DisclosureUpdate) -> dict:
    return set_disclosure(project_id, dataset_id, body.enabled)
