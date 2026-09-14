"""Predict submit and CSV download."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import Response
from starlette.datastructures import UploadFile as StarletteUploadFile

from app.models import AppError, PredictJson
from app.services.predict import download_prediction_csv, submit_predict
from app.settings import get_settings

router = APIRouter(prefix="/api/projects", tags=["predict"])


@router.post("/{project_id}/predict")
async def predict_submit(project_id: str, request: Request) -> dict:
    ctype = (request.headers.get("content-type") or "").lower()
    upload_bytes = None
    filename = "predict.csv"
    if "multipart/form-data" in ctype:
        form = await request.form()
        model_id = form.get("model_id")
        dataset_id = form.get("dataset_id")
        if not model_id or not isinstance(model_id, str):
            raise AppError("model_id is required")
        file = form.get("file")
        if isinstance(file, StarletteUploadFile):
            settings = get_settings()
            upload_bytes = await file.read(settings.max_upload_bytes + 1)
            filename = file.filename or filename
        dataset_id = dataset_id if isinstance(dataset_id, str) else None
    else:
        payload = PredictJson.model_validate(await request.json())
        model_id = payload.model_id
        dataset_id = payload.dataset_id
    return submit_predict(
        project_id,
        model_id=model_id,
        dataset_id=dataset_id,
        upload_bytes=upload_bytes,
        filename=filename,
    )


@router.get("/{project_id}/predictions/{prediction_id}/download")
def predict_download(project_id: str, prediction_id: str) -> Response:
    data, name = download_prediction_csv(project_id, prediction_id)
    return Response(
        content=data,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )
