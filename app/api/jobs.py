"""Job status, SSE events, cancel, reports."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, StreamingResponse

from app.models import AppError
from app.services.jobs import cancel_job, get_job, get_report, iter_job_events

router = APIRouter(prefix="/api/projects", tags=["jobs"])


@router.get("/{project_id}/jobs/{job_id}")
def job_get(project_id: str, job_id: str) -> dict:
    return get_job(project_id, job_id)


@router.get("/{project_id}/jobs/{job_id}/events")
def job_events(project_id: str, job_id: str) -> StreamingResponse:
    get_job(project_id, job_id)
    return StreamingResponse(
        iter_job_events(project_id, job_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/{project_id}/jobs/{job_id}/cancel")
def job_cancel(project_id: str, job_id: str) -> dict:
    return cancel_job(project_id, job_id)


@router.get("/{project_id}/reports/{job_id}")
def job_report(project_id: str, job_id: str) -> dict:
    return get_report(project_id, job_id)


@router.get("/{project_id}/reports/{job_id}/plots/{filename}")
def job_plot(project_id: str, job_id: str, filename: str) -> FileResponse:
    report = get_report(project_id, job_id)
    plot_dir = Path(report.get("plot_dir") or "")
    if not plot_dir.is_dir():
        raise AppError("plot not found", status_code=404)
    target = (plot_dir / Path(filename).name).resolve()
    if not str(target).startswith(str(plot_dir.resolve())) or not target.is_file():
        raise AppError("plot not found", status_code=404)
    return FileResponse(target)
