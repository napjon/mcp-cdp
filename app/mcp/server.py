"""MCP server and tool registration. import_local_file is stdio-only (see stdio_bridge)."""

from __future__ import annotations

from typing import Any

from mcp.server import MCPServer

from app.mcp import domain

try:
    from mcp.types import ToolAnnotations
except ImportError:  # pragma: no cover
    ToolAnnotations = None  # type: ignore

INSTRUCTIONS = (
    "Control the local AutoML app. Pass project_id. "
    "Training/predict return job_id; poll get_job. No SQL/shell/code."
)

mcp = MCPServer("mcp-cdp", instructions=INSTRUCTIONS)


def _annotations(**kwargs: Any):
    if ToolAnnotations is None:
        return None
    return ToolAnnotations(**kwargs)


_READ = _annotations(read_only_hint=True, open_world_hint=False, idempotent_hint=True)
_WRITE = _annotations(read_only_hint=False, destructive_hint=False, open_world_hint=False)
_SHEET = _annotations(read_only_hint=False, destructive_hint=False, open_world_hint=True)
_CANCEL = _annotations(read_only_hint=False, destructive_hint=True, open_world_hint=False)


@mcp.tool(annotations=_READ)
async def list_projects() -> dict[str, Any]:
    """List local projects."""
    return await domain.list_projects()


@mcp.tool(annotations=_READ)
async def get_project(project_id: str) -> dict[str, Any]:
    """Get a project and its datasets."""
    return await domain.get_project(project_id)


@mcp.tool(annotations=_READ)
async def list_datasets(project_id: str) -> dict[str, Any]:
    """List datasets in a project."""
    return await domain.list_datasets(project_id)


@mcp.tool(annotations=_READ)
async def inspect_dataset(
    project_id: str,
    dataset_id: str,
    sample_rows: int = 0,
) -> dict[str, Any]:
    """Inspect schema and aggregates. sample_rows defaults to 0 and is capped at 5. Raw rows require sample sharing."""
    return await domain.inspect_dataset(project_id, dataset_id, sample_rows)


@mcp.tool(annotations=_READ)
async def get_profile(project_id: str, dataset_id: str) -> dict[str, Any]:
    """Get column profile summary for a dataset (no full table)."""
    return await domain.get_profile(project_id, dataset_id)


@mcp.tool(annotations=_READ)
async def get_report(project_id: str, job_id: str) -> dict[str, Any]:
    """Get the metrics report for a training job."""
    return await domain.get_report(project_id, job_id)


@mcp.tool(annotations=_SHEET)
async def connect_google_sheet(
    project_id: str,
    url: str,
    confirm: bool = False,
    gid: str | None = None,
    header_row: int | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    """Connect a public Google Sheet. confirm must be true."""
    return await domain.connect_google_sheet(
        project_id, url, confirm=confirm, gid=gid, header_row=header_row, name=name
    )


@mcp.tool(annotations=_SHEET)
async def refresh_google_sheet(
    project_id: str,
    connection_id: str,
    confirm: bool = False,
) -> dict[str, Any]:
    """Refresh a Google Sheet connection. confirm must be true."""
    return await domain.refresh_google_sheet(project_id, connection_id, confirm=confirm)


@mcp.tool(annotations=_WRITE)
async def configure_experiment(
    project_id: str,
    dataset_id: str,
    config: dict[str, Any],
    confirm: bool = False,
) -> dict[str, Any]:
    """Save an experiment configuration. confirm must be true."""
    return await domain.configure_experiment(project_id, dataset_id, config, confirm=confirm)


@mcp.tool(annotations=_WRITE)
async def submit_training(
    project_id: str,
    experiment_id: str,
    confirm: bool = False,
) -> dict[str, Any]:
    """Queue training and return job_id immediately. confirm must be true."""
    return await domain.submit_training(project_id, experiment_id, confirm=confirm)


@mcp.tool(annotations=_READ)
async def get_job(project_id: str, job_id: str) -> dict[str, Any]:
    """Get job status. Poll this after submit_training or submit_predict."""
    return await domain.get_job(project_id, job_id)


@mcp.tool(annotations=_CANCEL)
async def cancel_job(project_id: str, job_id: str, confirm: bool = False) -> dict[str, Any]:
    """Cancel a queued or running job. confirm must be true."""
    return await domain.cancel_job(project_id, job_id, confirm=confirm)


@mcp.tool(annotations=_WRITE)
async def submit_predict(
    project_id: str,
    confirm: bool = False,
    model_id: str | None = None,
    dataset_id: str | None = None,
) -> dict[str, Any]:
    """Queue a prediction job and return job_id immediately. confirm must be true."""
    return await domain.submit_predict(
        project_id, confirm=confirm, model_id=model_id, dataset_id=dataset_id
    )


def http_tool_names() -> list[str]:
    return [t.name for t in mcp._tool_manager.list_tools()]
