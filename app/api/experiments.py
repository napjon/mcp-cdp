"""Experiment create and submit."""

from __future__ import annotations

from fastapi import APIRouter

from app.models import ExperimentCreate
from app.services.experiments import create_experiment, get_experiment, list_experiments
from app.services.jobs import submit_training

router = APIRouter(prefix="/api/projects", tags=["experiments"])


@router.get("/{project_id}/experiments")
def experiments_list(project_id: str) -> list[dict]:
    return list_experiments(project_id)


@router.get("/{project_id}/experiments/{experiment_id}")
def experiment_get(project_id: str, experiment_id: str) -> dict:
    return get_experiment(project_id, experiment_id)


@router.post("/{project_id}/experiments")
def experiment_create(project_id: str, body: ExperimentCreate) -> dict:
    return create_experiment(project_id, body.dataset_id, body.config)


@router.post("/{project_id}/experiments/{experiment_id}/submit")
def experiment_submit(project_id: str, experiment_id: str) -> dict:
    return submit_training(project_id, experiment_id)
