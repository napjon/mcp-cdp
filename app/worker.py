"""Single ML worker: one running job, lease heartbeat, ML engine hook."""

from __future__ import annotations

import importlib
import json
import logging
import multiprocessing
import shutil
import threading
from contextlib import suppress
from pathlib import Path

from app.db import get_db, new_id, utcnow
from app.services.jobs import (
    attach_job_process,
    attempt_is_current,
    claim_next_job,
    finish_job,
    heartbeat,
    job_status,
    recover_stale_jobs,
    terminate_all_job_processes,
    update_progress,
)
from app.services.security import redact_secrets
from app.settings import get_settings

log = logging.getLogger("mcp_cdp.worker")

ML_MISSING = "ml engine not loaded"


class Worker:
    def __init__(self, poll_interval: float = 0.25):
        self.poll_interval = poll_interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        recover_stale_jobs()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="mcp-cdp-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        terminate_all_job_processes()
        if self._thread:
            self._thread.join(timeout=timeout)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                recover_stale_jobs()
                job = claim_next_job()
            except Exception:
                log.exception("claim failed")
                job = None
            if job:
                self._run_job(job)
            else:
                self._stop.wait(self.poll_interval)

    def _run_job(self, job: dict) -> None:
        job_id = job["id"]
        attempt_id = job.get("attempt_id")
        hb_stop = threading.Event()

        def _beat() -> None:
            while not hb_stop.wait(5):
                try:
                    heartbeat(job_id, attempt_id=attempt_id)
                except Exception:
                    log.exception("heartbeat failed")

        def _finish(*, status: str, error: str | None = None) -> None:
            finish_job(job_id, status=status, error=error, attempt_id=attempt_id)

        def _abandoned() -> bool:
            return job_status(job_id) == "canceled" or not attempt_is_current(
                job_id, attempt_id
            )

        hb_thread = threading.Thread(target=_beat, name=f"hb-{job_id[:8]}", daemon=True)
        hb_thread.start()
        try:
            if _abandoned():
                return
            try:
                run_train, run_predict = load_ml()
            except ImportError:
                _finish(status="failed", error=ML_MISSING)
                return
            if job["type"] == "train" and run_train is None:
                _finish(status="failed", error=ML_MISSING)
                return
            if job["type"] == "predict" and run_predict is None:
                _finish(status="failed", error=ML_MISSING)
                return
            if job["type"] not in {"train", "predict"}:
                _finish(status="failed", error="unknown job type")
                return

            job_dict, paths = _build_call(job)
            update_progress(job_id, {**(job_dict.get("progress") or {}), "stage": job["type"]})
            if _abandoned():
                return
            result, outcome = _run_ml_child(self, job_id, attempt_id, job_dict, paths)
            if _abandoned() or outcome == "canceled":
                return
            if outcome != "ok":
                _finish(
                    status="failed",
                    error=redact_secrets(str(result) or "ml process failed", get_settings()),
                )
                return
            if job["type"] == "train":
                _store_train_result(job, paths, result or {}, attempt_id=attempt_id)
            else:
                _store_predict_result(job, paths, result or {}, attempt_id=attempt_id)
        except Exception as exc:  # noqa: BLE001 — job must not kill the worker thread
            if _abandoned():
                return
            _finish(
                status="failed",
                error=redact_secrets(str(exc) or type(exc).__name__, get_settings()),
            )
        finally:
            hb_stop.set()
            attach_job_process(job_id, None, attempt_id=attempt_id)


def _attempt_artifact_dir(job: dict) -> Path:
    settings = get_settings()
    attempt_id = job.get("attempt_id")
    if not attempt_id:
        return settings.artifacts_dir / job["id"]
    return settings.artifacts_dir / job["id"] / attempt_id


def _relocate_under(path: str | Path, src_root: Path, dst_root: Path) -> str:
    raw = Path(path)
    try:
        rel = raw.resolve().relative_to(src_root.resolve())
    except (ValueError, OSError):
        return str(raw)
    return str(dst_root / rel)


def _publish_current(job_id: str, attempt_id: str | None, artifact_dir: Path) -> Path:
    """Replace artifacts/{job_id}/current with this attempt's dir if it is current."""
    if not attempt_id:
        return artifact_dir
    settings = get_settings()
    expected = (settings.artifacts_dir / job_id / attempt_id).resolve()
    try:
        src = artifact_dir.resolve()
    except OSError:
        return artifact_dir
    if src != expected or not src.is_dir():
        return artifact_dir
    job_root = settings.artifacts_dir / job_id
    current = job_root / "current"
    staging = job_root / f".staging-{attempt_id}"
    backup = job_root / f".old-{attempt_id}"
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(src, staging)
    if backup.exists():
        shutil.rmtree(backup)
    if current.exists() or current.is_symlink():
        current.rename(backup)
    staging.rename(current)
    if backup.exists():
        shutil.rmtree(backup, ignore_errors=True)
    return current


def _build_call(job: dict) -> tuple[dict, dict]:
    artifact_dir = _attempt_artifact_dir(job)
    (artifact_dir / "plots").mkdir(parents=True, exist_ok=True)
    progress = {}
    if job.get("progress_json"):
        try:
            progress = json.loads(job["progress_json"])
        except json.JSONDecodeError:
            progress = {}

    config: dict = {}
    dataset_normalized_path = None
    predict_path = progress.get("predict_path")
    model_path = None

    with get_db() as conn:
        if job.get("experiment_revision_id"):
            rev = conn.execute(
                "SELECT * FROM experiment_revisions WHERE id = ?",
                (job["experiment_revision_id"],),
            ).fetchone()
            if rev:
                config = json.loads(rev["config_json"])
                exp = conn.execute(
                    "SELECT dataset_id FROM experiments WHERE id = ?",
                    (rev["experiment_id"],),
                ).fetchone()
                if exp:
                    version_id = progress.get("dataset_version_id")
                    if version_id:
                        ver = conn.execute(
                            "SELECT * FROM dataset_versions WHERE id = ?", (version_id,)
                        ).fetchone()
                    else:
                        ver = conn.execute(
                            """
                            SELECT * FROM dataset_versions
                            WHERE dataset_id = ? ORDER BY version DESC LIMIT 1
                            """,
                            (exp["dataset_id"],),
                        ).fetchone()
                    if ver:
                        dataset_normalized_path = ver["normalized_path"]
                        progress["dataset_version_id"] = ver["id"]
        if job.get("model_id"):
            model = conn.execute(
                "SELECT * FROM models WHERE id = ?", (job["model_id"],)
            ).fetchone()
            if model:
                model_path = model["artifact_path"]
                if not dataset_normalized_path:
                    ver = conn.execute(
                        "SELECT * FROM dataset_versions WHERE id = ?",
                        (model["dataset_version_id"],),
                    ).fetchone()
                    if ver:
                        dataset_normalized_path = ver["normalized_path"]
        if progress.get("input_version_id") and not predict_path:
            ver = conn.execute(
                "SELECT * FROM dataset_versions WHERE id = ?",
                (progress["input_version_id"],),
            ).fetchone()
            if ver:
                predict_path = ver["normalized_path"]

    job_dict = {
        "id": job["id"],
        "type": job["type"],
        "config": config,
        "dataset_normalized_path": dataset_normalized_path,
        "progress": progress,
    }
    if predict_path:
        job_dict["predict_path"] = predict_path
    if model_path:
        job_dict["model_path"] = model_path
        dest = artifact_dir / "model.joblib"
        src = Path(model_path)
        if src.is_file() and src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
    paths = {"artifact_dir": str(artifact_dir)}
    return job_dict, paths


def _store_train_result(
    job: dict, paths: dict, result: dict, *, attempt_id: str | None = None
) -> bool:
    artifact_dir = Path(paths["artifact_dir"])
    plot_dir = artifact_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    model_path = result.get("model_path") or str(artifact_dir / "model.joblib")
    metrics = result.get("metrics") or {}
    dataset_version_id = None
    task = result.get("task") or "classification"
    cfg: dict = {}
    progress = {}
    if job.get("progress_json"):
        try:
            progress = json.loads(job["progress_json"])
        except json.JSONDecodeError:
            progress = {}
    dataset_version_id = progress.get("dataset_version_id")
    with get_db() as conn:
        if not attempt_is_current(job["id"], attempt_id, conn=conn):
            return False
        if job.get("experiment_revision_id"):
            rev = conn.execute(
                "SELECT experiment_id, config_json FROM experiment_revisions WHERE id = ?",
                (job["experiment_revision_id"],),
            ).fetchone()
            if rev:
                cfg = json.loads(rev["config_json"])
                task = cfg.get("task") or task
                if not dataset_version_id:
                    exp = conn.execute(
                        "SELECT dataset_id FROM experiments WHERE id = ?",
                        (rev["experiment_id"],),
                    ).fetchone()
                    if exp:
                        ver = conn.execute(
                            """
                            SELECT id FROM dataset_versions
                            WHERE dataset_id = ? ORDER BY version DESC LIMIT 1
                            """,
                            (exp["dataset_id"],),
                        ).fetchone()
                        if ver:
                            dataset_version_id = ver["id"]
        if not dataset_version_id:
            raise RuntimeError("dataset version missing for model")
        report = _train_report(result, metrics, cfg, dataset_version_id, task)
        model_id = new_id()
        conn.execute(
            """
            INSERT INTO models (
              id, project_id, job_id, dataset_version_id, task, artifact_path,
              metrics_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model_id,
                job["project_id"],
                job["id"],
                dataset_version_id,
                task,
                str(model_path),
                json.dumps(metrics),
                utcnow(),
            ),
        )
        conn.execute(
            "INSERT INTO reports (id, job_id, report_json, plot_dir) VALUES (?, ?, ?, ?)",
            (new_id(), job["id"], json.dumps(report), str(plot_dir)),
        )
        conn.execute("UPDATE jobs SET model_id = ? WHERE id = ?", (model_id, job["id"]))
        if not finish_job(job["id"], status="succeeded", attempt_id=attempt_id, conn=conn):
            conn.rollback()
            return False
        published = _publish_current(job["id"], attempt_id, artifact_dir)
        if published.resolve() != artifact_dir.resolve():
            conn.execute(
                "UPDATE models SET artifact_path = ? WHERE id = ?",
                (_relocate_under(model_path, artifact_dir, published), model_id),
            )
            conn.execute(
                "UPDATE reports SET plot_dir = ? WHERE job_id = ?",
                (str(published / "plots"), job["id"]),
            )
        return True


def _store_predict_result(
    job: dict, paths: dict, result: dict, *, attempt_id: str | None = None
) -> bool:
    artifact_dir = Path(paths["artifact_dir"])
    plot_dir = artifact_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    output_path = result.get("output_path") or str(artifact_dir / "predictions.csv")
    progress = {}
    if job.get("progress_json"):
        try:
            progress = json.loads(job["progress_json"])
        except json.JSONDecodeError:
            progress = {}
    input_version_id = progress.get("input_version_id")
    model_id = job.get("model_id")
    if not model_id or not input_version_id:
        raise RuntimeError("predict job missing model or input version")
    pred_id = new_id()
    progress["prediction_id"] = pred_id
    progress["model_id"] = model_id
    metrics = result.get("metrics")
    with get_db() as conn:
        if not attempt_is_current(job["id"], attempt_id, conn=conn):
            return False
        conn.execute(
            """
            INSERT INTO predictions (id, job_id, model_id, input_version_id, output_path, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (pred_id, job["id"], model_id, input_version_id, str(output_path), utcnow()),
        )
        if metrics is not None:
            report = {
                "metrics": metrics,
                "warnings": result.get("warnings") or [],
                "selected_candidate": result.get("selected_candidate"),
                "plot_files": result.get("plot_files") or [],
            }
            if isinstance(metrics, dict) and metrics.get("task"):
                report["task"] = metrics["task"]
            conn.execute(
                "INSERT INTO reports (id, job_id, report_json, plot_dir) VALUES (?, ?, ?, ?)",
                (new_id(), job["id"], json.dumps(report), str(plot_dir)),
            )
        conn.execute(
            "UPDATE jobs SET progress_json = ? WHERE id = ?",
            (json.dumps(progress), job["id"]),
        )
        if not finish_job(job["id"], status="succeeded", attempt_id=attempt_id, conn=conn):
            conn.rollback()
            return False
        published = _publish_current(job["id"], attempt_id, artifact_dir)
        if published.resolve() != artifact_dir.resolve():
            conn.execute(
                """
                UPDATE predictions SET output_path = ?
                WHERE id = ?
                """,
                (_relocate_under(output_path, artifact_dir, published), pred_id),
            )
            conn.execute(
                "UPDATE reports SET plot_dir = ? WHERE job_id = ?",
                (str(published / "plots"), job["id"]),
            )
        return True


def _train_report(
    result: dict,
    metrics: dict,
    cfg: dict,
    dataset_version_id: str,
    task: str,
) -> dict:
    report = {
        "metrics": metrics,
        "selected_candidate": result.get("selected_candidate"),
        "warnings": result.get("warnings") or [],
        "plot_files": result.get("plot_files") or [],
        "dataset_version_id": dataset_version_id,
        "task": task,
        "units": _report_units(task, cfg),
    }
    split = cfg.get("split")
    if split:
        report["split"] = split
    n_test = _n_test(metrics)
    if n_test is not None:
        report["n_test"] = n_test
    baseline = _baseline_metrics(metrics)
    if baseline:
        report["baseline_metrics"] = baseline
    return report


def _report_units(task: str, cfg: dict) -> str:
    if task == "classification":
        return "none"
    for key in ("units", "target_units"):
        value = cfg.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "target units"


def _n_test(metrics: dict) -> int | None:
    if not isinstance(metrics, dict):
        return None
    split_idx = metrics.get("split_indices")
    if isinstance(split_idx, dict) and isinstance(split_idx.get("test"), list):
        return len(split_idx["test"])
    y_true = metrics.get("y_true")
    if isinstance(y_true, list):
        return len(y_true)
    return None


def _baseline_metrics(metrics: dict) -> dict | None:
    if not isinstance(metrics, dict):
        return None
    canonical = metrics.get("baseline")
    if isinstance(canonical, dict) and canonical:
        return canonical
    candidates = metrics.get("candidates")
    if not isinstance(candidates, dict):
        return None
    src = None
    for name in ("dummy", "last_value"):
        cand = candidates.get(name)
        if isinstance(cand, dict):
            src = cand
            break
    if not src:
        return None
    out: dict = {}
    for key in ("val", "validation", "test"):
        value = src.get(key)
        if isinstance(value, dict) and value:
            out[key] = value
    for key in (
        "val_score",
        "val_mae",
        "mae",
        "rmse",
        "r2",
        "macro_f1",
        "balanced_accuracy",
        "score",
    ):
        if key in src and src[key] is not None:
            out[key] = src[key]
    return out or None


def load_ml():
    from app.ml.runner import run_predict, run_train

    return run_train, run_predict


def run_ml_job(job_dict: dict, paths: dict) -> dict:
    run_train, run_predict = load_ml()
    job_type = job_dict.get("type")
    if job_type == "train":
        if run_train is None:
            raise RuntimeError(ML_MISSING)
        return run_train(job_dict, paths) or {}
    if job_type == "predict":
        if run_predict is None:
            raise RuntimeError(ML_MISSING)
        return run_predict(job_dict, paths) or {}
    raise RuntimeError("unknown job type")


def _ml_process_entry(job_dict: dict, paths: dict, conn, fn_qualname: str) -> None:
    try:
        module_name, func_name = fn_qualname.rsplit(".", 1)
        fn = getattr(importlib.import_module(module_name), func_name)
        conn.send(("ok", fn(job_dict, paths) or {}))
    except Exception as exc:  # noqa: BLE001 — child reports failure to parent
        with suppress(BrokenPipeError, EOFError, OSError):
            conn.send(("err", f"{type(exc).__name__}: {exc}"))
    finally:
        with suppress(BrokenPipeError, EOFError, OSError):
            conn.close()


def _stop_ml_proc(proc) -> None:
    if not proc.is_alive():
        return
    proc.terminate()
    proc.join(timeout=5)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=2)


def _run_ml_child(
    worker: Worker,
    job_id: str,
    attempt_id: str | None,
    job_dict: dict,
    paths: dict,
) -> tuple[object, str]:
    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    target = f"{run_ml_job.__module__}.{run_ml_job.__name__}"
    proc = ctx.Process(
        target=_ml_process_entry,
        args=(job_dict, paths, child_conn, target),
        name=f"ml-{job_id[:8]}",
        daemon=True,
    )
    proc.start()
    child_conn.close()
    attach_job_process(job_id, proc, attempt_id=attempt_id)

    def _abandoned() -> bool:
        return job_status(job_id) == "canceled" or not attempt_is_current(job_id, attempt_id)

    received = None
    try:
        # Recv while the child is alive so a large pickle cannot fill the pipe and deadlock.
        while True:
            if worker._stop.is_set() or _abandoned():
                _stop_ml_proc(proc)
                return None, "canceled"
            alive = proc.is_alive()
            if received is None:
                if parent_conn.poll(0.2 if alive else 0):
                    received = parent_conn.recv()
                    continue
                if not alive:
                    break
                continue
            if not alive:
                break
            proc.join(timeout=0.2)
        if _abandoned() or worker._stop.is_set():
            return None, "canceled"
        if received is None:
            if proc.exitcode not in (0, None):
                return f"ml process exited ({proc.exitcode})", "err"
            return "ml process produced no result", "err"
        kind, payload = received
        if kind == "ok":
            return payload, "ok"
        return payload, "err"
    finally:
        with suppress(BrokenPipeError, EOFError, OSError):
            parent_conn.close()
        attach_job_process(job_id, None, attempt_id=attempt_id)


def start_worker(*, poll_interval: float = 0.25) -> Worker:
    worker = Worker(poll_interval=poll_interval)
    worker.start()
    return worker
