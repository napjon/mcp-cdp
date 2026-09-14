"""Platform worker interface: run_train and run_predict."""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_val_score,
)
from sklearn.pipeline import Pipeline

from app.ml import classify, forecast, regress
from app.ml.metrics import (
    classification_metrics,
    jsonable,
    library_versions,
    regression_metrics,
)
from app.ml.plots import plot_confusion_matrix, plot_feature_importance, plot_residuals
from app.ml.preprocess import (
    FeatureSpec,
    build_feature_spec,
    build_preprocessor,
    coerce_features,
    drop_target_na,
    flag_target_leakage,
    grouping_values,
    parse_date_column,
    split_supervised,
)

MODEL_FILENAME = "model.joblib"
QUICK_BUDGET_S = 60.0
THOROUGH_BUDGET_S = 300.0
BASELINE_FAMILY = {"classification": "dummy", "regression": "dummy", "forecast": "last_value"}


def budget_seconds(budget: str) -> float:
    name = (budget or "quick").strip().lower()
    return THOROUGH_BUDGET_S if name == "thorough" else QUICK_BUDGET_S


def _budget_deadline(budget: str) -> float:
    return time.monotonic() + budget_seconds(budget)


def run_train(job: dict, paths: dict) -> dict:
    config = _config(job)
    artifact_dir = Path(paths["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "plots").mkdir(parents=True, exist_ok=True)
    task = (config.get("task") or "").strip().lower()
    if task not in {"classification", "regression", "forecast"}:
        raise ValueError("config['task'] must be classification, regression, or forecast")
    df = _read_table(job["dataset_normalized_path"], roles=config.get("roles"))
    deadline = _budget_deadline(config.get("budget") or "quick")
    if task == "forecast":
        result = forecast.train(df, config, artifact_dir, deadline=deadline)
        bundle = {
            "task": "forecast",
            "pipeline": None,
            "forecast_state": result.pop("forecast_state"),
            "feature_columns": result.get("feature_columns") or [],
            "feature_spec": None,
            "target": config.get("target"),
            "roles": config.get("roles") or {},
            "config": config,
            "classes": None,
            "seed": int(config.get("seed", 42)),
            "selected_candidate": result["selected_candidate"],
        }
        return _finalize_train(result, bundle, artifact_dir)

    result = _train_supervised(df, config, artifact_dir, task, deadline=deadline)
    bundle = {
        "task": task,
        "pipeline": result.pop("pipeline"),
        "forecast_state": None,
        "feature_columns": result.pop("feature_columns"),
        "feature_spec": result.pop("feature_spec"),
        "target": config["target"],
        "roles": result.pop("roles"),
        "config": config,
        "classes": result.pop("classes"),
        "seed": int(config.get("seed", 42)),
        "selected_candidate": result["selected_candidate"],
        "label_list": result.pop("label_list"),
    }
    return _finalize_train(result, bundle, artifact_dir)


def run_predict(job: dict, paths: dict) -> dict:
    artifact_dir = Path(paths["artifact_dir"])
    model_path = artifact_dir / MODEL_FILENAME
    if not model_path.exists():
        raise FileNotFoundError(f"No fitted model at {model_path}")
    bundle = joblib.load(model_path)
    src = job.get("predict_path") or job.get("dataset_normalized_path")
    if not src:
        raise ValueError("predict requires predict_path or dataset_normalized_path")
    roles = bundle.get("roles") or {}
    spec_raw = bundle.get("feature_spec")
    df = _read_table(src, roles=roles, feature_spec=spec_raw)
    output_path = artifact_dir / "predictions.csv"
    warnings: list[str] = []
    metrics = None
    selected = bundle.get("selected_candidate")

    if bundle.get("task") == "forecast":
        _, metrics, fw = forecast.predict_forecast(df, bundle, output_path)
        warnings.extend(fw)
    else:
        spec = FeatureSpec.from_dict(bundle["feature_spec"] or {})
        aligned = coerce_features(df, spec, roles)
        missing = [c for c in spec.columns if c not in df.columns]
        if missing:
            warnings.append(f"Missing columns filled with NA: {missing}")
        extra = [c for c in df.columns if c not in spec.columns and c != bundle.get("target")]
        if extra:
            warnings.append(f"Ignored extra columns: {extra}")
        pipeline = bundle["pipeline"]
        preds = pipeline.predict(aligned)
        out = df.copy()
        out["prediction"] = preds
        _stringify_identifiers(out, roles, spec_raw)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_path, index=False)
        target = bundle.get("target")
        if target and target in df.columns and df[target].notna().any():
            y_true_raw = df[target]
            mask = y_true_raw.notna()
            y_pred = np.asarray(preds)[mask.to_numpy()]
            if bundle.get("task") == "classification":
                y_true = y_true_raw.loc[mask].astype(str).to_numpy()
                y_pred = np.asarray(y_pred).astype(str)
                metrics = jsonable(
                    {
                        "task": "classification",
                        "test": classification_metrics(y_true, y_pred, labels=bundle.get("label_list")),
                        "y_true": y_true.tolist(),
                        "y_pred": y_pred.tolist(),
                    }
                )
            else:
                y_true = pd.to_numeric(y_true_raw.loc[mask], errors="coerce")
                good = y_true.notna().to_numpy()
                metrics = jsonable(
                    {
                        "task": "regression",
                        "test": regression_metrics(y_true.to_numpy()[good], np.asarray(y_pred, dtype=float)[good]),
                        "y_true": y_true.to_numpy()[good].tolist(),
                        "y_pred": np.asarray(y_pred, dtype=float)[good].tolist(),
                    }
                )

    result = {
        "output_path": str(output_path),
        "metrics": metrics,
        "warnings": warnings,
        "model_path": str(model_path),
        "plot_files": [],
        "selected_candidate": selected,
    }
    (artifact_dir / "predict_metrics.json").write_text(json.dumps(jsonable(result), indent=2))
    return result


def _finalize_train(result: dict, bundle: dict, artifact_dir: Path) -> dict:
    model_path = artifact_dir / MODEL_FILENAME
    joblib.dump(bundle, model_path, compress=3)
    metrics = result["metrics"]
    (artifact_dir / "metrics.json").write_text(json.dumps(jsonable(metrics), indent=2))
    (artifact_dir / "metadata.json").write_text(
        json.dumps(
            jsonable(
                {
                    "task": bundle["task"],
                    "selected_candidate": result["selected_candidate"],
                    "seed": bundle["seed"],
                    "library_versions": library_versions(),
                    "warnings": result.get("warnings") or [],
                }
            ),
            indent=2,
        )
    )
    return {
        "metrics": metrics,
        "model_path": str(model_path),
        "plot_files": list(result.get("plot_files") or []),
        "warnings": list(result.get("warnings") or []),
        "selected_candidate": result["selected_candidate"],
    }


def _config(job: dict) -> dict:
    cfg = job.get("config") or {}
    if isinstance(cfg, str):
        cfg = json.loads(cfg)
    return dict(cfg)


def _spec_dict(feature_spec) -> dict:
    if feature_spec is None:
        return {}
    if isinstance(feature_spec, dict):
        return feature_spec
    if hasattr(feature_spec, "to_dict"):
        return feature_spec.to_dict()
    return {}


def _identifier_columns(roles=None, feature_spec=None) -> list[str]:
    skip = set(_spec_dict(feature_spec).get("numeric") or [])
    cols: list[str] = []
    for col, role in dict(roles or {}).items():
        if role == "identifier" and col not in skip and col not in cols:
            cols.append(col)
    return cols


def _stringify_identifiers(df: pd.DataFrame, roles=None, feature_spec=None) -> pd.DataFrame:
    for col in _identifier_columns(roles, feature_spec):
        if col not in df.columns:
            continue
        df[col] = df[col].map(lambda v: v if pd.isna(v) else str(v))
    return df


def _read_table(path, *, roles=None, feature_spec=None) -> pd.DataFrame:
    dtype = {col: str for col in _identifier_columns(roles, feature_spec)} or None
    df = pd.read_csv(path, dtype=dtype)
    _stringify_identifiers(df, roles, feature_spec)
    df.reset_index(drop=True, inplace=True)
    return df


def _train_supervised(
    df: pd.DataFrame,
    config: dict,
    artifact_dir: Path,
    task: str,
    deadline: float | None = None,
) -> dict:
    target = config.get("target")
    if not target:
        raise ValueError("config['target'] is required")
    df = drop_target_na(df, target)
    spec, roles = build_feature_spec(df, config)
    X = coerce_features(df, spec, roles)
    if task == "classification":
        y = df[target].astype(str).to_numpy()
    else:
        y = pd.to_numeric(df[target], errors="coerce").to_numpy()
        if np.isnan(y).any():
            keep = ~np.isnan(y)
            X = X.loc[keep].reset_index(drop=True)
            y = y[keep]
            df = df.loc[keep].reset_index(drop=True)
        if len(y) < 4:
            raise ValueError("Need at least 4 rows with a numeric target")

    seed = int(config.get("seed", 42))
    test_size = float(config.get("test_size", 0.2))
    mode = (config.get("split") or "random").lower()
    groups = grouping_values(df, config)
    dates = None
    if mode == "time":
        date_column = config.get("date_column")
        if not date_column or date_column not in df.columns:
            raise ValueError("time split requires date_column in the dataset")
        dates = parse_date_column(df[date_column], date_column)

    train_fit, val_idx, test_idx = split_supervised(
        y,
        mode=mode,
        test_size=test_size,
        seed=seed,
        groups=groups,
        dates=dates,
        stratify=(task == "classification"),
    )
    train_all = np.sort(np.concatenate([train_fit, val_idx]))
    if set(train_all) & set(test_idx):
        raise RuntimeError("Train/test split leaked")

    budget = config.get("budget") or "quick"
    if deadline is None:
        deadline = _budget_deadline(budget)
    if task == "classification":
        candidates = classify.iter_candidates(budget, seed)
        scoring = "f1_macro"
    else:
        candidates = regress.iter_candidates(budget, seed)
        scoring = "neg_mean_absolute_error"
    baseline_family = BASELINE_FAMILY[task]

    if task == "classification" and not classify.HAS_XGBOOST or task == "regression" and not regress.HAS_XGBOOST:
        xgb_missing = True
    else:
        xgb_missing = False

    warnings: list[str] = []
    leak_flags = flag_target_leakage(df, target, spec.columns)
    if leak_flags and config.get("allow_target_leakage") is not True:
        raise ValueError("Target leakage detected: " + "; ".join(leak_flags))
    warnings.extend(leak_flags)
    failures: list[dict] = []
    if xgb_missing:
        failures.append({"candidate": "xgboost", "error": "xgboost unavailable"})

    family_best: dict[str, dict] = {}
    skipped_families: set[str] = set()
    thorough = budget == "thorough"
    cv = None
    if thorough:
        cv = _make_cv(task, mode, y[train_all], groups[train_all] if groups is not None else None, seed)

    budget_warned = False
    for cand in candidates:
        if cand.family != baseline_family and time.monotonic() >= deadline:
            skipped_families.add(cand.family)
            if not budget_warned:
                warnings.append("time budget exhausted; skipped remaining candidates")
                budget_warned = True
            continue
        try:
            pipe = Pipeline(
                [
                    ("preprocess", build_preprocessor(spec, scale_numeric=cand.scale_numeric)),
                    ("model", clone(cand.estimator)),
                ]
            )
            val_metrics = None
            if thorough and cv is not None:
                groups_cv = groups[train_all] if mode == "group" and groups is not None else None
                scores = cross_val_score(
                    pipe,
                    X.iloc[train_all],
                    y[train_all],
                    cv=cv,
                    scoring=scoring,
                    groups=groups_cv,
                    error_score="raise",
                )
                val_score = float(np.mean(scores))
            else:
                pipe.fit(X.iloc[train_fit], y[train_fit])
                pred_val = pipe.predict(X.iloc[val_idx])
                val_metrics = _task_metrics(task, y[val_idx], pred_val)
                val_score = _score_from_metrics(task, val_metrics)
            current = family_best.get(cand.family)
            if current is None or val_score > current["val_score"]:
                family_best[cand.family] = {
                    "val_score": val_score,
                    "candidate": cand,
                    "status": "ok",
                    "params": cand.params,
                    "val_metrics": val_metrics,
                }
        except Exception as exc:  # noqa: BLE001
            failures.append({"candidate": cand.name, "error": str(exc)})
            warnings.append(f"candidate {cand.name} failed: {exc}")

    if not family_best:
        if skipped_families:
            raise ValueError("time budget exhausted before any candidate finished")
        raise ValueError(f"All candidates failed: {failures}")

    selected_family = max(family_best, key=lambda fam: family_best[fam]["val_score"])
    selected = family_best[selected_family]["candidate"]
    if selected_family != baseline_family and time.monotonic() >= deadline:
        if baseline_family in family_best:
            warnings.append("time budget exhausted; skipping long final fit")
            selected_family = baseline_family
            selected = family_best[baseline_family]["candidate"]
        else:
            raise ValueError("time budget exhausted before final fit")

    final_pipe = Pipeline(
        [
            ("preprocess", build_preprocessor(spec, scale_numeric=selected.scale_numeric)),
            ("model", clone(selected.estimator)),
        ]
    )
    final_pipe.fit(X.iloc[train_all], y[train_all])
    y_pred_raw = final_pipe.predict(X.iloc[test_idx])
    test_metrics = _task_metrics(task, y[test_idx], y_pred_raw)
    if task == "classification":
        y_true = np.asarray(y[test_idx]).astype(str)
        y_pred = np.asarray(y_pred_raw).astype(str)
        labels = test_metrics["labels"]
    else:
        y_true = np.asarray(y[test_idx], dtype=float)
        y_pred = np.asarray(y_pred_raw, dtype=float)
        labels = None

    dummy_test = test_metrics if selected_family == baseline_family else None
    report_families = {selected_family, baseline_family}
    for fam in report_families:
        info = family_best.get(fam)
        if info is None:
            continue
        if info.get("val_metrics") is None:
            _, _, info["val_metrics"] = _fit_predict_metrics(
                info["candidate"], spec, X, y, train_fit, val_idx, task
            )
        if fam == baseline_family and dummy_test is None:
            _, _, dummy_test = _fit_predict_metrics(
                info["candidate"], spec, X, y, train_all, test_idx, task
            )

    plots_dir = Path(artifact_dir) / "plots"
    plot_files: list[str] = []
    try:
        if task == "classification":
            plot_files.append(
                plot_confusion_matrix(y_true, y_pred, labels, plots_dir / "confusion_matrix.png")
            )
        else:
            plot_files.append(plot_residuals(y_true, y_pred, plots_dir / "residuals.png"))
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"metric plot skipped: {exc}")
    try:
        extracted = _feature_importance(final_pipe)
        if extracted is not None:
            names, values = extracted
            path = plot_feature_importance(names, values, plots_dir / "feature_importance.png")
            if path:
                plot_files.append(path)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"feature importance plot skipped: {exc}")

    selected_val = _canonical_split_bag(
        task, family_best[selected_family].get("val_metrics")
    )
    dummy_val = None
    dummy_test_bag = None
    if baseline_family in family_best:
        dummy_val = _canonical_split_bag(task, family_best[baseline_family].get("val_metrics"))
        dummy_test_bag = _canonical_split_bag(task, dummy_test)

    candidates_out = {}
    for fam, info in family_best.items():
        entry = {
            "val_score": info["val_score"],
            "status": "ok",
            "params": info["params"],
            "name": info["candidate"].name,
        }
        val_bag = _canonical_split_bag(task, info.get("val_metrics"))
        if val_bag:
            entry["validation"] = val_bag
        if fam == selected_family:
            entry["test"] = _canonical_split_bag(task, test_metrics)
        elif fam == baseline_family and dummy_test_bag:
            entry["test"] = dummy_test_bag
        candidates_out[fam] = entry
    for fail in failures:
        candidates_out.setdefault(
            fail["candidate"].split("_")[0],
            {"status": "failed", "error": fail["error"]},
        )
    for family in skipped_families:
        candidates_out.setdefault(
            family, {"status": "skipped", "error": "time budget exhausted"}
        )
    for family in ("dummy", "linear", "xgboost"):
        if family in candidates_out:
            continue
        err = next(
            (
                fail["error"]
                for fail in failures
                if str(fail["candidate"]).split("_")[0] == family
            ),
            "not evaluated",
        )
        candidates_out[family] = {"status": "failed", "error": err}

    metrics_payload = {
        "task": task,
        "seed": seed,
        "selected_candidate": selected_family,
        "test": test_metrics,
        "validation": selected_val,
        "candidates": candidates_out,
        "candidate_failures": failures,
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "split_indices": {
            "train": train_all.tolist(),
            "val": val_idx.tolist(),
            "test": test_idx.tolist(),
        },
        "library_versions": library_versions(),
        "params": selected.params,
    }
    if dummy_val is not None and dummy_test_bag is not None:
        metrics_payload["baseline"] = {
            "family": baseline_family,
            "test": dummy_test_bag,
            "validation": dummy_val,
        }
    metrics = jsonable(metrics_payload)
    label_list = sorted(set(y.astype(str).tolist())) if task == "classification" else None
    return {
        "metrics": metrics,
        "warnings": warnings,
        "selected_candidate": selected_family,
        "plot_files": plot_files,
        "pipeline": final_pipe,
        "feature_columns": spec.columns,
        "feature_spec": spec.to_dict(),
        "roles": roles,
        "classes": labels,
        "label_list": label_list,
    }


def _task_metrics(task: str, y_true, y_pred) -> dict:
    if task == "classification":
        return classification_metrics(
            np.asarray(y_true).astype(str), np.asarray(y_pred).astype(str)
        )
    return regression_metrics(
        np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    )


def _canonical_split_bag(task: str, metrics: dict | None) -> dict:
    if not metrics:
        return {}
    keys = ("macro_f1", "balanced_accuracy") if task == "classification" else ("mae", "rmse", "r2")
    return {key: metrics.get(key) for key in keys}


def _score_from_metrics(task: str, metrics: dict) -> float:
    if task == "classification":
        return float(metrics["macro_f1"])
    return float(-metrics["mae"])


def _fit_predict_metrics(cand, spec, X, y, fit_idx, pred_idx, task: str):
    pipe = Pipeline(
        [
            ("preprocess", build_preprocessor(spec, scale_numeric=cand.scale_numeric)),
            ("model", clone(cand.estimator)),
        ]
    )
    pipe.fit(X.iloc[fit_idx], y[fit_idx])
    pred = pipe.predict(X.iloc[pred_idx])
    return pipe, pred, _task_metrics(task, y[pred_idx], pred)


def _make_cv(task: str, mode: str, y_train, groups_train, seed: int):
    n = len(y_train)
    if n < 6:
        return None
    if mode == "time":
        n_splits = min(3, n - 1)
        if n_splits < 2:
            return None
        return TimeSeriesSplit(n_splits=n_splits)
    if mode == "group" and groups_train is not None:
        n_groups = int(pd.Series(groups_train).nunique())
        n_splits = min(3, n_groups)
        if n_splits < 2:
            return None
        return GroupKFold(n_splits=n_splits)
    if task == "classification":
        counts = pd.Series(y_train).value_counts()
        n_splits = int(min(3, counts.min()))
        if n_splits < 2:
            return None
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return KFold(n_splits=min(3, n), shuffle=True, random_state=seed)


def _feature_importance(pipeline: Pipeline):
    prep = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    inner = getattr(model, "estimator_", None) or getattr(model, "estimator", None)
    if inner is not None and (
        hasattr(inner, "feature_importances_") or hasattr(inner, "coef_")
    ):
        model = inner
    try:
        names = list(prep.get_feature_names_out())
    except Exception:  # noqa: BLE001
        return None
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        values = np.abs(coef).ravel() if coef.ndim == 1 else np.abs(coef).mean(axis=0)
    else:
        return None
    n = min(len(names), len(values))
    if n == 0:
        return None
    return names[:n], values[:n]
