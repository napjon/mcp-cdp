"""Classification and regression train/predict tests on tiny frames."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from sklearn.base import clone
from sklearn.pipeline import Pipeline

from app.ml.preprocess import (
    MAX_CATEGORIES,
    FeatureSpec,
    build_preprocessor,
    split_supervised,
)
from app.ml.runner import run_predict, run_train

TRAIN_KEYS = {"metrics", "model_path", "plot_files", "warnings", "selected_candidate"}
PREDICT_KEYS = {
    "output_path",
    "metrics",
    "warnings",
    "model_path",
    "plot_files",
    "selected_candidate",
}


def _cls_frame(n: int = 48, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = (x1 + 0.4 * x2 > 0).astype(int)
    text = np.where(y == 1, "good alpha signal", "bad beta noise")
    cat = np.where(y == 1, "pos", "neg")
    return pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "cat": cat,
            "text": text,
            "target": y,
        }
    )


def _reg_frame(n: int = 48, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 3.0 * x1 - 1.5 * x2 + rng.normal(scale=0.05, size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "grp": np.where(x1 > 0, "a", "b"), "target": y})


def _train_job(path: Path, task: str, extra: dict | None = None) -> dict:
    cfg = {
        "task": task,
        "target": "target",
        "features": None,
        "roles": {},
        "split": "random",
        "test_size": 0.2,
        "budget": "quick",
        "seed": 42,
    }
    if extra:
        cfg.update(extra)
    return {
        "id": f"job-{task}",
        "type": "train",
        "config": cfg,
        "dataset_normalized_path": str(path),
    }


def test_classification_metrics_recompute(tmp_path: Path):
    csv_path = tmp_path / "cls.csv"
    _cls_frame().to_csv(csv_path, index=False)
    result = run_train(_train_job(csv_path, "classification"), {"artifact_dir": str(tmp_path / "art")})
    assert set(result.keys()) == TRAIN_KEYS
    metrics = result["metrics"]
    y_true = metrics["y_true"]
    y_pred = metrics["y_pred"]
    assert f1_score(y_true, y_pred, average="macro", zero_division=0) == pytest.approx(
        metrics["test"]["macro_f1"]
    )
    assert balanced_accuracy_score(y_true, y_pred) == pytest.approx(
        metrics["test"]["balanced_accuracy"]
    )
    assert "per_class" in metrics["test"]
    assert "confusion_matrix" in metrics["test"]
    assert Path(result["model_path"]).is_file()
    assert result["plot_files"]
    assert any(Path(p).is_file() and Path(p).stat().st_size > 0 for p in result["plot_files"])
    train_idx = set(metrics["split_indices"]["train"])
    test_idx = set(metrics["split_indices"]["test"])
    assert train_idx.isdisjoint(test_idx)
    assert result["selected_candidate"] in {"dummy", "linear", "xgboost"}


def test_classification_string_labels_xgboost_candidate(tmp_path: Path):
    from app.ml import classify

    df = pd.DataFrame(
        {
            "age": list(range(40)),
            "city": ["A", "B"] * 20,
            "survived": ["yes", "no"] * 20,
        }
    )
    csv_path = tmp_path / "str.csv"
    df.to_csv(csv_path, index=False)
    result = run_train(
        _train_job(
            csv_path,
            "classification",
            extra={
                "target": "survived",
                "features": ["age", "city"],
                "roles": {"age": "numeric", "city": "categorical", "survived": "categorical"},
            },
        ),
        {"artifact_dir": str(tmp_path / "art")},
    )
    assert set(result["metrics"]["y_true"]).issubset({"yes", "no"})
    cands = result["metrics"]["candidates"]
    for name in ("dummy", "linear", "xgboost"):
        assert name in cands
        status = cands[name]["status"]
        assert status in {"ok", "failed"}
        if status == "failed":
            assert cands[name].get("error")
    xgb = cands.get("xgboost")
    if classify.HAS_XGBOOST:
        assert xgb is not None
        assert xgb.get("status") == "ok"
        assert not any("xgboost" in w.lower() and "failed" in w.lower() for w in result["warnings"])
    else:
        assert xgb.get("status") == "failed"
        assert xgb.get("error")
        assert result["metrics"]["candidate_failures"]


def test_regression_metrics_recompute(tmp_path: Path):
    csv_path = tmp_path / "reg.csv"
    _reg_frame().to_csv(csv_path, index=False)
    result = run_train(_train_job(csv_path, "regression"), {"artifact_dir": str(tmp_path / "art")})
    assert set(result.keys()) == TRAIN_KEYS
    metrics = result["metrics"]
    y_true = np.asarray(metrics["y_true"], dtype=float)
    y_pred = np.asarray(metrics["y_pred"], dtype=float)
    assert mean_absolute_error(y_true, y_pred) == pytest.approx(metrics["test"]["mae"])
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    assert rmse == pytest.approx(metrics["test"]["rmse"])
    assert r2_score(y_true, y_pred) == pytest.approx(metrics["test"]["r2"])
    assert result["plot_files"]


def test_predict_aligns_columns_by_name(tmp_path: Path):
    csv_path = tmp_path / "cls.csv"
    df = _cls_frame(n=40)
    df.to_csv(csv_path, index=False)
    art = tmp_path / "art"
    train = run_train(
        _train_job(
            csv_path,
            "classification",
            extra={"features": ["x2", "x1", "cat", "text"], "roles": {"cat": "categorical", "text": "text"}},
        ),
        {"artifact_dir": str(art)},
    )
    shuffled = df[["text", "target", "x2", "cat", "x1"]].copy()
    shuffled["noise"] = 123
    pred_path = tmp_path / "pred_in.csv"
    shuffled.to_csv(pred_path, index=False)
    out = run_predict(
        {
            "id": "p1",
            "type": "predict",
            "config": {"task": "classification", "target": "target"},
            "predict_path": str(pred_path),
        },
        {"artifact_dir": str(art)},
    )
    assert set(out.keys()) == PREDICT_KEYS
    assert Path(out["output_path"]).is_file()
    pred_df = pd.read_csv(out["output_path"])
    assert len(pred_df) == len(shuffled)
    assert "prediction" in pred_df.columns
    assert out["metrics"] is not None
    assert "macro_f1" in out["metrics"]["test"]
    assert train["model_path"] == out["model_path"]


def test_regression_predict_without_actuals(tmp_path: Path):
    csv_path = tmp_path / "reg.csv"
    df = _reg_frame(n=40)
    df.to_csv(csv_path, index=False)
    art = tmp_path / "art"
    run_train(_train_job(csv_path, "regression"), {"artifact_dir": str(art)})
    holdout = df.drop(columns=["target"])[["x2", "grp", "x1"]]
    pred_path = tmp_path / "pred_in.csv"
    holdout.to_csv(pred_path, index=False)
    out = run_predict(
        {
            "id": "p2",
            "type": "predict",
            "config": {"task": "regression", "target": "target"},
            "predict_path": str(pred_path),
        },
        {"artifact_dir": str(art)},
    )
    assert out["metrics"] is None
    assert Path(out["output_path"]).is_file()
    assert isinstance(out["warnings"], list)


def test_unknown_split_raises(tmp_path: Path):
    y = np.arange(20)
    with pytest.raises(ValueError, match="Unknown split"):
        split_supervised(y, mode="kfold", test_size=0.2, seed=0)
    csv_path = tmp_path / "cls.csv"
    _cls_frame(n=24).to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="Unknown split"):
        run_train(
            _train_job(csv_path, "classification", extra={"split": "stratified"}),
            {"artifact_dir": str(tmp_path / "art")},
        )


def test_group_split_two_groups_no_overlap_or_raises(tmp_path: Path):
    groups = np.array(["A"] * 12 + ["B"] * 12)
    y = np.array([0, 1] * 12)
    try:
        train_fit, val_idx, test_idx = split_supervised(
            y, mode="group", test_size=0.2, seed=0, groups=groups
        )
    except ValueError as exc:
        assert "insufficient groups" in str(exc).lower()
    else:
        assert set(groups[train_fit]).isdisjoint(set(groups[val_idx]))
        assert set(groups[train_fit]).isdisjoint(set(groups[test_idx]))
        assert set(groups[val_idx]).isdisjoint(set(groups[test_idx]))

    df = _cls_frame(n=24)
    df["grp"] = groups
    csv_path = tmp_path / "groups.csv"
    df.to_csv(csv_path, index=False)
    job = _train_job(
        csv_path,
        "classification",
        extra={"split": "group", "group_columns": ["grp"], "roles": {"grp": "categorical"}},
    )
    try:
        result = run_train(job, {"artifact_dir": str(tmp_path / "art")})
    except ValueError as exc:
        assert "insufficient groups" in str(exc).lower()
        return
    split = result["metrics"]["split_indices"]
    val_groups = set(df.loc[split["val"], "grp"].astype(str))
    train_only = [i for i in split["train"] if i not in set(split["val"])]
    fit_groups = set(df.loc[train_only, "grp"].astype(str))
    assert fit_groups.isdisjoint(val_groups)


def test_predict_preserves_identifier_leading_zeros(tmp_path: Path):
    n = 40
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    y = (x > 0).astype(int)
    ids = [f"{i:05d}" for i in range(n)]
    ids[0] = "00123"
    df = pd.DataFrame({"id": ids, "x": x, "target": y})
    csv_path = tmp_path / "ids.csv"
    df.to_csv(csv_path, index=False)
    assert "00123" in csv_path.read_text(encoding="utf-8")
    art = tmp_path / "art"
    run_train(
        _train_job(
            csv_path,
            "classification",
            extra={
                "features": ["x"],
                "roles": {"id": "identifier", "x": "numeric"},
            },
        ),
        {"artifact_dir": str(art)},
    )
    pred = run_predict(
        {
            "id": "p-id",
            "type": "predict",
            "config": {"task": "classification", "target": "target"},
            "predict_path": str(csv_path),
        },
        {"artifact_dir": str(art)},
    )
    out_path = Path(pred["output_path"])
    out_text = out_path.read_text(encoding="utf-8")
    assert "00123" in out_text
    scored = pd.read_csv(out_path, dtype={"id": str})
    assert scored["id"].iloc[0] == "00123"
    assert "123" not in set(scored["id"].tolist())


def test_time_split_keeps_untouched_test_or_raises():
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    train_fit, val_idx, test_idx = split_supervised(
        np.arange(20),
        mode="time",
        test_size=0.2,
        seed=0,
        dates=dates,
    )
    assert len(test_idx) == 4
    assert set(train_fit).isdisjoint(test_idx)
    assert set(val_idx).isdisjoint(test_idx)
    assert max(test_idx) == 19
    with pytest.raises(ValueError, match="untouched test partition"):
        split_supervised(
            np.arange(4),
            mode="time",
            test_size=0.8,
            seed=0,
            dates=pd.date_range("2024-01-01", periods=4, freq="D"),
        )


def test_unseen_categories_do_not_crash_and_encoded_width_capped(tmp_path: Path):
    n_levels = MAX_CATEGORIES + 30
    cats = [f"c{i}" for i in range(n_levels)]
    df = pd.DataFrame(
        {
            "cat": cats + cats,
            "x": np.arange(n_levels * 2, dtype=float),
            "target": ([0, 1] * n_levels),
        }
    )
    spec = FeatureSpec(numeric=["x"], categorical=["cat"], text=[])
    prep = build_preprocessor(spec, scale_numeric=False)
    prep.fit(df[["x", "cat"]])
    transformed = prep.transform(df[["x", "cat"]])
    cat_width = transformed.shape[1] - len(spec.numeric)
    assert cat_width <= MAX_CATEGORIES + 1
    unseen = pd.DataFrame({"x": [0.0], "cat": ["brand_new_level"]})
    out = prep.transform(unseen)
    assert out.shape[0] == 1
    assert out.shape[1] == transformed.shape[1]

    csv_path = tmp_path / "cat.csv"
    small = pd.DataFrame(
        {
            "cat": ["a", "b"] * 16,
            "x": np.arange(32, dtype=float),
            "target": [0, 1] * 16,
        }
    )
    small.to_csv(csv_path, index=False)
    art = tmp_path / "art"
    run_train(
        _train_job(
            csv_path,
            "classification",
            extra={"features": ["cat", "x"], "roles": {"cat": "categorical", "x": "numeric"}},
        ),
        {"artifact_dir": str(art)},
    )
    pred_path = tmp_path / "pred.csv"
    pd.DataFrame({"cat": ["unseen_zzz", "a"], "x": [1.0, 2.0]}).to_csv(pred_path, index=False)
    pred = run_predict(
        {
            "id": "p-cat",
            "type": "predict",
            "config": {"task": "classification", "target": "target"},
            "predict_path": str(pred_path),
        },
        {"artifact_dir": str(art)},
    )
    scored = pd.read_csv(pred["output_path"])
    assert len(scored) == 2
    assert "prediction" in scored.columns


def test_linear_pipeline_has_scaler_dummy_and_xgb_do_not():
    from app.ml import classify, regress

    spec = FeatureSpec(numeric=["x"], categorical=["c"], text=[])
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "c": ["a", "b"] * 3})
    y_cls = np.array(["yes", "no", "yes", "no", "yes", "no"])
    y_reg = np.array([1.0, 2.0, 1.5, 2.5, 1.2, 2.2])

    def _has_scaler(module, y):
        found = {}
        for cand in module.iter_candidates("quick", 0):
            pipe = Pipeline(
                [
                    ("preprocess", build_preprocessor(spec, scale_numeric=cand.scale_numeric)),
                    ("model", clone(cand.estimator)),
                ]
            )
            pipe.fit(X, y)
            num = pipe.named_steps["preprocess"].named_transformers_["num"]
            found[cand.family] = "scaler" in num.named_steps
        return found

    for module, y in ((classify, y_cls), (regress, y_reg)):
        found = _has_scaler(module, y)
        assert found["linear"] is True
        assert found["dummy"] is False
        if module.HAS_XGBOOST:
            assert found["xgboost"] is False
        else:
            assert "xgboost" not in found
