"""Grouped forecast behavior on tiny synthetic series."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.ml.runner import run_predict, run_train


def _daily_frame(n: int = 24, groups: tuple[str, ...] = ("A", "B")) -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2024-01-01")
    for g_i, g in enumerate(groups):
        for t in range(n):
            rows.append(
                {
                    "date": (start + pd.Timedelta(days=t)).strftime("%Y-%m-%d"),
                    "sku": g,
                    "sales": float(10 * (g_i + 1) + t + (3 if t % 7 == 0 else 0)),
                }
            )
    return pd.DataFrame(rows)


def _job(path: Path, extra: dict | None = None) -> dict:
    cfg = {
        "task": "forecast",
        "target": "sales",
        "date_column": "date",
        "group_columns": ["sku"],
        "frequency": "D",
        "horizon": 4,
        "duplicate_timestamp": "reject",
        "budget": "quick",
        "seed": 42,
    }
    if extra:
        cfg.update(extra)
    return {
        "id": "forecast-1",
        "type": "train",
        "config": cfg,
        "dataset_normalized_path": str(path),
    }


def test_forecast_grouped_train_and_predict(tmp_path: Path):
    csv_path = tmp_path / "fc.csv"
    _daily_frame().to_csv(csv_path, index=False)
    art = tmp_path / "art"
    result = run_train(_job(csv_path), {"artifact_dir": str(art)})
    assert set(result.keys()) == {
        "metrics",
        "model_path",
        "plot_files",
        "warnings",
        "selected_candidate",
    }
    assert result["selected_candidate"] in {"last_value", "seasonal_naive", "xgboost"}
    metrics = result["metrics"]
    assert metrics["y_true"]
    assert len(metrics["y_true"]) == len(metrics["y_pred"])
    assert set(metrics["y_group"]) <= {"A", "B"}
    assert Path(result["model_path"]).is_file()
    assert result["plot_files"]

    pred = run_predict(
        {
            "id": "fp",
            "type": "predict",
            "config": result["metrics"].get("params") or {},
            "predict_path": str(csv_path),
        },
        {"artifact_dir": str(art)},
    )
    out = pd.read_csv(pred["output_path"])
    assert "prediction" in out.columns
    assert len(out) == 8  # 2 groups * horizon 4
    assert pred["metrics"] is None or "test" in pred["metrics"]


def test_ambiguous_dates_raise(tmp_path: Path):
    df = pd.DataFrame(
        {
            "date": ["01/02/2024", "03/04/2024", "05/06/2024", "07/08/2024"] * 2,
            "sku": ["A"] * 4 + ["B"] * 4,
            "sales": np.arange(8, dtype=float),
        }
    )
    path = tmp_path / "amb.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Ambiguous date"):
        run_train(_job(path, extra={"horizon": 1}), {"artifact_dir": str(tmp_path / "art")})


def test_duplicate_timestamp_reject_and_aggregate(tmp_path: Path):
    df = _daily_frame(n=16, groups=("A",))
    extra = df.iloc[[3]].copy()
    extra["sales"] = 99.0
    rejected = pd.concat([df, extra], ignore_index=True)
    path = tmp_path / "dup.csv"
    rejected.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Duplicate timestamps"):
        run_train(_job(path, extra={"group_columns": ["sku"]}), {"artifact_dir": str(tmp_path / "art1")})

    mean_job = _job(path, extra={"duplicate_timestamp": "aggregate_mean"})
    result = run_train(mean_job, {"artifact_dir": str(tmp_path / "art2")})
    assert result["selected_candidate"]


def test_insufficient_history_raise_and_exclude(tmp_path: Path):
    long = _daily_frame(n=20, groups=("A",))
    short = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3, freq="D").strftime("%Y-%m-%d"),
            "sku": ["B"] * 3,
            "sales": [1.0, 2.0, 3.0],
        }
    )
    df = pd.concat([long, short], ignore_index=True)
    path = tmp_path / "short.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Insufficient history"):
        run_train(_job(path, extra={"horizon": 4}), {"artifact_dir": str(tmp_path / "art1")})

    result = run_train(
        _job(path, extra={"horizon": 4, "allow_exclude_insufficient": True}),
        {"artifact_dir": str(tmp_path / "art2")},
    )
    assert "B" in result["metrics"]["groups_excluded"]
    assert any("Insufficient history" in w for w in result["warnings"])
    assert set(result["metrics"]["y_group"]) == {"A"}


def test_missing_periods_not_filled_with_zero(tmp_path: Path):
    rows = []
    start = pd.Timestamp("2024-01-01")
    for t in range(20):
        if t == 5:
            continue
        rows.append(
            {
                "date": (start + pd.Timedelta(days=t)).strftime("%Y-%m-%d"),
                "sku": "A",
                "sales": float(10 + t),
            }
        )
    df = pd.DataFrame(rows)
    path = tmp_path / "gap.csv"
    df.to_csv(path, index=False)
    result = run_train(_job(path, extra={"horizon": 3, "group_columns": ["sku"]}), {"artifact_dir": str(tmp_path / "art")})
    counts = result["metrics"]["missing_period_counts"]
    assert counts.get("A", 0) >= 1
    assert 0.0 not in result["metrics"]["y_true"]
    assert any("missing" in w.lower() for w in result["warnings"])


def test_weekly_train_predict_does_not_fail_insufficient_history(tmp_path: Path):
    n = 20
    dates = pd.date_range("2024-01-01", periods=n, freq="7D")
    df = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "sku": ["A"] * n,
            "sales": np.arange(n, dtype=float) + 10.0,
        }
    )
    path = tmp_path / "weekly.csv"
    df.to_csv(path, index=False)
    art = tmp_path / "art"
    result = run_train(
        _job(path, extra={"frequency": "W", "horizon": 4, "group_columns": ["sku"]}),
        {"artifact_dir": str(art)},
    )
    assert result["selected_candidate"]
    assert not any("Insufficient history" in w for w in result["warnings"])
    pred = run_predict(
        {
            "id": "fp-w",
            "type": "predict",
            "config": {"task": "forecast"},
            "predict_path": str(path),
        },
        {"artifact_dir": str(art)},
    )
    out = pd.read_csv(pred["output_path"])
    assert "prediction" in out.columns
    assert len(out) == 4


def test_horizon_and_series_limits(tmp_path: Path):
    path = tmp_path / "lim.csv"
    _daily_frame(n=10).to_csv(path, index=False)
    with pytest.raises(ValueError, match="horizon"):
        run_train(_job(path, extra={"horizon": 0}), {"artifact_dir": str(tmp_path / "h0")})
    with pytest.raises(ValueError, match="horizon"):
        run_train(_job(path, extra={"horizon": 91}), {"artifact_dir": str(tmp_path / "h91")})

    rows = []
    for i in range(101):
        for t in range(6):
            rows.append(
                {
                    "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=t),
                    "sku": f"s{i}",
                    "sales": float(t),
                }
            )
    many = pd.DataFrame(rows)
    many["date"] = pd.to_datetime(many["date"]).dt.strftime("%Y-%m-%d")
    many_path = tmp_path / "many.csv"
    many.to_csv(many_path, index=False)
    with pytest.raises(ValueError, match="maximum is 100"):
        run_train(_job(many_path, extra={"horizon": 1}), {"artifact_dir": str(tmp_path / "many")})


def test_predict_metrics_when_actuals_present(tmp_path: Path):
    df = _daily_frame(n=20)
    path = tmp_path / "fc.csv"
    df.to_csv(path, index=False)
    art = tmp_path / "art"
    run_train(_job(path, extra={"horizon": 3}), {"artifact_dir": str(art)})
    future_rows = []
    last = pd.Timestamp("2024-01-20")
    for g, base in (("A", 10), ("B", 20)):
        for h in range(1, 4):
            future_rows.append(
                {
                    "date": (last + pd.Timedelta(days=h)).strftime("%Y-%m-%d"),
                    "sku": g,
                    "sales": float(base + 19 + h),
                }
            )
    fut = pd.DataFrame(future_rows)
    fut_path = tmp_path / "future.csv"
    fut.to_csv(fut_path, index=False)
    pred = run_predict(
        {
            "id": "fp",
            "type": "predict",
            "config": {"task": "forecast"},
            "predict_path": str(fut_path),
        },
        {"artifact_dir": str(art)},
    )
    assert pred["metrics"] is not None
    assert pred["metrics"]["test"]["mae"] >= 0


def test_predict_rejects_unseen_forecast_groups(tmp_path: Path):
    csv_path = tmp_path / "fc.csv"
    _daily_frame(groups=("A", "B")).to_csv(csv_path, index=False)
    art = tmp_path / "art"
    run_train(_job(csv_path), {"artifact_dir": str(art)})
    unseen = _daily_frame(n=6, groups=("C",))
    pred_path = tmp_path / "unseen.csv"
    unseen.to_csv(pred_path, index=False)
    with pytest.raises(ValueError, match="Unseen forecast group"):
        run_predict(
            {
                "id": "fp-unseen",
                "type": "predict",
                "config": {"task": "forecast"},
                "predict_path": str(pred_path),
            },
            {"artifact_dir": str(art)},
        )


def test_forecast_last_value_nested_metric_bags(tmp_path: Path):
    csv_path = tmp_path / "fc.csv"
    _daily_frame().to_csv(csv_path, index=False)
    result = run_train(_job(csv_path), {"artifact_dir": str(tmp_path / "art")})
    metrics = result["metrics"]
    keys = ("mae", "rmse", "r2")
    lv = metrics["candidates"]["last_value"]
    selected = metrics["candidates"][result["selected_candidate"]]
    bags = (
        metrics["test"],
        metrics["validation"],
        lv["test"],
        lv["validation"],
        selected["test"],
        selected["validation"],
    )
    for bag in bags:
        for key in keys:
            assert key in bag
    baseline = metrics["baseline"]
    assert baseline["family"] == "last_value"
    for key in keys:
        assert key in baseline["test"]
        assert key in baseline["validation"]
    if result["selected_candidate"] != "last_value":
        assert lv["test"] != metrics["validation"]
