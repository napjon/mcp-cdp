"""Preprocessor and lag features must not leak future or test information."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.ml.forecast import lag_rolling_features
from app.ml.preprocess import FeatureSpec, build_preprocessor, flag_target_leakage
from app.ml.runner import run_train


def test_imputer_and_scaler_fit_on_train_only():
    train = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0],
            "cat": ["a", "b", "a"],
            "text": ["hello world", "hello", "world"],
        }
    )
    spec = FeatureSpec(numeric=["x"], categorical=["cat"], text=["text"])
    prep = build_preprocessor(spec, scale_numeric=True)
    prep.fit(train)
    num = prep.named_transformers_["num"]
    median = float(num.named_steps["imputer"].statistics_[0])
    mean = float(num.named_steps["scaler"].mean_[0])
    assert median == pytest.approx(2.0)
    assert mean == pytest.approx(2.0)
    test = pd.DataFrame({"x": [1e9], "cat": ["unseen"], "text": ["secretxyz onlyintest"]})
    transformed = prep.transform(test)
    assert transformed.shape[0] == 1
    num_after = prep.named_transformers_["num"]
    assert float(num_after.named_steps["imputer"].statistics_[0]) == pytest.approx(2.0)
    assert float(num_after.named_steps["scaler"].mean_[0]) == pytest.approx(2.0)


def test_tfidf_vocabulary_ignores_test_tokens():
    train = pd.DataFrame(
        {
            "x": [0.0, 1.0, 0.0, 1.0],
            "text": ["alpha beta", "alpha", "beta gamma", "alpha gamma"],
        }
    )
    spec = FeatureSpec(numeric=["x"], categorical=[], text=["text"])
    prep = build_preprocessor(spec, scale_numeric=False)
    prep.fit(train)
    vec = prep.named_transformers_["text"]
    vocab = vec.vocabulary_
    assert "alpha" in vocab
    assert "secretxyz" not in vocab
    test = pd.DataFrame({"x": [0.0], "text": ["secretxyz alpha"]})
    prep.transform(test)
    assert "secretxyz" not in vec.vocabulary_


def test_saved_pipeline_ignores_test_partition(tmp_path: Path):
    rng = np.random.default_rng(0)
    n = 40
    x = rng.normal(size=n)
    y = (x > 0).astype(int)
    text = np.where(y == 1, "good token", "bad token")
    df = pd.DataFrame({"x": x, "text": text, "target": y})
    # Unique token only on rows that a 20% tail split would isolate if we use time split.
    df["date"] = pd.date_range("2024-01-01", periods=n, freq="D")
    df.loc[df.index[-8:], "text"] = "onlyintesttoken " + df.loc[df.index[-8:], "text"]
    df.loc[df.index[-8:], "x"] = 1e6
    csv_path = tmp_path / "leak.csv"
    df.to_csv(csv_path, index=False)
    result = run_train(
        {
            "id": "leak",
            "type": "train",
            "config": {
                "task": "classification",
                "target": "target",
                "features": ["x", "text"],
                "roles": {"text": "text", "x": "numeric"},
                "split": "time",
                "date_column": "date",
                "test_size": 0.2,
                "budget": "quick",
                "seed": 0,
            },
            "dataset_normalized_path": str(csv_path),
        },
        {"artifact_dir": str(tmp_path / "art")},
    )
    import joblib

    bundle = joblib.load(result["model_path"])
    prep = bundle["pipeline"].named_steps["preprocess"]
    num = prep.named_transformers_["num"]
    train_idx = result["metrics"]["split_indices"]["train"]
    train_median = float(pd.to_numeric(df.loc[train_idx, "x"]).median())
    assert float(num.named_steps["imputer"].statistics_[0]) == pytest.approx(train_median, rel=1e-6)
    vec = prep.named_transformers_["text"]
    vocab = vec.vocabulary_ if hasattr(vec, "vocabulary_") else vec.named_steps["tfidf"].vocabulary_
    assert "onlyintesttoken" not in vocab


def test_forecast_backtest_origins_do_not_enter_holdout():
    from app.ml.forecast import _train_origins

    n, horizon = 24, 4
    train_end = n - horizon
    origins = _train_origins(n, horizon, 4)
    assert origins
    for origin in origins:
        assert origin < train_end
        assert origin + horizon <= train_end


def test_forecast_lags_no_future_and_group_isolated():
    df = pd.DataFrame(
        {
            "g": ["a", "a", "a", "a", "b", "b", "b", "b"],
            "d": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                ]
            ),
            "y": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
        }
    )
    out = lag_rolling_features(
        df,
        target="y",
        group_columns=["g"],
        lags=[1, 2],
        roll_windows=[2],
    )
    assert pd.isna(out.loc[0, "lag_1"])
    assert out.loc[1, "lag_1"] == 1.0
    assert out.loc[2, "lag_1"] == 2.0
    assert out.loc[3, "lag_1"] == 3.0
    assert out.loc[3, "lag_2"] == 2.0
    assert pd.isna(out.loc[4, "lag_1"])
    assert out.loc[5, "lag_1"] == 10.0
    assert out.loc[1, "lag_1"] != 2.0
    assert out.loc[4, "lag_1"] != 4.0
    # rolling uses shifted values so current y is excluded
    assert out.loc[2, "roll_mean_2"] == pytest.approx(1.5)
    assert out.loc[6, "roll_mean_2"] == pytest.approx(15.0)


def test_identical_feature_target_warns(tmp_path: Path):
    df = pd.DataFrame(
        {
            "x": np.arange(24, dtype=float),
            "target": [0, 1] * 12,
        }
    )
    df["copy_target"] = df["target"]
    warnings = flag_target_leakage(df, "target", ["x", "copy_target"])
    assert any("copy_target" in w and "identical" in w.lower() for w in warnings)
    assert not any("feature 'x'" in w for w in warnings)

    csv_path = tmp_path / "leak.csv"
    df.to_csv(csv_path, index=False)
    job = {
        "id": "leak-ident",
        "type": "train",
        "config": {
            "task": "classification",
            "target": "target",
            "features": ["x", "copy_target"],
            "roles": {"x": "numeric", "copy_target": "numeric"},
            "split": "random",
            "test_size": 0.2,
            "budget": "quick",
            "seed": 0,
        },
        "dataset_normalized_path": str(csv_path),
    }
    with pytest.raises(ValueError, match="leakage"):
        run_train(job, {"artifact_dir": str(tmp_path / "art-default")})
    job["config"] = dict(job["config"], allow_target_leakage=True)
    result = run_train(job, {"artifact_dir": str(tmp_path / "art-allow")})
    assert any("copy_target" in w and "identical" in w.lower() for w in result["warnings"])
