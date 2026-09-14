"""Metric helpers. Values are JSON-serializable."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)


def jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return jsonable(float(obj))
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if hasattr(obj, "item") and not isinstance(obj, (bytes, bytearray)):
        try:
            return jsonable(obj.item())
        except (ValueError, AttributeError):
            pass
    return str(obj)


def classification_metrics(y_true, y_pred, labels=None) -> dict:
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    if labels is None:
        labels = list(dict.fromkeys(list(y_true) + list(y_pred)))
    else:
        labels = list(labels)
    p, r, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class = {
        str(lab): {"precision": float(p[i]), "recall": float(r[i])}
        for i, lab in enumerate(labels)
    }
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "labels": [str(x) for x in labels],
    }


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    if len(y_true) < 2:
        r2 = None
    else:
        r2_val = float(r2_score(y_true, y_pred))
        r2 = None if (np.isnan(r2_val) or np.isinf(r2_val)) else r2_val
    return {"mae": mae, "rmse": rmse, "r2": r2}


def library_versions() -> dict[str, str]:
    import joblib
    import numpy
    import pandas
    import sklearn

    versions = {
        "sklearn": sklearn.__version__,
        "pandas": pandas.__version__,
        "numpy": numpy.__version__,
        "joblib": joblib.__version__,
    }
    try:
        import xgboost

        versions["xgboost"] = xgboost.__version__
    except Exception:  # noqa: BLE001
        versions["xgboost"] = "missing"
    return versions
