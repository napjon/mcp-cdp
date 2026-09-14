"""Classification candidate estimators."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except Exception:  # noqa: BLE001  # missing package or native lib (libomp)
    XGBClassifier = None
    HAS_XGBOOST = False


class LabelMappedClassifier(BaseEstimator, ClassifierMixin):
    """Map string labels to 0..n-1 so XGBoost can fit, then invert predictions."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        y_str = np.asarray(y).astype(str)
        self.encoder_ = LabelEncoder()
        encoded = self.encoder_.fit_transform(y_str)
        self.classes_ = self.encoder_.classes_
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, encoded)
        return self

    def predict(self, X):
        pred = np.asarray(self.estimator_.predict(X))
        if np.issubdtype(pred.dtype, np.floating):
            pred = np.rint(pred).astype(int)
        else:
            pred = pred.astype(int, copy=False)
        return self.encoder_.inverse_transform(pred)


@dataclass
class Candidate:
    family: str
    name: str
    estimator: object
    scale_numeric: bool
    params: dict = field(default_factory=dict)


def _mapped(estimator):
    return LabelMappedClassifier(estimator)


def iter_candidates(budget: str, seed: int) -> list[Candidate]:
    quick = (budget or "quick") != "thorough"
    out = [
        Candidate(
            family="dummy",
            name="dummy",
            estimator=_mapped(DummyClassifier(strategy="most_frequent")),
            scale_numeric=False,
            params={"strategy": "most_frequent"},
        )
    ]
    if quick:
        out.append(
            Candidate(
                family="linear",
                name="linear",
                estimator=_mapped(
                    LogisticRegression(
                        C=1.0, max_iter=250, solver="lbfgs", random_state=seed
                    )
                ),
                scale_numeric=True,
                params={"C": 1.0, "max_iter": 250},
            )
        )
        if HAS_XGBOOST:
            out.append(
                Candidate(
                    family="xgboost",
                    name="xgboost",
                    estimator=_mapped(
                        XGBClassifier(
                            n_estimators=20,
                            max_depth=3,
                            learning_rate=0.3,
                            subsample=1.0,
                            colsample_bytree=1.0,
                            random_state=seed,
                            n_jobs=1,
                            verbosity=0,
                        )
                    ),
                    scale_numeric=False,
                    params={"n_estimators": 20, "max_depth": 3, "learning_rate": 0.3},
                )
            )
        return out

    for c_value in (0.1, 1.0, 10.0):
        out.append(
            Candidate(
                family="linear",
                name=f"linear_C{c_value}",
                estimator=_mapped(
                    LogisticRegression(
                        C=c_value, max_iter=400, solver="lbfgs", random_state=seed
                    )
                ),
                scale_numeric=True,
                params={"C": c_value, "max_iter": 400},
            )
        )
    if HAS_XGBOOST:
        for n_est, depth, lr in ((40, 3, 0.1), (50, 4, 0.05)):
            out.append(
                Candidate(
                    family="xgboost",
                    name=f"xgboost_n{n_est}_d{depth}",
                    estimator=_mapped(
                        XGBClassifier(
                            n_estimators=n_est,
                            max_depth=depth,
                            learning_rate=lr,
                            subsample=1.0,
                            colsample_bytree=1.0,
                            random_state=seed,
                            n_jobs=1,
                            verbosity=0,
                        )
                    ),
                    scale_numeric=False,
                    params={
                        "n_estimators": n_est,
                        "max_depth": depth,
                        "learning_rate": lr,
                    },
                )
            )
    return out
