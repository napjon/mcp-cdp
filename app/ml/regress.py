"""Regression candidate estimators."""

from __future__ import annotations

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge

from app.ml.classify import Candidate

try:
    from xgboost import XGBRegressor

    HAS_XGBOOST = True
except Exception:  # noqa: BLE001  # missing package or native lib (libomp)
    XGBRegressor = None
    HAS_XGBOOST = False


def iter_candidates(budget: str, seed: int) -> list[Candidate]:
    quick = (budget or "quick") != "thorough"
    out = [
        Candidate(
            family="dummy",
            name="dummy",
            estimator=DummyRegressor(strategy="mean"),
            scale_numeric=False,
            params={"strategy": "mean"},
        )
    ]
    if quick:
        out.append(
            Candidate(
                family="linear",
                name="linear",
                estimator=Ridge(alpha=1.0),
                scale_numeric=True,
                params={"alpha": 1.0},
            )
        )
        if HAS_XGBOOST:
            out.append(
                Candidate(
                    family="xgboost",
                    name="xgboost",
                    estimator=XGBRegressor(
                        n_estimators=20,
                        max_depth=3,
                        learning_rate=0.3,
                        subsample=1.0,
                        colsample_bytree=1.0,
                        random_state=seed,
                        n_jobs=1,
                        verbosity=0,
                        objective="reg:squarederror",
                    ),
                    scale_numeric=False,
                    params={"n_estimators": 20, "max_depth": 3, "learning_rate": 0.3},
                )
            )
        return out

    for alpha in (0.1, 1.0, 10.0):
        out.append(
            Candidate(
                family="linear",
                name=f"linear_a{alpha}",
                estimator=Ridge(alpha=alpha),
                scale_numeric=True,
                params={"alpha": alpha},
            )
        )
    if HAS_XGBOOST:
        for n_est, depth, lr in ((40, 3, 0.1), (50, 4, 0.05)):
            out.append(
                Candidate(
                    family="xgboost",
                    name=f"xgboost_n{n_est}_d{depth}",
                    estimator=XGBRegressor(
                        n_estimators=n_est,
                        max_depth=depth,
                        learning_rate=lr,
                        subsample=1.0,
                        colsample_bytree=1.0,
                        random_state=seed,
                        n_jobs=1,
                        verbosity=0,
                        objective="reg:squarederror",
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
