"""ML engine: train and predict entrypoints used by the platform worker."""

from app.ml.runner import run_predict, run_train

__all__ = ["run_predict", "run_train"]
