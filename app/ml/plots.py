"""PNG plots under artifact_dir/plots. No uncertainty bands."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def _save(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return str(path)


def plot_confusion_matrix(y_true, y_pred, labels, path: Path) -> str:
    labels = list(labels)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(x) for x in labels])
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Confusion matrix")
    return _save(fig, path)


def plot_residuals(y_true, y_pred, path: Path) -> str:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))
    axes[0].scatter(y_pred, resid, s=18, alpha=0.8)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Residual")
    axes[0].set_title("Residuals")
    axes[1].scatter(y_true, y_pred, s=18, alpha=0.8)
    lo = float(np.nanmin([y_true.min(), y_pred.min()]))
    hi = float(np.nanmax([y_true.max(), y_pred.max()]))
    axes[1].plot([lo, hi], [lo, hi], color="black", linewidth=0.8)
    axes[1].set_xlabel("Actual")
    axes[1].set_ylabel("Predicted")
    axes[1].set_title("Actual vs predicted")
    return _save(fig, path)


def plot_feature_importance(names, values, path: Path, top_n: int = 20) -> str | None:
    names = list(names)
    values = np.asarray(values, dtype=float)
    if len(names) == 0 or len(values) == 0:
        return None
    n = min(len(names), len(values), top_n)
    order = np.argsort(np.abs(values))[-n:]
    fig, ax = plt.subplots(figsize=(6.5, max(2.5, 0.28 * n + 1.0)))
    ax.barh([str(names[i]) for i in order], values[order])
    ax.set_title("Feature importance")
    ax.set_xlabel("Importance")
    return _save(fig, path)


def plot_forecast(dates, y_true, y_pred, path: Path, title: str = "Forecast") -> str:
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot(dates, y_true, marker="o", label="Actual")
    ax.plot(dates, y_pred, marker="o", label="Predicted")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    fig.autofmt_xdate()
    return _save(fig, path)
