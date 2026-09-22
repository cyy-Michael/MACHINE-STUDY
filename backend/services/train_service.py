from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from backend.algorithms.registry import create_algorithm
from backend.data.preprocessing import k_fold_indices
from .evaluation import classification_metrics, regression_metrics


def run_experiment(name: str, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray | None = None, y_test: np.ndarray | None = None, **params: Any) -> dict[str, Any]:
    """Run one model through the common fit/predict lifecycle used by future APIs."""
    model = create_algorithm(name, **params)
    started = perf_counter()
    if getattr(model, "task_type", None) in {"clustering", "dimensionality_reduction"}:
        model.fit(X_train)
        predictions = model.predict(X_test)
    else:
        if y_train is None:
            raise ValueError("supervised algorithms require y_train")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
    predict_seconds = perf_counter() - started
    result: dict[str, Any] = {
        "algorithm": name,
        "task_type": model.task_type,
        "parameters": model.get_params(),
        "prediction_time_seconds": predict_seconds,
        "predictions": predictions.tolist(),
        "model": model,
    }
    if y_test is not None and model.task_type == "classification":
        result["metrics"] = classification_metrics(y_test, predictions)
    elif y_test is not None and model.task_type == "regression":
        result["metrics"] = regression_metrics(y_test, predictions)
    return result


def run_cross_validation(name: str, X: np.ndarray, y: np.ndarray | None = None, n_splits: int = 5, **params: Any) -> dict[str, Any]:
    """Run stratified K-fold cross-validation for one supervised model.

    Returns per-fold metrics plus the mean/std of every numeric metric so
    the frontend can show both quality and stability.
    """
    model = create_algorithm(name, **params)
    if getattr(model, "task_type", None) not in {"classification", "regression"}:
        raise ValueError(f"cross-validation only supports supervised algorithms, got: {name}")
    if y is None:
        raise ValueError("supervised algorithms require y")
    y = np.asarray(y)

    fold_results = []
    for fold, (train_indices, test_indices) in enumerate(k_fold_indices(X, y, n_splits=n_splits), start=1):
        metrics = run_experiment(
            name, X[train_indices], X[test_indices], y[train_indices], y[test_indices], **params
        ).get("metrics", {})
        fold_results.append({"fold": fold, "n_train": int(len(train_indices)), "n_test": int(len(test_indices)), "metrics": metrics})

    numeric_keys = [key for key, value in fold_results[0]["metrics"].items() if isinstance(value, (int, float))]
    summary = {
        key: {
            "mean": float(np.mean([fold["metrics"][key] for fold in fold_results])),
            "std": float(np.std([fold["metrics"][key] for fold in fold_results])),
        }
        for key in numeric_keys
    }
    return {"algorithm": name, "task_type": model.task_type, "parameters": model.get_params(), "n_splits": n_splits, "folds": fold_results, "summary": summary}
