from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from backend.algorithms.registry import create_algorithm
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
