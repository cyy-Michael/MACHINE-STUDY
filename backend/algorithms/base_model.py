from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseModel(ABC):
    """Small common interface shared by every model exposed to the platform."""

    algorithm_name = "base"
    task_type = "unknown"

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "BaseModel":
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_params(self) -> dict[str, Any]:
        """Return constructor parameters so the API can render a parameter form."""
        return {
            key: value
            for key, value in vars(self).items()
            if not key.endswith("_") and not key.startswith("_")
        }


def as_float_matrix(X: np.ndarray) -> np.ndarray:
    array = np.asarray(X, dtype=float)
    if array.ndim != 2:
        raise ValueError("X must be a 2-dimensional numeric matrix")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("X must contain at least one sample and one feature")
    if not np.isfinite(array).all():
        raise ValueError("X contains NaN or infinite values; preprocess it first")
    return array


def require_y(y: np.ndarray | None, n_samples: int) -> np.ndarray:
    if y is None:
        raise ValueError("y is required for supervised learning")
    array = np.asarray(y)
    if array.ndim != 1 or len(array) != n_samples:
        raise ValueError("y must be a 1-dimensional array with one value per sample")
    return array
