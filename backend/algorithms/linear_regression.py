from __future__ import annotations

import numpy as np

from .base_model import BaseModel, as_float_matrix, require_y


class LinearRegression(BaseModel):
    """Least-squares linear regression solved with the normal equation.

    The bias term is folded in by prepending a column of ones; the
    Moore-Penrose pseudo-inverse keeps the solution stable when X^T X
    is singular.
    """

    algorithm_name = "linear_regression"
    task_type = "regression"

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "LinearRegression":
        X = as_float_matrix(X)
        y = require_y(y, len(X)).astype(float)
        X_b = np.c_[np.ones(len(X)), X]
        self.w_ = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "w_"):
            raise RuntimeError("fit must be called before prediction")
        X = as_float_matrix(X)
        return np.c_[np.ones(len(X)), X] @ self.w_
