from __future__ import annotations

import numpy as np

from ..data.preprocessing import standardize
from .base_model import BaseModel, as_float_matrix, require_y


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


class LogisticRegression(BaseModel):
    """Logistic regression trained by gradient descent on cross-entropy loss.

    Multi-class problems are handled with one-vs-rest: one sigmoid
    classifier per class, prediction picks the highest probability.
    Features are standardized inside fit so the optimizer converges on
    any input scale.
    """

    algorithm_name = "logistic_regression"
    task_type = "classification"

    def __init__(self, learning_rate: float = 0.1, n_iters: int = 1000, reg_lambda: float = 0.0) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if n_iters < 1:
            raise ValueError("n_iters must be at least 1")
        if reg_lambda < 0:
            raise ValueError("reg_lambda must be non-negative")
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.reg_lambda = reg_lambda

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "LogisticRegression":
        X, self._mean, self._scale = standardize(as_float_matrix(X))
        y = require_y(y, len(X))
        X_b = np.c_[np.ones(len(X)), X]
        self.classes_ = np.unique(y)
        weights = []
        for c in self.classes_:
            target = (y == c).astype(float)
            w = np.zeros(X_b.shape[1])
            for _ in range(self.n_iters):
                error = _sigmoid(X_b @ w) - target
                grad = X_b.T @ error / len(X) + self.reg_lambda * w
                w -= self.learning_rate * grad
            weights.append(w)
        self.w_ = np.vstack(weights)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "classes_"):
            raise RuntimeError("fit must be called before prediction")
        X = (as_float_matrix(X) - self._mean) / self._scale
        return _sigmoid(np.c_[np.ones(len(X)), X] @ self.w_.T)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
