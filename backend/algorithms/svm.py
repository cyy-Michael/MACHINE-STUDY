from __future__ import annotations

import numpy as np

from ..data.preprocessing import standardize
from .base_model import BaseModel, as_float_matrix, require_y


class SVM(BaseModel):
    """Linear soft-margin SVM trained by subgradient descent on hinge loss.

    Objective: min (1/2)||w||^2 + C * sum(max(0, 1 - y*(x@w + b))).
    Multi-class problems use one-vs-rest; prediction picks the largest
    decision value. Features are standardized inside fit so the
    gradient step works on any input scale.
    """

    algorithm_name = "svm"
    task_type = "classification"

    def __init__(self, C: float = 1.0, learning_rate: float = 0.001, n_iters: int = 2000) -> None:
        if C <= 0:
            raise ValueError("C must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if n_iters < 1:
            raise ValueError("n_iters must be at least 1")
        self.C = C
        self.learning_rate = learning_rate
        self.n_iters = n_iters

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "SVM":
        X, self._mean, self._scale = standardize(as_float_matrix(X))
        y = require_y(y, len(X))
        self.classes_ = np.unique(y)
        self.w_ = np.zeros((len(self.classes_), X.shape[1]))
        self.b_ = np.zeros(len(self.classes_))
        for index, c in enumerate(self.classes_):
            target = np.where(y == c, 1.0, -1.0)
            w, b = np.zeros(X.shape[1]), 0.0
            for _ in range(self.n_iters):
                margin = target * (X @ w + b)
                violating = margin < 1
                grad_w = w - self.C * X[violating].T @ target[violating]
                grad_b = -self.C * np.sum(target[violating])
                w -= self.learning_rate * grad_w
                b -= self.learning_rate * grad_b
            self.w_[index], self.b_[index] = w, b
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "classes_"):
            raise RuntimeError("fit must be called before prediction")
        X = (as_float_matrix(X) - self._mean) / self._scale
        return X @ self.w_.T + self.b_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Softmax over decision values; margins are not calibrated probabilities."""
        scores = self.decision_function(X)
        shifted = scores - scores.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]
