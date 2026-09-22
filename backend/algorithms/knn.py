from __future__ import annotations

import numpy as np

from ..data.preprocessing import standardize
from .base_model import BaseModel, as_float_matrix, require_y


class KNN(BaseModel):
    """K-nearest-neighbours classifier with Euclidean distance and majority vote.

    fit only memorizes the (standardized) training set; prediction
    computes distances with array broadcasting. predict_proba returns
    the neighbour vote fractions.
    """

    algorithm_name = "knn"
    task_type = "classification"

    def __init__(self, k: int = 5) -> None:
        if k < 1:
            raise ValueError("k must be at least 1")
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "KNN":
        X, self._mean, self._scale = standardize(as_float_matrix(X))
        y = require_y(y, len(X))
        if self.k > len(X):
            raise ValueError("k cannot exceed the number of training samples")
        self._X_train = X
        self._y_train = y
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_X_train"):
            raise RuntimeError("fit must be called before prediction")
        X = (as_float_matrix(X) - self._mean) / self._scale
        diff = X[:, None, :] - self._X_train[None, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=2))
        nearest = np.argpartition(distances, self.k - 1, axis=1)[:, : self.k]

        lookup = {label: index for index, label in enumerate(self.classes_)}
        proba = np.zeros((len(X), len(self.classes_)))
        for row, neighbour_indices in enumerate(nearest):
            votes = np.array([lookup[label] for label in self._y_train[neighbour_indices]])
            counts = np.bincount(votes, minlength=len(self.classes_))
            proba[row] = counts / counts.sum()
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
